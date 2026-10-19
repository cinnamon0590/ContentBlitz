"""Agent 2 — Deep Research Agent.

Flow (agent ↔ ToolNode loop, as in the content-writer template):

1. ``research_agent`` plans 3–4 strategic queries and calls tools:
   * ``web_search``: cheap, can run several times (SerpAPI → Tavily → Perplexity chain, cached 30 minutes).
   * ``deep_research``: expensive synthesized research (Perplexity/Tavily advanced), capped at once per request.
   * ``trending_topics``: the trend radar — rising Google Trends searches with their growth.
     It also runs automatically for trend / "what should I write about" requests.
2. When the model stops calling tools (or the iteration cap is hit), the node synthesizes
   the results. Every retrieved source goes into the thread's source registry and gets a
   stable citation id ``[n]``. The LLM gets a numbered evidence list (without URLs) and must
   return findings that reference those ids. Ids that don't exist are dropped.
3. If no tool ran (model error, fake LLM), the agent runs its own fallback queries.
   If every provider is down, the report is labeled "model knowledge only".
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.agents import (
    AgentDeps,
    add_output,
    add_warning,
    clear_scratch,
    format_sources,
    tool_loop_step,
    tool_results,
    topic_of,
    wants_tools,
)
from src.core import prompts
from src.integrations.serp_client import SearchOutcome, topic_stems, trends_explore_url, window_label
from src.utils.observability import get_logger
from src.workflow.state_management import (
    ContentState,
    Provenance,
    ResearchFinding,
    ResearchReport,
    Source,
    SourceRegistry,
    TrendSignal,
)

log = get_logger("research")
MAX_SOURCES = 12
_TREND_REQUEST = re.compile(r"\btrend|\bwhat (should|can|do) (i|we) (write|post|blog|talk)|\b(content|topic|post) ideas\b"
                            r"|\bwhat'?s (hot|popular)\b", re.I)


def _outcome_payload(outcome: SearchOutcome) -> str:
    return json.dumps({
        "query": outcome.query,
        "provider": outcome.provider,
        "from_cache": outcome.from_cache,
        "stale": outcome.stale,
        "answer": outcome.answer[:1500],
        "results": [s.model_dump(include={"title", "url", "publisher", "snippet", "retrieved_at", "provider"})
                    for s in outcome.sources],
        "errors": outcome.errors,
    })


def make_research_tools(deps: AgentDeps) -> list:
    num = deps.config.research.get("results_per_query", 5)

    @tool
    def web_search(query: str) -> str:
        """Quick web search (cheap). Returns top results with title, URL and snippet. Use several times."""
        return _outcome_payload(deps.search.search(query, num=num))

    @tool
    def deep_research(question: str) -> str:
        """In-depth, synthesized research with citations (EXPENSIVE). Call at most once per request."""
        if deps.turn.deep_research_calls >= deps.config.guardrails.max_deep_research_calls:
            return json.dumps({"error": "deep_research limit reached for this request; use web_search instead"})
        deps.turn.deep_research_calls += 1
        outcome = deps.search.search(question, num=num, depth="advanced", order=["perplexity", "tavily", "serpapi"])
        return _outcome_payload(outcome)

    trends = deps.config.research.get("trends", {})

    @tool
    def trending_topics(niche: str) -> str:
        """Trend radar: rising Google searches related to a niche, with growth in search interest. Use once."""
        return _trend_payload(*deps.search.find_trends(
            niche, geo=trends.get("geo", "US"), window=trends.get("window", "today 12-m"),
            limit=trends.get("max_signals", 8)))

    return [web_search, deep_research, trending_topics]


def _trend_payload(niche: str, signals: list[TrendSignal]) -> str:
    return json.dumps({"niche": niche, "signals": [s.model_dump() for s in signals], "provider": "google_trends"})


def wants_trends(state: ContentState) -> bool:
    request = state.get("request") or {}
    return request.get("template") == "trend report" or bool(_TREND_REQUEST.search(state.get("user_input", "")))


def trend_source(niche: str, signals: list[TrendSignal], geo: str, window: str) -> Source:
    """The Google Trends explore page as a citable source, so growth figures carry a real [n]."""
    rising = ", ".join(f"{s.query} ({s.growth})" for s in signals if s.kind == "rising")[:400]
    top = ", ".join(s.query for s in signals if s.kind == "top")[:200]
    snippet = f"Rising related searches: {rising or 'none'}. Top related searches: {top or 'none'}."
    return Source(title=f"Google Trends: related searches for “{niche}” ({geo}, {window_label(window)})",
                  url=trends_explore_url(niche, geo, window), publisher="trends.google.com", snippet=snippet,
                  provider="serpapi")


class ResearchSynthesis(BaseModel):
    summary: str = ""
    key_findings: list[ResearchFinding] = Field(default_factory=list)
    statistics: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


def fallback_queries(topic: str, year: int | None = None) -> list[str]:
    year = year or datetime.now().year
    return [f"{topic} statistics {year}", f"{topic} trends {year}", f"{topic} best practices for small businesses"]


def render_research_markdown(report: ResearchReport) -> str:
    lines = [f"# Research Brief: {report.topic}", "", "## Executive Summary", "", report.summary or "_No summary._", ""]
    if report.key_findings:
        lines += ["## Key Findings", ""]
        lines += [f"- {f.claim} " + "".join(f"[{i}]" for i in f.source_ids) for f in report.key_findings]
        lines.append("")
    if report.statistics:
        lines += ["## Notable Statistics", ""] + [f"- {s}" for s in report.statistics] + [""]
    if report.trends:
        lines += ["## Emerging Trends", ""] + [f"- {t}" for t in report.trends] + [""]
    if report.trend_signals:
        lines += render_trend_radar(report.trend_signals, report.trend_source_id) + [""]
    if report.keywords:
        lines += ["## Suggested Keywords", "", ", ".join(report.keywords), ""]
    if report.queries:
        lines += ["## Research Queries", ""] + [f"- {q}" for q in report.queries] + [""]
    lines.append(report.provenance.to_markdown())
    return "\n".join(lines)


def drop_off_topic(topic: str, payloads: list[dict]) -> list[dict]:
    """Remove results that share no meaningful word with the topic (search engines pad niche queries with
    loosely matching pages). Keeps everything if the filter would leave nothing."""
    stems = topic_stems(topic)
    if not stems:
        return payloads
    filtered = [{**p, "results": [r for r in p.get("results", [])
                                  if stems & topic_stems(f"{r.get('title', '')} {r.get('snippet', '')}")]}
                for p in payloads]
    return filtered if any(p["results"] for p in filtered) else payloads


def render_trend_radar(signals: list[TrendSignal], source_id: int) -> list[str]:
    cite = f" [{source_id}]" if source_id else ""
    rising = [s for s in signals if s.kind == "rising"]
    lines = ["## Trend Radar", "", f"Searches gaining interest on Google (\"Breakout\" = more than +5000%){cite}.", ""]
    if rising:
        lines += ["| Rising search | Growth |", "|---|---|"] + [f"| {s.query} | {s.growth}{cite} |" for s in rising]
    top = [s.query for s in signals if s.kind == "top"]
    if top:
        lines += ["", f"Most-searched related topics: {', '.join(top)}{cite}."]
    return lines


def _strip_invalid_ids(text: str, valid: set[int]) -> str:
    import re

    return re.sub(r"\[(\d{1,3})\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", text)


def synthesize(deps: AgentDeps, state: ContentState, payloads: list[dict], queries: list[str],
               trend_payloads: list[dict] | None = None) -> dict:
    topic = topic_of(state)
    warnings: list[str] = []
    trend_cfg = deps.config.research.get("trends", {})
    geo, window = trend_cfg.get("geo", "US"), trend_cfg.get("window", "today 12-m")
    if trend_payloads:
        niche = trend_payloads[0].get("niche") or topic
        signals = [TrendSignal.model_validate(s) for p in trend_payloads for s in p.get("signals", [])]
    elif wants_trends(state):
        niche, signals = deps.search.find_trends(topic, geo=geo, window=window, limit=trend_cfg.get("max_signals", 8))
    else:
        niche, signals = topic, []
    if not any(p.get("results") for p in payloads):
        # The model didn't (or couldn't) search: run the agent's own strategic queries.
        queries = queries or fallback_queries(topic)
        payloads = [json.loads(_outcome_payload(deps.search.search(q))) for q in queries]

    payloads = drop_off_topic(topic, payloads)
    registry = SourceRegistry(state.get("sources"))
    used: list[Source] = []
    answers: list[str] = []
    cached_flags: list[bool] = []
    errors: list[str] = []
    for payload in payloads:
        errors.extend(payload.get("errors", []))
        if payload.get("answer"):
            answers.append(payload["answer"])
        for item in payload.get("results", []):
            source = registry.add(Source.model_validate(item))
            if all(s.id != source.id for s in used) and len(used) < MAX_SOURCES:
                used.append(source)
                cached_flags.append(bool(payload.get("from_cache")))

    trend_id = 0
    if signals:
        trend = registry.add(trend_source(niche, signals, geo, window))
        trend_id = trend.id
        if all(s.id != trend_id for s in used):
            used.append(trend)  # beyond MAX_SOURCES on purpose: the radar must stay citable
            cached_flags.append(all(cached_flags) if cached_flags else False)
    elif wants_trends(state):
        warnings.append("Trend radar unavailable: Google Trends needs a SerpAPI key." if not deps.search.trends_available
                        else f"Trend radar: Google Trends has no related searches for “{topic}” or a shorter version "
                             "of it, so the report has no trend table.")

    if not used:
        grounding = "model_knowledge_only"
        warnings.append("Live research unavailable (all providers failed); report is based on model knowledge. "
                        + "; ".join(sorted(set(errors)))[:300])
    elif all(cached_flags):
        grounding = "cached_research"
    else:
        grounding = "live_research"

    valid_ids = {s.id for s in used}
    user = (f"TOPIC: {topic}\nAUDIENCE: {(state.get('brand') or {}).get('audience', 'small businesses')}\n\n"
            f"SOURCES:\n{format_sources(used)}\n\nPROVIDER SYNTHESES (may reference other numbering; "
            f"re-cite with the SOURCES ids above):\n" + "\n---\n".join(a[:1500] for a in answers[:2]))
    synthesis = deps.llm.structured(ResearchSynthesis, prompts.RESEARCH_SYNTHESIS, user, tier="smart")
    if synthesis is None:
        warnings.append("Research synthesis model failed; showing source snippets.")
        synthesis = ResearchSynthesis(
            summary=" ".join(f"{s.snippet[:200]} [{s.id}]" for s in used[:3]),
            key_findings=[ResearchFinding(claim=s.snippet[:200] or s.title, source_ids=[s.id]) for s in used[:6]],
        )

    findings = []
    for finding in synthesis.key_findings:
        ids = [i for i in finding.source_ids if i in valid_ids]
        if ids or grounding == "model_knowledge_only":
            findings.append(ResearchFinding(claim=_strip_invalid_ids(finding.claim, valid_ids), source_ids=ids))
    statistics = [_strip_invalid_ids(s, valid_ids) for s in synthesis.statistics if grounding != "model_knowledge_only"]
    cited_ids = sorted({i for f in findings for i in f.source_ids})
    registry.mark_used(cited_ids or sorted(valid_ids), "research")

    provenance = Provenance(
        sources=[registry.get(s.id) for s in used],
        generated_by=deps.llm.describe("smart"),
        grounding=grounding,
        notes=[f"Research providers used: {', '.join(sorted({s.provider for s in used})) or 'none'}"],
    )
    report = ResearchReport(
        topic=topic, queries=queries, summary=_strip_invalid_ids(synthesis.summary, valid_ids),
        key_findings=findings, statistics=statistics, trends=synthesis.trends, keywords=synthesis.keywords,
        trend_signals=signals, trend_source_id=trend_id, provenance=provenance,
    )
    report.markdown = render_research_markdown(report)
    return {
        "research": report.model_dump(),
        "sources": registry.dump(),
        "agent_scratch": clear_scratch(),
        "outputs_this_turn": add_output(state, "research"),
        "warnings": add_warning(state, *warnings),
    }


def make_research_agent(deps: AgentDeps, tools: list | None = None):
    tools = tools or make_research_tools(deps)

    def research_agent(state: ContentState) -> dict:
        topic = topic_of(state)
        brief = (f"Research topic: {topic}\nUser request: {state.get('user_input', '')}\n"
                 f"Current year: {datetime.now().year}. Plan your queries, then search.")
        ai, new_messages = tool_loop_step(deps, state, prompts.RESEARCH_AGENT, brief, tools)
        if wants_tools(ai):
            return {"agent_scratch": new_messages}
        scratch = [*state.get("agent_scratch", []), *new_messages]
        payloads = [*tool_results(scratch, "web_search"), *tool_results(scratch, "deep_research")]
        queries = [p["query"] for p in payloads if p.get("query")]
        return synthesize(deps, state, payloads, queries, tool_results(scratch, "trending_topics"))

    return research_agent
