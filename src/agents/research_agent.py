"""Agent 2 — Deep Research Agent.

Flow (agent ↔ ToolNode loop, as in the content-writer template):

1. ``research_agent`` plans 3–4 strategic queries and calls tools:
   * ``web_search``: cheap, can run several times (SerpAPI → Tavily → Perplexity chain, cached 30 minutes).
   * ``deep_research``: expensive synthesized research (Perplexity/Tavily advanced), capped at once per request.
2. When the model stops calling tools (or the iteration cap is hit), the node synthesizes
   the results. Every retrieved source goes into the thread's source registry and gets a
   stable citation id ``[n]``. The LLM gets a numbered evidence list (without URLs) and must
   return findings that reference those ids. Ids that don't exist are dropped.
3. If no tool ran (model error, fake LLM), the agent runs its own fallback queries.
   If every provider is down, the report is labeled "model knowledge only".
"""

from __future__ import annotations

import json
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
from src.integrations.serp_client import SearchOutcome
from src.utils.observability import get_logger
from src.workflow.state_management import (
    ContentState,
    Provenance,
    ResearchFinding,
    ResearchReport,
    Source,
    SourceRegistry,
)

log = get_logger("research")
MAX_SOURCES = 12


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

    return [web_search, deep_research]


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
    if report.keywords:
        lines += ["## Suggested Keywords", "", ", ".join(report.keywords), ""]
    if report.queries:
        lines += ["## Research Queries", ""] + [f"- {q}" for q in report.queries] + [""]
    lines.append(report.provenance.to_markdown())
    return "\n".join(lines)


def _strip_invalid_ids(text: str, valid: set[int]) -> str:
    import re

    return re.sub(r"\[(\d{1,3})\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", text)


def synthesize(deps: AgentDeps, state: ContentState, payloads: list[dict], queries: list[str]) -> dict:
    topic = topic_of(state)
    warnings: list[str] = []
    if not any(p.get("results") for p in payloads):
        # The model didn't (or couldn't) search: run the agent's own strategic queries.
        queries = queries or fallback_queries(topic)
        payloads = [json.loads(_outcome_payload(deps.search.search(q))) for q in queries]

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
        provenance=provenance,
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
        payloads = tool_results(scratch)
        queries = [p["query"] for p in payloads if p.get("query")]
        return synthesize(deps, state, payloads, queries)

    return research_agent
