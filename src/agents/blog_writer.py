"""Agent 3 — SEO Blog Writer.

Flow:
1. **Keyword loop** (agent ↔ ToolNode): the model calls ``keyword_research`` (SerpAPI related
   searches + People-also-ask) to find secondary keywords and FAQ questions. It's skipped on
   revisions and refinements, because the keyword data is already there.
2. **Compose**: structured output (title, meta description, Markdown, secondary keywords,
   internal-link ideas) grounded in the numbered research sources.
3. **Deterministic SEO post-processing**:
   * fit the meta description to 150–160 characters,
   * slugify the URL,
   * make sure exactly one H1 sits at the top,
   * remove invented URLs and citation ids,
   * fall back to H2-based internal-link suggestions,
   * append Sources & Provenance from the source registry.

The Quality Gate scores the result next and can send it back with suggestions.
"""

from __future__ import annotations

import json
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.agents import (
    AgentDeps,
    add_output,
    add_warning,
    brand_of,
    clear_scratch,
    format_sources,
    request_of,
    scrub_body,
    tool_loop_step,
    tool_results,
    topic_of,
    wants_tools,
)
from src.core import prompts
from src.core.templates import outline, suggest_template
from src.utils import content_optimization as co
from src.workflow.state_management import BlogPost, ContentState, Provenance, ResearchReport, SourceRegistry


class BlogDraft(BaseModel):
    title: str
    meta_description: str = ""
    markdown: str
    secondary_keywords: list[str] = Field(default_factory=list)
    internal_link_suggestions: list[str] = Field(default_factory=list)


def make_blog_tools(deps: AgentDeps) -> list:
    @tool
    def keyword_research(seed: str) -> str:
        """Find related searches and 'People also ask' questions for a seed keyword (SEO research)."""
        return json.dumps({"seed": seed, **deps.search.keyword_ideas(seed)})

    return [keyword_research]


def _keyword_context(payloads: list[dict], research: ResearchReport) -> tuple[str, list[str]]:
    related = list(dict.fromkeys([k for p in payloads for k in p.get("related_searches", [])] + research.keywords))
    questions = list(dict.fromkeys(q for p in payloads for q in p.get("people_also_ask", [])))
    text = (f"Related searches / keyword ideas: {', '.join(related[:15]) or 'n/a'}\n"
            f"People also ask: {' | '.join(questions[:6]) or 'n/a'}")
    return text, related


def _strip_existing_provenance(markdown: str) -> str:
    return markdown.split("## Sources & Provenance")[0].rstrip()


def compose_blog(deps: AgentDeps, state: ContentState, keyword_payloads: list[dict]) -> dict:
    brand = brand_of(state)
    request = request_of(state)
    research = ResearchReport.model_validate(state.get("research") or {"topic": topic_of(state)})
    registry = SourceRegistry(state.get("sources"))
    sources = research.provenance.sources if research.provenance.sources else []
    valid_ids = {s.id for s in sources}
    keyword = request.target_keyword or research.topic
    template = suggest_template("blog", state.get("user_input", ""), request.template)
    keyword_text, related = _keyword_context(keyword_payloads, research)

    system = prompts.with_brand(prompts.BLOG_WRITER, brand, outline=outline("blog", template))
    findings = "\n".join(f"- {f.claim} {''.join(f'[{i}]' for i in f.source_ids)}" for f in research.key_findings)
    stats = "\n".join(f"- {s}" for s in research.statistics)
    user = (f"TOPIC: {request.topic or research.topic}\nTARGET KEYWORD: {keyword}\nAUDIENCE: "
            f"{request.audience or brand.audience}\nTONE: {request.tone or brand.tone}\nTEMPLATE: {template}\n"
            f"{keyword_text}\n\nRESEARCH SUMMARY: {research.summary}\nKEY FINDINGS:\n{findings}\n"
            f"STATISTICS:\n{stats}\n\nSOURCES:\n{format_sources(sources)}")

    previous = state.get("blog") or {}
    feedback = state.get("revision_feedback", "")
    if feedback and state.get("quality_target") == "blog" and previous:
        user += prompts.REVISION_NOTE.format(feedback=feedback,
                                             previous=_strip_existing_provenance(previous.get("markdown", ""))[:9000])
    elif request.refine_target == "blog" and previous:
        user += prompts.REVISION_NOTE.format(feedback=request.refine_instructions or state.get("user_input", ""),
                                             previous=_strip_existing_provenance(previous.get("markdown", ""))[:9000])

    warnings: list[str] = []
    draft = deps.llm.structured(BlogDraft, system, user, tier="smart")
    if draft is None:
        try:
            markdown = deps.llm.text(system + "\nReturn only the Markdown post.", user, tier="smart")
        except RuntimeError as exc:
            return {"warnings": add_warning(state, f"Blog writer unavailable: {exc}"), "agent_scratch": clear_scratch(),
                    "quality_target": ""}
        h1 = re.search(r"^#\s+(.+)$", markdown, re.M)
        draft = BlogDraft(title=h1.group(1).strip() if h1 else (request.topic or keyword).title(), markdown=markdown)
        warnings.append("Structured output failed; blog metadata was derived automatically.")

    markdown, notes = scrub_body(draft.markdown, registry, valid_ids)
    markdown = re.sub(r"^#\s+.+\n+", "", markdown.lstrip(), count=1) if markdown.lstrip().startswith("# ") else markdown
    markdown = re.sub(r"^#\s+", "## ", markdown, flags=re.M)  # demote any extra H1s
    title = draft.title.strip().strip("#").strip()
    byline = f"*By the {brand.name} team · Fact-check and add your expert reviewer before publishing.*"
    body = f"# {title}\n\n{byline}\n\n{markdown.strip()}"

    first_para = next((p for p in co.strip_markdown(markdown).split("\n") if len(p) > 60), "")
    meta = co.fit_meta_description(draft.meta_description or first_para, fallback=first_para)
    cited = sorted(co.citation_ids(body) & valid_ids)
    registry.mark_used(cited, "blog")
    provenance = Provenance(
        sources=registry.by_ids(cited) or [registry.get(s.id) or s for s in sources],
        generated_by=deps.llm.describe("smart"),
        grounding=research.provenance.grounding if sources else "model_knowledge_only",
        revision_count=state.get("revision_count", 0),
        notes=notes,
    )
    blog = BlogPost(
        title=title,
        slug=co.slugify(keyword or title),  # Moz: short, hyphenated, lowercase, keyword-bearing URL
        meta_description=meta,
        target_keyword=keyword,
        secondary_keywords=draft.secondary_keywords or related[:6],
        markdown=f"{body}\n\n{provenance.to_markdown()}",
        internal_link_suggestions=draft.internal_link_suggestions or co.suggest_internal_links(body),
        template=template,
        provenance=provenance,
    )
    return {
        "blog": blog.model_dump(),
        "sources": registry.dump(),
        "agent_scratch": clear_scratch(),
        "quality_target": "blog",
        "revision_feedback": "",
        "outputs_this_turn": add_output(state, "blog"),
        "warnings": add_warning(state, *warnings),
    }


def make_blog_writer(deps: AgentDeps, tools: list | None = None):
    tools = tools or make_blog_tools(deps)

    def blog_writer(state: ContentState) -> dict:
        request = request_of(state)
        revising = bool(state.get("revision_feedback")) or (request.refine_target == "blog" and state.get("blog"))
        if not revising:
            keyword = request.target_keyword or topic_of(state)
            ai, new_messages = tool_loop_step(deps, state, prompts.BLOG_KEYWORDS,
                                              f"Topic: {topic_of(state)}\nTarget keyword: {keyword}", tools)
            if wants_tools(ai):
                return {"agent_scratch": new_messages}
            scratch = [*state.get("agent_scratch", []), *new_messages]
            payloads = tool_results(scratch, "keyword_research")
            if not payloads:  # the model skipped the tool → do one lookup ourselves
                payloads = [{"seed": keyword, **deps.search.keyword_ideas(keyword)}]
        else:
            payloads = []
        return compose_blog(deps, state, payloads)

    return blog_writer
