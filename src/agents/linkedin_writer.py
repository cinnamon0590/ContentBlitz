"""Agent 4 — LinkedIn Post Writer.

Flow:
1. **Trends loop** (agent ↔ ToolNode): ``trending_hashtags`` searches for current angles
   and hashtag candidates on the topic.
2. **Compose**: structured output (hook, body, hashtags).
3. **Deterministic platform formatting**:
   * strip links from the body (they hurt reach; sources go into a suggested first comment),
   * remove ``[n]`` markers,
   * normalize and dedupe hashtags into 8–12 CamelCase tags,
   * split long paragraphs, keep the post within 1600 characters.
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
    advance_pending,
    brand_of,
    clear_scratch,
    format_sources,
    request_of,
    tool_loop_step,
    tool_results,
    topic_of,
    wants_tools,
)
from src.core import prompts
from src.core.templates import outline, suggest_template
from src.utils import content_optimization as co
from src.utils.guardrails import strip_links
from src.workflow.state_management import (
    ContentState,
    LinkedInPost,
    Provenance,
    ResearchReport,
    SourceRegistry,
)


class LinkedInDraft(BaseModel):
    hook: str
    body: str
    hashtags: list[str] = Field(default_factory=list)


def make_linkedin_tools(deps: AgentDeps) -> list:
    @tool
    def trending_hashtags(topic: str) -> str:
        """Find trending LinkedIn angles and hashtag candidates for a topic (cheap web search)."""
        outcome = deps.search.search(f"{topic} LinkedIn trends {datetime.now().year}", num=5)
        text = " ".join(f"{s.title} {s.snippet}" for s in outcome.sources)
        found = co.extract_hashtags(text)
        related = deps.search.keyword_ideas(topic).get("related_searches", [])[:6]
        candidates = list(dict.fromkeys([*found, *(co.normalize_hashtag(r) for r in related)]))
        return json.dumps({"topic": topic, "angles": [s.title for s in outcome.sources][:5],
                           "hashtag_candidates": [c for c in candidates if c][:15], "provider": outcome.provider})

    return [trending_hashtags]


def build_first_comment(post_sources, limit: int = 3) -> str:
    if not post_sources:
        return ""
    lines = ["📚 Sources for this post:"]
    lines += [f"• {s.title or s.publisher} — {s.url}" for s in post_sources[:limit]]
    return "\n".join(lines)


def compose_linkedin(deps: AgentDeps, state: ContentState, trend_payloads: list[dict]) -> dict:
    brand = brand_of(state)
    request = request_of(state)
    research = ResearchReport.model_validate(state.get("research") or {"topic": topic_of(state)})
    registry = SourceRegistry(state.get("sources"))
    sources = research.provenance.sources
    template = suggest_template("linkedin", state.get("user_input", ""), request.template)
    angles = [a for p in trend_payloads for a in p.get("angles", [])]
    candidates = list(dict.fromkeys(c for p in trend_payloads for c in p.get("hashtag_candidates", [])))

    system = prompts.with_brand(prompts.LINKEDIN_WRITER, brand, outline=outline("linkedin", template))
    findings = "\n".join(f"- {f.claim} {f.source_ids}" for f in research.key_findings[:6])
    user = (f"TOPIC: {request.topic or research.topic}\nTEMPLATE: {template}\nAUDIENCE: {request.audience or brand.audience}"
            f"\nTRENDING ANGLES: {' | '.join(angles[:5]) or 'n/a'}\nHASHTAG IDEAS: {' '.join(candidates) or 'n/a'}\n\n"
            f"RESEARCH SUMMARY: {research.summary}\nKEY FINDINGS:\n{findings}\n\nSOURCES:\n{format_sources(sources)}")
    if blog := state.get("blog"):
        user += f"\n\nRELATED BLOG (keep the voice consistent): {blog.get('title', '')} — {blog.get('meta_description', '')}"
    previous = state.get("linkedin") or {}
    feedback = state.get("revision_feedback", "")
    if feedback and state.get("quality_target") == "linkedin" and previous:
        user += prompts.REVISION_NOTE.format(feedback=feedback, previous=previous.get("text", ""))
    elif request.refine_target == "linkedin" and previous:
        user += prompts.REVISION_NOTE.format(feedback=request.refine_instructions or state.get("user_input", ""),
                                             previous=previous.get("text", ""))

    warnings: list[str] = []
    draft = deps.llm.structured(LinkedInDraft, system, user, tier="smart")
    if draft is None:
        try:
            text = deps.llm.text(system + "\nReturn only the post text.", user, tier="smart")
        except RuntimeError as exc:
            return {"warnings": add_warning(state, f"LinkedIn writer unavailable: {exc}"),
                    "agent_scratch": clear_scratch(), "quality_target": "",
                    "pending_formats": advance_pending(state, "linkedin")}
        hook, _, body = text.strip().partition("\n")
        draft = LinkedInDraft(hook=hook, body=body, hashtags=co.extract_hashtags(text))
        warnings.append("Structured output failed; hook/hashtags were derived automatically.")

    hook = draft.hook.strip()
    body = draft.body.strip()
    text = body if body.startswith(hook) else f"{hook}\n\n{body}"
    text, link_count = strip_links(text)
    text = re.sub(r"\s?\[\d{1,3}\]", "", text)
    hashtags = draft.hashtags or candidates
    if len(set(h.lower() for h in hashtags)) < 8:  # top up from trend candidates
        hashtags = [*hashtags, *candidates]
    text, tags, notes = co.format_linkedin(text, hashtags)
    if link_count:
        notes.append(f"Moved {link_count} link(s) out of the post body (sources are in the first comment).")

    used_ids = sorted({i for f in research.key_findings for i in f.source_ids}) or [s.id for s in sources]
    post_sources = registry.by_ids(used_ids[:5])
    registry.mark_used([s.id for s in post_sources], "linkedin")
    provenance = Provenance(
        sources=post_sources,
        generated_by=deps.llm.describe("smart"),
        grounding=research.provenance.grounding if sources else "model_knowledge_only",
        revision_count=state.get("revision_count", 0),
        notes=notes,
    )
    post = LinkedInPost(text=text, hook=hook, hashtags=tags, first_comment=build_first_comment(post_sources),
                        template=template, provenance=provenance)
    return {
        "linkedin": post.model_dump(),
        "sources": registry.dump(),
        "agent_scratch": clear_scratch(),
        "quality_target": "linkedin",
        "revision_feedback": "",
        "outputs_this_turn": add_output(state, "linkedin"),
        "warnings": add_warning(state, *warnings),
    }


def make_linkedin_writer(deps: AgentDeps, tools: list | None = None):
    tools = tools or make_linkedin_tools(deps)

    def linkedin_writer(state: ContentState) -> dict:
        request = request_of(state)
        revising = bool(state.get("revision_feedback")) or (request.refine_target == "linkedin" and state.get("linkedin"))
        payloads: list[dict] = []
        if not revising:
            ai, new_messages = tool_loop_step(deps, state, prompts.LINKEDIN_TRENDS, f"Topic: {topic_of(state)}", tools)
            if wants_tools(ai):
                return {"agent_scratch": new_messages}
            payloads = tool_results([*state.get("agent_scratch", []), *new_messages], "trending_hashtags")
        return compose_linkedin(deps, state, payloads)

    return linkedin_writer
