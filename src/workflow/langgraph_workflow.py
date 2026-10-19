"""LangGraph assembly: nodes, edges, quality gate, guardrail nodes and checkpointer.

``build_graph(deps, checkpointer)`` wires the six agents into one ``StateGraph``.
Following the content-writer template:

* a router with conditional edges,
* agent ↔ ``ToolNode`` loops driven by ``should_continue`` functions (on the
  ``agent_scratch`` channel, so tool chatter stays out of the user-visible memory),
* a checkpointer keyed by ``thread_id`` for multi-turn memory.

ContentBlitz adds these on top:

* **input_guardrail**: blocks injection/policy/moderation hits, redacts PII, resets per-turn state.
* **quality_gate**: deterministic scoring, plus a revision loop back to the writer
  (at most ``max_revisions``, and only while under budget).
* **finalize**: output moderation, the user-facing summary, and usage/cost reporting.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents import AgentDeps, add_warning, advance_pending, brand_of, clear_scratch
from src.agents.blog_writer import make_blog_tools, make_blog_writer
from src.agents.content_strategist import make_content_strategist
from src.agents.image_generator import make_image_generator
from src.agents.linkedin_writer import make_linkedin_tools, make_linkedin_writer
from src.agents.query_handler import make_general_handler, make_post_research_router, make_query_handler
from src.agents.research_agent import make_research_agent, make_research_tools
from src.core import router
from src.utils.guardrails import check_input, check_output
from src.utils.observability import get_logger
from src.workflow.state_management import (
    BlogPost,
    BrandProfile,
    ContentState,
    LinkedInPost,
    QualityReport,
    SourceRegistry,
    StrategyReport,
)

log = get_logger("workflow")
LABELS = {"blog": "SEO blog post", "linkedin": "LinkedIn post", "report": "research report"}


# --------------------------------------------------------------------------------------
# Guardrail / gate / finalize nodes
# --------------------------------------------------------------------------------------
def make_input_guardrail(deps: AgentDeps):
    def input_guardrail(state: ContentState) -> dict:
        deps.turn.reset()
        g = deps.config.guardrails
        check = check_input(state.get("user_input", ""), g.max_input_chars, deps.moderator)
        update: dict = {
            "outputs_this_turn": [], "warnings": [], "errors": [], "final_response": "",
            "agent_scratch": clear_scratch(), "revision_feedback": "", "quality_target": "",
            "revision_count": 0, "pending_formats": [], "next_after_research": "",
        }
        if not state.get("brand"):
            update["brand"] = BrandProfile().model_dump()
        if check.redactions:
            update["warnings"] = [f"Removed personal data ({', '.join(check.redactions)}) from your message "
                                  "before processing."]
        if not check.allowed:
            log.info("input blocked: %s", check.category)
            update.update(route="blocked", final_response=check.user_message,
                          messages=[HumanMessage(f"[message blocked by guardrails: {check.category}]")])
        else:
            update.update(user_input=check.text, messages=[HumanMessage(check.text)])
        return update

    return input_guardrail


def make_quality_gate(deps: AgentDeps):
    def quality_gate(state: ContentState) -> dict:
        target = state.get("quality_target", "")
        artifact = state.get(target) if target else None
        if not artifact:  # the writer failed; skip scoring
            return {"revision_feedback": "", "quality_target": "",
                    "pending_formats": advance_pending(state, target) if target else state.get("pending_formats", [])}
        brand = brand_of(state)
        validator = deps.validator
        if target == "blog":
            known = {s["id"] for s in SourceRegistry(state.get("sources")).dump()}
            report = validator.score_blog(BlogPost.model_validate(artifact), brand, known)
        elif target == "linkedin":
            report = validator.score_linkedin(LinkedInPost.model_validate(artifact), brand)
        else:
            report = validator.score_report(StrategyReport.model_validate(artifact), brand)
        quality = {**state.get("quality", {}), target: report.model_dump()}

        revisions = state.get("revision_count", 0)
        over_budget = deps.tracker.over_budget(deps.config.guardrails.request_budget_usd)
        if not report.passed and revisions < deps.config.guardrails.max_revisions and not over_budget:
            log.info("%s scored %.1f — revision %d", target, report.overall, revisions + 1)
            feedback = "\n".join(f"- {s}" for s in report.suggestions[:8]) or (
                f"- Overall quality {report.overall:.0f}/100 is below the {validator.pass_threshold:.0f} target: "
                "sharpen the hook, add concrete examples, and tighten the structure.")
            return {"quality": quality, "revision_count": revisions + 1, "revision_feedback": feedback,
                    "quality_target": target}

        warnings = []
        if not report.passed:
            reason = "budget limit reached" if over_budget else f"after {revisions} revision(s)"
            warnings.append(f"{LABELS.get(target, target).capitalize()} scored {report.overall:.0f}/100 ({reason}); "
                            "see the Content Dashboard for improvement suggestions.")
        pending = advance_pending(state, target)
        if over_budget and pending:
            warnings.append(f"Per-request budget reached; skipped: {', '.join(pending)}.")
            pending = []
        return {"quality": quality, "revision_count": 0, "revision_feedback": "", "quality_target": "",
                "pending_formats": pending, "warnings": add_warning(state, *warnings)}

    return quality_gate


def _quality_line(state: ContentState, kind: str) -> str:
    q = (state.get("quality") or {}).get(kind)
    if not q:
        return ""
    report = QualityReport.model_validate(q)
    return f"**Quality score:** {report.overall:.0f}/100 {'✅' if report.passed else '⚠️'}"


def render_response(state: ContentState) -> str:
    parts: list[str] = []
    outputs = state.get("outputs_this_turn", [])
    if "report" in outputs and state.get("report"):
        parts.append(f"## 📊 Research Report\n{_quality_line(state, 'report')}\n\n{state['report']['markdown']}")
    elif "research" in outputs and state.get("research") and not {"blog", "linkedin", "image"} & set(outputs):
        parts.append(f"## 🔎 Research\n\n{state['research']['markdown']}")
    if "blog" in outputs and state.get("blog"):
        b = state["blog"]
        parts.append(f"## 📝 SEO Blog Post\n**Title:** {b['title']}  \n**Meta description:** {b['meta_description']}"
                     f"  \n**URL slug:** /blog/{b['slug']}  \n**Target keyword:** {b['target_keyword']}  \n"
                     f"{_quality_line(state, 'blog')}\n\n---\n\n{b['markdown']}")
    if "linkedin" in outputs and state.get("linkedin"):
        li = state["linkedin"]
        comment = f"\n\n**Suggested first comment:**\n\n{li['first_comment']}" if li.get("first_comment") else ""
        alternatives = [o for o in li.get("hook_options", []) if o["text"] != li.get("hook")]
        if alternatives:
            comment += ("\n\n**Alternative hooks** (swap one in from the Content Dashboard):\n\n"
                        + "\n".join(f"- _{o['style']}_ ({o['score']:.0f}/100): {o['text']}" for o in alternatives))
        parts.append(f"## 💼 LinkedIn Post ({len(li['text'])} characters, {len(li['hashtags'])} hashtags)\n"
                     f"{_quality_line(state, 'linkedin')}\n\n{li['text']}{comment}\n\n"
                     f"_{li['provenance'].get('grounding', '').replace('_', ' ')}_")
    if "image" in outputs and state.get("image"):
        img = state["image"]
        parts.append(f"## 🖼️ Image\n**Generated with:** {img['provider']}  \n**Enhanced prompt:** "
                     f"{img['enhanced_prompt']}  \n**Alt text:** {img['alt_text']}  \n**Saved to:** `{img['path']}`")
    if not parts:
        parts.append("I couldn't produce content for that request. Please try rephrasing.")
    return "\n\n---\n\n".join(parts)


def make_finalize(deps: AgentDeps):
    def finalize(state: ContentState) -> dict:
        update: dict = {}
        warnings = list(state.get("warnings", []))
        texts = {"blog": ("blog", "markdown"), "linkedin": ("linkedin", "text"), "report": ("report", "markdown")}
        for kind, (key, field) in texts.items():
            if kind in state.get("outputs_this_turn", []) and state.get(key):
                safe, categories = check_output(state[key][field], deps.moderator)
                if not safe:
                    update[key] = {}
                    warnings.append(f"The generated {LABELS[kind]} was withheld by the safety filter "
                                    f"({', '.join(categories)}).")
        merged = {**state, **update}
        response = state.get("final_response") or render_response(merged)
        if warnings:
            response += "\n\n" + "\n".join(f"> ⚠️ {w}" for w in dict.fromkeys(warnings))
        usage = deps.tracker.summary() if deps.tracker else {}
        update.update(final_response=response, warnings=list(dict.fromkeys(warnings)), usage=usage,
                      messages=[AIMessage(response)])
        return update

    return finalize


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------
def build_graph(deps: AgentDeps, checkpointer=None):
    research_tools = make_research_tools(deps)
    blog_tools = make_blog_tools(deps)
    linkedin_tools = make_linkedin_tools(deps)

    g = StateGraph(ContentState)
    g.add_node("input_guardrail", make_input_guardrail(deps))
    g.add_node("query_handler", make_query_handler(deps))
    g.add_node("general", make_general_handler(deps))
    g.add_node("research_agent", make_research_agent(deps, research_tools))
    g.add_node("research_tools", ToolNode(research_tools, messages_key="agent_scratch"))
    g.add_node("post_research_router", make_post_research_router(deps))
    g.add_node("blog_writer", make_blog_writer(deps, blog_tools))
    g.add_node("blog_tools", ToolNode(blog_tools, messages_key="agent_scratch"))
    g.add_node("linkedin_writer", make_linkedin_writer(deps, linkedin_tools))
    g.add_node("linkedin_tools", ToolNode(linkedin_tools, messages_key="agent_scratch"))
    g.add_node("image_generator", make_image_generator(deps))
    g.add_node("content_strategist", make_content_strategist(deps))
    g.add_node("quality_gate", make_quality_gate(deps))
    g.add_node("finalize", make_finalize(deps))

    writers = ["blog_writer", "linkedin_writer", "image_generator", "content_strategist"]
    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", router.route_input, ["query_handler", "finalize"])
    g.add_conditional_edges("query_handler", router.route_initial,
                            ["research_agent", "general", "finalize", *writers])
    g.add_edge("general", "finalize")

    g.add_conditional_edges("research_agent", router.should_continue_tools,
                            {"tools": "research_tools", "done": "post_research_router"})
    g.add_edge("research_tools", "research_agent")
    g.add_conditional_edges("post_research_router", router.route_after_research, writers)

    g.add_conditional_edges("blog_writer", router.should_continue_tools, {"tools": "blog_tools", "done": "quality_gate"})
    g.add_edge("blog_tools", "blog_writer")
    g.add_conditional_edges("linkedin_writer", router.should_continue_tools,
                            {"tools": "linkedin_tools", "done": "quality_gate"})
    g.add_edge("linkedin_tools", "linkedin_writer")
    g.add_edge("content_strategist", "quality_gate")

    g.add_conditional_edges("quality_gate", router.route_after_quality, [*writers, "finalize"])
    g.add_conditional_edges("image_generator", router.route_next_pending, [*writers, "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def make_checkpointer(config):
    """``memory`` (default; in-process) or ``sqlite`` (durable across restarts)."""
    if config.checkpointer == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = config.resolve_path(config.sqlite_path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    return InMemorySaver()


def make_studio_graph():
    """Factory for ``langgraph dev`` / LangGraph Platform (see ``langgraph.json``).

    The platform provides its own persistence, so no checkpointer is attached here.
    In Studio, invoke with input ``{"user_input": "..."}``.
    """
    from src.core.config import get_config

    return build_graph(AgentDeps.from_config(get_config()))
