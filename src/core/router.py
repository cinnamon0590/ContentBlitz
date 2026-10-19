"""Conditional-edge functions for the LangGraph workflow.

These are pure functions of state, so they're trivially unit-testable. Each one
returns the name of the next node.

Routing overview::

    input_guardrail ──blocked──────────────────────────────────────────────▶ finalize
          │ ok
    query_handler ──research─▶ research_agent ⇄ research_tools ─▶ post_research_router
          │                                                             │
          ├─blog──────▶ blog_writer ⇄ blog_tools ─────┐                 │
          ├─linkedin──▶ linkedin_writer ⇄ linkedin_tools ─┤◀────────────┤
          ├─report────▶ content_strategist ───────────┤                 │
          │                                            ▼                 │
          │                                       quality_gate ──revise──▶ (same writer)
          │                                            │ pass / max revisions
          ├─image─────▶ image_generator ─▶ next pending format (campaign) or finalize
          └─general───▶ general ─────────────────────────────────────────▶ finalize
"""

from __future__ import annotations

from src.agents import should_continue_tools
from src.workflow.state_management import ContentState

WRITER_NODES = {
    "blog": "blog_writer",
    "linkedin": "linkedin_writer",
    "image": "image_generator",
    "report": "content_strategist",
}

__all__ = [
    "WRITER_NODES",
    "route_after_quality",
    "route_after_research",
    "route_initial",
    "route_input",
    "route_next_pending",
    "should_continue_tools",
]


def route_input(state: ContentState) -> str:
    return "finalize" if state.get("route") == "blocked" else "query_handler"


def route_initial(state: ContentState) -> str:
    route = state.get("route", "general")
    if route == "research":
        return "research_agent"
    if route == "campaign":
        pending = state.get("pending_formats", [])
        return WRITER_NODES.get(pending[0], "finalize") if pending else "finalize"
    if route in WRITER_NODES:
        return WRITER_NODES[route]
    return "general"


def route_after_research(state: ContentState) -> str:
    return WRITER_NODES.get(state.get("next_after_research", "report"), "content_strategist")


def route_next_pending(state: ContentState) -> str:
    """After an artifact is finished: continue the campaign queue, or finish the turn."""
    pending = state.get("pending_formats", [])
    return WRITER_NODES.get(pending[0], "finalize") if pending else "finalize"


def route_after_quality(state: ContentState) -> str:
    if state.get("revision_feedback") and state.get("quality_target") in WRITER_NODES:
        return WRITER_NODES[state["quality_target"]]
    return route_next_pending(state)
