"""Streamlit UI components. Each ``render_*`` function draws one tab of the app."""

from __future__ import annotations

import uuid

import streamlit as st

from src.workflow.state_management import BrandProfile

NODE_LABELS = {
    "input_guardrail": "🛡️ Checking your request",
    "query_handler": "🧭 Understanding what you need",
    "research_agent": "🔎 Researching",
    "research_tools": "🌐 Searching the web",
    "post_research_router": "🧭 Choosing the best format",
    "blog_writer": "📝 Writing the SEO blog post",
    "blog_tools": "🔑 Researching keywords",
    "linkedin_writer": "💼 Writing the LinkedIn post",
    "linkedin_tools": "📈 Checking trending hashtags",
    "image_generator": "🖼️ Creating the image",
    "content_strategist": "📊 Formatting the research report",
    "quality_gate": "✅ Scoring quality",
    "general": "💬 Replying",
    "finalize": "📦 Final safety checks",
}


def get_app():
    """One ContentBlitz instance per browser session (keeps memory and cost tracking separate)."""
    if "app" not in st.session_state:
        from src.core.workflow import ContentBlitz

        st.session_state.app = ContentBlitz()
    return st.session_state.app


def thread_id() -> str:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
    return st.session_state.thread_id


def new_thread() -> None:
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
    st.session_state.pop("last_result", None)


def brand() -> BrandProfile:
    if "brand" not in st.session_state:
        st.session_state.brand = BrandProfile().model_dump()
    return BrandProfile.model_validate(st.session_state.brand)


def thread_state() -> dict:
    return get_app().state(thread_id())
