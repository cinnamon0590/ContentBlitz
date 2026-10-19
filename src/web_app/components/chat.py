"""💬 Chat tab: multi-turn conversation with live agent progress."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.web_app.components import NODE_LABELS, brand, get_app, thread_id

EXAMPLES = [
    "Research the latest trends in sustainable packaging for small e-commerce brands",
    "Write an SEO blog post about email marketing for local bakeries",
    "Create a LinkedIn post about why fast replies win more customers",
    "Make an image of a cozy coffee shop at sunrise for our autumn campaign",
    "Create a full campaign (blog, LinkedIn, image) about our new vegan menu",
]


def run_turn(message: str) -> None:
    """Stream one turn through the graph, showing each agent step as it happens."""
    app = get_app()
    with st.status("Working on it…", expanded=True) as status:
        result = None
        for node, update in app.stream(message, thread_id=thread_id(), brand=brand()):
            if node == "__result__":
                result = update["result"]
                continue
            st.write(NODE_LABELS.get(node, node))
        status.update(label="Done" if result and not result.error else "Finished with issues",
                      state="complete" if result and not result.error else "error", expanded=False)
    st.session_state.last_result = result


def render_chat() -> None:
    app = get_app()
    if not app.config.secrets.openai_api_key:
        st.warning("**OPENAI_API_KEY is not set.** Copy `.env.example` to `.env`, add your key and restart. "
                   "Without it, the assistant can only use fallbacks.", icon="🔑")

    history = app.history(thread_id())
    if not history:
        st.markdown("#### 👋 What would you like to create today?")
        st.caption("Try one of these, or type your own request below:")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, use_container_width=True, key=f"ex-{example[:20]}"):
                st.session_state.pending_prompt = example
                st.rerun()

    for role, content in history:
        with st.chat_message(role):
            st.markdown(content)

    image = app.state(thread_id()).get("image")
    last = st.session_state.get("last_result")
    if image and last and "image" in last.outputs and Path(image["path"]).exists():
        st.image(image["path"], caption=image.get("alt_text", ""), alt=image.get("alt_text") or "Generated marketing image",
                 use_container_width=True)

    prompt = st.chat_input("Ask for research, a blog post, a LinkedIn post, an image — or a full campaign…")
    prompt = prompt or st.session_state.pop("pending_prompt", None)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            run_turn(prompt)
        st.rerun()
