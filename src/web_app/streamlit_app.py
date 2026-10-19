"""ContentBlitz — Streamlit web interface.

Run from the project root:

    streamlit run src/web_app/streamlit_app.py

Tabs: 💬 Chat · 📝 Content Dashboard · 🔎 Research · 🖼️ Images · 📦 Export · ⚙️ Settings
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # allow `streamlit run src/web_app/streamlit_app.py` from the project root
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.web_app.components import brand, get_app, new_thread, thread_id, thread_state  # noqa: E402
from src.web_app.components.chat import render_chat  # noqa: E402
from src.web_app.components.dashboard import render_dashboard, render_gallery  # noqa: E402
from src.web_app.components.export_panel import render_export_panel  # noqa: E402
from src.web_app.components.research_panel import render_research_panel  # noqa: E402
from src.web_app.components.settings_panel import render_settings_panel  # noqa: E402


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### ⚡ ContentBlitz")
        st.caption("Multi-agent content marketing for small businesses")
        st.markdown(f"**Brand:** {brand().name}  \n**Conversation:** `{thread_id()}`")
        if st.button("🆕 New conversation", use_container_width=True):
            new_thread()
            st.rerun()
        state = thread_state()
        produced = [k for k in ("research", "report", "blog", "linkedin") if state.get(k)]
        produced += [f"{len(state.get('images', []))} image(s)"] if state.get("images") else []
        st.markdown("**In this conversation:** " + (", ".join(produced) or "nothing yet"))
        last = st.session_state.get("last_result")
        if last and last.usage:
            st.caption(f"Last request: {last.usage.get('total_tokens', 0):,} tokens · "
                       f"${last.usage.get('estimated_cost_usd', 0):.4f}")
        st.divider()
        st.caption("Six agents: Query Handler · Deep Research · SEO Blog Writer · LinkedIn Writer · "
                   "Image Generator · Content Strategist")
        st.caption("AI-assisted content — review facts before publishing.")


def main() -> None:
    st.set_page_config(page_title="ContentBlitz", page_icon="⚡", layout="wide")
    css = Path(__file__).parent / "static" / "styles.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)
    get_app()
    sidebar()
    st.markdown('<div class="cb-hero"><h1>⚡ ContentBlitz</h1><span class="cb-tag">research · blog · LinkedIn · '
                "images</span></div>", unsafe_allow_html=True)
    tabs = st.tabs(["💬 Chat", "📝 Content Dashboard", "🔎 Research", "🖼️ Images", "📦 Export", "⚙️ Settings"])
    with tabs[0]:
        render_chat()
    with tabs[1]:
        render_dashboard()
    with tabs[2]:
        render_research_panel()
    with tabs[3]:
        render_gallery()
    with tabs[4]:
        render_export_panel()
    with tabs[5]:
        render_settings_panel()


main()
