"""⚙️ Settings: brand voice profile, brand-voice learning, service status, usage and cost."""

from __future__ import annotations

import streamlit as st

from src.web_app.components import brand, get_app, new_thread
from src.workflow.state_management import BrandProfile


def _split(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def render_settings_panel() -> None:
    app = get_app()
    current = brand()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Brand voice")
        st.caption("Every agent uses this profile, and the quality score checks against it.")
        with st.form("brand-form"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Business name", current.name)
            industry = c2.text_input("Industry", current.industry)
            location = c1.text_input("Location (for local SEO)", current.location, placeholder="e.g. Austin, TX")
            audience = c2.text_input("Target audience", current.audience)
            tone = st.text_input("Tone (adjectives)", current.tone)
            values = st.text_input("Brand values (comma-separated)", ", ".join(current.values))
            banned = st.text_input("Words to avoid (comma-separated)", ", ".join(current.banned_words))
            required = st.text_input("Signature phrases to include (comma-separated)",
                                     ", ".join(current.required_phrases))
            cta = st.text_input("Default call to action", current.cta)
            style = st.text_input("Visual style for images", current.visual_style)
            colors = st.columns(3)
            palette = [colors[i].color_picker(f"Brand color {i + 1}", (current.color_palette + ["#FFFFFF"] * 3)[i])
                       for i in range(3)]
            if st.form_submit_button("Save brand profile", type="primary"):
                st.session_state.brand = current.model_copy(update={
                    "name": name, "industry": industry, "location": location, "audience": audience, "tone": tone,
                    "values": _split(values), "banned_words": _split(banned), "required_phrases": _split(required),
                    "cta": cta, "visual_style": style, "color_palette": palette,
                }).model_dump()
                st.toast("Brand profile saved")

        with st.expander("✨ Learn my brand voice from sample content"):
            sample = st.text_area("Paste 2–3 paragraphs of your existing content (website, newsletter, posts)",
                                  height=160)
            if st.button("Analyze voice") and sample.strip():
                with st.spinner("Analyzing your writing…"):
                    learned = app.learn_brand_voice(sample, current)
                st.session_state.brand = learned.model_dump()
                st.success(f"Learned tone: **{learned.tone}** · audience: **{learned.audience}**")
                st.rerun()
        if st.button("Reset brand profile"):
            st.session_state.brand = BrandProfile().model_dump()
            st.rerun()

    with right:
        st.subheader("Services")
        status = app.service_status()
        st.markdown(f"**LLM:** `{status['llm']}`  \n**Memory:** `{status['checkpointer']}`")
        for key, present in status["keys"].items():
            st.markdown(f"{'✅' if present else '⬜'} {key.upper()} key")
        st.markdown("**Research fallback chain:** " + (" → ".join(status["research_chain"]) or "_none configured_"))
        st.markdown("**Image fallback chain:** " + " → ".join(status["image_chain"]))

        st.subheader("Last request usage")
        usage = (st.session_state.get("last_result").usage if st.session_state.get("last_result") else {}) or {}
        c1, c2 = st.columns(2)
        c1.metric("Tokens", f"{usage.get('total_tokens', 0):,}")
        c2.metric("Est. cost", f"${usage.get('estimated_cost_usd', 0):.4f}")
        st.caption(f"Searches: {usage.get('searches', {})} · Images: {usage.get('images', {})} · "
                   f"Cache hits: {usage.get('cache_hits', 0)} · Time: {usage.get('elapsed_s', 0)}s")

        st.subheader("Maintenance")
        if st.button("🧹 Clear research cache"):
            st.toast(f"Removed {app.deps.search.cache.clear() if app.deps.search.cache else 0} cached results")
        if st.button("🆕 Start a new conversation"):
            new_thread()
            st.rerun()
