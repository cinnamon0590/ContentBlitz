"""⚙️ Settings: brand voice profile, brand-voice learning, service status, usage and cost."""

from __future__ import annotations

import streamlit as st

from src.web_app.components import brand, get_app, new_thread
from src.workflow.state_management import VOICE_ARCHETYPES, BrandProfile

EMOJI_LABELS = {"none": "None", "light": "Light (0–3 per post)", "heavy": "Heavy (emoji bullets)"}


def _split(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def render_settings_panel() -> None:
    app = get_app()
    current = brand()
    left, right = st.columns([3, 2])

    with left:
        st.header("Brand voice")
        st.caption("Every agent uses this profile, and the quality score checks against it.")
        with st.form("brand-form"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Business name", current.name)
            industry = c2.text_input("Industry", current.industry)
            location = c1.text_input("Location (for local SEO)", current.location, placeholder="e.g. Austin, TX")
            audience = c2.text_input("Target audience", current.audience)
            tone = st.text_input("Tone (adjectives)", current.tone)
            v1, v2, v3 = st.columns(3)
            archetypes = ["", *VOICE_ARCHETYPES]
            archetype = v1.selectbox("Writing persona", archetypes,
                                     index=archetypes.index(current.voice_archetype)
                                     if current.voice_archetype in archetypes else 0,
                                     format_func=lambda a: a.title() if a else "None",
                                     help="\n".join(f"**{k.title()}**: {v}" for k, v in VOICE_ARCHETYPES.items()))
            emoji = v2.selectbox("Emoji use", list(EMOJI_LABELS), index=list(EMOJI_LABELS).index(current.emoji_level),
                                 format_func=EMOJI_LABELS.get)
            sentence = v3.number_input("Words per sentence", 0.0, 40.0, float(current.avg_sentence_words), 1.0,
                                       help="Typical sentence length of your writing (0 = don't check).")
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
                    "voice_archetype": archetype, "emoji_level": emoji, "avg_sentence_words": sentence,
                    "values": _split(values), "banned_words": _split(banned), "required_phrases": _split(required),
                    "cta": cta, "visual_style": style, "color_palette": palette,
                }).model_dump()
                st.toast("Brand profile saved")

        learned_note = st.session_state.pop("voice_learned", "")  # shown after the rerun that applies it
        with st.expander("✨ Learn my brand voice from sample content", expanded=bool(learned_note)):
            if learned_note:
                st.success(learned_note)
            sample = st.text_area("Paste 2–3 paragraphs of your existing content (website, newsletter, posts)",
                                  height=160)
            if st.button("Analyze voice") and sample.strip():
                with st.spinner("Analyzing your writing…"):
                    learned = app.learn_brand_voice(sample, current)
                st.session_state.brand = learned.model_dump()
                st.session_state.voice_learned = (
                    f"Learned tone: **{learned.tone}** · audience: **{learned.audience}** · persona: "
                    f"**{learned.voice_archetype or 'none'}** · ~{learned.avg_sentence_words:.0f} words per "
                    f"sentence · emojis: **{learned.emoji_level}**. The fields above now use it.")
                st.rerun()
        if st.button("Reset brand profile"):
            st.session_state.brand = BrandProfile().model_dump()
            st.rerun()

    with right:
        st.header("Services")
        status = app.service_status()
        st.markdown(f"**LLM:** `{status['llm']}`  \n**Memory:** `{status['checkpointer']}`")
        for key, present in status["keys"].items():
            st.markdown(f"{'✅' if present else '⬜'} {key.upper()} key")
        for warning in status.get("key_warnings", []):
            st.warning(warning, icon="🔑")
        st.markdown("**Research fallback chain:** " + (" → ".join(status["research_chain"]) or "_none configured_"))
        st.markdown("**Image fallback chain:** " + " → ".join(status["image_chain"]))

        st.header("Last request usage")
        usage = (st.session_state.get("last_result").usage if st.session_state.get("last_result") else {}) or {}
        c1, c2 = st.columns(2)
        c1.metric("Tokens", f"{usage.get('total_tokens', 0):,}")
        c2.metric("Est. cost", f"${usage.get('estimated_cost_usd', 0):.4f}")
        st.caption(f"Searches: {usage.get('searches', {})} · Images: {usage.get('images', {})} · "
                   f"Cache hits: {usage.get('cache_hits', 0)} · Time: {usage.get('elapsed_s', 0)}s")

        st.header("Maintenance")
        if st.button("🧹 Clear research cache"):
            st.toast(f"Removed {app.deps.search.cache.clear() if app.deps.search.cache else 0} cached results")
        if st.button("🆕 Start a new conversation"):
            new_thread()
            st.rerun()
