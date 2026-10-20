"""📝 Content Dashboard (preview, edit, score, refine) and 🖼️ image gallery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from src.web_app.components import brand, get_app, thread_id, thread_state
from src.workflow.state_management import Provenance, QualityReport

KINDS = {"blog": ("📝 SEO Blog", "markdown"), "linkedin": ("💼 LinkedIn", "text"), "report": ("📊 Report", "markdown")}


def _quality_panel(report: QualityReport) -> None:
    # Full width: in a quarter-width column the value was truncated to "83/…" on laptop screens.
    st.metric("Quality score", f"{report.overall:.0f}/100", "passed" if report.passed else "needs work",
              delta_color="normal" if report.passed else "inverse")
    for name, score in report.scores.items():
        label = "SEO" if name == "seo" else name.replace("_", " ").title()
        st.progress(min(1.0, score / 100), text=f"{label}: {score:.0f}")
    if report.suggestions:
        st.markdown("**Suggestions to improve**")
        for suggestion in report.suggestions:
            st.markdown(f"- {suggestion}")
    if report.llm_feedback:
        st.info(f"Editor's note (advisory): {report.llm_feedback}")
    with st.expander("Raw metrics"):
        st.json(report.metrics)


def _content_version(data: dict) -> str:
    """Fingerprint of the stored artifact, used in widget keys.

    Streamlit keeps a widget's value for as long as its key is unchanged, so with fixed keys the editor would keep
    showing old text after a hook swap, an AI refine or a newly generated post. Keying by content resets them.
    """
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _hook_picker(data: dict) -> None:
    """Pick one of the scored hook variants; the post's opening line is swapped and re-scored."""
    options = data.get("hook_options") or []
    if len(options) < 2:
        return
    st.markdown("**Hook options**")
    labels = [f"{o['score']:.0f}/100 · {o['style']} — {o['text']}" for o in options]
    current = next((i for i, o in enumerate(options) if o["text"] == data.get("hook")), 0)
    choice = st.radio("Opening line", range(len(options)), index=current, format_func=lambda i: labels[i],
                      key=f"hook-choice-{_content_version(data)}", label_visibility="collapsed")
    if options[choice]["issues"]:
        st.caption("⚠️ " + " ".join(options[choice]["issues"]))
    if st.button("Use this hook", key="hook-apply", disabled=choice == current):
        report = get_app().apply_hook(thread_id(), options[choice]["text"], brand())
        st.toast(f"Hook swapped. New score: {report.overall:.0f}/100")
        st.rerun()


def _provenance_panel(data: dict) -> None:
    provenance = Provenance.model_validate(data.get("provenance") or {})
    with st.expander(f"Sources & provenance — {provenance.label}"):
        st.markdown(provenance.to_markdown())


def render_dashboard() -> None:
    state = thread_state()
    available = [k for k in KINDS if state.get(k)]
    if not available:
        st.info("Nothing to preview yet. Ask for a blog post, a LinkedIn post or research in the Chat tab.")
        return
    app = get_app()
    kind = st.segmented_control("Content", available, format_func=lambda k: KINDS[k][0], default=available[0],
                                key="dash-kind") or available[0]
    data = dict(state[kind])
    field = KINDS[kind][1]
    version = _content_version(data)
    quality = (state.get("quality") or {}).get(kind)

    left, right = st.columns([3, 2])
    with left:
        if kind == "blog":
            st.text_input("Title", value=data["title"], key=f"title-{kind}-{version}")
            st.text_area("Meta description", value=data["meta_description"], key=f"meta-{kind}-{version}", height=80,
                         help="150–160 characters works best in search results.")
            st.caption(f"{len(st.session_state[f'meta-{kind}-{version}'])} characters · slug: /blog/{data['slug']}")
        edited = st.text_area("Edit content", value=data[field], height=520, key=f"edit-{kind}-{version}")
        c1, c2 = st.columns(2)
        if c1.button("💾 Save edits & re-score", key=f"save-{kind}", width="stretch"):
            data[field] = edited
            if kind == "blog":
                data["title"] = st.session_state[f"title-{kind}-{version}"]
                data["meta_description"] = st.session_state[f"meta-{kind}-{version}"]
            report = app.rescore(kind, data, brand(), thread_id())
            app.update_artifact(thread_id(), kind, data, quality=report)
            st.toast(f"Saved. New score: {report.overall:.0f}/100")
            st.rerun()
        with c2.popover("✨ Ask AI to refine", width="stretch"):
            instructions = st.text_input("What should change?", key=f"refine-{kind}",
                                         placeholder="e.g. make it more casual and add a local example")
            if st.button("Refine", key=f"refine-go-{kind}") and instructions:
                label = {"blog": "blog post", "linkedin": "LinkedIn post", "report": "report"}[kind]
                from src.web_app.components.chat import run_turn

                run_turn(f"Revise the {label}: {instructions}")
                st.rerun()
    with right:
        st.markdown("**Preview**")
        with st.container(border=True, height=420):
            st.markdown(edited)
        if kind == "linkedin":
            _hook_picker(data)
        if kind == "linkedin" and data.get("first_comment"):
            st.markdown("**Suggested first comment**")
            st.code(data["first_comment"], language=None)
        if quality:
            _quality_panel(QualityReport.model_validate(quality))
        _provenance_panel(data)


def render_gallery() -> None:
    images = [i for i in thread_state().get("images", []) if Path(i["path"]).exists()]
    if not images:
        st.info("No images yet. Try: “Make an image of a cozy coffee shop at sunrise”.")
        return
    for image in reversed(images):
        with st.container(border=True):
            cols = st.columns([3, 2])
            cols[0].image(image["path"], caption=image.get("alt_text", ""),
                          alt=image.get("alt_text") or "Generated marketing image", width="stretch")
            with cols[1]:
                st.markdown(f"**Provider:** `{image['provider']}`  \n**Size:** {image['width']}×{image['height']}")
                st.markdown(f"**Your idea:** {image['original_prompt']}")
                st.markdown(f"**Enhanced prompt:** {image['enhanced_prompt']}")
                if image.get("attempts"):
                    st.warning("Fallbacks used: " + "; ".join(image["attempts"]))
                st.download_button("⬇️ Download PNG", Path(image["path"]).read_bytes(),
                                   file_name=Path(image["path"]).name, mime="image/png", key=f"dl-{image['path']}")
