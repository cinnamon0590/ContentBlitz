"""📦 Export tab: download any artifact in Markdown / HTML / WordPress / TXT / PDF / PNG / JPG, or a ZIP of everything."""

from __future__ import annotations

import streamlit as st

from src.utils.export_tools import bundle_zip, export_artifact
from src.web_app.components import thread_state

FORMATS = {
    "blog": ["md", "html", "wordpress.html", "txt", "pdf"],
    "linkedin": ["txt", "md", "html", "pdf"],
    "report": ["md", "html", "txt", "pdf"],
    "research": ["md", "html", "txt", "pdf"],
    "image": ["png", "jpg", "md"],
}
LABELS = {"blog": "SEO blog post", "linkedin": "LinkedIn post", "report": "Research report",
          "research": "Research brief", "image": "Latest image"}
FORMAT_LABELS = {"md": "Markdown", "html": "HTML", "wordpress.html": "WordPress blocks", "txt": "Plain text",
                 "pdf": "PDF", "png": "PNG", "jpg": "JPG"}


def render_export_panel() -> None:
    state = thread_state()
    artifacts = {k: state[k] for k in FORMATS if state.get(k)}
    if not artifacts:
        st.info("Nothing to export yet.")
        return
    st.caption("Every export includes its Sources & Provenance section and an AI-assisted-content note.")
    for kind, data in artifacts.items():
        with st.container(border=True):
            st.markdown(f"**{LABELS[kind]}** — {data.get('title') or data.get('topic') or ''}")
            cols = st.columns(len(FORMATS[kind]))
            for col, fmt in zip(cols, FORMATS[kind], strict=True):
                try:
                    payload, mime, name = export_artifact(kind, data, fmt)
                except Exception as exc:  # noqa: BLE001 - one bad format shouldn't break the tab
                    col.button(FORMAT_LABELS[fmt], disabled=True, help=str(exc), key=f"x-{kind}-{fmt}")
                    continue
                col.download_button(FORMAT_LABELS[fmt], payload, file_name=name, mime=mime,
                                    key=f"dl-{kind}-{fmt}", use_container_width=True)
    st.download_button("⬇️ Download everything (ZIP)", bundle_zip(artifacts, state.get("sources", [])),
                       file_name="contentblitz-export.zip", mime="application/zip", type="primary",
                       use_container_width=True)
