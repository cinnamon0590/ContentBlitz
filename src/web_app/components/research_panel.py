"""🔎 Research Panel: findings plus the thread-wide source registry with attribution."""

from __future__ import annotations

import streamlit as st

from src.web_app.components import thread_state
from src.workflow.state_management import ResearchReport


def render_research_panel() -> None:
    state = thread_state()
    research = state.get("research")
    sources = state.get("sources", [])
    if not research and not sources:
        st.info("No research yet. Try: “Research the latest trends in sustainable packaging”.")
        return
    if research:
        report = ResearchReport.model_validate(research)
        st.subheader(f"Research: {report.topic}")
        st.caption(report.provenance.label)
        st.markdown(report.summary)
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Key findings**")
            for finding in report.key_findings:
                st.markdown(f"- {finding.claim} " + " ".join(f"`[{i}]`" for i in finding.source_ids))
        with cols[1]:
            if report.statistics:
                st.markdown("**Notable statistics**")
                for stat in report.statistics:
                    st.markdown(f"- {stat}")
            if report.trends:
                st.markdown("**Trends**")
                for trend in report.trends:
                    st.markdown(f"- {trend}")
        if report.queries:
            with st.expander("Search queries the agent ran"):
                for q in report.queries:
                    st.markdown(f"- {q}")

    st.subheader(f"Sources ({len(sources)})")
    st.caption("Every source retrieved in this conversation, the outputs that used it, and where it came from. "
               "Citation numbers [n] stay the same across follow-up requests.")
    rows = [{"#": s["id"], "Title": s.get("title") or s.get("publisher"), "URL": s["url"],
             "Publisher": s.get("publisher", ""), "Found via": s.get("provider", ""),
             "Used in": ", ".join(s.get("used_in", [])) or "—", "Retrieved": s.get("retrieved_at", "")[:16]}
            for s in sorted(sources, key=lambda x: x["id"])]
    st.dataframe(rows, hide_index=True, use_container_width=True,
                 column_config={"URL": st.column_config.LinkColumn("URL", display_text="open ↗")})
