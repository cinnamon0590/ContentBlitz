"""🔎 Research Panel: findings plus the thread-wide source registry with attribution."""

from __future__ import annotations

import streamlit as st

from src.web_app.components import thread_state
from src.workflow.state_management import ResearchReport, TrendSignal


def _trend_radar(report: ResearchReport) -> None:
    """Rising Google searches for the niche, each one a click away from a draft."""
    rising = [s for s in report.trend_signals if s.kind == "rising"]
    top = [s.query for s in report.trend_signals if s.kind == "top"]
    st.subheader("📈 Trend radar")
    st.caption(f"Searches gaining interest on Google Trends (“Breakout” = more than +5000%). "
               f"Cited as [{report.trend_source_id}] in reports.")
    for i, signal in enumerate(rising):
        cols = st.columns([5, 2, 2, 2])
        cols[0].markdown(f"**{signal.query}**")
        cols[1].markdown(f"`{signal.growth}`")
        if cols[2].button("💼 LinkedIn", key=f"trend-li-{i}", use_container_width=True):
            _draft(signal, "LinkedIn post")
        if cols[3].button("📝 Blog", key=f"trend-blog-{i}", use_container_width=True):
            _draft(signal, "SEO blog post")
    if top:
        st.markdown("**Most-searched related topics:** " + ", ".join(top))


def _draft(signal: TrendSignal, label: str) -> None:
    from src.web_app.components.chat import run_turn

    run_turn(f"Write a {label} about {signal.query} (search interest {signal.growth} on Google Trends)")
    st.rerun()


def render_research_panel() -> None:
    state = thread_state()
    research = state.get("research")
    sources = state.get("sources", [])
    if not research and not sources:
        st.info("No research yet. Try: “Research the latest trends in sustainable packaging” or "
                "“What should I post about this month for my bakery?” for a trend radar.")
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
        if report.trend_signals:
            _trend_radar(report)
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
