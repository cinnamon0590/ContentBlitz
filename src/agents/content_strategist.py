"""Agent 6 — Content Strategist: formats research into a readable report.

Used when the user asked for research but no specific output format. Templates:
trend report, market analysis, or competitive analysis. The strategist:

* writes from the numbered research sources (no URLs are given to the model),
* removes any URL that isn't in the source registry, and any citation id that doesn't exist,
* appends a Sources & Provenance section built from the registry (not from model output).
"""

from __future__ import annotations

import re

from src.agents import (
    AgentDeps,
    add_output,
    add_warning,
    brand_of,
    format_sources,
    request_of,
    scrub_body,
    topic_of,
)
from src.core import prompts
from src.core.templates import outline, suggest_template
from src.utils.content_optimization import citation_ids
from src.workflow.state_management import (
    ContentState,
    Provenance,
    ResearchReport,
    SourceRegistry,
    StrategyReport,
)


def make_content_strategist(deps: AgentDeps):
    def content_strategist(state: ContentState) -> dict:
        brand = brand_of(state)
        request = request_of(state)
        research = ResearchReport.model_validate(state.get("research") or {"topic": topic_of(state)})
        registry = SourceRegistry(state.get("sources"))
        sources = research.provenance.sources
        template = suggest_template("report", state.get("user_input", ""), request.template)
        system = prompts.with_brand(prompts.CONTENT_STRATEGIST, brand, outline=outline("report", template))
        feedback = state.get("revision_feedback", "")
        user = (f"TOPIC: {research.topic}\nUSER REQUEST: {state.get('user_input', '')}\nTEMPLATE: {template}\n\n"
                f"RESEARCH SUMMARY: {research.summary}\nKEY FINDINGS:\n"
                + "\n".join(f"- {f.claim} {f.source_ids}" for f in research.key_findings)
                + "\nSTATISTICS:\n" + "\n".join(f"- {s}" for s in research.statistics)
                + "\nTRENDS:\n" + "\n".join(f"- {t}" for t in research.trends)
                + f"\n\nSOURCES:\n{format_sources(sources)}")
        if feedback and state.get("quality_target") == "report":
            previous = (state.get("report") or {}).get("markdown", "")
            user += prompts.REVISION_NOTE.format(feedback=feedback, previous=previous[:6000])

        warnings = []
        try:
            markdown = deps.llm.text(system, user, tier="smart")
        except RuntimeError as exc:
            warnings.append(f"Content Strategist model unavailable ({exc}); showing the research brief instead.")
            markdown = research.markdown.split("## Sources & Provenance")[0]

        markdown, notes = scrub_body(markdown, registry, {s.id for s in sources})
        if not markdown.lstrip().startswith("# "):
            markdown = f"# {template.title()}: {research.topic}\n\n{markdown}"
        title = re.match(r"#\s+(.+)", markdown.lstrip()).group(1).strip()
        cited = sorted(citation_ids(markdown))
        registry.mark_used(cited, "report")
        provenance = Provenance(
            sources=[registry.get(s.id) or s for s in sources],
            generated_by=deps.llm.describe("smart"),
            grounding=research.provenance.grounding,
            revision_count=state.get("revision_count", 0),
            notes=notes,
        )
        report = StrategyReport(title=title, markdown=f"{markdown.rstrip()}\n\n{provenance.to_markdown()}",
                                template=template, provenance=provenance)
        return {
            "report": report.model_dump(),
            "sources": registry.dump(),
            "quality_target": "report",
            "revision_feedback": "",
            "outputs_this_turn": add_output(state, "report"),
            "warnings": add_warning(state, *warnings),
        }

    return content_strategist
