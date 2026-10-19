import json

from src.agents.research_agent import fallback_queries, make_research_agent, make_research_tools
from src.workflow.state_management import MODEL_KNOWLEDGE_LABEL, ResearchReport
from tests.conftest import FakeSearchProvider, default_roles, drive_agent

STATE = {"user_input": "research bakery email marketing", "request": {"intent": "research",
                                                                      "topic": "bakery email marketing"}}


def run(deps, state=STATE):
    tools = make_research_tools(deps)
    return drive_agent(make_research_agent(deps, tools), tools, dict(state))


def test_tool_loop_produces_cited_report(make_deps):
    deps = make_deps()
    state = run(deps)
    report = ResearchReport.model_validate(state["research"])
    assert report.queries == ["bakery email marketing statistics 2026"]
    assert report.provenance.grounding == "live_research"
    assert {s.id for s in report.provenance.sources} == {1, 2, 3}
    assert all(i in {1, 2, 3} for f in report.key_findings for i in f.source_ids)  # id 99 dropped
    assert "## Sources & Provenance" in report.markdown and "https://example1.com/article" in report.markdown
    assert state["agent_scratch"] == []  # scratch cleared after the loop
    assert "research" in state["outputs_this_turn"]
    assert any("research" in s["used_in"] for s in state["sources"])


def test_fallback_queries_when_model_skips_tools(make_deps):
    roles = default_roles()
    roles["research_agent"] = "RESEARCH COMPLETE"  # never calls a tool
    state = run(make_deps(roles))
    assert len(state["research"]["queries"]) == 3


def test_all_providers_down_labels_model_knowledge(make_deps):
    state = run(make_deps(search_fail=True))
    report = ResearchReport.model_validate(state["research"])
    assert report.provenance.grounding == "model_knowledge_only"
    assert MODEL_KNOWLEDGE_LABEL in report.markdown
    assert report.statistics == []  # no numbers without evidence
    assert any("Live research unavailable" in w for w in state["warnings"])


def test_cached_research_grounding(make_deps):
    deps = make_deps()
    run(deps)
    state = run(deps)  # identical query → served from cache
    assert state["research"]["provenance"]["grounding"] == "cached_research"


def test_synthesis_failure_uses_snippets(make_deps):
    roles = default_roles()
    roles["research_synthesis"] = "garbage"
    state = run(make_deps(roles))
    assert state["research"]["key_findings"] and any("synthesis model failed" in w for w in state["warnings"])


def test_deep_research_capped_once(make_deps):
    deps = make_deps(providers=[FakeSearchProvider("perplexity", __import__("tests.conftest").conftest.sample_sources("perplexity"), answer="A [1]")])
    _, deep = make_research_tools(deps)
    first = json.loads(deep.invoke({"question": "q"}))
    second = json.loads(deep.invoke({"question": "q2"}))
    assert first["provider"] == "perplexity" and first["answer"] == "A [1]"
    assert "limit reached" in second["error"]
    deps.turn.reset()
    assert "error" not in json.loads(deep.invoke({"question": "q3"}))


def test_source_ids_stable_across_turns(make_deps):
    deps = make_deps()
    first = run(deps)
    second = run(deps, {**STATE, "sources": first["sources"]})
    assert [s["id"] for s in second["sources"]] == [1, 2, 3]


def test_fallback_queries_include_year():
    qs = fallback_queries("coffee", 2026)
    assert qs[0] == "coffee statistics 2026" and len(qs) == 3
