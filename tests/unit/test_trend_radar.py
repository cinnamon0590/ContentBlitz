"""Trend radar: Google Trends via SerpAPI → relevance filter → cited research + report section."""

from src.agents.content_strategist import make_content_strategist
from src.agents.research_agent import make_research_agent, make_research_tools, wants_trends
from src.integrations.serp_client import SearchChain, SerpAPIClient, relevant_trends, trends_explore_url
from src.utils.cache import TTLCache
from src.utils.content_optimization import uncited_statistics
from src.utils.observability import CostTracker
from src.workflow.state_management import ResearchReport, TrendSignal
from tests.conftest import FakeKeywords, default_roles, drive_agent
from tests.unit.test_search_clients import fake_session

TREND_STATE = {"user_input": "What should I post about this month for my bakery?",
               "request": {"intent": "research", "topic": "bakery marketing", "template": "trend report"}}


def run(deps, state):
    tools = make_research_tools(deps)
    return drive_agent(make_research_agent(deps, tools), tools, dict(state))


def test_serpapi_trending_queries_parsing():
    payload = {"related_queries": {
        "rising": [{"query": "ai bakery ideas", "value": "Breakout", "extracted_value": 5000},
                   {"query": "bakery newsletter", "value": "+190%", "extracted_value": 190}],
        "top": [{"query": "bakery near me", "value": "100", "extracted_value": 100}, {"value": "1"}]}}
    session = fake_session(payload)
    signals = SerpAPIClient("k", session=session).trending_queries("bakery marketing")
    assert [(s.query, s.growth, s.kind) for s in signals] == [
        ("ai bakery ideas", "Breakout", "rising"), ("bakery newsletter", "+190%", "rising"),
        ("bakery near me", "100", "top")]
    params = session.request.call_args.kwargs["params"]
    assert params["engine"] == "google_trends" and params["geo"] == "US" and params["data_type"] == "RELATED_QUERIES"


def test_relevance_filter_drops_viral_noise():
    signals = [TrendSignal(query="what is a 401k", growth="Breakout", growth_value=9000),
               TrendSignal(query="bakeries instagram", growth="+300%", growth_value=300),
               TrendSignal(query="Bakeries Instagram", growth="+300%", growth_value=300),
               TrendSignal(query="marketing tips", growth="+50%", growth_value=50),
               TrendSignal(query="bakery near me", growth="100", growth_value=100, kind="top")]
    kept = relevant_trends("bakery marketing", signals)
    assert [s.query for s in kept] == ["bakeries instagram", "marketing tips", "bakery near me"]
    assert len(relevant_trends("bakery marketing", signals, limit=1)) == 1


def test_chain_caches_trends_and_degrades_quietly(tmp_path):
    tracker = CostTracker()
    chain = SearchChain([], cache=TTLCache(tmp_path), tracker=tracker, keyword_provider=FakeKeywords())
    first = chain.trending_queries("bakery marketing")
    assert [s.query for s in first] == ["ai bakery ideas", "bakery newsletter", "bakery near me"]
    assert chain.trending_queries("bakery marketing") == first and tracker.searches == {"serpapi": 1}

    class Broken(FakeKeywords):
        def trending_queries(self, *args, **kwargs):
            raise RuntimeError("trends engine down")

    assert SearchChain([], keyword_provider=Broken()).trending_queries("x") == []
    assert SearchChain([], keyword_provider=None).trending_queries("x") == []
    assert trends_explore_url("bakery marketing").startswith("https://trends.google.com/trends/explore?q=bakery")


def test_trend_request_adds_cited_radar(make_deps):
    deps = make_deps()
    state = run(deps, TREND_STATE)
    report = ResearchReport.model_validate(state["research"])
    assert [s.query for s in report.trend_signals] == ["ai bakery ideas", "bakery newsletter", "bakery near me"]
    trend = next(s for s in report.provenance.sources if s.id == report.trend_source_id)
    assert trend.publisher == "trends.google.com" and "bakery newsletter (+190%)" in trend.snippet
    assert any(s["url"] == trend.url for s in state["sources"])  # in the registry, so the URL survives scrubbing
    radar = report.markdown.split("## Trend Radar")[1].split("##")[0]
    assert f"| bakery newsletter | +190% [{trend.id}] |" in radar and "401k" not in report.markdown
    assert uncited_statistics(radar) == []


def test_trending_topics_tool_payload_is_used(make_deps):
    roles = default_roles("research")
    roles["research_agent"] = lambda msgs: ("DONE" if any(getattr(m, "type", "") == "tool" for m in msgs)
                                            else __import__("tests.conftest").conftest.tool_call(
                                                "trending_topics", niche="bakery"))
    deps = make_deps(roles)
    state = run(deps, {"user_input": "research bakery marketing", "request": {"intent": "research",
                                                                               "topic": "bakery marketing"}})
    report = ResearchReport.model_validate(state["research"])
    assert report.trend_signals and "“bakery”" in report.provenance.sources[-1].title


def test_plain_research_skips_trends(make_deps):
    deps = make_deps()
    state = run(deps, {"user_input": "research bakery email marketing",
                       "request": {"intent": "research", "topic": "bakery email marketing"}})
    assert state["research"]["trend_signals"] == [] and "Trend Radar" not in state["research"]["markdown"]
    assert "serpapi" not in deps.tracker.searches or deps.tracker.searches["serpapi"] == 1  # web search only


def test_trend_request_without_data_warns(make_deps):
    deps = make_deps()
    deps.search.keyword_provider = None
    state = run(deps, TREND_STATE)
    assert any("Trend radar unavailable" in w for w in state["warnings"])


def test_wants_trends_detection():
    assert wants_trends({"user_input": "what should I write about next week?"})
    assert wants_trends({"user_input": "give me content ideas"})
    assert wants_trends({"user_input": "x", "request": {"template": "trend report"}})
    assert not wants_trends({"user_input": "research bakery email marketing"})


def test_strategist_keeps_radar_when_model_drops_it(make_deps):
    deps = make_deps()
    state = run(deps, TREND_STATE)
    out = make_content_strategist(deps)(state)  # the fake strategist reply has no Trend Radar section
    md = out["report"]["markdown"]
    assert md.index("## Trend Radar") < md.index("## Recommendations")
    report_id = state["research"]["trend_source_id"]
    assert f"[{report_id}]" in md.split("## Trend Radar")[1]  # the citation survived scrub_body


def test_off_topic_results_dropped_but_never_all():
    from src.agents.research_agent import drop_off_topic

    payloads = [{"query": "q", "results": [{"title": "Social Security Administration", "snippet": "Benefits"},
                                           {"title": "October Social Media Ideas for Bakeries", "snippet": ""}]}]
    kept = drop_off_topic("content ideas for bakery", payloads)
    assert [r["title"] for r in kept[0]["results"]] == ["October Social Media Ideas for Bakeries"]
    nothing = [{"results": [{"title": "Unrelated", "snippet": "x"}]}]
    assert drop_off_topic("bakery", nothing) == nothing and drop_off_topic("the best", payloads) == payloads


def test_window_label_and_router_context():
    from src.agents.query_handler import _context_summary
    from src.integrations.serp_client import window_label

    assert window_label("today 12-m") == "past 12 months" and window_label("today 1-y") == "past 1 year"
    assert window_label("2026-01-01 2026-06-30") == "2026-01-01 2026-06-30"
    state = {"research": {"topic": "bakery", "trend_signals": [
        {"query": "viral caramel apple", "growth": "+1,800%", "kind": "rising"},
        {"query": "bakery near me", "growth": "100", "kind": "top"}]}}
    summary = _context_summary(state)
    assert "trend radar (fastest-rising first): viral caramel apple (+1,800%)" in summary
    assert "bakery near me" not in summary
