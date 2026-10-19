from unittest.mock import MagicMock

import pytest
import requests

from src.integrations import ProviderError, request_json
from src.integrations.perplexity_client import PerplexityClient
from src.integrations.serp_client import SearchChain, SerpAPIClient
from src.integrations.tavily_client import TavilyClient
from src.utils.cache import TTLCache
from src.utils.observability import CostTracker
from tests.conftest import FakeSearchProvider, sample_sources


def fake_session(payload=None, status=200, side_effect=None):
    session = MagicMock(spec=requests.Session)
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload or {}
    response.text = "err"
    session.request.return_value = response
    if side_effect:
        session.request.side_effect = side_effect
    return session


# ---- request_json --------------------------------------------------------------------
def test_request_json_ok():
    assert request_json("p", "GET", "u", session=fake_session({"a": 1})) == {"a": 1}


def test_request_json_http_error():
    with pytest.raises(ProviderError) as exc:
        request_json("p", "GET", "u", session=fake_session(status=401))
    assert exc.value.status == 401


def test_request_json_retries_transient_then_fails(monkeypatch):
    session = fake_session(status=503)
    with pytest.raises(ProviderError, match="after retries"):
        request_json("p", "GET", "u", session=session)
    assert session.request.call_count == 3


def test_request_json_connection_error():
    with pytest.raises(ProviderError):
        request_json("p", "GET", "u", session=fake_session(side_effect=requests.ConnectionError("x")))


def test_request_json_bad_json():
    session = fake_session()
    session.request.return_value.json.side_effect = ValueError("bad")
    with pytest.raises(ProviderError, match="invalid JSON"):
        request_json("p", "GET", "u", session=session)


# ---- provider parsing ---------------------------------------------------------------------
def test_serpapi_parsing_and_keywords():
    payload = {
        "organic_results": [{"title": "T", "link": "https://a.com/x", "snippet": "S", "source": "A"}, {"title": "no link"}],
        "answer_box": {"snippet": "answer"},
        "related_searches": [{"query": "k1"}, {"query": ""}],
        "related_questions": [{"question": "q1?"}],
    }
    client = SerpAPIClient("key", session=fake_session(payload))
    sources, answer = client.search("q")
    assert [s.url for s in sources] == ["https://a.com/x"] and answer == "answer"
    assert sources[0].provider == "serpapi" and sources[0].publisher == "A"
    assert client.keyword_ideas("q") == {"related_searches": ["k1"], "people_also_ask": ["q1?"]}


def test_serpapi_error_and_missing_key():
    with pytest.raises(ProviderError):
        SerpAPIClient(None).search("q")
    with pytest.raises(ProviderError, match="Invalid API key"):
        SerpAPIClient("k", session=fake_session({"error": "Invalid API key"})).search("q")


def test_tavily_parsing():
    payload = {"results": [{"title": "T", "url": "https://t.com", "content": "C"}], "answer": "A"}
    sources, answer = TavilyClient("k", session=fake_session(payload)).search("q", depth="advanced")
    assert sources[0].publisher == "t.com" and answer == "A"
    with pytest.raises(ProviderError):
        TavilyClient(None).search("q")


def test_perplexity_agent_api_parsing():
    payload = {"output": [
        {"type": "search_results", "results": [{"id": 1, "title": "P", "url": "https://p.com", "snippet": "s"},
                                                {"id": 2, "title": "no url"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "Answer [1]"}]},
    ]}
    session = fake_session(payload)
    sources, answer = PerplexityClient("k", session=session).search("q")
    assert answer == "Answer [1]" and [s.title for s in sources] == ["P"]
    assert PerplexityClient("k", session=fake_session({})).search("q") == ([], "")
    with pytest.raises(ProviderError):
        PerplexityClient(None).search("q")


# ---- fallback chain -----------------------------------------------------------------------
def test_chain_falls_back_to_second_provider(tmp_path):
    tracker = CostTracker()
    primary = FakeSearchProvider("serpapi", fail=True)
    secondary = FakeSearchProvider("tavily", sample_sources("tavily"))
    chain = SearchChain([primary, secondary], cache=TTLCache(tmp_path), tracker=tracker)
    outcome = chain.search("ai marketing")
    assert outcome.ok and outcome.provider == "tavily"
    assert any("simulated outage" in e for e in outcome.errors)
    assert tracker.summary()["searches"] == {"tavily": 1}


def test_chain_uses_cache_on_second_call(tmp_path):
    provider = FakeSearchProvider("serpapi", sample_sources())
    chain = SearchChain([provider], cache=TTLCache(tmp_path))
    chain.search("q")
    second = chain.search("q")
    assert second.from_cache and len(provider.calls) == 1


def test_chain_skips_unavailable_and_empty(tmp_path):
    chain = SearchChain([FakeSearchProvider("serpapi", available=False), FakeSearchProvider("tavily", [])],
                        cache=TTLCache(tmp_path))
    outcome = chain.search("q")
    assert not outcome.ok and outcome.provider == "offline"
    assert "serpapi: no API key" in outcome.errors and "tavily: no results" in outcome.errors


def test_chain_stale_cache_during_outage(tmp_path):
    cache = TTLCache(tmp_path, ttl_seconds=1800)
    good = FakeSearchProvider("serpapi", sample_sources())
    SearchChain([good], cache=cache).search("q")
    cache.ttl_seconds = -1  # everything is now expired
    outcome = SearchChain([FakeSearchProvider("serpapi", fail=True)], cache=cache).search("q")
    assert outcome.ok and outcome.stale and outcome.from_cache


def test_chain_order_override_and_unexpected_error(tmp_path):
    class Boom(FakeSearchProvider):
        def search(self, *a, **k):
            raise ValueError("bug")

    perplexity = FakeSearchProvider("perplexity", sample_sources("perplexity"))
    chain = SearchChain([Boom("serpapi"), perplexity], cache=None)
    assert chain.search("q", order=["perplexity", "serpapi"]).provider == "perplexity"
    assert chain.search("q2").provider == "perplexity"  # Boom falls through
    assert chain.available_providers == ["serpapi", "perplexity"]


def test_keyword_ideas_cached_and_degrades(tmp_path):
    serp = MagicMock()
    serp.available = True
    serp.name = "serpapi"
    serp.keyword_ideas.return_value = {"related_searches": ["a"], "people_also_ask": []}
    chain = SearchChain([], cache=TTLCache(tmp_path), keyword_provider=serp)
    assert chain.keyword_ideas("x")["related_searches"] == ["a"]
    chain.keyword_ideas("x")
    assert serp.keyword_ideas.call_count == 1
    serp.keyword_ideas.side_effect = ProviderError("serpapi", "down")
    assert chain.keyword_ideas("y") == {"related_searches": [], "people_also_ask": []}
    assert SearchChain([]).keyword_ideas("z") == {"related_searches": [], "people_also_ask": []}
