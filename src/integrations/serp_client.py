"""SerpAPI (Google results) client and the research-provider fallback chain.

``SerpAPIClient`` is the primary research provider, as the FAQ recommends. It
also supplies SEO keyword data (related searches and "People also ask") and
search-demand trends (Google Trends related queries with their growth).

``SearchChain`` provides redundancy: it tries providers in the order set in
``config/services.yaml`` (SerpAPI → Tavily → Perplexity). Each result is cached
for 30 minutes. If every live provider fails, it falls back to stale cache, and
then to an explicit "offline" outcome, so the agents can label their output
"model knowledge only" rather than invent sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlencode

import requests

from src.integrations import ProviderError, request_json
from src.utils.cache import TTLCache
from src.utils.observability import CostTracker, get_logger
from src.workflow.state_management import Source, TrendSignal

log = get_logger("search")
SERPAPI_URL = "https://serpapi.com/search.json"
_TREND_STOPWORDS = {"the", "a", "an", "and", "or", "for", "of", "in", "on", "to", "with", "how", "what", "is",
                    "best", "my", "your", "small", "business", "businesses", "tips", "ideas", "near", "me",
                    "content", "post", "posts", "social", "media", "write", "this", "month", "2026", "2027"}


def topic_stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:5] for w in words if w not in _TREND_STOPWORDS and len(w) > 1}


def relevant_trends(topic: str, signals: list[TrendSignal], limit: int = 8) -> list[TrendSignal]:
    """Keep related queries that share a meaningful word with the topic.

    Google Trends' related-query lists for niche topics often contain unrelated
    viral searches ("what is a 401k"), so the relevance filter matters. Rising
    queries come first, then top queries as context.
    """
    stems = topic_stems(topic)
    kept, seen = [], set()
    for signal in sorted(signals, key=lambda s: (s.kind != "rising", -s.growth_value)):
        key = signal.query.lower().strip()
        if key in seen or not (stems & topic_stems(signal.query)):
            continue
        seen.add(key)
        kept.append(signal)
    return kept[:limit]


def window_label(window: str) -> str:
    """'today 12-m' → 'past 12 months' (Google Trends date syntax to plain English)."""
    match = re.fullmatch(r"today (\d+)-([dmy])", window.strip())
    if not match:
        return window
    unit = {"d": "day", "m": "month", "y": "year"}[match.group(2)]
    return f"past {match.group(1)} {unit}{'s' if match.group(1) != '1' else ''}"


def trends_explore_url(topic: str, geo: str = "US", window: str = "today 12-m") -> str:
    return "https://trends.google.com/trends/explore?" + urlencode({"q": topic, "geo": geo, "date": window})


class SearchProvider(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def search(self, query: str, num: int = 5, depth: str = "basic") -> tuple[list[Source], str]: ...


class SerpAPIClient:
    name = "serpapi"

    def __init__(self, api_key: str | None, timeout: float = 25, session: requests.Session | None = None):
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _query(self, query: str, num: int) -> dict:
        if not self.available:
            raise ProviderError(self.name, "SERPAPI_API_KEY is not set")
        data = request_json(
            self.name, "GET", SERPAPI_URL, session=self.session, timeout=self.timeout,
            params={"engine": "google", "q": query, "num": num, "api_key": self.api_key},
        )
        if data.get("error"):
            raise ProviderError(self.name, data["error"])
        return data

    def search(self, query: str, num: int = 5, depth: str = "basic") -> tuple[list[Source], str]:
        data = self._query(query, num)
        sources = [
            Source(title=item.get("title", ""), url=item["link"], snippet=item.get("snippet", ""),
                   publisher=item.get("source", ""), provider=self.name)
            for item in data.get("organic_results", [])[:num]
            if item.get("link")
        ]
        answer = (data.get("answer_box") or {}).get("snippet") or (data.get("answer_box") or {}).get("answer") or ""
        return sources, answer

    def keyword_ideas(self, query: str) -> dict[str, list[str]]:
        """Related searches and People-also-ask questions, for SEO keyword research."""
        data = self._query(query, 10)
        return {
            "related_searches": [r.get("query", "") for r in data.get("related_searches", []) if r.get("query")],
            "people_also_ask": [q.get("question", "") for q in data.get("related_questions", []) if q.get("question")],
        }

    def trending_queries(self, topic: str, geo: str = "US", window: str = "today 12-m") -> list[TrendSignal]:
        """Google Trends related queries (rising with % growth, and top) for ``topic``."""
        if not self.available:
            raise ProviderError(self.name, "SERPAPI_API_KEY is not set")
        data = request_json(
            self.name, "GET", SERPAPI_URL, session=self.session, timeout=self.timeout,
            params={"engine": "google_trends", "q": topic, "geo": geo, "hl": "en", "date": window,
                    "data_type": "RELATED_QUERIES", "api_key": self.api_key},
        )
        if data.get("error"):
            raise ProviderError(self.name, data["error"])
        related = data.get("related_queries") or {}
        return [TrendSignal(query=item["query"], growth=str(item.get("value", "")),
                            growth_value=int(item.get("extracted_value") or 0), kind=kind)
                for kind in ("rising", "top") for item in related.get(kind) or [] if item.get("query")]


@dataclass
class SearchOutcome:
    query: str
    sources: list[Source] = field(default_factory=list)
    answer: str = ""
    provider: str = "none"
    from_cache: bool = False
    stale: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.sources)


class SearchChain:
    """Ordered provider fallback with TTL caching and cost tracking."""

    def __init__(
        self,
        providers: list[SearchProvider],
        cache: TTLCache | None = None,
        tracker: CostTracker | None = None,
        keyword_provider: SerpAPIClient | None = None,
    ):
        self.providers = providers
        self.cache = cache
        self.tracker = tracker
        self.keyword_provider = keyword_provider

    @property
    def available_providers(self) -> list[str]:
        return [p.name for p in self.providers if p.available]

    def _from_cache(self, key: str, query: str, stale: bool = False) -> SearchOutcome | None:
        if not self.cache:
            return None
        entry = self.cache.get_stale(key) if stale else self.cache.get(key)
        if not entry:
            return None
        value, stored_at = entry
        retrieved = datetime.fromtimestamp(stored_at, UTC).isoformat(timespec="seconds")
        sources = [Source.model_validate({**s, "retrieved_at": s.get("retrieved_at", retrieved)})
                   for s in value["sources"]]
        if self.tracker:
            self.tracker.record_cache_hit()
        return SearchOutcome(query=query, sources=sources, answer=value.get("answer", ""),
                             provider=value.get("provider", "cache"), from_cache=True, stale=stale)

    def search(self, query: str, num: int = 5, depth: str = "basic",
               order: list[str] | None = None) -> SearchOutcome:
        key = TTLCache.make_key("search", query.lower().strip(), num, depth, order)
        cached = self._from_cache(key, query)
        if cached:
            return cached

        providers = self.providers
        if order:
            rank = {name: i for i, name in enumerate(order)}
            providers = sorted(providers, key=lambda p: rank.get(p.name, len(rank)))

        errors: list[str] = []
        for provider in providers:
            if not provider.available:
                errors.append(f"{provider.name}: no API key")
                continue
            try:
                sources, answer = provider.search(query, num=num, depth=depth)
            except ProviderError as exc:
                errors.append(str(exc))
                log.warning("search provider failed, falling back: %s", exc)
                continue
            except Exception as exc:  # noqa: BLE001 - unexpected provider bug → still fall back
                errors.append(f"{provider.name}: {exc}")
                log.exception("unexpected search provider error")
                continue
            if self.tracker:
                self.tracker.record_search(provider.name)
            if not sources:
                errors.append(f"{provider.name}: no results")
                continue
            if self.cache:
                self.cache.set(key, {"sources": [s.model_dump() for s in sources], "answer": answer,
                                     "provider": provider.name})
            return SearchOutcome(query=query, sources=sources, answer=answer, provider=provider.name, errors=errors)

        stale = self._from_cache(key, query, stale=True)
        if stale:
            stale.errors = errors
            return stale
        return SearchOutcome(query=query, provider="offline", errors=errors)

    def trending_queries(self, topic: str, geo: str = "US", window: str = "today 12-m",
                         limit: int = 8) -> list[TrendSignal]:
        """Relevant rising/top searches for ``topic`` (cached). Empty when Trends is unavailable."""
        key = TTLCache.make_key("trends", topic.lower().strip(), geo, window)
        if self.cache and (entry := self.cache.get(key)):
            signals = [TrendSignal.model_validate(s) for s in entry[0]]
        else:
            if not self.keyword_provider or not self.keyword_provider.available:
                return []
            try:
                signals = self.keyword_provider.trending_queries(topic, geo=geo, window=window)
            except Exception as exc:  # noqa: BLE001 - the radar is optional; never fail the research turn
                log.warning("trend data unavailable: %s", exc)
                return []
            if self.tracker:
                self.tracker.record_search(self.keyword_provider.name)
            if self.cache:
                self.cache.set(key, [s.model_dump() for s in signals])
        return relevant_trends(topic, signals, limit)

    def keyword_ideas(self, query: str) -> dict[str, list[str]]:
        key = TTLCache.make_key("keywords", query.lower().strip())
        if self.cache and (entry := self.cache.get(key)):
            return entry[0]
        if not self.keyword_provider or not self.keyword_provider.available:
            return {"related_searches": [], "people_also_ask": []}
        try:
            ideas = self.keyword_provider.keyword_ideas(query)
        except ProviderError as exc:
            log.warning("keyword research unavailable: %s", exc)
            return {"related_searches": [], "people_also_ask": []}
        if self.tracker:
            self.tracker.record_search(self.keyword_provider.name)
        if self.cache:
            self.cache.set(key, ideas)
        return ideas
