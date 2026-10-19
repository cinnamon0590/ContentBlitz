"""Tavily search client (research fallback #1; also an option for ``deep_research``)."""

from __future__ import annotations

import requests

from src.integrations import ProviderError, request_json
from src.workflow.state_management import Source

TAVILY_URL = "https://api.tavily.com/search"


class TavilyClient:
    name = "tavily"

    def __init__(self, api_key: str | None, timeout: float = 25, session: requests.Session | None = None):
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, num: int = 5, depth: str = "basic") -> tuple[list[Source], str]:
        """Return ``(sources, answer)``. ``depth`` is ``basic`` (cheap) or ``advanced`` (deep)."""
        if not self.available:
            raise ProviderError(self.name, "TAVILY_API_KEY is not set")
        data = request_json(
            self.name,
            "POST",
            TAVILY_URL,
            session=self.session,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "search_depth": depth,
                "max_results": num,
                "include_answer": depth == "advanced",
            },
        )
        sources = [
            Source(title=item.get("title", ""), url=item["url"], snippet=(item.get("content") or "")[:500],
                   provider=self.name)
            for item in data.get("results", [])
            if item.get("url")
        ]
        return sources, data.get("answer") or ""
