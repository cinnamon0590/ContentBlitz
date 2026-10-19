"""Perplexity Sonar client: research + analysis in one call, with citations.

It's used for the expensive ``deep_research`` tool (at most once per request) and as
the last live provider in the research fallback chain.
"""

from __future__ import annotations

import requests

from src.integrations import ProviderError, request_json
from src.workflow.state_management import Source

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityClient:
    name = "perplexity"

    def __init__(self, api_key: str | None, model: str = "sonar-pro", timeout: float = 45,
                 session: requests.Session | None = None):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, num: int = 5, depth: str = "advanced") -> tuple[list[Source], str]:
        """Return ``(sources, answer)``; the answer contains ``[n]`` markers referring to ``sources``."""
        if not self.available:
            raise ProviderError(self.name, "PERPLEXITY_API_KEY is not set")
        data = request_json(
            self.name,
            "POST",
            PERPLEXITY_URL,
            session=self.session,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a meticulous research analyst. Cite sources."},
                    {"role": "user", "content": query},
                ],
            },
        )
        answer = ""
        choices = data.get("choices") or []
        if choices:
            answer = choices[0].get("message", {}).get("content", "")

        sources: list[Source] = []
        for item in data.get("search_results") or []:
            if item.get("url"):
                sources.append(Source(title=item.get("title", ""), url=item["url"],
                                      snippet=(item.get("snippet") or "")[:500], provider=self.name))
        if not sources:  # older response shape: a bare list of citation URLs
            sources = [Source(url=url, provider=self.name) for url in data.get("citations") or [] if url]
        # Perplexity decides how many sources to use; ``num`` is only a lower bound for callers.
        return sources, answer
