"""Perplexity Agent API client: research + analysis in one call, with citations.

Perplexity replaced the Sonar chat-completions endpoint with the Agent API
(``POST /v1/agent``, ``preset`` + ``input``). ``sonar-pro`` maps to the ``fast`` preset.

It's used for the expensive ``deep_research`` tool (at most once per request) and as
the last live provider in the research fallback chain.
"""

from __future__ import annotations

import requests

from src.integrations import ProviderError, request_json
from src.workflow.state_management import Source

PERPLEXITY_URL = "https://api.perplexity.ai/v1/agent"


class PerplexityClient:
    name = "perplexity"

    def __init__(self, api_key: str | None, preset: str = "fast", timeout: float = 45,
                 session: requests.Session | None = None):
        self.api_key = api_key
        self.preset = preset
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
                "preset": self.preset,
                "instructions": "You are a meticulous research analyst. Cite sources.",
                "input": query,
            },
        )
        answer = ""
        sources: list[Source] = []
        for item in data.get("output") or []:
            if item.get("type") == "search_results":
                for result in item.get("results") or []:
                    if result.get("url"):
                        sources.append(Source(title=result.get("title", ""), url=result["url"],
                                              snippet=(result.get("snippet") or "")[:500], provider=self.name))
            elif item.get("type") == "message":
                answer += "".join(part.get("text", "") for part in item.get("content") or []
                                  if part.get("type") == "output_text")
        # Perplexity decides how many sources to use; ``num`` is only a lower bound for callers.
        return sources, answer
