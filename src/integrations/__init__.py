"""External service clients (LLMs, web research, image generation).

``request_json`` is the shared HTTP helper. It retries transient failures
(timeouts, connection errors, HTTP 429/5xx) with exponential backoff, and raises
:class:`ProviderError` for anything else, so the caller can fall through to the
next provider in the chain.
"""

from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


class ProviderError(RuntimeError):
    """A provider call failed in a way that should trigger fallback."""

    def __init__(self, provider: str, message: str, status: int | None = None):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status = status


class _Transient(Exception):
    pass


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (_Transient, requests.ConnectionError, requests.Timeout))


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
def _send(session: requests.Session, method: str, url: str, timeout: float, **kwargs: Any) -> requests.Response:
    response = session.request(method, url, timeout=timeout, **kwargs)
    if response.status_code == 429 or response.status_code >= 500:
        raise _Transient(f"HTTP {response.status_code}")
    return response


def request_json(
    provider: str,
    method: str,
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 25,
    **kwargs: Any,
) -> dict[str, Any]:
    session = session or requests.Session()
    try:
        response = _send(session, method, url, timeout, **kwargs)
    except (_Transient, requests.RequestException) as exc:
        raise ProviderError(provider, f"request failed after retries: {exc}") from exc
    if response.status_code >= 400:
        raise ProviderError(provider, f"HTTP {response.status_code}: {response.text[:200]}", response.status_code)
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(provider, "invalid JSON response") from exc
