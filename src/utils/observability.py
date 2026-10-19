"""Logging and cost/usage tracking.

* :func:`setup_logging` configures plain or JSON logs (production uses JSON).
* :class:`CostTracker` is a LangChain callback that records token usage per model
  and converts it to USD using ``config/services.yaml`` pricing. It also records
  search and image calls, so the UI can show cost per request and the
  guardrails can enforce a per-request budget.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

LOGGER_NAME = "contentblitz"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            payload["data"] = extra
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_logs: bool = False) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            _JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


class CostTracker(BaseCallbackHandler):
    """Accumulates token usage and estimated cost. Call :meth:`reset` per request."""

    def __init__(
        self,
        pricing: dict[str, list[float]] | None = None,
        image_pricing: dict[str, float] | None = None,
        search_pricing: dict[str, float] | None = None,
    ):
        self.pricing = pricing or {}
        self.image_pricing = image_pricing or {}
        self.search_pricing = search_pricing or {}
        self.reset()

    def reset(self) -> None:
        self.tokens: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})
        self.searches: dict[str, int] = defaultdict(int)
        self.images: dict[str, int] = defaultdict(int)
        self.cache_hits = 0
        self.started_at = time.time()

    # ---- LangChain callback -------------------------------------------------------
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        model = "unknown"
        input_tokens = output_tokens = 0
        llm_output = response.llm_output or {}
        model = llm_output.get("model_name") or llm_output.get("model") or model
        usage = llm_output.get("token_usage") or {}
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0
        if not (input_tokens or output_tokens):
            for generations in response.generations:
                for gen in generations:
                    message = getattr(gen, "message", None)
                    meta = getattr(message, "usage_metadata", None) or {}
                    input_tokens += meta.get("input_tokens", 0)
                    output_tokens += meta.get("output_tokens", 0)
                    resp_meta = getattr(message, "response_metadata", None) or {}
                    model = resp_meta.get("model_name", model)
        self.record_llm(model, input_tokens, output_tokens)

    # ---- manual recording ---------------------------------------------------------
    def record_llm(self, model: str, input_tokens: int, output_tokens: int) -> None:
        entry = self.tokens[model]
        entry["input"] += int(input_tokens)
        entry["output"] += int(output_tokens)
        entry["calls"] += 1

    def record_search(self, provider: str) -> None:
        self.searches[provider] += 1

    def record_image(self, model: str) -> None:
        self.images[model] += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    # ---- reporting ---------------------------------------------------------------
    def _price(self, model: str) -> list[float] | None:
        if model in self.pricing:
            return self.pricing[model]
        # Versioned names such as "gpt-4o-2024-08-06" -> longest matching prefix.
        matches = [name for name in self.pricing if model.startswith(name)]
        return self.pricing[max(matches, key=len)] if matches else None

    @property
    def total_cost(self) -> float:
        cost = 0.0
        for model, t in self.tokens.items():
            price = self._price(model)
            if price:
                cost += t["input"] / 1_000_000 * price[0] + t["output"] / 1_000_000 * price[1]
        cost += sum(self.image_pricing.get(m, 0.0) * n for m, n in self.images.items())
        cost += sum(self.search_pricing.get(p, 0.0) * n for p, n in self.searches.items())
        return round(cost, 6)

    def over_budget(self, budget_usd: float) -> bool:
        return budget_usd > 0 and self.total_cost > budget_usd

    def summary(self) -> dict[str, Any]:
        return {
            "llm": {m: dict(t) for m, t in self.tokens.items()},
            "unpriced_models": sorted(m for m in self.tokens if self._price(m) is None),
            "searches": dict(self.searches),
            "images": dict(self.images),
            "cache_hits": self.cache_hits,
            "total_tokens": sum(t["input"] + t["output"] for t in self.tokens.values()),
            "estimated_cost_usd": self.total_cost,
            "elapsed_s": round(time.time() - self.started_at, 2),
        }
