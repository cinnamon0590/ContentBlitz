"""LLM provider abstraction (OpenAI by default) with model fallbacks.

Every agent talks to the LLM through :class:`LLMService`, never to a provider SDK
directly. That gives us:

* **Provider switching** — ``LLM_PROVIDER=openai|anthropic|gemini`` (no code changes).
* **Model tiers** — ``smart`` for writing and synthesis, ``fast`` (cheaper) for routing,
  scoring and prompt enhancement. This is the main cost-control lever.
* **Fallbacks** — if the primary model errors, the configured fallback models are tried.
* **Testability** — tests inject fake chat models via ``chat_models=``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from src.core.config import AppConfig
from src.utils.observability import CostTracker, get_logger

log = get_logger("llm")
T = TypeVar("T", bound=BaseModel)


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider's package or API key is missing."""


def create_chat_model(provider: str, model: str, temperature: float, config: AppConfig) -> BaseChatModel:
    """Factory for LangChain chat models. Optional providers are imported lazily."""
    timeout = config.llm.get("request_timeout_s", 60)
    retries = config.llm.get("max_retries", 2)
    secrets = config.secrets
    if provider == "openai":
        if not secrets.openai_api_key:
            raise ProviderUnavailableError("OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=retries,
            api_key=secrets.openai_api_key,
        )
    if provider == "anthropic":
        if not secrets.anthropic_api_key:
            raise ProviderUnavailableError("ANTHROPIC_API_KEY is not set")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailableError("pip install langchain-anthropic") from exc
        return ChatAnthropic(model=model, temperature=temperature, timeout=timeout, max_retries=retries,
                             api_key=secrets.anthropic_api_key)
    if provider == "gemini":
        if not secrets.google_api_key:
            raise ProviderUnavailableError("GOOGLE_API_KEY is not set")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailableError("pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, timeout=timeout,
                                      max_retries=retries, google_api_key=secrets.google_api_key)
    raise ProviderUnavailableError(f"Unknown LLM provider '{provider}'")


def _extract_json(text: str) -> Any:
    """Parse the first JSON object/array found in ``text`` (tolerates ```json fences)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in model output")


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
    return "".join(parts)


class LLMService:
    """Tiered, fallback-aware access to chat models."""

    def __init__(
        self,
        config: AppConfig,
        tracker: CostTracker | None = None,
        chat_models: dict[str, BaseChatModel] | None = None,
        fallback_models: dict[str, list[BaseChatModel]] | None = None,
    ):
        self.config = config
        self.tracker = tracker
        self._models: dict[str, BaseChatModel] = dict(chat_models or {})
        self._fallbacks: dict[str, list[BaseChatModel]] = dict(fallback_models or {})
        self._injected = chat_models is not None

    # ---- model construction -------------------------------------------------------
    @property
    def provider(self) -> str:
        return "fake" if self._injected else self.config.llm_provider

    def model_name(self, tier: str = "smart") -> str:
        if self._injected:
            return f"fake-{tier}"
        return self.config.model_for(tier)

    def describe(self, tier: str = "smart") -> str:
        return f"{self.provider}:{self.model_name(tier)}"

    def model(self, tier: str = "smart") -> BaseChatModel:
        if tier not in self._models:
            if self._injected:
                # Tests may inject only one tier; reuse it.
                self._models[tier] = next(iter(self._models.values()))
            else:
                self._models[tier] = create_chat_model(
                    self.config.llm_provider, self.config.model_for(tier), self.config.temperature_for(tier),
                    self.config,
                )
        return self._models[tier]

    def fallbacks(self, tier: str = "smart") -> list[BaseChatModel]:
        if tier not in self._fallbacks:
            if self._injected:
                self._fallbacks[tier] = []
            else:
                built = []
                for name in self.config.model_fallbacks():
                    try:
                        built.append(create_chat_model(self.config.llm_provider, name,
                                                       self.config.temperature_for(tier), self.config))
                    except ProviderUnavailableError as exc:  # pragma: no cover - defensive
                        log.warning("fallback model %s unavailable: %s", name, exc)
                self._fallbacks[tier] = built
        return self._fallbacks[tier]

    def _callbacks(self) -> list:
        return [self.tracker] if self.tracker else []

    # ---- invocation ---------------------------------------------------------------
    def invoke(self, messages: Sequence[BaseMessage], tier: str = "smart") -> AIMessage:
        """Invoke the tier's model, falling back to the configured fallback models on error."""
        candidates = [self.model(tier), *self.fallbacks(tier)]
        last_error: Exception | None = None
        for index, model in enumerate(candidates):
            try:
                result = model.invoke(list(messages), config={"callbacks": self._callbacks()})
                if index:
                    log.warning("LLM fallback #%d succeeded", index)
                return result
            except Exception as exc:  # noqa: BLE001 - any provider error triggers fallback
                last_error = exc
                log.warning("LLM call failed on candidate %d: %s", index, exc)
        raise RuntimeError(f"All LLM candidates failed: {last_error}") from last_error

    def text(self, system: str, user: str, tier: str = "smart") -> str:
        return message_text(self.invoke([SystemMessage(system), HumanMessage(user)], tier))

    def structured(self, schema: type[T], system: str, user: str, tier: str = "fast") -> T | None:
        """Return a validated ``schema`` instance, or ``None`` if every strategy fails.

        Strategy 1: native structured output (function calling).
        Strategy 2: ask for JSON in plain text and parse it (works with any model,
        including the fakes used in tests).
        """
        messages = [SystemMessage(system), HumanMessage(user)]
        try:
            runnable = self.model(tier).with_structured_output(schema, method="function_calling")
            result = runnable.invoke(messages, config={"callbacks": self._callbacks()})
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except NotImplementedError:
            pass
        except Exception as exc:  # noqa: BLE001
            log.info("native structured output failed (%s); trying JSON mode", exc)

        json_hint = (
            "\n\nRespond ONLY with a JSON object matching this JSON schema (no prose):\n"
            + json.dumps(schema.model_json_schema())
        )
        try:
            raw = message_text(self.invoke([SystemMessage(system + json_hint), HumanMessage(user)], tier))
            return schema.model_validate(_extract_json(raw))
        except (ValueError, ValidationError, RuntimeError) as exc:
            log.warning("structured output failed for %s: %s", schema.__name__, exc)
            return None

    def bind_tools(self, tools: Sequence[Any], tier: str = "smart") -> Runnable:
        """Model with tools bound (plus bound fallbacks) for agent ↔ ToolNode loops."""
        primary = self.model(tier).bind_tools(list(tools))
        fallbacks = [m.bind_tools(list(tools)) for m in self.fallbacks(tier)]
        bound = primary.with_fallbacks(fallbacks) if fallbacks else primary
        return bound.with_config({"callbacks": self._callbacks()})


class ModerationClient:
    """Thin wrapper over the (free) OpenAI moderation endpoint. Fails open with a warning."""

    def __init__(self, api_key: str | None, enabled: bool = True, client: Any = None):
        self.enabled = enabled and bool(api_key or client)
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def check(self, text: str) -> tuple[bool, list[str]]:
        """Return ``(flagged, categories)``."""
        if not self.enabled or not text.strip():
            return False, []
        try:
            result = self._get_client().moderations.create(model="omni-moderation-latest", input=text[:8000])
            item = result.results[0]
            categories = item.categories.model_dump() if hasattr(item.categories, "model_dump") else dict(item.categories)
            flagged_cats = [name for name, hit in categories.items() if hit]
            return bool(item.flagged), flagged_cats
        except Exception as exc:  # noqa: BLE001 - moderation outage must not block the app
            log.warning("moderation API unavailable: %s", exc)
            return False, []
