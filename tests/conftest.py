"""Shared test fixtures: offline config, a scriptable fake chat model, fake providers.

Nothing here touches the network. Each agent's system prompt starts with
``ROLE: <name>``, so :class:`FakeChatModel` can serve as every agent at once by
dispatching on that marker (see ``responder_by_role``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.core.config import load_config
from src.integrations.openai_client import LLMService
from src.workflow.state_management import Source

Responder = Callable[[list[BaseMessage]], AIMessage | str]


class FakeChatModel(BaseChatModel):
    """Chat model whose replies come from ``responder(messages)``. Supports ``bind_tools``."""

    responder: Any
    calls: list = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.calls.append(messages)
        reply = self.responder(messages)
        if isinstance(reply, str):
            reply = AIMessage(content=reply)
        return ChatResult(generations=[ChatGeneration(message=reply)],
                          llm_output={"model_name": "fake", "token_usage": {"prompt_tokens": 10,
                                                                             "completion_tokens": 5}})

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:  # type: ignore[override]
        return self


class FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("provider down")

    def bind_tools(self, tools: Any, **kwargs: Any) -> FailingChatModel:  # type: ignore[override]
        return self


def system_role(messages: list[BaseMessage]) -> str:
    for m in messages:
        if isinstance(m, SystemMessage) and isinstance(m.content, str) and m.content.startswith("ROLE:"):
            return m.content.split("\n", 1)[0].removeprefix("ROLE:").strip()
    return ""


def has_tool_result(messages: list[BaseMessage]) -> bool:
    return any(getattr(m, "type", "") == "tool" for m in messages)


def responder_by_role(table: dict[str, Responder | str | dict]) -> Responder:
    """Build a responder from ``{role: reply}``. Dict replies are JSON-encoded."""

    def respond(messages: list[BaseMessage]) -> AIMessage | str:
        role = system_role(messages)
        reply = table.get(role, table.get("*", "ok"))
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, dict | list):
            reply = json.dumps(reply)
        return reply

    return respond


@pytest.fixture
def config(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "SERPAPI_API_KEY", "TAVILY_API_KEY", "PERPLEXITY_API_KEY",
                "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER"):
        monkeypatch.setenv(key, "")
    cfg = load_config("development")
    cfg.cache.dir = str(tmp_path / "cache")
    cfg.output_dir = str(tmp_path / "outputs")
    cfg.sqlite_path = str(tmp_path / "checkpoints.sqlite")
    cfg.guardrails.use_moderation_api = False
    return cfg


@pytest.fixture
def make_llm(config):
    def _make(responder: Responder, tracker=None) -> LLMService:
        return LLMService(config, tracker=tracker, chat_models={"smart": FakeChatModel(responder=responder),
                                                                 "fast": FakeChatModel(responder=responder)})

    return _make


class FakeSearchProvider:
    def __init__(self, name: str, sources: list[Source] | None = None, fail: bool = False,
                 available: bool = True, answer: str = ""):
        self.name = name
        self._sources = sources or []
        self.fail = fail
        self._available = available
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    @property
    def available(self) -> bool:
        return self._available

    def search(self, query: str, num: int = 5, depth: str = "basic"):
        from src.integrations import ProviderError

        self.calls.append((query, depth))
        if self.fail:
            raise ProviderError(self.name, "simulated outage")
        return [s.model_copy() for s in self._sources[:num]], self.answer


def sample_sources(provider: str = "serpapi", n: int = 3) -> list[Source]:
    return [
        Source(title=f"Study {i}", url=f"https://example{i}.com/article", snippet=f"Finding {i}: 4{i}% growth",
               provider=provider)
        for i in range(1, n + 1)
    ]


@pytest.fixture
def sources():
    return sample_sources()


class FakeImageClient:
    """Mimics OpenAIImageClient. ``fail_models`` simulates provider outages."""

    def __init__(self, fail_models: set[str] | None = None):
        self.fail_models = fail_models or set()
        self.calls: list[str] = []

    available = True

    def generate(self, model: str, prompt: str, size: str) -> bytes:
        import io

        from PIL import Image

        self.calls.append(model)
        if model in self.fail_models:
            raise RuntimeError(f"{model} unavailable (simulated)")
        w, h = (int(x) for x in size.split("x"))
        buf = io.BytesIO()
        Image.new("RGB", (w // 4, h // 4), (10, 120, 200)).save(buf, format="PNG")
        return buf.getvalue()


# --------------------------------------------------------------------------------------
# Content fixtures
# --------------------------------------------------------------------------------------
_FILLER = ("Small teams can plan a simple weekly routine. Start with one clear goal and a short list of tasks. "
           "Write down what worked and what did not. Keep each step small so it is easy to repeat. "
           "Ask your customers what they need and use their words in your copy. ")


def good_blog_markdown(keyword: str = "email marketing") -> str:
    """~1300-word blog that satisfies the SEO / structure / citation rules."""
    sections = []
    for i, heading in enumerate(["Why " + keyword.title() + " Still Works", "Build Your List the Right Way",
                                 "Write Emails People Open", "Measure What Matters"]):
        paras = "\n\n".join(_FILLER * 2 for _ in range(3))
        sections.append(f"## {heading}\n\n{paras} Good {keyword} keeps this simple.\n\n### Quick tip {i + 1}\n\n"
                        f"Brands that segment lists see 30% more opens [1]. {_FILLER}")
    faq = (f"## FAQ\n\n### Is {keyword} worth it in 2026?\n\nYes. Most owners see steady results [2].\n\n"
           f"### How often should I send?\n\nOnce a week is a good start.")
    intro = (f"# The Small Business Guide to {keyword.title()}\n\n"
             f"This guide to {keyword} shows small teams how to grow sales with simple steps. "
             f"You will learn practical {keyword} habits that fit a busy week.")
    outro = f"## Conclusion\n\nStart small and stay consistent with {keyword}. Get in touch to learn how we can help."
    return "\n\n".join([intro, *sections, faq, outro])
