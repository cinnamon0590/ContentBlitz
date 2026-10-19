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


# --------------------------------------------------------------------------------------
# Scripted multi-agent responder + deps factory
# --------------------------------------------------------------------------------------
import itertools  # noqa: E402

_ids = itertools.count(1)


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def _loop(name: str, **args):
    """First call requests the tool; once a tool result is present, finish."""
    return lambda msgs: "DONE" if has_tool_result(msgs) else tool_call(name, **args)


LINKEDIN_BODY = "\n\n".join([
    "Last spring we tested a 5-minute reply rule with three local clients.",
    "Bookings rose within a month. Customers told us speed felt like care.",
    "Here is the simple system we used, step by step, so you can copy it.",
    "First, set up instant alerts. Second, write three reply templates. Third, review weekly.",
    "None of this needs new software. It needs a habit and a shared inbox. A recent report found most buyers "
    "choose the first business that replies, which matches what we saw with every client we work with.",
    "The surprising part was how much calmer the team felt with a clear rule. Fewer dropped leads, fewer "
    "late-night catch-ups, and more time for the work customers actually pay for.",
    "If you run a small team, start with one channel and measure reply time for two weeks. Then add the next "
    "channel only when the first one is consistently fast. Small, boring habits beat big campaigns.",
    "What is your average response time today? Tell me in the comments.",
])


def default_roles(intent: str = "blog", **request_overrides) -> dict:
    request = {"intent": intent, "topic": "email marketing for bakeries", "target_keyword": "email marketing",
               "needs_research": True, "formats": ["blog", "linkedin", "image"] if intent == "campaign" else [],
               **request_overrides}
    return {
        "query_handler": request,
        "post_research_router": {"next": "report", "reason": "research only"},
        "general": "Hello! I can write blogs, LinkedIn posts, images and research.",
        "research_agent": _loop("web_search", query="bakery email marketing statistics 2026"),
        "research_synthesis": {
            "summary": "Email remains a top channel for bakeries [1]. Segmentation lifts opens [2].",
            "key_findings": [{"claim": "Segmented emails get 30% more opens [1]", "source_ids": [1]},
                             {"claim": "Weekly sends perform best for food brands", "source_ids": [2, 99]}],
            "statistics": ["30% more opens with segmentation [1]"],
            "trends": ["AI-personalised subject lines"], "keywords": ["bakery newsletter", "email list growth"],
        },
        "blog_writer.research": _loop("keyword_research", seed="email marketing"),
        "blog_writer": {
            "title": "Email Marketing for Bakeries: A Practical Guide",
            "meta_description": "Email marketing for bakeries made simple: build your list, write emails people "
                                "open, and measure what matters with this practical, step-by-step guide.",
            "markdown": good_blog_markdown() + "\n\n## Sources\n\n- https://invented.example/fake",
            "secondary_keywords": ["bakery newsletter"],
            "internal_link_suggestions": ["/blog/bakery-newsletter-ideas"],
        },
        "linkedin_writer.research": _loop("trending_hashtags", topic="email marketing"),
        "linkedin_writer": {
            "hook": "Most small shops lose leads to slow replies.",
            "body": LINKEDIN_BODY + " See https://not-allowed.example [1]",
            "hashtags": ["SmallBusiness", "EmailMarketing", "#Bakery", "Marketing", "CustomerService",
                         "Leads", "Growth", "LocalBusiness", "Entrepreneur"],
        },
        "image_prompt_enhancer": {"enhanced_prompt": "Warm bakery counter at sunrise, croissants, soft light",
                                  "alt_text": "Fresh croissants on a bakery counter at sunrise"},
        "content_strategist": ("# Trend Report: Bakery Email\n\n## Executive Summary\n\nEmail works [1].\n\n"
                               "## Key Findings\n\n- Segmentation lifts opens 30% [1] [42]\n\n## Recommendations\n\n"
                               "- Send weekly. See https://made-up.example\n\n## Sources\n\n- junk"),
        "quality_critic": {"feedback": "Solid and on-brand.", "suggestions": []},
        "brand_voice_learner": {"tone": "warm", "audience": "locals", "values": [], "signature_phrases": [],
                                "avoid_words": []},
    }


@pytest.fixture
def make_deps(config, tmp_path):
    from src.agents import AgentDeps
    from src.core.prompts import QUALITY_CRITIC
    from src.integrations.image_clients import ImageChain
    from src.integrations.serp_client import SearchChain
    from src.utils.cache import TTLCache
    from src.utils.observability import CostTracker
    from src.utils.quality_validation import QualityValidator

    class FakeKeywords:
        name = "serpapi"
        available = True

        def keyword_ideas(self, query):
            return {"related_searches": [f"{query} tips", f"{query} examples"],
                    "people_also_ask": [f"What is {query}?"]}

    def _make(roles: dict | None = None, search_fail: bool = False, image_fail: set | None = None,
              threshold: float = 70, providers: list | None = None):
        tracker = CostTracker(config.pricing, config.image_pricing, config.search_pricing)
        responder = responder_by_role(roles if roles is not None else default_roles())
        llm = LLMService(config, tracker=tracker, chat_models={"smart": FakeChatModel(responder=responder),
                                                                "fast": FakeChatModel(responder=responder)})
        providers = providers or [FakeSearchProvider("serpapi", sample_sources(), fail=search_fail),
                                  FakeSearchProvider("tavily", sample_sources("tavily"), fail=search_fail)]
        search = SearchChain(providers, cache=TTLCache(tmp_path / "cache"), tracker=tracker,
                             keyword_provider=FakeKeywords())
        images = ImageChain(FakeImageClient(image_fail or set()), config.images["chain"], config.images["sizes"],
                            tracker)
        return AgentDeps(config=config, llm=llm, search=search, images=images,
                         validator=QualityValidator(threshold, llm=llm, critique_prompt=QUALITY_CRITIC),
                         tracker=tracker)

    return _make


def apply_update(state: dict, update: dict) -> dict:
    """Apply a node's partial update using the same reducers as ContentState."""
    from langgraph.graph.message import add_messages

    from src.workflow.state_management import merge_sources

    new = dict(state)
    for key, value in update.items():
        if key in ("messages", "agent_scratch"):
            new[key] = add_messages(new.get(key, []), value)
        elif key == "sources":
            new[key] = merge_sources(new.get(key, []), value)
        else:
            new[key] = value
    return new


def drive_agent(node, tools: list, state: dict, max_steps: int = 10) -> dict:
    """Run agent ↔ ToolNode until the agent stops requesting tools (mirrors the graph wiring)."""
    from langchain_core.messages import ToolMessage

    by_name = {t.name: t for t in tools}
    for _ in range(max_steps):
        state = apply_update(state, node(state))
        scratch = state.get("agent_scratch", [])
        if scratch and getattr(scratch[-1], "tool_calls", None):
            results = [ToolMessage(content=str(by_name[c["name"]].invoke(c["args"])), name=c["name"],
                                   tool_call_id=c["id"]) for c in scratch[-1].tool_calls]
            state = apply_update(state, {"agent_scratch": results})
            continue
        return state
    raise AssertionError("tool loop did not terminate")
