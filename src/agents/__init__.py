"""The six ContentBlitz agents plus their shared plumbing.

Every agent module exposes a ``make_<agent>(deps)`` factory that returns a LangGraph
node function ``node(state) -> partial_state``. All external services come in
through :class:`AgentDeps`, so each agent can be unit-tested on its own with fake
LLMs and fake providers (see ``tests/unit/test_agent_*.py``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.core.config import AppConfig
from src.integrations.image_clients import GeminiImageClient, ImageChain, ImageClients, OpenAIImageClient
from src.integrations.openai_client import LLMService, ModerationClient
from src.integrations.perplexity_client import PerplexityClient
from src.integrations.serp_client import SearchChain, SerpAPIClient
from src.integrations.tavily_client import TavilyClient
from src.utils.cache import TTLCache
from src.utils.guardrails import enforce_citation_integrity
from src.utils.observability import CostTracker
from src.utils.quality_validation import QualityValidator
from src.workflow.state_management import BrandProfile, ContentRequest, ContentState, Source, SourceRegistry


@dataclass
class TurnBudget:
    """Per-request counters used by execution guardrails. Reset at the start of each turn."""

    deep_research_calls: int = 0

    def reset(self) -> None:
        self.deep_research_calls = 0


@dataclass
class AgentDeps:
    config: AppConfig
    llm: LLMService
    search: SearchChain
    images: ImageChain
    validator: QualityValidator
    tracker: CostTracker
    moderator: ModerationClient | None = None
    turn: TurnBudget = field(default_factory=TurnBudget)

    @property
    def output_dir(self) -> Path:
        path = self.config.resolve_path(self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def from_config(cls, config: AppConfig) -> AgentDeps:
        """Wire up the real service clients from configuration."""
        from src.core.prompts import QUALITY_CRITIC

        tracker = CostTracker(config.pricing, config.image_pricing, config.search_pricing)
        secrets = config.secrets
        timeout = config.research.get("request_timeout_s", 25)
        serp = SerpAPIClient(secrets.serpapi_api_key, timeout=timeout)
        providers = {
            "serpapi": serp,
            "tavily": TavilyClient(secrets.tavily_api_key, timeout=timeout),
            "perplexity": PerplexityClient(secrets.perplexity_api_key,
                                           preset=config.research.get("perplexity_preset", "fast")),
        }
        chain = [providers[name] for name in config.research.get("chain", providers) if name in providers]
        cache = TTLCache(config.resolve_path(config.cache.dir), config.cache.ttl_seconds, config.cache.enabled)
        llm = LLMService(config, tracker=tracker)
        return cls(
            config=config,
            llm=llm,
            search=SearchChain(chain, cache=cache, tracker=tracker, keyword_provider=serp),
            images=ImageChain(ImageClients(OpenAIImageClient(secrets.openai_api_key,
                                                             quality=config.images.get("quality", "medium")),
                                           GeminiImageClient(secrets.google_api_key)),
                              config.images.get("chain", []),
                              config.images.get("sizes", {}), tracker,
                              config.images.get("min_dimension_px", 256)),
            validator=QualityValidator(config.quality.pass_threshold, llm=llm, critique_prompt=QUALITY_CRITIC),
            tracker=tracker,
            moderator=ModerationClient(secrets.openai_api_key, enabled=config.guardrails.use_moderation_api),
        )


# --------------------------------------------------------------------------------------
# State helpers
# --------------------------------------------------------------------------------------
def brand_of(state: ContentState) -> BrandProfile:
    return BrandProfile.model_validate(state.get("brand") or {})


def request_of(state: ContentState) -> ContentRequest:
    return ContentRequest.model_validate(state.get("request") or {})


def topic_of(state: ContentState) -> str:
    request = request_of(state)
    return request.topic or (state.get("research") or {}).get("topic", "") or state.get("user_input", "")


def add_output(state: ContentState, name: str) -> list[str]:
    return list(dict.fromkeys([*state.get("outputs_this_turn", []), name]))


def add_warning(state: ContentState, *warnings: str) -> list[str]:
    return [*state.get("warnings", []), *[w for w in warnings if w]]


def advance_pending(state: ContentState, finished: str) -> list[str]:
    """Pop ``finished`` off the campaign queue (no-op when it isn't next)."""
    pending = list(state.get("pending_formats", []))
    if pending and pending[0] == finished:
        pending.pop(0)
    return pending


def format_sources(sources: list[Source]) -> str:
    """Numbered evidence list for prompts. URLs are deliberately omitted so the model cannot copy or alter them."""
    if not sources:
        return "(no sources available — write from general knowledge without statistics)"
    return "\n".join(
        f"[{s.id}] {s.title or s.publisher} — {s.publisher} ({s.retrieved_at[:10]}): {s.snippet[:400]}"
        for s in sorted(sources, key=lambda x: x.id)
    )


# --------------------------------------------------------------------------------------
# Agent ↔ ToolNode loop helpers (pattern from the content-writer template)
# --------------------------------------------------------------------------------------
def clear_scratch() -> list[BaseMessage]:
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES)]


def tool_loop_step(deps: AgentDeps, state: ContentState, system_prompt: str, brief: str, tools: list,
                   tier: str = "fast") -> tuple[AIMessage | None, list[BaseMessage]]:
    """Run one agent step of a tool loop.

    Returns ``(ai_message, new_messages)``. ``ai_message`` is ``None`` when the loop must stop
    (the iteration cap was reached or the model errored); the caller then composes its output.
    """
    scratch = list(state.get("agent_scratch", []))
    new: list[BaseMessage] = []
    if not scratch:
        new = [SystemMessage(system_prompt), HumanMessage(brief)]
        scratch = list(new)
    iterations = sum(1 for m in scratch if isinstance(m, AIMessage))
    if iterations >= deps.config.guardrails.max_tool_iterations:
        return None, new
    try:
        ai = deps.llm.bind_tools(tools, tier).invoke(scratch)
    except Exception:  # noqa: BLE001 - tool-loop failure degrades to direct composition
        return None, new
    return ai, [*new, ai]


def wants_tools(message: AIMessage | None) -> bool:
    return bool(message is not None and getattr(message, "tool_calls", None))


def tool_results(messages: list[BaseMessage], tool_name: str | None = None) -> list[dict[str, Any]]:
    """Parse JSON payloads from ToolMessages (optionally filtered by tool name)."""
    results = []
    for message in messages:
        if isinstance(message, ToolMessage) and (tool_name is None or message.name == tool_name):
            try:
                results.append(json.loads(message.content if isinstance(message.content, str) else "{}"))
            except json.JSONDecodeError:
                continue
    return results


def should_continue_tools(state: ContentState) -> str:
    """Conditional edge: go to the agent's ToolNode if the last scratch message requested tools."""
    scratch = state.get("agent_scratch", [])
    return "tools" if scratch and wants_tools(scratch[-1]) else "done"


# --------------------------------------------------------------------------------------
# Output scrubbing (shared by every writer)
# --------------------------------------------------------------------------------------
def scrub_body(markdown: str, registry: SourceRegistry, valid_ids: set[int]) -> tuple[str, list[str]]:
    """Apply output guardrails to generated Markdown.

    * drop any Sources/References section the model wrote (ours is generated from the registry),
    * remove URLs that are not in the source registry (no invented links),
    * remove ``[n]`` citations that do not correspond to a real source.
    """
    notes: list[str] = []
    markdown = re.split(r"^#{1,3}\s*(Sources|References|Bibliography)\b.*$", markdown, maxsplit=1,
                        flags=re.M | re.I)[0].rstrip()
    markdown, removed = enforce_citation_integrity(markdown, registry.urls)
    if removed:
        notes.append(f"Removed {len(removed)} unverified link(s) the model produced.")
    bad_ids: set[int] = set()

    def fix(match: re.Match) -> str:
        n = int(match.group(1))
        if n in valid_ids:
            return match.group(0)
        bad_ids.add(n)
        return ""

    markdown = re.sub(r"\[(\d{1,3})\](?!\()", fix, markdown)
    if bad_ids:
        notes.append("Removed citation id(s) " + ", ".join(map(str, sorted(bad_ids))) + " that did not match any source.")
    return markdown, notes
