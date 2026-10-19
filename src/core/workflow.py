"""``ContentBlitz``: the single entry point for the UI, the notebook and the tests.

    from src.core.workflow import ContentBlitz
    app = ContentBlitz()
    result = app.run("Write an SEO blog about email marketing for bakeries", thread_id="demo")
    print(result.response)

It hides graph construction, checkpointer choice, per-request cost tracking,
recursion limits and error handling. Any unexpected failure comes back as a
friendly message on ``TurnResult``; it never raises into the UI.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from src.agents import AgentDeps
from src.agents.linkedin_writer import swap_hook
from src.core.config import AppConfig, get_config
from src.core.prompts import BRAND_VOICE_LEARNER
from src.integrations.openai_client import message_text
from src.utils.observability import get_logger, setup_logging
from src.utils.quality_validation import learn_brand_voice
from src.workflow.langgraph_workflow import build_graph, make_checkpointer
from src.workflow.state_management import BlogPost, BrandProfile, LinkedInPost, QualityReport, StrategyReport

log = get_logger("app")
ARTIFACT_KEYS = ("research", "report", "blog", "linkedin", "image")


@dataclass
class TurnResult:
    response: str
    route: str = ""
    outputs: list[str] = field(default_factory=list)
    artifacts: dict[str, dict] = field(default_factory=dict)
    quality: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @classmethod
    def from_state(cls, state: dict) -> TurnResult:
        return cls(
            response=state.get("final_response", ""),
            route=state.get("route", ""),
            outputs=list(state.get("outputs_this_turn", [])),
            artifacts={k: state[k] for k in ARTIFACT_KEYS if state.get(k)},
            quality=dict(state.get("quality") or {}),
            warnings=list(state.get("warnings", [])),
            sources=list(state.get("sources", [])),
            usage=dict(state.get("usage") or {}),
        )


class ContentBlitz:
    def __init__(self, config: AppConfig | None = None, deps: AgentDeps | None = None, checkpointer=None):
        self.config = config or get_config()
        setup_logging(self.config.log_level, self.config.log_json)
        self.deps = deps or AgentDeps.from_config(self.config)
        self.checkpointer = checkpointer if checkpointer is not None else make_checkpointer(self.config)
        self.graph = build_graph(self.deps, self.checkpointer)

    def _run_config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": self.config.guardrails.recursion_limit}

    @staticmethod
    def _inputs(message: str, brand: BrandProfile | dict | None) -> dict:
        inputs: dict[str, Any] = {"user_input": message}
        if brand is not None:
            inputs["brand"] = brand.model_dump() if isinstance(brand, BrandProfile) else dict(brand)
        return inputs

    # ---- conversation -------------------------------------------------------------
    def run(self, message: str, thread_id: str = "default", brand: BrandProfile | dict | None = None) -> TurnResult:
        """Process one user turn and return the outcome."""
        self.deps.tracker.reset()
        try:
            state = self.graph.invoke(self._inputs(message, brand), self._run_config(thread_id))
        except GraphRecursionError:
            log.exception("recursion limit hit")
            return TurnResult(response="That request needed too many steps, so I stopped it to protect your "
                                       "budget. Try a narrower request.", error="recursion_limit")
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            log.exception("workflow failed")
            return TurnResult(response=f"Sorry — something went wrong while processing that request ({exc}). "
                                       "Please try again.", error=str(exc))
        return TurnResult.from_state(state)

    def stream(self, message: str, thread_id: str = "default",
               brand: BrandProfile | dict | None = None) -> Iterator[tuple[str, dict]]:
        """Yield ``(node_name, update)`` as each node finishes (for live progress in the UI).

        The final item is ``("__result__", {"result": TurnResult})``.
        """
        self.deps.tracker.reset()
        try:
            for chunk in self.graph.stream(self._inputs(message, brand), self._run_config(thread_id),
                                           stream_mode="updates"):
                for node, update in chunk.items():
                    yield node, update or {}
            result = TurnResult.from_state(self.state(thread_id))
        except GraphRecursionError:
            result = TurnResult(response="That request needed too many steps, so I stopped it.",
                                error="recursion_limit")
        except Exception as exc:  # noqa: BLE001
            log.exception("workflow failed")
            result = TurnResult(response=f"Sorry — something went wrong ({exc}).", error=str(exc))
        yield "__result__", {"result": result}

    # ---- thread state ---------------------------------------------------------------
    def state(self, thread_id: str = "default") -> dict:
        snapshot = self.graph.get_state(self._run_config(thread_id))
        return dict(snapshot.values) if snapshot and snapshot.values else {}

    def history(self, thread_id: str = "default") -> list[tuple[str, str]]:
        messages = self.state(thread_id).get("messages", [])
        return [("user" if isinstance(m, HumanMessage) else "assistant", message_text(m))
                for m in messages if isinstance(m, HumanMessage | AIMessage)]

    def update_artifact(self, thread_id: str, kind: str, data: dict, quality: QualityReport | None = None) -> None:
        """Persist a user's manual edit (Content Dashboard), and optionally its new score, into the thread."""
        update: dict[str, Any] = {kind: data}
        if quality is not None:
            update["quality"] = {**(self.state(thread_id).get("quality") or {}), kind: quality.model_dump()}
        self.graph.update_state(self._run_config(thread_id), update, as_node="finalize")

    def rescore(self, kind: str, data: dict, brand: BrandProfile | dict | None = None,
                thread_id: str = "default") -> QualityReport:
        """Re-run the deterministic quality scoring on edited content."""
        brand = BrandProfile.model_validate(brand or self.state(thread_id).get("brand") or {})
        validator = self.deps.validator
        if kind == "blog":
            known = {s["id"] for s in self.state(thread_id).get("sources", [])}
            return validator.score_blog(BlogPost.model_validate(data), brand, known)
        if kind == "linkedin":
            return validator.score_linkedin(LinkedInPost.model_validate(data), brand)
        return validator.score_report(StrategyReport.model_validate(data), brand)

    def apply_hook(self, thread_id: str, hook: str, brand: BrandProfile | dict | None = None) -> QualityReport:
        """Swap the LinkedIn post's opening line for ``hook`` (Content Dashboard), re-score and persist."""
        post = swap_hook(LinkedInPost.model_validate(self.state(thread_id)["linkedin"]), hook)
        report = self.rescore("linkedin", post.model_dump(), brand, thread_id)
        self.update_artifact(thread_id, "linkedin", post.model_dump(), quality=report)
        return report

    def learn_brand_voice(self, sample_text: str, base: BrandProfile | None = None) -> BrandProfile:
        return learn_brand_voice(sample_text, base or BrandProfile(), self.deps.llm, BRAND_VOICE_LEARNER)

    def service_status(self) -> dict[str, Any]:
        return {
            "llm": self.deps.llm.describe("smart"),
            "keys": self.config.secrets.status(),
            "key_warnings": self.config.secrets.format_warnings(),
            "research_chain": self.deps.search.available_providers,
            "image_chain": self.deps.images.chain,
            "checkpointer": self.config.checkpointer,
        }
