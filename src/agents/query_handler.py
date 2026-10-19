"""Agent 1 — Query Handler: the smart router.

Two routing decisions (per the project spec):

* **Initial routing** (:func:`make_query_handler`): an LLM classifier with structured
  output and conversation context (follow-ups such as "now make it a LinkedIn post").
  It extracts a :class:`ContentRequest`: intent, topic, keyword, template and whether
  research is needed. If the LLM fails, a keyword heuristic takes over (redundancy).
* **Post-research routing** (:func:`make_post_research_router`): once research is
  done, it decides between blog, LinkedIn, image, or a research report from the
  Content Strategist.

It also owns the **general / clarify** path (:func:`make_general_handler`), which
answers greetings, questions about earlier messages (via checkpointer memory) and
ambiguous requests.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.agents import AgentDeps, request_of
from src.core import prompts
from src.core.templates import REPORT_TEMPLATES
from src.integrations.openai_client import message_text
from src.utils.observability import get_logger
from src.workflow.state_management import MODEL_KNOWLEDGE_LABEL, ContentRequest, ContentState

log = get_logger("query_handler")

_HEURISTICS: list[tuple[str, str]] = [
    (r"\b(campaign|content series|series of|all formats|blog (and|\+|&) linkedin|everything)\b", "campaign"),
    (r"\b(image|picture|photo|visual|graphic|banner|illustration|thumbnail)\b", "image"),
    (r"\blinked ?in\b", "linkedin"),
    (r"\b(blog|article|seo post|long[- ]form)\b", "blog"),
    (r"\b(report|summari[sz]e|summary)\b", "report"),
    (r"\b(research|trends?|trending|statistics|stats|analy[sz]e|analysis|insights?|market)\b"
     r"|\bwhat (should|can) (i|we) (write|post)\b|\b(content|topic|post) ideas\b", "research"),
]
# "What should I post about?" asks for ideas, not content: answer with a trend report unless a format is named.
_TOPIC_DISCOVERY = r"\bwhat (should|can|do) (i|we) (write|post|blog|talk) about\b|\b(content|topic|post) ideas\b"
_NAMED_FORMAT = r"\b(blog|article|linked ?in|image|picture|photo|visual|campaign|series)\b"
_REFINE = r"\b(make it|make the|shorter|longer|rewrite|revise|edit|tweak|more (casual|formal|fun)|change the tone)\b"


def heuristic_request(text: str, state: ContentState) -> ContentRequest:
    """Keyword-rule fallback used when the LLM router is unavailable."""
    lowered = text.lower()
    intent = next((name for pattern, name in _HEURISTICS if re.search(pattern, lowered)), "general")
    refine_target = ""
    if re.search(_REFINE, lowered):
        if "linkedin" in lowered and state.get("linkedin"):
            refine_target, intent = "linkedin", "linkedin"
        elif state.get("blog") and intent in ("blog", "general"):
            refine_target, intent = "blog", "blog"
    topic = re.sub(r"\b(write|create|make|generate|an?|the|seo|blog|post|linkedin|about|on|for|me|please)\b", " ",
                   lowered)
    topic = re.sub(r"\s+", " ", topic).strip(" .?!") or (state.get("research") or {}).get("topic", "")
    return ContentRequest(intent=intent, topic=topic[:80], target_keyword=" ".join(topic.split()[:3]),
                          refine_target=refine_target, refine_instructions=text if refine_target else "",
                          formats=["blog", "linkedin", "image"] if intent == "campaign" else [],
                          reasoning="heuristic fallback")


def _context_summary(state: ContentState, max_messages: int = 6) -> str:
    lines = []
    for message in state.get("messages", [])[-max_messages - 1 : -1]:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"{role}: {message_text(message)[:300]}")
    artifacts = []
    if research := state.get("research"):
        artifacts.append(f"research on '{research.get('topic', '')}'")
        rising = [s for s in research.get("trend_signals", []) if s.get("kind") == "rising"][:5]
        if rising:  # lets "write about the fastest-rising one" resolve to a concrete topic
            artifacts.append("trend radar (fastest-rising first): "
                             + "; ".join(f"{s['query']} ({s['growth']})" for s in rising))
    if blog := state.get("blog"):
        artifacts.append(f"blog draft '{blog.get('title', '')}'")
    if state.get("linkedin"):
        artifacts.append("LinkedIn draft")
    if state.get("image"):
        artifacts.append("image")
    return ("Conversation so far:\n" + ("\n".join(lines) or "(none)") +
            "\n\nExisting artifacts: " + (", ".join(artifacts) or "none"))


def _topics_overlap(a: str, b: str) -> bool:
    stop = {"the", "a", "an", "for", "of", "and", "in", "on", "to", "about", "small", "business", "businesses"}
    wa = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if w not in stop}
    wb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if w not in stop}
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= 0.5


def finalize_request(request: ContentRequest, state: ContentState) -> ContentRequest:
    """Deterministic post-processing of the classifier output (guards against LLM inconsistency)."""
    research = state.get("research") or {}
    text = state.get("user_input", "")
    if re.search(_TOPIC_DISCOVERY, text, re.I) and not re.search(_NAMED_FORMAT, text, re.I):
        request.intent, request.template, request.formats = "research", "trend report", []
    if not request.topic and research:
        request.topic = research.get("topic", "")
    if request.refine_target and not state.get(request.refine_target):
        request.refine_target = ""  # nothing to refine yet → write fresh
    if request.intent in ("blog", "linkedin", "campaign", "research"):
        has_matching_research = bool(research) and _topics_overlap(request.topic, research.get("topic", ""))
        if request.refine_target:
            request.needs_research = False
        elif request.intent == "research":
            request.needs_research = True
        elif has_matching_research:
            request.needs_research = False
        else:
            request.needs_research = True
    if request.intent == "report":
        request.needs_research = not research
    if request.intent == "image":
        request.needs_research = False
        request.image_prompt = request.image_prompt or request.topic
    if request.intent == "campaign":
        allowed = [f for f in request.formats if f in ("blog", "linkedin", "image")]
        request.formats = list(dict.fromkeys(allowed)) or ["blog", "linkedin", "image"]
    if not request.target_keyword and request.topic:
        request.target_keyword = " ".join(request.topic.split()[:4])
    return request


def make_query_handler(deps: AgentDeps):
    def query_handler(state: ContentState) -> dict:
        text = state.get("user_input", "")
        request = deps.llm.structured(
            ContentRequest, prompts.QUERY_HANDLER, f"{_context_summary(state)}\n\nLATEST USER MESSAGE:\n{text}",
            tier="fast",
        )
        if request is None:
            log.warning("LLM router failed; using heuristic routing")
            request = heuristic_request(text, state)
        request = finalize_request(request, state)
        route = "research" if request.needs_research and request.intent != "general" else request.intent
        log.info("route=%s intent=%s topic=%r", route, request.intent, request.topic)
        return {
            "request": request.model_dump(),
            "route": route,
            "pending_formats": list(request.formats) if request.intent == "campaign" else [],
            "revision_count": 0,
            "revision_feedback": "",
        }

    return query_handler


class PostResearchDecision(BaseModel):
    next: Literal["blog", "linkedin", "image", "report"] = "report"
    reason: str = ""


def make_post_research_router(deps: AgentDeps):
    def post_research_router(state: ContentState) -> dict:
        request = request_of(state)
        pending = state.get("pending_formats", [])
        if pending:
            decision = pending[0]
        elif request.intent in ("blog", "linkedin", "image"):
            decision = request.intent
        elif request.intent == "report" or request.template in REPORT_TEMPLATES:
            decision = "report"
        else:  # research or unclear → ask the LLM; default to the research report
            research = state.get("research") or {}
            result = deps.llm.structured(
                PostResearchDecision, prompts.POST_RESEARCH_ROUTER,
                f"User request: {state.get('user_input', '')}\nResearch summary: {research.get('summary', '')[:800]}",
                tier="fast",
            )
            decision = result.next if result else "report"
        return {"next_after_research": decision}

    return post_research_router


def make_general_handler(deps: AgentDeps):
    def general_handler(state: ContentState) -> dict:
        request = request_of(state)
        # Recent turns only, each truncated: enough for "what did I ask earlier?" without huge prompts.
        history = [type(m)(content=message_text(m)[:1500])
                   for m in state.get("messages", []) if isinstance(m, HumanMessage | AIMessage)][-12:]
        system = prompts.GENERAL
        if request.clarifying_question:
            system += f"\nAsk this clarifying question: {request.clarifying_question}"
        try:
            reply = message_text(deps.llm.invoke([SystemMessage(system), *history], tier="fast"))
        except RuntimeError:
            reply = ("I'm having trouble reaching the language model right now. I can create research reports, "
                     "SEO blog posts, LinkedIn posts and images — please try again in a moment.")
        if request.clarifying_question and request.clarifying_question not in reply:
            reply = f"{reply}\n\n{request.clarifying_question}".strip()
        return {"final_response": f"{reply}\n\n_{MODEL_KNOWLEDGE_LABEL}_",
                "outputs_this_turn": [*state.get("outputs_this_turn", []), "general"]}

    return general_handler
