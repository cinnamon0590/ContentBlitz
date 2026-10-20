from langchain_core.messages import AIMessage, HumanMessage

from src.agents.query_handler import (
    finalize_request,
    heuristic_request,
    make_general_handler,
    make_post_research_router,
    make_query_handler,
)
from src.workflow.state_management import MODEL_KNOWLEDGE_LABEL, ContentRequest
from tests.conftest import default_roles


def test_routes_blog_to_research_first(make_deps):
    out = make_query_handler(make_deps())({"user_input": "Write an SEO blog about email marketing for bakeries"})
    assert out["route"] == "research" and out["request"]["intent"] == "blog"
    assert out["revision_count"] == 0


def test_skips_research_when_matching_research_exists(make_deps):
    deps = make_deps(default_roles("linkedin"))
    state = {"user_input": "now a LinkedIn post", "research": {"topic": "email marketing for bakeries"}}
    out = make_query_handler(deps)(state)
    assert out["route"] == "linkedin" and out["request"]["needs_research"] is False


def test_campaign_sets_pending_formats(make_deps):
    out = make_query_handler(make_deps(default_roles("campaign")))({"user_input": "full campaign on croissants"})
    assert out["route"] == "research" and out["pending_formats"] == ["blog", "linkedin", "image"]


def test_image_and_general_routes(make_deps):
    out = make_query_handler(make_deps(default_roles("image", image_prompt="a latte")))({"user_input": "image"})
    assert out["route"] == "image" and out["request"]["image_prompt"] == "a latte"
    out = make_query_handler(make_deps(default_roles("general")))({"user_input": "hi"})
    assert out["route"] == "general"


def test_llm_failure_falls_back_to_heuristics(make_deps):
    deps = make_deps({"query_handler": "not json at all"})
    out = make_query_handler(deps)({"user_input": "Create a LinkedIn post about remote work"})
    assert out["request"]["intent"] == "linkedin" and out["request"]["reasoning"] == "heuristic fallback"


def test_heuristics_cover_intents():
    cases = {
        "make me an image of a cafe": "image",
        "blog post on SEO": "blog",
        "research coffee trends": "research",
        "summarize it as a report": "report",
        "a full campaign please": "campaign",
        "hello": "general",
    }
    for text, intent in cases.items():
        assert heuristic_request(text, {}).intent == intent, text
    refine = heuristic_request("make it shorter", {"blog": {"title": "x"}})
    assert refine.refine_target == "blog" and refine.intent == "blog"
    refine_li = heuristic_request("make the linkedin post more casual", {"linkedin": {"text": "x"}})
    assert refine_li.refine_target == "linkedin"


def test_finalize_request_rules():
    state = {"research": {"topic": "vegan menu ideas"}, "blog": {"title": "x"}}
    r = finalize_request(ContentRequest(intent="blog", refine_target="blog", needs_research=True), state)
    assert r.needs_research is False and r.topic == "vegan menu ideas"
    r = finalize_request(ContentRequest(intent="linkedin", refine_target="linkedin", topic="t"), {})
    assert r.refine_target == "" and r.needs_research is True
    r = finalize_request(ContentRequest(intent="report"), state)
    assert r.needs_research is False
    r = finalize_request(ContentRequest(intent="campaign", formats=["video", "blog", "blog"]), {})
    assert r.formats == ["blog"]
    r = finalize_request(ContentRequest(intent="campaign", formats=["video"]), {})
    assert r.formats == ["blog", "linkedin", "image"]


def test_post_research_router(make_deps):
    router = make_post_research_router(make_deps())
    assert router({"pending_formats": ["linkedin"], "request": {"intent": "campaign"}})["next_after_research"] == "linkedin"
    assert router({"request": {"intent": "blog"}})["next_after_research"] == "blog"
    assert router({"request": {"intent": "report"}})["next_after_research"] == "report"
    assert router({"request": {"intent": "research"}, "research": {"summary": "s"}})["next_after_research"] == "report"
    bad = make_post_research_router(make_deps({"post_research_router": "???"}))
    assert bad({"request": {"intent": "research"}})["next_after_research"] == "report"


def test_general_handler_uses_history_and_labels(make_deps):
    roles = default_roles()
    seen = {}

    def general(messages):
        seen["n"] = len(messages)
        return "Your last request was about bakery emails."

    roles["general"] = general
    state = {"messages": [HumanMessage("blog about bakery emails"), AIMessage("done"), HumanMessage("what did I ask?")],
             "request": {"intent": "general", "clarifying_question": "Which format would you like?"}}
    out = make_general_handler(make_deps(roles))(state)
    assert "bakery emails" in out["final_response"] and MODEL_KNOWLEDGE_LABEL in out["final_response"]
    assert "Which format" in out["final_response"] and seen["n"] == 4


def test_general_handler_llm_down(config):
    from src.agents import AgentDeps
    from src.integrations.openai_client import LLMService
    from tests.conftest import FailingChatModel

    deps = AgentDeps(config=config, llm=LLMService(config, chat_models={"fast": FailingChatModel()}), search=None,
                     images=None, validator=None, tracker=None)
    out = make_general_handler(deps)({"messages": [HumanMessage("hi")]})
    assert "trouble reaching" in out["final_response"]


def test_topic_discovery_becomes_trend_report():
    from src.agents.query_handler import finalize_request
    from src.workflow.state_management import ContentRequest

    campaign = ContentRequest(intent="campaign", topic="bakery content", formats=["blog", "linkedin"])
    request = finalize_request(campaign, {"user_input": "What should I post about this month for my bakery?"})
    assert (request.intent, request.template, request.formats, request.needs_research) == (
        "research", "trend report", [], True)
    named = ContentRequest(intent="linkedin", topic="bakery")
    assert finalize_request(named, {"user_input": "Give me LinkedIn post ideas for my bakery"}).intent == "linkedin"


def test_report_template_always_routes_to_report(make_deps):
    from src.agents.query_handler import make_post_research_router

    roles = default_roles("research")
    roles["post_research_router"] = {"next": "blog", "reason": "model would pick a blog"}
    node = make_post_research_router(make_deps(roles))
    state = {"request": {"intent": "research", "template": "trend report"}, "user_input": "what should I post?"}
    assert node(state) == {"next_after_research": "report"}
    assert node({"request": {"intent": "research"}, "user_input": "x"}) == {"next_after_research": "blog"}


def test_refine_keeps_the_posts_keyword_and_default_keywords_end_cleanly():
    from src.agents.query_handler import default_keyword

    state = {"blog": {"target_keyword": "local bakery email marketing", "markdown": "# x"}}
    r = finalize_request(ContentRequest(intent="blog", refine_target="blog", topic="email marketing for local bakeries",
                                        target_keyword="email marketing for local"), state)
    assert r.target_keyword == "local bakery email marketing"
    assert default_keyword("email marketing for local bakeries in Austin") == "email marketing for local bakeries"
    assert default_keyword("ideas for the") == "ideas"
    assert default_keyword("guide to the best croissants") == "guide to the best croissants"
