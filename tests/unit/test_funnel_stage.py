"""Buyer's-journey targeting (HubSpot content-marketing guide): stage inference, prompts, stored stage."""

import pytest

from src.agents.blog_writer import make_blog_tools, make_blog_writer
from src.agents.linkedin_writer import make_linkedin_tools, make_linkedin_writer
from src.core.templates import FUNNEL_STAGES, funnel_note, suggest_funnel_stage
from tests.conftest import default_roles, drive_agent


@pytest.mark.parametrize(("text", "template", "stage"), [
    ("What is email marketing and how do I start?", "how-to", "awareness"),
    ("Best email tools for bakeries compared", "listicle", "consideration"),
    ("Square vs Mailchimp for a small cafe", "", "consideration"),
    ("A post announcing our weekend pre-order special", "", "decision"),
    ("Blog about our wedding cake pricing", "", "decision"),
    ("Write about our new menu", "announcement", "decision"),  # template fallback
    ("Write about sourdough", "comparison", "consideration"),
])
def test_suggest_funnel_stage(text, template, stage):
    assert suggest_funnel_stage(text, template) == stage


def test_explicit_stage_wins_and_bad_values_fall_back():
    assert suggest_funnel_stage("best tools compared", "", "Decision") == "decision"
    assert suggest_funnel_stage("how to start", "", "top of funnel") == "awareness"
    assert set(FUNNEL_STAGES) == {"awareness", "consideration", "decision"}
    assert funnel_note("decision").startswith("FUNNEL STAGE: decision.") and "call to action" in funnel_note("decision")


def capture(roles, role):
    captured, original = {}, roles[role]

    def writer(messages):
        captured["prompt"] = messages[-1].content
        return original

    roles[role] = writer
    return captured


def test_writers_target_the_stage_and_store_it(make_deps):
    roles = default_roles()
    blog_prompt, li_prompt = capture(roles, "blog_writer"), capture(roles, "linkedin_writer")
    deps = make_deps(roles)
    state = {"user_input": "Blog about our holiday pre-order special",
             "request": {"intent": "blog", "topic": "holiday pre-orders", "target_keyword": "holiday pre-order"}}
    tools = make_blog_tools(deps)
    blog = drive_agent(make_blog_writer(deps, tools), tools, state)["blog"]
    assert blog["funnel_stage"] == "decision" and "FUNNEL STAGE: decision." in blog_prompt["prompt"]

    campaign = {"user_input": "Create a full campaign about our bakery", "blog": blog,
                "request": {"intent": "campaign", "topic": "bakery"}}
    li_tools = make_linkedin_tools(deps)
    post = drive_agent(make_linkedin_writer(deps, li_tools), li_tools, campaign)["linkedin"]
    assert post["funnel_stage"] == "decision" and "FUNNEL STAGE: decision." in li_prompt["prompt"]  # follows the blog


def test_refine_keeps_the_posts_stage(make_deps):
    roles = default_roles()
    deps = make_deps(roles)
    state = {"user_input": "Revise the blog post: make the intro warmer",
             "request": {"intent": "blog", "topic": "t", "target_keyword": "email marketing", "refine_target": "blog"},
             "blog": {"title": "Old", "markdown": "# Old\n\nold", "funnel_stage": "consideration"}}
    assert make_blog_writer(deps)(state)["blog"]["funnel_stage"] == "consideration"
