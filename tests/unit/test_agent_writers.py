from src.agents.blog_writer import make_blog_tools, make_blog_writer
from src.agents.content_strategist import make_content_strategist
from src.agents.image_generator import fallback_enhancement, make_image_generator
from src.agents.linkedin_writer import build_first_comment, make_linkedin_tools, make_linkedin_writer
from src.agents.research_agent import make_research_agent, make_research_tools
from src.utils.content_optimization import linkedin_metrics
from src.workflow.state_management import BlogPost, BrandProfile, ImageAsset, LinkedInPost, StrategyReport
from tests.conftest import default_roles, drive_agent


def researched_state(deps, intent="blog"):
    tools = make_research_tools(deps)
    state = {"user_input": f"write a {intent} about email marketing for bakeries",
             "request": {"intent": intent, "topic": "email marketing for bakeries",
                         "target_keyword": "email marketing"}}
    return drive_agent(make_research_agent(deps, tools), tools, state)


# ---- Blog -----------------------------------------------------------------------------
def test_blog_writer_full_flow(make_deps):
    deps = make_deps()
    state = researched_state(deps)
    tools = make_blog_tools(deps)
    state = drive_agent(make_blog_writer(deps, tools), tools, state)
    blog = BlogPost.model_validate(state["blog"])
    assert blog.markdown.count("\n# ") + blog.markdown.startswith("# ") == 1  # exactly one H1
    assert "invented.example" not in blog.markdown  # model's fake Sources section dropped
    assert "## Sources & Provenance" in blog.markdown
    assert 150 <= len(blog.meta_description) <= 160
    assert blog.slug == "email-marketing" and blog.internal_link_suggestions
    assert {s.id for s in blog.provenance.sources} <= {1, 2, 3}
    assert state["quality_target"] == "blog" and state["agent_scratch"] == []
    report = deps.validator.score_blog(blog, BrandProfile(), {1, 2, 3})
    assert report.overall > 60


def test_blog_revision_skips_tools_and_includes_feedback(make_deps):
    roles = default_roles()
    captured = {}
    original = roles["blog_writer"]

    def writer(messages):
        captured["prompt"] = messages[-1].content
        return original

    roles["blog_writer"] = writer
    roles["blog_writer.research"] = lambda m: (_ for _ in ()).throw(AssertionError("tools must be skipped"))
    deps = make_deps(roles)
    state = {"request": {"intent": "blog", "topic": "t", "target_keyword": "email marketing"},
             "blog": {"title": "Old", "markdown": "# Old\n\nold body\n\n## Sources & Provenance\n\nx"},
             "revision_feedback": "- Add an FAQ", "quality_target": "blog", "revision_count": 1}
    out = make_blog_writer(deps)(state)
    assert "REVISION REQUEST" in captured["prompt"] and "Add an FAQ" in captured["prompt"]
    assert "Sources & Provenance" not in captured["prompt"].split("PREVIOUS DRAFT:")[1]
    assert out["blog"]["provenance"]["revision_count"] == 1
    assert out["blog"]["provenance"]["grounding"] == "model_knowledge_only"  # no research in state


def test_blog_structured_failure_falls_back_to_text(make_deps):
    roles = default_roles()
    roles["blog_writer"] = "# Plain Title Here\n\nBody about email marketing."
    deps = make_deps(roles)
    out = make_blog_writer(deps)({"request": {"intent": "blog", "topic": "x", "refine_target": "blog"},
                                  "blog": {"title": "x", "markdown": "# x"}})
    assert out["blog"]["title"] == "Plain Title Here"
    assert any("Structured output failed" in w for w in out["warnings"])


# ---- LinkedIn -------------------------------------------------------------------------
def test_linkedin_writer_formats_and_attributes(make_deps):
    deps = make_deps()
    state = researched_state(deps, "linkedin")
    tools = make_linkedin_tools(deps)
    state = drive_agent(make_linkedin_writer(deps, tools), tools, state)
    post = LinkedInPost.model_validate(state["linkedin"])
    metrics = linkedin_metrics(post.text)
    assert not metrics["has_link"] and "[1]" not in post.text
    assert 8 <= len(post.hashtags) <= 12 and metrics["chars"] <= 1600
    assert post.first_comment.startswith("📚 Sources") and "https://example1.com/article" in post.first_comment
    assert any("Moved 1 link" in n for n in post.provenance.notes)
    assert post.provenance.grounding == "live_research"


def test_linkedin_without_research_has_no_first_comment(make_deps):
    roles = default_roles()
    roles["linkedin_writer.research"] = "TRENDS READY"
    out = make_linkedin_writer(make_deps(roles))({"request": {"intent": "linkedin", "topic": "remote work"}})
    assert out["linkedin"]["first_comment"] == ""
    assert out["linkedin"]["provenance"]["grounding"] == "model_knowledge_only"
    assert build_first_comment([]) == ""


def test_linkedin_text_fallback(make_deps):
    roles = default_roles()
    roles["linkedin_writer"] = "Hook line\nBody text #One #Two"
    out = make_linkedin_writer(make_deps(roles))({"request": {"intent": "linkedin", "topic": "x",
                                                              "refine_target": "linkedin"},
                                                  "linkedin": {"text": "old"}})
    assert out["linkedin"]["hook"] == "Hook line"


# ---- Image ----------------------------------------------------------------------------
def test_image_generator_primary(make_deps):
    deps = make_deps()
    out = make_image_generator(deps)({"request": {"intent": "image", "topic": "bakery", "image_prompt": "croissants"},
                                      "pending_formats": ["image"]})
    asset = ImageAsset.model_validate(out["image"])
    assert asset.provider == "gpt-image-1" and asset.enhanced_prompt.startswith("Warm bakery")
    assert asset.provenance.image_model == "gpt-image-1"
    assert out["pending_formats"] == [] and len(out["images"]) == 1
    assert __import__("pathlib").Path(asset.path).exists() and __import__("pathlib").Path(asset.metadata_path).exists()


def test_image_generator_fallback_chain_and_enhancer_failure(make_deps):
    roles = default_roles()
    roles["image_prompt_enhancer"] = "nope"
    deps = make_deps(roles, image_fail={"gpt-image-1"})
    out = make_image_generator(deps)({"request": {"intent": "image", "topic": "bakery"},
                                      "blog": {"title": "Bakery Emails"}})
    asset = ImageAsset.model_validate(out["image"])
    assert asset.provider == "gpt-image-1-mini" and len(asset.attempts) == 1
    assert any("fallback" in w.lower() for w in out["warnings"])
    assert any("Fallback attempt failed" in n for n in asset.provenance.notes)


def test_image_generator_placeholder_and_moderation(make_deps):
    deps = make_deps(image_fail={"gpt-image-1", "gpt-image-1-mini", "gemini-2.5-flash-image"})
    out = make_image_generator(deps)({"request": {"intent": "image", "topic": "bakery"}})
    assert out["image"]["provider"] == "placeholder"

    class Block:
        def check(self, text):
            return True, ["violence"]

    deps.moderator = Block()
    out = make_image_generator(deps)({"request": {"intent": "image", "topic": "x"}, "pending_formats": ["image"]})
    assert "blocked by moderation" in out["warnings"][0] and out["pending_formats"] == []


def test_image_chain_exhausted(make_deps):
    deps = make_deps()
    deps.images.chain = ["gpt-image-1"]
    deps.images.client.fail_models = {"gpt-image-1"}
    out = make_image_generator(deps)({"request": {"intent": "image", "topic": "x"}})
    assert "Image generation failed" in out["warnings"][0]


def test_fallback_enhancement_uses_brand():
    spec = fallback_enhancement("a cafe", BrandProfile())
    assert "#1F4E79" in spec.enhanced_prompt and "No text" in spec.enhanced_prompt


# ---- Content Strategist -----------------------------------------------------------------
def test_content_strategist_scrubs_and_attributes(make_deps):
    deps = make_deps()
    state = researched_state(deps, "research")
    out = make_content_strategist(deps)(state)
    report = StrategyReport.model_validate(out["report"])
    assert "made-up.example" not in report.markdown and "[42]" not in report.markdown
    assert report.markdown.count("Sources & Provenance") == 1 and "junk" not in report.markdown
    assert any("Removed" in n for n in report.provenance.notes)
    assert out["quality_target"] == "report"
    quality = deps.validator.score_report(report, BrandProfile())
    assert quality.scores["citations"] > 0


def test_content_strategist_adds_title_and_handles_llm_failure(make_deps, config):
    roles = default_roles()
    roles["content_strategist"] = "No heading here [1]."
    deps = make_deps(roles)
    state = researched_state(deps, "research")
    out = make_content_strategist(deps)(state)
    assert out["report"]["markdown"].startswith("# Trend Report")

    from src.integrations.openai_client import LLMService
    from tests.conftest import FailingChatModel

    deps.llm = LLMService(config, chat_models={"smart": FailingChatModel()})
    out = make_content_strategist(deps)(state)
    assert any("unavailable" in w for w in out["warnings"]) and out["report"]["markdown"].startswith("# Research Brief")


# ---- Blog: section-by-section expansion of short drafts -----------------------------------
SHORT_DRAFT = ("# Email Marketing Guide\n\nShort intro about email marketing. Too brief to rank.\n\n"
               "## Build Your List\n\nAsk at the till [1].\n\n## Write Emails People Open\n\nKeep it short.\n\n"
               "## FAQ\n\n### How often?\n\nWeekly.\n\n## Conclusion\n\nGet in touch to learn how we can help.")
FILLER = "Small teams can plan a simple weekly routine and keep each step easy to repeat. " * 40


def _grow_section(messages):
    """Fake editor: echo the section's heading, then add plain-language detail."""
    section = messages[-1].content.split("SECTION TO EXPAND:\n", 1)[1]
    heading, _, rest = section.partition("\n")
    return f"{heading}\n\n{rest.strip()} {FILLER}"


def _blog_state():
    return {"request": {"intent": "blog", "topic": "t", "target_keyword": "email marketing", "refine_target": "blog"},
            "blog": {"title": "x", "markdown": "# x"}}


def test_short_blog_draft_is_expanded_section_by_section(make_deps):
    roles = default_roles()
    roles["blog_writer"] = {**roles["blog_writer"], "markdown": SHORT_DRAFT}
    asked = []
    roles["blog_writer.expand"] = lambda m: (asked.append(m[-1].content), _grow_section(m))[1]
    blog = BlogPost.model_validate(make_blog_writer(make_deps(roles))(_blog_state())["blog"])
    assert len(asked) == 2  # the two body sections; FAQ and Conclusion are left alone
    assert all("Build Your List" in a or "Write Emails People Open" in a for a in asked)
    assert blog.markdown.count("Small teams can plan") >= 80 and "Ask at the till" in blog.markdown  # [1] is scrubbed: no research, no valid ids
    assert "### How often?\n\nWeekly." in blog.markdown
    assert any("editing pass expanded it" in n for n in blog.provenance.notes)


def test_bad_expansions_keep_the_original(make_deps):
    for reply in ("Sure! Here is a longer version." + FILLER,  # heading not kept
                  lambda m: m[-1].content.split("SECTION TO EXPAND:\n", 1)[1]):  # not longer
        roles = default_roles()
        roles["blog_writer"] = {**roles["blog_writer"], "markdown": SHORT_DRAFT}
        roles["blog_writer.expand"] = reply
        blog = BlogPost.model_validate(make_blog_writer(make_deps(roles))(_blog_state())["blog"])
        assert "Too brief to rank" in blog.markdown and "Small teams can plan" not in blog.markdown
        assert not any("expanded" in n for n in blog.provenance.notes)
