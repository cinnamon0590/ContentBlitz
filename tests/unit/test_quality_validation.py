from src.utils.quality_validation import QualityValidator, brand_voice_check, learn_brand_voice
from src.workflow.state_management import (
    BlogPost,
    BrandProfile,
    LinkedInPost,
    Provenance,
    Source,
    StrategyReport,
)
from tests.conftest import good_blog_markdown

BRAND = BrandProfile()


def make_blog(md=None, meta=None):
    md = md or good_blog_markdown()
    meta = meta or ("Email marketing for small businesses: a practical guide to building lists, writing emails "
                    "people open, and measuring results that matter this year.")
    return BlogPost(title="Guide", target_keyword="email marketing", meta_description=meta, markdown=md)


def test_good_blog_passes():
    report = QualityValidator(70).score_blog(make_blog(), BRAND, known_source_ids={1, 2})
    assert report.passed, (report.overall, report.scores, report.suggestions)
    assert report.metrics["word_count"] >= 1200
    assert report.metrics["invalid_citations"] == []


def test_weak_blog_fails_with_suggestions():
    md = "# Title\n\nShort post with 45% growth and no citations. It is a revolutionary game-changer."
    report = QualityValidator(70).score_blog(make_blog(md, meta="too short"), BRAND, known_source_ids={1})
    assert not report.passed
    text = " ".join(report.suggestions)
    assert "keyword density" in text and "citations" in text and "off-brand" in text and "Expand" in text


def test_invalid_citation_ids_penalised():
    md = good_blog_markdown().replace("[2]", "[9]")
    report = QualityValidator().score_blog(make_blog(md), BRAND, known_source_ids={1, 2})
    assert report.metrics["invalid_citations"] == [9]
    assert report.scores["citations"] < 100


def test_linkedin_scoring_good_and_bad():
    paragraphs = ["Most small shops lose 30% of leads to slow replies.",
                  "Last spring we tested a 5-minute response rule with three local clients.",
                  "Bookings rose within a month. Customers told us speed felt like care.",
                  "Here is the simple system we used, step by step, so you can copy it.",
                  "First, set up instant alerts. Second, write three reply templates. Third, review weekly.",
                  "None of this needs new software. It needs a habit and a shared inbox.",
                  "The surprising part was how much calmer the team felt with a clear rule.",
                  "If you run a small team, start with one channel and measure reply time for two weeks.",
                  "What is your average response time today? Tell me in the comments."]
    text = "\n\n".join(paragraphs) + "\n\n" + " ".join(f"#Tag{i}" for i in range(10))
    good = QualityValidator().score_linkedin(LinkedInPost(text=text), BRAND)
    assert good.scores["hashtags"] == 100 and good.scores["cta"] == 100
    bad = QualityValidator().score_linkedin(LinkedInPost(text="In today's world see https://x.com #a"), BRAND)
    assert not bad.passed and any("first comment" in s for s in bad.suggestions)
    assert any("generic opener" in s for s in bad.suggestions)


def test_report_scoring():
    md = "# Report\n\n## Executive Summary\n\nGrowth hit 20% [1].\n\n## Key Findings\n\n- x\n\n## Recommendations\n\n- y\n\n## Sources\n"
    prov = Provenance(sources=[Source(id=1, url="https://a.com")], grounding="live_research")
    report = QualityValidator().score_report(StrategyReport(title="R", markdown=md, provenance=prov), BRAND)
    assert report.scores["structure"] == 100 and report.scores["citations"] == 100
    no_sources = QualityValidator().score_report(StrategyReport(title="R", markdown="# R\n\ntext"), BRAND)
    assert no_sources.scores["citations"] == 40


def test_brand_voice_rules():
    brand = BrandProfile(required_phrases=["family-owned"])
    score, suggestions, metrics = brand_voice_check("We leverage synergy!!!!! Wow!", brand)
    assert score < 60 and metrics["banned_words_found"] == ["leverage", "synergy"]
    assert metrics["missing_phrases"] == ["family-owned"]


def test_llm_critique_is_advisory(make_llm):
    llm = make_llm(lambda m: '{"feedback": "Nice tone", "suggestions": ["Add a story"]}')
    validator = QualityValidator(70, llm=llm, critique_prompt="ROLE: critic")
    report = validator.score_blog(make_blog(), BRAND, known_source_ids={1, 2})
    baseline = QualityValidator(70).score_blog(make_blog(), BRAND, known_source_ids={1, 2})
    assert report.overall == baseline.overall  # LLM cannot change the score
    assert report.llm_feedback == "Nice tone" and "Add a story" in report.suggestions


def test_learn_brand_voice(make_llm):
    llm = make_llm(lambda m: '{"tone": "warm, witty", "audience": "home bakers", "values": ["craft"], '
                             '"signature_phrases": ["baked with love"], "avoid_words": ["cheap"]}')
    profile = learn_brand_voice("We bake bread every morning. Our loaves are baked with love.", BrandProfile(), llm,
                                "ROLE: voice")
    assert profile.tone == "warm, witty" and "cheap" in profile.banned_words
    assert profile.required_phrases == ["baked with love"] and profile.sample_text
    unchanged = learn_brand_voice("Sample text here.", BrandProfile(), None, "x")
    assert unchanged.tone == BrandProfile().tone


def test_readability_feedback_gives_the_numbers():
    dense = ("Notwithstanding multifaceted organizational considerations, comprehensive omnichannel personalization "
             "methodologies necessitate sophisticated infrastructural investments. ") * 40
    report = QualityValidator(70).score_blog(make_blog(md=f"# Email marketing\n\n{dense}"), BRAND)
    tip = next(s for s in report.suggestions if s.startswith("Reading ease is"))
    assert "the brand targets 60+" in tip and "under 15 words" in tip
