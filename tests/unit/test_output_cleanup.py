"""Output clean-up found in live runs: merged citations, location placeholders, text in images."""

import pytest

from src.agents import scrub_body
from src.agents.image_generator import NO_TEXT_SUFFIX, make_image_generator
from src.utils import content_optimization as co
from src.workflow.state_management import Source, SourceRegistry


@pytest.mark.parametrize(("text", "expected"), [
    ("Opens rose 20% [7, 8].", "Opens rose 20% [7][8]."),
    ("See [3; 5].", "See [3][5]."),
    ("A range [2-4] and [2–4].", "A range [2][3][4] and [2][3][4]."),
    ("A wide span [1-50] is not expanded.", "A wide span [1][50] is not expanded."),
    ("Duplicates [4, 4] collapse.", "Duplicates [4] collapse."),
    ("Single [9] and a link [1, 2](https://x.example) stay.", "Single [9] and a link [1, 2](https://x.example) stay."),
])
def test_normalize_citations(text, expected):
    assert co.normalize_citations(text) == expected


def test_merged_citations_count_as_citations():
    assert co.citation_ids("Sales rose 20% [7, 8].") == {7, 8}
    assert co.uncited_statistics("Sales rose 20% [7, 8].") == []
    assert "[" not in co.strip_markdown("Sales rose [7, 8].")


def test_location_placeholders():
    text = "Bakeries in [Your City] and [neighborhood] win. > **Your experience:** [Add a real example]"
    assert co.fill_location_placeholders(text, "Austin, TX").startswith("Bakeries in Austin, TX and Austin, TX win.")
    filled = co.fill_location_placeholders(text)
    assert "in your area and your area" in filled and "[Add a real example]" in filled  # the E-E-A-T prompt stays


def test_scrub_body_checks_each_merged_id_and_fills_location():
    registry = SourceRegistry()
    for n in (1, 2):
        registry.add(Source(title=f"S{n}", url=f"https://s{n}.example"))
    markdown, notes = scrub_body("Open rates hit 40% [1, 2, 9] in [Your City].", registry, {1, 2}, "Boise")
    assert markdown == "Open rates hit 40% [1][2] in Boise."
    assert any("9" in n for n in notes)


def test_every_image_prompt_forbids_text(make_deps):
    deps = make_deps()
    out = make_image_generator(deps)({"request": {"intent": "image", "topic": "bakery", "image_prompt": "storefront"}})
    assert deps.images.client.prompts[-1].endswith(NO_TEXT_SUFFIX)
    assert not out["image"]["enhanced_prompt"].endswith(NO_TEXT_SUFFIX)  # the shown prompt stays readable


def test_meta_closer_only_when_it_lands_in_range():
    assert co.fit_meta_description("x" * 100) == "x" * 100  # too far short for any closer: left as written
    lifted = co.fit_meta_description("Email marketing for local bakeries: build a list, send weekly specials and "
                                     "track what brings people through the door.")
    assert 150 <= len(lifted) <= 160 and lifted.endswith(".")


def test_refine_feedback_carries_open_quality_suggestions():
    from src.agents import refine_feedback

    state = {"quality": {"blog": {"suggestions": ["Remove off-brand words: leverage.", "Shorten sentences."]}}}
    feedback = refine_feedback(state, "blog", "Make the intro warmer")
    assert feedback.startswith("Make the intro warmer") and "- Remove off-brand words: leverage." in feedback
    assert refine_feedback({}, "blog", "Make the intro warmer") == "Make the intro warmer"


def test_revision_that_drops_the_faq_gets_it_back():
    previous = "# T\n\nIntro.\n\n## Why\n\nBody.\n\n## FAQ\n\n### Q1?\n\nA1.\n\n## Conclusion\n\nJoin us."
    revised = "# T\n\nWarmer intro.\n\n## Why\n\nBody.\n\n## Conclusion\n\nJoin us today."
    fixed, restored = co.keep_faq_section(previous, revised)
    assert restored and fixed.index("## FAQ") < fixed.index("## Conclusion") and "### Q1?" in fixed
    assert "Warmer intro." in fixed and "Join us today." in fixed
    assert co.keep_faq_section(previous, previous) == (previous, False)  # FAQ still there: unchanged
    no_conclusion, _ = co.keep_faq_section(previous, "# T\n\nOnly intro.")
    assert no_conclusion.rstrip().endswith("A1.")
