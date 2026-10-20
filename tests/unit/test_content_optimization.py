from src.utils import content_optimization as co
from tests.conftest import good_blog_markdown


def test_keyword_density_and_word_count():
    text = "email marketing works. " + "word " * 96
    assert co.word_count(text) == 99
    assert co.keyword_density(text, "email marketing") == round(2 / 99 * 100, 2)
    assert co.keyword_density("", "x") == 0.0
    assert co.keyword_density("abc", "") == 0.0


def test_readability_bounds():
    r = co.readability("The cat sat on the mat. It was happy.")
    assert r["flesch_reading_ease"] > 80
    assert co.readability("") == {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}


def test_heading_structure_good_and_bad():
    good = co.heading_structure(good_blog_markdown(), "email marketing")
    assert good["h1"] == 1 and good["h2"] >= 3 and good["issues"] == []
    bad = co.heading_structure("# A\n# B\n### skipped", "seo")
    assert len(bad["issues"]) >= 4


def test_meta_description_check_and_fit():
    meta = "Learn email marketing " + "x" * 140
    assert co.check_meta_description(meta[:155], "email marketing")["ok"]
    assert not co.check_meta_description("short", "kw")["ok"]
    long = "word " * 60
    fitted = co.fit_meta_description(long)
    assert len(fitted) <= 160 and fitted.endswith("…")
    sentences = ("Build your list at the till with a simple sign-up card. Send one helpful email each week and "
                 "track what works. Share seasonal specials with subscribers first.")
    extended = co.fit_meta_description("Email marketing for bakeries, made simple.", fallback=sentences)
    assert 150 <= len(extended) <= 160 and extended.endswith(".")  # whole sentences only


def test_meta_description_never_ends_mid_sentence():
    """Regression: padding with raw first-paragraph text produced "...start today! In the bustling world…"."""
    meta = ("Explore local bakery email marketing trends for 2026. Learn practical tips to boost engagement and sales. "
            "Get inspired and start today!")
    intro = ("In the bustling world of local bakeries, staying connected with your customers is key to thriving in "
             "2026. Local bakery email marketing is increasingly a cornerstone.")
    fitted = co.fit_meta_description(meta, fallback=intro)
    assert fitted.startswith(meta) and 150 <= len(fitted) <= 160  # no intro sentence fits, so a short closer does
    assert fitted.removeprefix(meta).strip() in co.META_CLOSERS
    long = ("Bakery email marketing made simple. Build your list, write emails people open, and measure results. "
            "Start this week with three easy steps. Then keep going every single week for steady growth.")
    trimmed = co.fit_meta_description(long)
    assert trimmed.endswith("three easy steps.") and len(trimmed) <= 160  # cut at a sentence end, no fragment
    assert co.fit_meta_description("word " * 60).endswith("…")  # no sentence end at all: word boundary + ellipsis


def test_slugify():
    assert co.slugify("Hello, World! Café Tips 2026") == "hello-world-cafe-tips-2026"
    assert len(co.slugify("word " * 40)) <= 60


def test_uncited_statistics():
    md = ("# T\n\nSales rose 40% last year.\n\nRevenue hit $2 million [1].\n\n1. A numbered item\n\n"
          "## Sources\n\nSomething 99% here.")
    flagged = co.uncited_statistics(md)
    assert flagged == ["Sales rose 40% last year."]
    assert co.uncited_statistics("In 2026, the focus is on personalization.") == []  # a year is not a statistic
    assert co.uncited_statistics("Since 2020, opens grew 25%.") == ["Since 2020, opens grew 25%."]
    assert co.citation_ids("a [1] b [12]") == {1, 12}


def test_faq_cta_internal_links():
    md = good_blog_markdown()
    assert co.has_faq_section(md) and co.has_cta(md)
    links = co.suggest_internal_links(md)
    assert links and all(link.startswith("/blog/") for link in links)
    assert not any("faq" in link for link in links)
    assert co.first_paragraph_has_keyword(md, "email marketing")


def test_hashtag_normalization():
    assert co.normalize_hashtag("small business") == "#SmallBusiness"
    assert co.normalize_hashtag("#marketing") == "#marketing"
    assert co.normalize_hashtag("!!!") == ""


def test_format_linkedin_enforces_limits():
    body = ("This is a long sentence about marketing. " * 60) + " #inline"
    tags = [f"tag{i}" for i in range(15)]
    text, out_tags, notes = co.format_linkedin(body, tags)
    assert len(text) <= 1600
    assert len(out_tags) == 12
    assert any("Trimmed hashtags" in n for n in notes) and any("Trimmed post body" in n for n in notes)
    metrics = co.linkedin_metrics(text)
    assert metrics["max_paragraph_chars"] <= 280


def test_format_linkedin_flags_too_few_tags():
    _, tags, notes = co.format_linkedin("Short post.\n\nWhat do you think?", ["one", "#One", "two"])
    assert tags == ["#one", "#two"]
    assert any("Only 2 hashtags" in n for n in notes)


def test_linkedin_metrics_fields():
    m = co.linkedin_metrics("Hook line 🚀\n\nBody https://x.com\n\n#a #b")
    assert m["hashtag_count"] == 2 and m["has_link"] and m["emoji_count"] == 1 and m["hook"] == "Hook line 🚀"
