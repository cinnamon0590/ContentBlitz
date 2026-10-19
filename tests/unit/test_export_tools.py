import io
import zipfile

import pytest
from PIL import Image

from src.utils import export_tools as ex
from src.workflow.state_management import (
    BlogPost,
    ImageAsset,
    LinkedInPost,
    Provenance,
    ResearchReport,
    Source,
    StrategyReport,
)

PROV = Provenance(sources=[Source(id=1, title="Study", url="https://a.com/s", provider="serpapi")],
                  generated_by="openai:gpt-4o", grounding="live_research")


@pytest.fixture
def blog():
    return BlogPost(title="Coffee Guide ☕", slug="coffee-guide", meta_description="Meta",
                    markdown="# Coffee Guide ☕\n\nGrowth 20% [1].\n\n## Brewing\n\nText.", provenance=PROV).model_dump()


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (300, 300), (1, 2, 3)).save(path)
    return ImageAsset(path=str(path), original_prompt="cup", enhanced_prompt="a cup, studio light",
                      provider="dall-e-3", alt_text="A cup", provenance=PROV).model_dump()


def test_artifact_markdown_includes_provenance(blog, image):
    md = ex.artifact_markdown("blog", blog)
    assert "Sources & Provenance" in md and "slug: coffee-guide" in md and "AI-assisted" in md
    li = LinkedInPost(text="Post", first_comment="Sources: https://a.com/s", provenance=PROV).model_dump()
    assert "Suggested first comment" in ex.artifact_markdown("linkedin", li)
    assert "dall-e-3" in ex.artifact_markdown("image", image)
    assert "Sources & Provenance" in ex.artifact_markdown("research", ResearchReport(topic="t", markdown="# R").model_dump())
    assert "model knowledge" in ex.artifact_markdown("report", StrategyReport(title="r", markdown="# R").model_dump())
    with pytest.raises(ValueError):
        ex.artifact_markdown("nope", {})


def test_html_wordpress_text(blog):
    md = ex.artifact_markdown("blog", blog)
    html = ex.to_html(md, "Coffee")
    assert "<h1>" in html and 'class="provenance"' in html
    wp = ex.to_wordpress_html("# H\n\nPara\n\n- a\n\n1. b\n\n> q")
    assert "wp:heading" in wp and "wp:paragraph" in wp and '"ordered":true' in wp and "wp:quote" in wp
    text = ex.to_text(md)
    assert "https://a.com/s" in text and "#" not in text.splitlines()[0]


def test_pdf_export_unicode(blog):
    pdf = ex.to_pdf(ex.artifact_markdown("blog", blog), "Coffee ☕")
    assert pdf.startswith(b"%PDF")


def test_pdf_export_latin1_fallback(blog, monkeypatch):
    monkeypatch.setattr(ex, "_find_unicode_font", lambda: None)
    assert ex.to_pdf("# Title ☕\n\nBody 🚀", "T ☕").startswith(b"%PDF")


def test_export_artifact_formats(blog, image):
    for fmt in ("md", "html", "txt", "pdf", "wordpress.html"):
        data, mime, name = ex.export_artifact("blog", blog, fmt)
        assert data and name.endswith(fmt) and mime
    png, mime, _ = ex.export_artifact("image", image, "png")
    jpg, mime_j, _ = ex.export_artifact("image", image, "jpg")
    assert mime == "image/png" and mime_j == "image/jpeg" and jpg[:2] == b"\xff\xd8"


def test_bundle_zip(blog, image):
    data = ex.bundle_zip({"blog": blog, "image": image, "linkedin": {}}, sources=[{"id": 1}])
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "sources.json" in names and any(n.endswith(".wordpress.html") for n in names)
    assert any(n.startswith("image/") and n.endswith(".png") for n in names)
    assert ex.provenance_of(blog).grounding == "live_research"


def test_pdf_drops_emoji_the_font_cannot_draw(caplog):
    from src.utils.export_tools import strip_emoji, to_pdf

    assert strip_emoji("📚 Sources for this post: ⭐⭐ great 👍🏽") == " Sources for this post: great "
    pdf = to_pdf("# Title ✨\n\n📚 Sources for this post:\n\n- one")
    assert pdf[:4] == b"%PDF" and "missing the following glyphs" not in caplog.text
