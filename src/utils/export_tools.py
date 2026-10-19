"""Export generated content as Markdown, HTML, WordPress HTML, plain text, PDF, images or a ZIP bundle.

Every export includes the artifact's **Sources & Provenance** block and an
AI-assisted-content disclosure.
"""

from __future__ import annotations

import html
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import markdown as md_lib
from PIL import Image

from src.utils.content_optimization import strip_markdown
from src.workflow.state_management import (
    BlogPost,
    ImageAsset,
    LinkedInPost,
    Provenance,
    ResearchReport,
    StrategyReport,
)

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:760px;margin:2rem auto;
padding:0 1rem;line-height:1.65;color:#1b1b1b}
h1,h2,h3{line-height:1.25}a{color:#1F4E79}code{background:#f3f3f3;padding:0 .25rem}
.provenance{border-top:1px solid #ddd;margin-top:2rem;font-size:.9rem;color:#444}
"""

_PDF_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


# --------------------------------------------------------------------------------------
# Artifact → Markdown
# --------------------------------------------------------------------------------------
def artifact_markdown(kind: str, data: dict[str, Any], absolute_images: bool = False) -> str:
    """Render any stored artifact (as dict) to publish-ready Markdown, with provenance.

    ``absolute_images`` embeds the full image path (needed for PDF rendering); otherwise the
    bare filename is used so Markdown/HTML work next to the image inside the ZIP bundle.
    """
    if kind == "blog":
        blog = BlogPost.model_validate(data)
        header = f"<!-- meta description: {blog.meta_description} -->\n<!-- slug: {blog.slug} -->\n\n"
        body = blog.markdown
        if "Sources & Provenance" not in body:
            body = f"{body}\n\n{blog.provenance.to_markdown()}"
        return header + body
    if kind == "linkedin":
        post = LinkedInPost.model_validate(data)
        parts = ["# LinkedIn Post", "", post.text, ""]
        if post.first_comment:
            parts += ["## Suggested first comment (sources)", "", post.first_comment, ""]
        alternatives = [o for o in post.hook_options if o.text != post.hook]
        if alternatives:
            parts += ["## Alternative hooks", ""] + [f"- *{o.style}* ({o.score:.0f}/100): {o.text}"
                                                     for o in alternatives] + [""]
        parts.append(post.provenance.to_markdown())
        return "\n".join(parts)
    if kind == "research":
        report = ResearchReport.model_validate(data)
        body = report.markdown
        return body if "Sources & Provenance" in body else f"{body}\n\n{report.provenance.to_markdown()}"
    if kind == "report":
        report = StrategyReport.model_validate(data)
        body = report.markdown
        return body if "Sources & Provenance" in body else f"{body}\n\n{report.provenance.to_markdown()}"
    if kind == "image":
        asset = ImageAsset.model_validate(data)
        return "\n".join([
            "# Image", "", f"![{asset.alt_text}]({asset.path if absolute_images else Path(asset.path).name})", "",
            f"- **Provider:** {asset.provider}", f"- **Original prompt:** {asset.original_prompt}",
            f"- **Enhanced prompt:** {asset.enhanced_prompt}", f"- **Alt text:** {asset.alt_text}", "",
            asset.provenance.to_markdown(),
        ])
    raise ValueError(f"unknown artifact kind '{kind}'")


def title_of(kind: str, data: dict[str, Any]) -> str:
    return data.get("title") or data.get("topic") or {"linkedin": "LinkedIn Post", "image": "Image"}.get(kind, kind)


# --------------------------------------------------------------------------------------
# Format converters
# --------------------------------------------------------------------------------------
def to_html(markdown_text: str, title: str = "ContentBlitz export") -> str:
    body = md_lib.markdown(markdown_text, extensions=["extra", "sane_lists"])
    body = body.replace("<h2>Sources &amp; Provenance</h2>", '<div class="provenance"><h2>Sources &amp; Provenance</h2>')
    if '<div class="provenance">' in body:
        body += "</div>"
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>\n{body}\n</body></html>")


def to_wordpress_html(markdown_text: str) -> str:
    """Gutenberg block markup: paste into the WordPress code editor."""
    body = md_lib.markdown(markdown_text, extensions=["extra", "sane_lists"])
    blocks = []
    for element in re.findall(r"(<h[1-6][^>]*>.*?</h[1-6]>|<p>.*?</p>|<[ou]l>.*?</[ou]l>|<blockquote>.*?</blockquote>)",
                              body, re.S):
        if element.startswith("<h"):
            level = element[2]
            blocks.append(f'<!-- wp:heading {{"level":{level}}} -->\n{element}\n<!-- /wp:heading -->')
        elif element.startswith("<p>"):
            blocks.append(f"<!-- wp:paragraph -->\n{element}\n<!-- /wp:paragraph -->")
        elif element.startswith("<ol>"):
            blocks.append(f'<!-- wp:list {{"ordered":true}} -->\n{element}\n<!-- /wp:list -->')
        elif element.startswith("<ul>"):
            blocks.append(f"<!-- wp:list -->\n{element}\n<!-- /wp:list -->")
        else:
            blocks.append(f"<!-- wp:quote -->\n{element}\n<!-- /wp:quote -->")
    return "\n\n".join(blocks)


def to_text(markdown_text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", markdown_text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)  # keep URLs visible in plain text
    lines = [strip_markdown(line) if line.strip() else "" for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def _find_unicode_font() -> str | None:
    return next((p for p in _PDF_FONT_CANDIDATES if Path(p).exists()), None)


def to_pdf(markdown_text: str, title: str = "ContentBlitz export") -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font_path = _find_unicode_font()
    if font_path:
        for style in ("", "B", "I", "BI"):  # one TTF serves every style write_html may request
            pdf.add_font("Body", style, font_path)
        family = "Body"
    else:  # core fonts are Latin-1 only → sanitize
        family = "Helvetica"
        markdown_text = markdown_text.encode("latin-1", "replace").decode("latin-1")
        title = title.encode("latin-1", "replace").decode("latin-1")
    pdf.set_title(title)
    html_body = md_lib.markdown(re.sub(r"<!--.*?-->", "", markdown_text, flags=re.S), extensions=["extra"])
    html_body = re.sub(r"<(/?)(code|span|div)[^>]*>", "", html_body)  # tags write_html doesn't need
    html_body = re.sub(r'<img [^>]*src="([^"]+)"[^>]*>',
                       lambda m: f'<img src="{m.group(1)}" width="400">' if Path(m.group(1)).exists() else "",
                       html_body)
    pdf.set_font(family, size=11)
    pdf.write_html(html_body, font_family=family)
    return bytes(pdf.output())


def image_bytes(path: str | Path, fmt: str = "PNG") -> bytes:
    with Image.open(path) as img:
        out = io.BytesIO()
        (img.convert("RGB") if fmt.upper() in ("JPEG", "JPG") else img).save(
            out, format="JPEG" if fmt.upper() in ("JPEG", "JPG") else fmt.upper(), quality=92)
        return out.getvalue()


EXPORTERS = {
    "md": ("text/markdown", lambda md, title: md.encode()),
    "html": ("text/html", lambda md, title: to_html(md, title).encode()),
    "wordpress.html": ("text/html", lambda md, title: to_wordpress_html(md).encode()),
    "txt": ("text/plain", lambda md, title: to_text(md).encode()),
    "pdf": ("application/pdf", to_pdf),
}


def export_artifact(kind: str, data: dict[str, Any], fmt: str) -> tuple[bytes, str, str]:
    """Return ``(bytes, mime_type, filename)`` for one artifact in one format."""
    title = title_of(kind, data)
    stem = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or kind
    if kind == "image" and fmt in ("png", "jpg"):
        mime = "image/png" if fmt == "png" else "image/jpeg"
        return image_bytes(data["path"], "PNG" if fmt == "png" else "JPEG"), mime, f"{stem}.{fmt}"
    mime, fn = EXPORTERS[fmt]
    return fn(artifact_markdown(kind, data, absolute_images=fmt == "pdf"), title), mime, f"{stem}.{fmt}"


def bundle_zip(artifacts: dict[str, dict[str, Any]], sources: list[dict] | None = None) -> bytes:
    """ZIP with every artifact in every text format, images, and a provenance JSON."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for kind, data in artifacts.items():
            if not data:
                continue
            for fmt in ("md", "html", "txt", "pdf") + (("wordpress.html",) if kind == "blog" else ()):
                content, _, name = export_artifact(kind, data, fmt)
                zf.writestr(f"{kind}/{name}", content)
            if kind == "image" and Path(data.get("path", "")).exists():
                zf.write(data["path"], f"image/{Path(data['path']).name}")
        zf.writestr("sources.json", json.dumps(sources or [], indent=2))
        zf.writestr("README.txt", "Generated with ContentBlitz (AI-assisted). Review facts before publishing.\n"
                                  "Each file includes a 'Sources & Provenance' section.\n")
    return buf.getvalue()


def provenance_of(data: dict[str, Any]) -> Provenance:
    return Provenance.model_validate(data.get("provenance") or {})
