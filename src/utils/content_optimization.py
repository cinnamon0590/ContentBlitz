"""Deterministic content-optimization helpers (the SEO and platform-formatting engine).

These functions measure content and fix it where that can be done without an
LLM: keyword density, readability, heading structure, meta descriptions, slugs,
LinkedIn length, hashtags, and line breaks. They're based on the Moz SEO
Beginner's Guide (one H1, a descriptive heading hierarchy, a 150–160 character
meta description with the keyword) and on the LinkedIn guidance in the project
FAQ (1300–1600 characters, 8–12 hashtags, short paragraphs, a strong hook).
"""

from __future__ import annotations

import re
import unicodedata

import textstat

BLOG_TARGETS = {
    "keyword_density": (1.0, 2.0),  # percent
    "meta_description": (150, 160),  # characters
    "min_words": 1200,
    "min_h2": 3,
}
LINKEDIN_TARGETS = {"chars": (1300, 1600), "hashtags": (8, 12), "hook_max_chars": 210, "max_paragraph_chars": 280}

CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")
_STAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?\s?%|\$\s?\d|\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?\s?(?:million|billion|percent|x)\b"
                           r"|\b(?:19|20)\d{2}\b)", re.I)
_EMOJI_PATTERN = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


# --------------------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------------------
def strip_markdown(markdown: str) -> str:
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_`>|]", "", text)
    text = CITATION_PATTERN.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def body_without_sources(markdown: str) -> str:
    """Drop the trailing 'Sources & Provenance' / 'Sources' section before measuring."""
    return re.split(r"^#{1,3}\s*Sources\b.*$", markdown, maxsplit=1, flags=re.M | re.I)[0]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", strip_markdown(text)))


def slugify(title: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", text)
    if len(slug) > max_len:
        slug = slug[:max_len].rsplit("-", 1)[0]
    return slug.strip("-")


def keyword_density(text: str, keyword: str) -> float:
    """Percentage of words in ``text`` accounted for by occurrences of ``keyword`` (a phrase)."""
    keyword = keyword.strip().lower()
    plain = strip_markdown(text).lower()
    total = len(re.findall(r"\b[\w'-]+\b", plain))
    if not keyword or not total:
        return 0.0
    occurrences = len(re.findall(rf"(?<!\w){re.escape(keyword)}(?!\w)", plain))
    return round(occurrences * len(keyword.split()) / total * 100, 2)


def readability(text: str) -> dict[str, float]:
    plain = strip_markdown(text)
    if not plain:
        return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(plain), 1),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(plain), 1),
    }


# --------------------------------------------------------------------------------------
# Blog / SEO
# --------------------------------------------------------------------------------------
def heading_structure(markdown: str, keyword: str = "") -> dict:
    headings = [(len(m.group(1)), m.group(2).strip()) for m in re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.M)]
    counts = {f"h{n}": sum(1 for level, _ in headings if level == n) for n in (1, 2, 3)}
    issues: list[str] = []
    if counts["h1"] != 1:
        issues.append(f"Use exactly one H1 (found {counts['h1']}).")
    if counts["h2"] < BLOG_TARGETS["min_h2"]:
        issues.append(f"Add more H2 sections (found {counts['h2']}, want ≥{BLOG_TARGETS['min_h2']}).")
    previous = 0
    for level, text in headings:
        if previous and level > previous + 1:
            issues.append(f"Heading level skipped before '{text[:40]}' (H{previous} → H{level}).")
        previous = level
    kw = keyword.lower().strip()
    h1_text = next((t for level, t in headings if level == 1), "")
    keyword_in_h1 = bool(kw) and kw in h1_text.lower()
    keyword_in_h2 = bool(kw) and any(kw in t.lower() for level, t in headings if level == 2)
    if kw and not keyword_in_h1:
        issues.append("Include the target keyword in the H1.")
    if kw and not keyword_in_h2:
        issues.append("Include the target keyword (or a close variant) in at least one H2.")
    return {**counts, "h1_text": h1_text, "keyword_in_h1": keyword_in_h1, "keyword_in_h2": keyword_in_h2,
            "issues": issues, "headings": [t for _, t in headings]}


def check_meta_description(meta: str, keyword: str = "") -> dict:
    lo, hi = BLOG_TARGETS["meta_description"]
    length = len(meta.strip())
    issues = []
    if not lo <= length <= hi:
        issues.append(f"Meta description should be {lo}–{hi} characters (is {length}).")
    has_kw = bool(keyword) and keyword.lower() in meta.lower()
    if keyword and not has_kw:
        issues.append("Include the target keyword in the meta description.")
    return {"length": length, "has_keyword": has_kw, "ok": not issues, "issues": issues}


def check_title(title: str, keyword: str = "") -> list[str]:
    """Moz title-tag guidance: ~50–60 characters, keyword near the front."""
    issues = []
    if len(title) > 60:
        issues.append(f"Shorten the title to ≤ 60 characters (is {len(title)}) so it isn't truncated in search.")
    elif len(title) < 30:
        issues.append(f"Make the title more descriptive (is {len(title)} characters; aim for 50–60).")
    if keyword:
        pos = title.lower().find(keyword.lower())
        if pos == -1:
            issues.append("Include the target keyword in the title.")
        elif pos > len(title) / 2:
            issues.append("Move the target keyword closer to the front of the title.")
    return issues


def fit_meta_description(meta: str, fallback: str = "") -> str:
    """Trim to ≤160 chars at a word boundary; extend from ``fallback`` text if too short."""
    lo, hi = BLOG_TARGETS["meta_description"]
    meta = re.sub(r"\s+", " ", meta).strip()
    if len(meta) < lo and fallback:
        extra = strip_markdown(fallback)
        meta = f"{meta} {extra}".strip() if meta else extra
    if len(meta) > hi:
        cut = meta[: hi - 1].rsplit(" ", 1)[0].rstrip(",;:-")
        meta = cut + "…"
    return meta


def first_paragraph_has_keyword(markdown: str, keyword: str, words: int = 100) -> bool:
    body = re.sub(r"^#\s+.+$", "", markdown, count=1, flags=re.M)
    first = " ".join(strip_markdown(body).split()[:words]).lower()
    return bool(keyword) and keyword.lower() in first


def citation_ids(text: str) -> set[int]:
    return {int(n) for n in CITATION_PATTERN.findall(text)}


def uncited_statistics(markdown: str) -> list[str]:
    """Sentences that contain a statistic (%, $, large number, year) but no ``[n]`` citation."""
    body = body_without_sources(markdown)
    flagged = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stripped = re.sub(r"^(\d+\.|[-*])\s+", "", stripped)  # list markers are not statistics
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            if _STAT_PATTERN.search(CITATION_PATTERN.sub("", sentence)) and not CITATION_PATTERN.search(sentence):
                flagged.append(sentence.strip()[:200])
    return flagged


def has_faq_section(markdown: str) -> bool:
    return bool(re.search(r"^#{2,3}\s*(FAQ|Frequently Asked Questions)", markdown, re.M | re.I))


def has_cta(text: str, cta_hint: str = "") -> bool:
    lowered = text.lower()
    phrases = ["contact", "get in touch", "book", "sign up", "subscribe", "download", "try", "learn more",
               "call us", "visit", "schedule", "start", "join", "reach out", "comment", "share", "dm", "let me know",
               "what do you think", "?"]
    tail = lowered[-600:]
    return any(p in tail for p in phrases) or (bool(cta_hint) and cta_hint.lower()[:30] in lowered)


def suggest_internal_links(markdown: str, max_links: int = 5) -> list[str]:
    """Deterministic fallback: suggest a /blog/<slug> page for each H2 topic."""
    h2s = re.findall(r"^##\s+(.+)$", markdown, re.M)
    return [f"/blog/{slugify(h)}" for h in h2s if not re.match(r"(faq|sources|conclusion)", h, re.I)][:max_links]


# --------------------------------------------------------------------------------------
# LinkedIn
# --------------------------------------------------------------------------------------
def normalize_hashtag(tag: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", tag.lstrip("#"))
    if not words:
        return ""
    if len(words) == 1:
        return "#" + words[0]
    return "#" + "".join(w[:1].upper() + w[1:] for w in words)


def extract_hashtags(text: str) -> list[str]:
    return re.findall(r"(?<!\w)#[A-Za-z][A-Za-z0-9_]*", text)


def linkedin_metrics(text: str) -> dict:
    hashtags = extract_hashtags(text)
    body = re.sub(r"(?<!\w)#[A-Za-z][A-Za-z0-9_]*", "", text).strip()
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    first_line = body.strip().split("\n", 1)[0] if body.strip() else ""
    return {
        "chars": len(text),
        "hashtag_count": len(hashtags),
        "unique_hashtags": len({h.lower() for h in hashtags}),
        "paragraphs": len(paragraphs),
        "max_paragraph_chars": max((len(p) for p in paragraphs), default=0),
        "hook": first_line,
        "hook_chars": len(first_line),
        "line_breaks": text.count("\n"),
        "emoji_count": len(_EMOJI_PATTERN.findall(text)),
        "has_link": bool(re.search(r"https?://", text)),
    }


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    if len(paragraph) <= limit:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def format_linkedin(body: str, hashtags: list[str]) -> tuple[str, list[str], list[str]]:
    """Normalize a LinkedIn post: short paragraphs, 8–12 hashtags at the end, ≤1600 characters.

    Returns ``(text, hashtags, notes)``. ``notes`` describes any automatic fixes.
    """
    lo_tags, hi_tags = LINKEDIN_TARGETS["hashtags"]
    _, hi_chars = LINKEDIN_TARGETS["chars"]
    notes: list[str] = []

    inline = extract_hashtags(body)
    body = re.sub(r"(?<!\w)#[A-Za-z][A-Za-z0-9_]*", lambda m: m.group(0)[1:], body)  # keep words, drop '#'
    seen, tags = set(), []
    for tag in [*hashtags, *inline]:
        normalized = normalize_hashtag(tag)
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            tags.append(normalized)
    if len(tags) > hi_tags:
        notes.append(f"Trimmed hashtags from {len(tags)} to {hi_tags}.")
        tags = tags[:hi_tags]

    paragraphs = []
    for p in re.split(r"\n\s*\n", body.strip()):
        paragraphs.extend(_split_long_paragraph(re.sub(r"[ \t]*\n[ \t]*", "\n", p.strip()),
                                                LINKEDIN_TARGETS["max_paragraph_chars"]))
    text_body = "\n\n".join(p for p in paragraphs if p)
    tag_line = " ".join(tags)

    budget = hi_chars - len(tag_line) - 2
    if len(text_body) > budget:
        trimmed = text_body[:budget]
        cut = max(trimmed.rfind(". "), trimmed.rfind("? "), trimmed.rfind("! "), trimmed.rfind("\n"))
        text_body = (trimmed[: cut + 1] if cut > budget * 0.6 else trimmed.rsplit(" ", 1)[0]).rstrip()
        notes.append(f"Trimmed post body to fit the {hi_chars}-character limit.")
    if len(tags) < lo_tags:
        notes.append(f"Only {len(tags)} hashtags (target {lo_tags}–{hi_tags}).")
    text = f"{text_body}\n\n{tag_line}".strip()
    return text, tags, notes
