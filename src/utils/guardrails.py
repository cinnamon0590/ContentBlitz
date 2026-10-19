"""Input, execution, and output guardrails.

Input  : length cap, prompt-injection screen, PII redaction, policy screen
         (e.g. fake reviews), and an OpenAI moderation check.
Output : moderation, citation integrity (only URLs that were actually
         retrieved may appear), and stripping links from LinkedIn post bodies.

Everything here is plain Python, so it's deterministic and unit-testable. The
moderation client is injected and fails open with a logged warning, so an
outage never blocks the app.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from src.utils.observability import get_logger
from src.workflow.state_management import normalize_url

log = get_logger("guardrails")


class Moderator(Protocol):
    def check(self, text: str) -> tuple[bool, list[str]]: ...


# --------------------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
    r"\bdisregard\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|system)?\s*(instructions|prompts|rules)",
    r"\bforget\s+(all\s+|your\s+)?(previous\s+|prior\s+)?(instructions|rules)",
    r"\b(reveal|show|print|repeat|output)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"\byou\s+are\s+now\s+(?:in\s+)?(dan|developer\s+mode|jailbroken|unfiltered)",
    r"\b(jailbreak|do\s+anything\s+now)\b",
    r"\boverride\s+(your\s+)?(safety|guardrails|rules)",
    r"<\|?(im_start|im_end|system)\|?>",
    r"^\s*#{2,}\s*system\b",
    r"\bbegin\s+system\s+prompt\b",
]

_REVIEWS = r"(reviews?|testimonials?|ratings?)"
POLICY_PATTERNS = {
    # "fake reviews", but also "fake five-star customer reviews" (up to four words in between)
    "fake_reviews": rf"\b(fake|fabricated|made[- ]up|false|phony|bogus|invented)\b(?:[\s-]+[\w-]+){{0,4}}?[\s-]+{_REVIEWS}\b",
    # reviews written in customers' voices, or written to be posted on review sites
    "reviews_as_customers": rf"\b{_REVIEWS}\b.{{0,60}}\b(as if (they were|written by|from)|pretending to be|posing as|in the "
                            r"voice of|signed as)\b.{0,20}\b(customers?|clients?|guests?|patients?|buyers?)\b",
    "reviews_for_review_sites": rf"\b(write|generate|create|draft|make)\b.{{0,40}}\b{_REVIEWS}\b.{{0,40}}\b(post|publish|"
                                r"upload|submit)\b.{0,25}\b(google|yelp|tripadvisor|amazon|facebook|trustpilot|etsy)\b",
    "impersonation": r"\b(pretend|pose|posing)\s+(to\s+be|as)\s+(a\s+)?(real\s+)?(customer|competitor|doctor|lawyer)\b",
    "competitor_defamation": r"\b(defame|smear|spread\s+(rumou?rs|lies))\b.*\bcompetitor",
}

PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,19}\b",
    "phone": r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)",
}

URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+", re.I)
MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with ``[REDACTED_<KIND>]``; return the new text and the kinds found."""
    found: list[str] = []

    def sub(kind: str, pattern: str, value: str) -> str:
        def repl(match: re.Match) -> str:
            raw = match.group(0)
            if kind == "credit_card":
                digits = re.sub(r"\D", "", raw)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    return raw
            found.append(kind)
            return f"[REDACTED_{kind.upper()}]"

        return re.sub(pattern, repl, value)

    for kind in ("email", "ssn", "credit_card", "phone"):
        text = sub(kind, PII_PATTERNS[kind], text)
    return text, sorted(set(found))


def detect_injection(text: str) -> list[str]:
    return [p for p in INJECTION_PATTERNS if re.search(p, text, re.I | re.M)]


# Asking how to deal with fake reviews is legitimate and must not be blocked.
DEFENSIVE_REVIEW_CONTEXT = re.compile(
    r"\b(spot|spotting|detect|identify|recogni[sz]e|respond(ing)? to|reply(ing)? to|report(ing)?|flag(ging)?|remove|"
    r"removing|deal(ing)? with|handle|handling|fight|combat|protect|against|about|policy on|stop)\b(?:\W+\w+){0,3}\W+"
    r"(fake|fabricated|false|bogus|phony)\b", re.I)


def detect_policy_violation(text: str) -> list[str]:
    hits = [name for name, p in POLICY_PATTERNS.items() if re.search(p, text, re.I)]
    if hits == ["fake_reviews"] and DEFENSIVE_REVIEW_CONTEXT.search(text):
        return []  # e.g. "how do I respond to fake reviews?"
    return hits


# --------------------------------------------------------------------------------------
# Input guardrail
# --------------------------------------------------------------------------------------
@dataclass
class InputCheck:
    allowed: bool
    text: str
    category: str = "ok"  # ok | empty | too_long | injection | policy | moderation
    reasons: list[str] = field(default_factory=list)
    redactions: list[str] = field(default_factory=list)

    @property
    def user_message(self) -> str:
        messages = {
            "empty": "Please tell me what you'd like to create — e.g. a blog post, LinkedIn post, image or research.",
            "too_long": "That message is too long. Please shorten it (or paste only the key brief).",
            "injection": "I can't follow instructions that try to change how I operate. "
                         "Tell me about the content you need and I'll help.",
            "policy": "I can't help create deceptive content (such as fake reviews, impersonation or defamation). "
                      "I'm happy to help with authentic marketing content instead.",
            "moderation": "I can't help with that request because it may violate content policies.",
        }
        return messages.get(self.category, "")


def check_input(text: str, max_chars: int = 4000, moderator: Moderator | None = None) -> InputCheck:
    stripped = (text or "").strip()
    if not stripped:
        return InputCheck(False, "", "empty", ["empty input"])
    if len(stripped) > max_chars:
        return InputCheck(False, stripped[:200], "too_long", [f"{len(stripped)} chars > {max_chars}"])
    if hits := detect_injection(stripped):
        log.warning("prompt injection blocked: %s", hits)
        return InputCheck(False, stripped, "injection", hits)
    if hits := detect_policy_violation(stripped):
        return InputCheck(False, stripped, "policy", hits)
    redacted, kinds = redact_pii(stripped)
    if moderator:
        flagged, categories = moderator.check(redacted)
        if flagged:
            return InputCheck(False, redacted, "moderation", categories, kinds)
    return InputCheck(True, redacted, "ok", [], kinds)


# --------------------------------------------------------------------------------------
# Output guardrails
# --------------------------------------------------------------------------------------
def enforce_citation_integrity(text: str, allowed_urls: set[str]) -> tuple[str, list[str]]:
    """Remove any URL that is not in the source registry (prevents invented links).

    Markdown links ``[text](bad-url)`` are reduced to ``text``; bare bad URLs are removed.
    """
    allowed = {normalize_url(u) for u in allowed_urls}
    removed: list[str] = []

    def md_repl(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        if normalize_url(url) in allowed:
            return match.group(0)
        removed.append(url)
        return label

    text = MD_LINK_PATTERN.sub(md_repl, text)

    def bare_repl(match: re.Match) -> str:
        url = match.group(0).rstrip(".,;:")
        trailing = match.group(0)[len(url):]
        if normalize_url(url) in allowed:
            return match.group(0)
        removed.append(url)
        return trailing

    # Only bare URLs not already inside an allowed markdown link.
    parts = re.split(r"(\[[^\]]+\]\(https?://[^)\s]+\))", text)
    text = "".join(p if MD_LINK_PATTERN.fullmatch(p) else URL_PATTERN.sub(bare_repl, p) for p in parts)
    if removed:
        log.warning("stripped %d unverified URL(s): %s", len(removed), removed)
    return text, removed


def strip_links(text: str) -> tuple[str, int]:
    """Remove all URLs (LinkedIn body: links reduce reach; sources go in the first comment)."""
    count = len(MD_LINK_PATTERN.findall(text))
    text = MD_LINK_PATTERN.sub(r"\1", text)
    count += len(URL_PATTERN.findall(text))
    text = URL_PATTERN.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text), count


def check_output(text: str, moderator: Moderator | None = None) -> tuple[bool, list[str]]:
    """Return ``(safe, categories)`` for generated text."""
    if not moderator:
        return True, []
    flagged, categories = moderator.check(text)
    return not flagged, categories
