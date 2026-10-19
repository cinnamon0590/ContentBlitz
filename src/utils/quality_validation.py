"""Content quality scoring: the gate in the content enhancement pipeline.

Every score is computed **deterministically in Python** from measurable signals
(see :mod:`src.utils.content_optimization`). An optional LLM critique adds
advisory feedback and suggestions, but it can never change the score. That's a
guardrail against score drift and against the model grading its own homework.

Each category is scored 0–100, then combined with per-content-type weights. If
``overall < pass_threshold``, the workflow sends the draft back to its writer,
along with the suggestions, for up to ``max_revisions`` rewrites.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.utils import content_optimization as co
from src.workflow.state_management import (
    BlogPost,
    BrandProfile,
    LinkedInPost,
    QualityReport,
    StrategyReport,
)

BLOG_WEIGHTS = {"seo": 0.30, "structure": 0.20, "readability": 0.15, "citations": 0.15, "brand_voice": 0.10,
                "length": 0.10}
LINKEDIN_WEIGHTS = {"length": 0.25, "hashtags": 0.15, "hook": 0.15, "formatting": 0.20, "cta": 0.10,
                    "brand_voice": 0.15}
REPORT_WEIGHTS = {"structure": 0.35, "citations": 0.35, "readability": 0.15, "brand_voice": 0.15}

GENERIC_HOOKS = ("in today's", "in this day and age", "in the ever-evolving", "are you ready", "have you ever wondered")


def _band(value: float, lo: float, hi: float, tolerance: float) -> float:
    """100 inside [lo, hi]; linear decay to 0 at ``tolerance`` outside the band."""
    if lo <= value <= hi:
        return 100.0
    distance = lo - value if value < lo else value - hi
    return max(0.0, 100.0 * (1 - distance / tolerance))


def _weighted(scores: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(scores[k] * w for k, w in weights.items()), 1)


# --------------------------------------------------------------------------------------
# Brand voice (rules)
# --------------------------------------------------------------------------------------
def brand_voice_check(text: str, brand: BrandProfile) -> tuple[float, list[str], dict]:
    lowered = text.lower()
    suggestions: list[str] = []
    banned_hits = sorted({w for w in brand.banned_words if re.search(rf"\b{re.escape(w.lower())}\b", lowered)})
    missing_required = [p for p in brand.required_phrases if p.lower() not in lowered]
    ease = co.readability(text)["flesch_reading_ease"]
    score = 100.0
    if banned_hits:
        score -= min(60, 15 * len(banned_hits))
        suggestions.append(f"Remove off-brand words: {', '.join(banned_hits)}.")
    if missing_required:
        score -= min(30, 10 * len(missing_required))
        suggestions.append(f"Include brand phrases: {', '.join(missing_required)}.")
    if ease < brand.target_reading_ease - 15:
        score -= 20
        suggestions.append("Simplify the language to match the brand's plain, approachable voice.")
    exclamations = text.count("!")
    if exclamations > 4:
        score -= 10
        suggestions.append("Tone down exclamation marks for a professional voice.")
    return max(0.0, score), suggestions, {"banned_words_found": banned_hits, "missing_phrases": missing_required}


class Critique(BaseModel):
    """Advisory LLM critique: it gives feedback only and never sets the score."""

    feedback: str = ""
    suggestions: list[str] = Field(default_factory=list)


class QualityValidator:
    def __init__(self, pass_threshold: float = 70, llm=None, critique_prompt: str | None = None):
        self.pass_threshold = pass_threshold
        self.llm = llm
        self.critique_prompt = critique_prompt

    # ---- shared -----------------------------------------------------------------
    def _critique(self, content_type: str, text: str, brand: BrandProfile) -> tuple[str, list[str]]:
        if not self.llm or not self.critique_prompt:
            return "", []
        critique = self.llm.structured(
            Critique,
            self.critique_prompt,
            f"Content type: {content_type}\nBrand: {brand.model_dump_json()}\n\nCONTENT:\n{text[:6000]}",
            tier="fast",
        )
        if not critique:
            return "", []
        return critique.feedback, critique.suggestions[:3]

    def _finish(self, content_type: str, scores: dict, weights: dict, metrics: dict, suggestions: list[str],
                text: str, brand: BrandProfile) -> QualityReport:
        overall = _weighted(scores, weights)
        feedback, llm_suggestions = self._critique(content_type, text, brand)
        return QualityReport(
            content_type=content_type,
            overall=overall,
            passed=overall >= self.pass_threshold,
            scores={k: round(v, 1) for k, v in scores.items()},
            metrics=metrics,
            suggestions=list(dict.fromkeys([*suggestions, *llm_suggestions])),
            llm_feedback=feedback,
        )

    # ---- blog ---------------------------------------------------------------------
    def score_blog(self, blog: BlogPost, brand: BrandProfile, known_source_ids: set[int] | None = None) -> QualityReport:
        md = blog.markdown
        body = co.body_without_sources(md)
        kw = blog.target_keyword
        suggestions: list[str] = []

        density = co.keyword_density(body, kw)
        headings = co.heading_structure(body, kw)
        meta = co.check_meta_description(blog.meta_description, kw)
        kw_intro = co.first_paragraph_has_keyword(body, kw)

        lo, hi = co.BLOG_TARGETS["keyword_density"]
        seo = 0.45 * _band(density, lo, hi, 1.5) + 0.25 * (100 if meta["ok"] else 40 if meta["length"] else 0)
        seo += 0.15 * (100 if headings["keyword_in_h1"] else 0) + 0.15 * (100 if kw_intro else 0)
        if not lo <= density <= hi:
            suggestions.append(f"Adjust keyword density for '{kw}' to {lo}–{hi}% (currently {density}%).")
        if not kw_intro:
            suggestions.append("Mention the target keyword in the first 100 words.")
        suggestions.extend(meta["issues"])
        title_issues = co.check_title(blog.title, kw)
        seo -= 5 * len(title_issues)
        suggestions.extend(title_issues)
        if kw and blog.slug and co.slugify(kw) not in blog.slug:
            seo -= 5
            suggestions.append("Include the target keyword in the URL slug.")

        structure = 100.0 - 20 * len(headings["issues"])
        faq, cta = co.has_faq_section(md), co.has_cta(body, brand.cta)
        structure -= 0 if faq else 15
        structure -= 0 if cta else 15
        suggestions.extend(headings["issues"])
        if not faq:
            suggestions.append("Add an FAQ section (helps 'People also ask' visibility).")
        if not cta:
            suggestions.append("End with a clear call to action.")

        read = co.readability(body)
        readability_score = _band(read["flesch_reading_ease"], brand.target_reading_ease, 100, 40)
        if readability_score < 80:
            suggestions.append("Shorten sentences and use simpler words to improve readability.")

        uncited = co.uncited_statistics(md)
        cited = co.citation_ids(body)
        invalid = sorted(cited - known_source_ids) if known_source_ids is not None else []
        citations = 100.0 - min(60, 15 * len(uncited)) - min(40, 20 * len(invalid))
        if not cited and known_source_ids:
            citations -= 30
            suggestions.append("Cite your research inline with [n] markers.")
        if uncited:
            suggestions.append(f"Add [n] citations to {len(uncited)} statistic(s), e.g. “{uncited[0][:80]}”.")
        if invalid:
            suggestions.append(f"Citation ids {invalid} do not match any source — fix or remove them.")

        brand_score, brand_suggestions, brand_metrics = brand_voice_check(body, brand)
        suggestions.extend(brand_suggestions)

        words = co.word_count(body)
        min_words = co.BLOG_TARGETS["min_words"]
        length = 100.0 if words >= min_words else max(0.0, 100 * words / min_words)
        if words < min_words:
            suggestions.append(f"Expand to at least {min_words} words (currently {words}).")

        scores = {"seo": seo, "structure": max(0, structure), "readability": readability_score,
                  "citations": max(0, citations), "brand_voice": brand_score, "length": length}
        metrics = {"keyword_density": density, "word_count": words, **read, "meta_length": meta["length"],
                   "h1": headings["h1"], "h2": headings["h2"], "h3": headings["h3"], "has_faq": faq, "has_cta": cta,
                   "uncited_statistics": uncited, "invalid_citations": invalid, "citations_used": sorted(cited),
                   **brand_metrics}
        return self._finish("blog", scores, BLOG_WEIGHTS, metrics, suggestions, md, brand)

    # ---- LinkedIn -----------------------------------------------------------------
    def score_linkedin(self, post: LinkedInPost, brand: BrandProfile) -> QualityReport:
        m = co.linkedin_metrics(post.text)
        suggestions: list[str] = []
        lo, hi = co.LINKEDIN_TARGETS["chars"]
        length = _band(m["chars"], lo, hi, 600)
        if not lo <= m["chars"] <= hi:
            suggestions.append(f"Aim for {lo}–{hi} characters (currently {m['chars']}).")

        tlo, thi = co.LINKEDIN_TARGETS["hashtags"]
        hashtags = _band(m["unique_hashtags"], tlo, thi, 6)
        if not tlo <= m["unique_hashtags"] <= thi:
            suggestions.append(f"Use {tlo}–{thi} relevant hashtags (currently {m['unique_hashtags']}).")

        hook = 100.0
        if m["hook_chars"] > co.LINKEDIN_TARGETS["hook_max_chars"]:
            hook -= 40
            suggestions.append("Shorten the opening line so the hook shows before 'see more'.")
        if m["hook"].lower().startswith(GENERIC_HOOKS):
            hook -= 40
            suggestions.append("Replace the generic opener with a specific, surprising hook.")
        if m["hook_chars"] < 15:
            hook -= 30
            suggestions.append("Open with a stronger, more substantive hook line.")

        formatting = 100.0
        if m["paragraphs"] < 4:
            formatting -= 30
            suggestions.append("Break the post into more short paragraphs for mobile readability.")
        if m["max_paragraph_chars"] > co.LINKEDIN_TARGETS["max_paragraph_chars"]:
            formatting -= 25
            suggestions.append("Split long paragraphs (keep each under ~3 lines).")
        if m["has_link"]:
            formatting -= 30
            suggestions.append("Move links to the first comment (links in the body reduce reach).")
        ease = co.readability(post.text)["flesch_reading_ease"]
        if ease < 50:
            formatting -= 15

        cta = 100.0 if co.has_cta(post.text) else 30.0
        if cta < 100:
            suggestions.append("End with a question or call to action to invite comments.")

        brand_score, brand_suggestions, brand_metrics = brand_voice_check(post.text, brand)
        suggestions.extend(brand_suggestions)

        scores = {"length": length, "hashtags": hashtags, "hook": max(0, hook), "formatting": max(0, formatting),
                  "cta": cta, "brand_voice": brand_score}
        metrics = {**m, "flesch_reading_ease": ease, **brand_metrics}
        return self._finish("linkedin", scores, LINKEDIN_WEIGHTS, metrics, suggestions, post.text, brand)

    # ---- research / strategy report -------------------------------------------------
    def score_report(self, report: StrategyReport, brand: BrandProfile) -> QualityReport:
        md = report.markdown
        suggestions = []
        required = ["summary", "finding", "recommend", "source"]
        headings = " ".join(co.heading_structure(md)["headings"]).lower()
        missing = [r for r in required if r not in headings and r not in md.lower()[:200]]
        structure = 100.0 - 20 * len(missing)
        if missing:
            suggestions.append(f"Add sections covering: {', '.join(missing)}.")
        uncited = co.uncited_statistics(md)
        has_sources = bool(report.provenance.sources)
        citations = (100.0 - min(60, 15 * len(uncited))) if has_sources else 40.0
        if uncited:
            suggestions.append(f"Cite {len(uncited)} statistic(s) with [n] markers.")
        read = co.readability(md)
        readability_score = _band(read["flesch_reading_ease"], 45, 100, 40)
        brand_score, brand_suggestions, _ = brand_voice_check(md, brand)
        suggestions.extend(brand_suggestions)
        scores = {"structure": max(0, structure), "citations": max(0, citations), "readability": readability_score,
                  "brand_voice": brand_score}
        metrics = {"uncited_statistics": uncited, "source_count": len(report.provenance.sources), **read}
        return self._finish("report", scores, REPORT_WEIGHTS, metrics, suggestions, md, brand)


# --------------------------------------------------------------------------------------
# Brand voice learning (bonus feature)
# --------------------------------------------------------------------------------------
class LearnedVoice(BaseModel):
    tone: str = ""
    audience: str = ""
    values: list[str] = Field(default_factory=list)
    signature_phrases: list[str] = Field(default_factory=list)
    avoid_words: list[str] = Field(default_factory=list)


def learn_brand_voice(sample_text: str, base: BrandProfile, llm, system_prompt: str) -> BrandProfile:
    """Infer tone, audience and vocabulary from sample content and merge them into ``base``."""
    learned = llm.structured(LearnedVoice, system_prompt, sample_text[:6000], tier="fast") if llm else None
    profile = base.model_copy(deep=True)
    profile.sample_text = sample_text[:2000]
    profile.target_reading_ease = max(30.0, min(80.0, co.readability(sample_text)["flesch_reading_ease"]))
    if learned:
        profile.tone = learned.tone or profile.tone
        profile.audience = learned.audience or profile.audience
        profile.values = learned.values or profile.values
        profile.banned_words = sorted(set(profile.banned_words) | {w.lower() for w in learned.avoid_words})
        if learned.signature_phrases and not profile.required_phrases:
            profile.required_phrases = learned.signature_phrases[:1]
    return profile
