"""Reusable content templates (project FAQ #8).

The Query Handler picks a template for each request: the user can name one, or
it's auto-suggested from keywords in the request. The writers then follow the
template's outline.
"""

from __future__ import annotations

import re

BLOG_TEMPLATES: dict[str, str] = {
    "how-to": "Intro (problem + promise) → Why it matters → Step-by-step H2 sections (one per step, H3 tips) "
              "→ Common mistakes → FAQ → Conclusion with CTA",
    "listicle": "Intro → Numbered H2 items (each: what, why, example) → How to choose → FAQ → Conclusion with CTA",
    "comparison": "Intro → Quick verdict table → H2 per option (pros, cons, best for) → Head-to-head criteria "
                  "→ FAQ → Recommendation + CTA",
    "thought leadership": "Contrarian or fresh thesis → Evidence from research → Implications for the reader "
                          "→ Practical framework → FAQ → Call to action",
}

LINKEDIN_TEMPLATES: dict[str, str] = {
    "insights": "Hook with a surprising stat or claim → 3–5 short insight paragraphs → takeaway → question CTA",
    "story": "Hook in one line → short personal/customer story (situation, tension, turning point) "
             "→ lesson → question CTA",
    "tips": "Hook promising value → numbered tips (one line each, emoji bullets optional) → summary → CTA",
    "announcement": "Hook with the news → why it matters to the audience → what's next → CTA",
}

REPORT_TEMPLATES: dict[str, str] = {
    "trend report": "Executive Summary → Key Findings → Emerging Trends → Implications → Recommendations "
                    "→ Content Ideas → Sources",
    "market analysis": "Executive Summary → Market Overview (size, growth) → Customer Segments → Key Findings "
                       "→ Opportunities & Risks → Recommendations → Sources",
    "competitive analysis": "Executive Summary → Competitor Landscape → Strengths/Weaknesses → Positioning Gaps "
                            "→ Key Findings → Recommendations → Sources",
}

_SUGGEST_RULES = {
    "blog": [(r"\bhow (to|do|can)\b|\bguide\b|\bsteps?\b", "how-to"),
             (r"\b(top|best)\s+\d+|\b\d+\s+(ways|tips|ideas|tools|reasons)\b|\blist\b", "listicle"),
             (r"\bvs\.?\b|\bversus\b|\bcompar", "comparison"),
             (r"\bfuture\b|\bopinion\b|\bwhy .* (matters|wrong)\b|\bthought leadership\b", "thought leadership")],
    "linkedin": [(r"\bstory\b|\bjourney\b|\blearned\b|\blesson", "story"),
                 (r"\btips?\b|\bways\b|\bsteps\b|\bhacks?\b", "tips"),
                 (r"\blaunch|announc|introduc|excited to share", "announcement")],
    "report": [(r"\bmarket\b|\bindustry size\b|\bTAM\b", "market analysis"),
               (r"\bcompetit|\brivals?\b|\bvs\.?\b", "competitive analysis")],
}
_DEFAULTS = {"blog": "how-to", "linkedin": "insights", "report": "trend report"}
_CATALOG = {"blog": BLOG_TEMPLATES, "linkedin": LINKEDIN_TEMPLATES, "report": REPORT_TEMPLATES}


def suggest_template(kind: str, text: str, requested: str = "") -> str:
    """Pick a template name: the user's explicit choice if valid, otherwise keyword rules, otherwise the default."""
    catalog = _CATALOG[kind]
    if requested and requested.lower() in catalog:
        return requested.lower()
    for pattern, name in _SUGGEST_RULES[kind]:
        if re.search(pattern, text, re.I):
            return name
    return _DEFAULTS[kind]


# Buyer's-journey stages, after HubSpot's content-marketing guide: match the angle and the call to action to
# where the reader is. "What is X / how to" content serves awareness; comparisons serve consideration; pricing,
# offers and launches serve the decision.
FUNNEL_STAGES: dict[str, str] = {
    "awareness": "Top of funnel: the reader has a problem or a question, not a plan to buy. Teach and answer it; "
                 "mention the business only where it genuinely helps. Call to action: one low-commitment next "
                 "step (subscribe to the newsletter, read a related guide, save or follow).",
    "consideration": "Middle of funnel: the reader is weighing options. Show how the business's approach fits "
                     "their needs, with specifics, examples and proof. Call to action: a trust-building step "
                     "(see an example or customer story, try a sample, join the newsletter, ask a question).",
    "decision": "Bottom of funnel: the reader is ready to act. Answer the last objections (price, timing, how "
                "it works) and make acting easy. Call to action: a direct action (order, book, visit, get a "
                "quote) with a concrete reason to act now.",
}
_FUNNEL_RULES = [
    (r"\bpric(e|es|ing)\b|\bcost\b|\bbuy\b|\border\b|\bbook(ing)?\b|\bnear me\b|\bdiscount|\bcoupon|\bdeal\b"
     r"|\bpromo|\bspecial offer|\bquote\b|\bsign[- ]?up\b|\blaunch|\bannounc|\bgrand opening|\bpre-?order", "decision"),
    (r"\bvs\.?\b|\bversus\b|\bcompar|\bbest\b|\btop\s+\d+|\breviews?\b|\balternatives?\b|\bwhich\b"
     r"|\bchoos|\bcase stud|\boptions\b", "consideration"),
]
_TEMPLATE_STAGE = {"comparison": "consideration", "announcement": "decision"}


def suggest_funnel_stage(text: str, template: str = "", requested: str = "") -> str:
    """The buyer's-journey stage: the user's choice if valid, else keyword rules, else the template's, else awareness."""
    if requested and requested.lower() in FUNNEL_STAGES:
        return requested.lower()
    for pattern, stage in _FUNNEL_RULES:
        if re.search(pattern, text, re.I):
            return stage
    return _TEMPLATE_STAGE.get(template, "awareness")


def funnel_note(stage: str) -> str:
    return (f"FUNNEL STAGE: {stage}. {FUNNEL_STAGES[stage]} Adapt the brand's default call to action to this "
            "stage rather than repeating it word for word.")


def outline(kind: str, name: str) -> str:
    catalog = _CATALOG[kind]
    return catalog.get(name, catalog[_DEFAULTS[kind]])
