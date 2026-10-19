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


def outline(kind: str, name: str) -> str:
    catalog = _CATALOG[kind]
    return catalog.get(name, catalog[_DEFAULTS[kind]])
