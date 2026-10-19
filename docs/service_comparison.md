# Service Comparison Analysis

This document compares the AI services ContentBlitz integrates, explains why each one sits where it does in
the fallback chains, and gives recommendations by budget.

> **About the numbers.** Pricing and free tiers below are approximate list values used for the cost tracker
> (`config/services.yaml`). Providers change prices often, so verify them on each provider's pricing page
> before you rely on them. **Measured** latency and quality figures come from `scripts/benchmark_services.py`
> run with your own keys (see §5). None are invented here.

## 1. Research providers

| | **SerpAPI** (primary) | **Tavily** (fallback 1) | **Perplexity Sonar** (fallback 2 / deep research) |
|---|---|---|---|
| What it returns | Raw Google results: organic links, answer box, **related searches**, **People also ask** | Search results with extracted page content; optional synthesized answer (`advanced`) | An LLM-synthesized answer with citations (`search_results` / `citations`) |
| Free tier | ~100 searches/month | ~1,000 credits/month | Paid (per request + tokens) |
| Control over sources | Highest: we choose and cite | High | Lower: the provider pre-selects and pre-analyzes |
| SEO value | **Best**: related searches and PAA feed the keyword tool and the FAQ sections | Medium | Low |
| Synthesis | None (our LLM synthesizes) | Basic answer | Strong (saves an LLM call) |
| Role in ContentBlitz | `web_search` default and `keyword_research` | First fallback; `deep_research` second choice | `deep_research` first choice (≤ 1 per request); last live fallback |

**Why this order:** the FAQ recommends SerpAPI + GPT as the primary for control and custom analysis. Tavily's
generous free tier makes it a cheap, resilient backup. Perplexity is the most expensive per call but gives
research and analysis in one step, so it's reserved for the single `deep_research` call and for emergencies.

## 2. LLM providers

| | **OpenAI gpt-4o** (smart) | **OpenAI gpt-4o-mini** (fast) | Claude Sonnet / Haiku (optional) | Gemini Pro / Flash (optional) |
|---|---|---|---|---|
| Used for | Blog, LinkedIn, report writing; research synthesis | Routing, tool loops, prompt enhancement, critique, brand-voice learning | Drop-in via `LLM_PROVIDER=anthropic` | Drop-in via `LLM_PROVIDER=gemini` |
| List price (USD per 1M tokens, in/out) | ~2.50 / 10.00 | ~0.15 / 0.60 | see Anthropic pricing | see Google pricing |
| Strengths | Reliable structured output, strong long-form quality | Very cheap, fast, good at classification | Strong reasoning and long-form tone (FAQ notes Claude for complex content) | Large free tier for development |
| Program availability | Provided | Provided | Bring your own key | Bring your own key |

**Tiering impact:** most calls in a request are routing, tool decisions, prompt enhancement and critique. Running
them on the fast tier keeps the expensive model for the few long-form generations. The usage panel shows actual
tokens and estimated cost per request.

## 3. Image models

| | **gpt-image-1** | **DALL-E 3** | **DALL-E 2** | Pillow placeholder |
|---|---|---|---|---|
| Quality / prompt adherence | Highest | High | Moderate | Branded gradient card |
| Requirements | **Verified OpenAI organization** | Standard key | Standard key | None |
| Sizes used | 1536×1024 | 1792×1024 | 1024×1024 | 1536×1024 |
| Approx. cost / image | ~$0.04–0.17 by quality (tracked as 0.063) | ~$0.08 (standard landscape) | ~$0.02 | $0 |
| Role | Primary | Fallback 1 (the default when the org is unverified) | Fallback 2 | Final fallback (never fails) |

Alternatives worth adding (the interface is `client.generate(model, prompt, size) -> bytes`): Stability AI
(SDXL / SD3) for open-model control, Google Imagen, Midjourney (no official API; Discord-based).

## 4. Cost-benefit summary per request (illustrative)

The estimates assume typical prompt sizes. Measure your own with the usage panel or the benchmark script.

| Request | Main cost drivers | Ways to reduce cost |
|---|---|---|
| Research report | 2–4 searches (free tier), 1 synthesis + 1 report on the smart tier | Cache hits are free; skip `deep_research` |
| SEO blog | Research + keyword lookup + 1 long smart generation (+ up to 2 revisions) | Lower `max_revisions`, raise or lower `pass_threshold`, reuse thread research |
| LinkedIn post | Research (or reuse) + 1 smart generation | Write it as a follow-up to existing research |
| Image | 1 image call + 1 fast prompt call | DALL-E 2 for drafts; placeholder for layout tests |
| Campaign | All of the above, sharing one research pass | Already optimized: research once, write three times |

## 5. Measured benchmarks

Run the benchmark with your keys and paste the printed tables here:

```bash
python scripts/benchmark_services.py --queries 3            # research + LLM tiers
python scripts/benchmark_services.py --queries 3 --images   # + image models (≈ $0.15–0.25)
```

The script reports, for each research provider: median latency, average result count, unique source domains,
and error count. For each LLM tier it reports latency, tokens and cost; for each image model, latency, output
size and cost. Raw JSON is saved to `outputs/benchmarks/`.

| Provider | median latency (s) | avg results | unique domains | errors |
|---|---|---|---|---|
| serpapi | _run script_ | | | |
| tavily | _run script_ | | | |
| perplexity | _run script_ | | | |

## 6. Recommendations

| Situation | Recommended configuration |
|---|---|
| **Student / free tier** | OpenAI (program key) + SerpAPI free + Tavily free; images on DALL-E 3 (or DALL-E 2 for drafts); `request_budget_usd: 0.25` |
| **Small business, quality-first** | gpt-4o smart tier, SerpAPI paid, Perplexity for deep research, gpt-image-1 (verified org), `pass_threshold: 75` |
| **Lowest cost** | `smart: gpt-4o-mini`, Tavily only, DALL-E 2, `max_revisions: 0`, longer cache TTL |
| **Provider outage resilience** | Keep at least two research keys configured; leave the placeholder at the end of the image chain |
| **Strong long-form voice** | Try `LLM_PROVIDER=anthropic` for the writers; compare drafts in the Content Dashboard |
| **Privacy-sensitive** | Keep moderation on, and keep PII redaction (always on); consider self-hosted models via a custom `create_chat_model` branch |
