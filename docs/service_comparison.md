# Service Comparison Analysis

This document compares the AI services ContentBlitz integrates, explains why each one sits where it does in
the fallback chains, and gives recommendations by budget.

> **About the numbers.** Pricing and free tiers below are approximate list values used for the cost tracker
> (`config/services.yaml`). Providers change prices often, so verify them on each provider's pricing page
> before you rely on them. **Measured** latency and quality figures come from `scripts/benchmark_services.py`
> run with your own keys (see §5). None are invented here.

## 1. Research providers

| | **SerpAPI** (primary) | **Tavily** (fallback 1) | **Perplexity Agent API** (fallback 2 / deep research) |
|---|---|---|---|
| What it returns | Raw Google results: organic links, answer box, **related searches**, **People also ask** | Search results with extracted page content; optional synthesized answer (`advanced`) | An LLM-synthesized answer with citations (`search_results` + `message` output items; `/v1/agent`, preset `fast`) |
| Free tier | 250 searches/month (Starter plan: $25 for 1,000) | ~1,000 credits/month | Paid (~$0.002 per `fast` call, measured) |
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
| List price (USD per 1M tokens, in/out) | ~2.50 / 10.00 | ~0.15 / 0.60 | Sonnet 5.5: 2.00 / 10.00 · Haiku 5.5: 0.10 / 0.50 | see Google pricing (not tracked) |
| Strengths | Reliable structured output, strong long-form quality | Very cheap, fast, good at classification | Strong reasoning and long-form tone (FAQ notes Claude for complex content) | Large free tier for development |
| Program availability | Provided | Provided | Bring your own key | Bring your own key |

**Measured live, 2026-10-19** (the same request on each provider: "Write a LinkedIn post about why fast
replies win customers for local bakeries", which runs routing, the research tool loop, synthesis, the LinkedIn
writer and the quality gate):

| Provider (smart / fast) | Result | LinkedIn quality | Tokens | Tracked cost | Time |
|---|---|---|---|---|---|
| Anthropic `claude-sonnet-5-5` / `claude-haiku-5-5` | research + post, no warnings | 100/100 | 31,356 | $0.063 | 106 s |
| Gemini `gemini-pro-latest` / `gemini-3.8-flash` | research + post, no warnings | 100/100 | 21,310 | not priced | 106 s |

Claude Sonnet 5.5 and Haiku 5.5 reject any `temperature` and forced tool calls; the `anthropic` block in
`config/services.yaml` handles both (`temperature: null`, `structured_output: json_schema`). Before that fix
every Claude call failed and the router silently fell back to keyword rules, a gap only the live test caught.

**Tiering impact:** most calls in a request are routing, tool decisions, prompt enhancement and critique. Running
them on the fast tier keeps the expensive model for the few long-form generations. The usage panel shows actual
tokens and estimated cost per request.

## 3. Image models

| | **gpt-image-1** | **gpt-image-1-mini** | **Gemini 2.5 Flash Image** | Pillow placeholder |
|---|---|---|---|---|
| Provider | OpenAI | OpenAI | Google | Local (Pillow) |
| Quality / prompt adherence | Highest | Good (draft quality) | High; follows "no text" reliably | Branded gradient card |
| Requirements | **Verified OpenAI organization** | Standard key | `GOOGLE_API_KEY` on a **billing-enabled** project (free tier: 0 images) | None |
| Sizes used | 1536×1024 | 1536×1024 | 3:2 aspect ratio → 1248×832 | 1536×1024 |
| Approx. cost / image | ~$0.063 (`quality: medium`) | ~$0.015 (`quality: medium`) | ~$0.039 (~1290 output tokens) | $0 |
| Role | Primary | Fallback 1 (default when the org is unverified) | Fallback 2: **cross-provider redundancy** | Final fallback (never fails) |

DALL-E 2 and DALL-E 3 have been retired by OpenAI (the API returns "model does not exist"), so they were removed
from the chain. Quality is pinned to `medium` in `config/services.yaml`: the API default is `high`, which costs
about four times as much. Newer OpenAI models (`gpt-image-1.5`, `gpt-image-2`) can be put at the top of the chain.

Gemini sits after both OpenAI models, so an OpenAI outage, key problem or billing issue still produces a real
image. `ImageClients` routes `gemini-*` models to `GeminiImageClient` (REST, no SDK) and everything else to
OpenAI. `gemini-3.1-flash-image` also works on this key and looks more photographic, but it was slower (9.8 s vs
5.8 s) and painted text despite the no-text prompt, which matters for the image guardrails. Other options
(the interface is `client.generate(model, prompt, size) -> bytes`): Stability AI (SDXL / SD3), Midjourney
(no official API).

## 4. Cost-benefit summary per request (illustrative)

The estimates assume typical prompt sizes. Measure your own with the usage panel or the benchmark script.

| Request | Main cost drivers | Ways to reduce cost |
|---|---|---|
| Research report | 2–4 searches (free tier), 1 synthesis + 1 report on the smart tier | Cache hits are free; skip `deep_research` |
| SEO blog | Research + keyword lookup + 1 long smart generation (+ up to 2 revisions) | Lower `max_revisions`, raise or lower `pass_threshold`, reuse thread research |
| LinkedIn post | Research (or reuse) + 1 smart generation | Write it as a follow-up to existing research |
| Image | 1 image call + 1 fast prompt call | gpt-image-1-mini for drafts; placeholder for layout tests |
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

Measured 2026-10-19 (3 queries per provider; SerpAPI/Tavily figures are from a warm 30-minute cache, so their
cold latency is higher, about 0.9 s and 2.4 s on the first run):

| Provider | median latency (s) | avg results | unique domains | errors | cost / call (USD) |
|---|---|---|---|---|---|
| serpapi | 0.88 (cold) | 5 | 13 | 0 | 0 on the free plan; 0.025 on Starter |
| tavily | 2.39 (cold) | 5 | 13 | 0 | 0 (free tier) |
| perplexity | 6.16 | 13.3 | 30 | 0 | 0.002 |

| LLM tier | model | latency (s) | cost (USD, short probe) |
|---|---|---|---|
| fast | gpt-4o-mini | 1.65–2.03 | 0.00003 |
| smart | gpt-4o | 0.91 | 0.0006 |

| Image model | latency (s) | size | cost (USD) |
|---|---|---|---|
| gpt-image-1 | 29.8–82.5 | 1536×1024 | 0.063 |
| gpt-image-1-mini | 15.6 | 1536×1024 | 0.015 |
| gemini-2.5-flash-image | 5.8 | 1248×832 | 0.039 |

## 6. Recommendations

| Situation | Recommended configuration |
|---|---|
| **Student / free tier** | OpenAI (program key) + SerpAPI free + Tavily free; images on gpt-image-1-mini; `request_budget_usd: 0.25` |
| **Small business, quality-first** | gpt-4o smart tier, SerpAPI paid, Perplexity for deep research, gpt-image-1 (verified org), `pass_threshold: 75` |
| **Lowest cost** | `smart: gpt-4o-mini`, Tavily only, gpt-image-1-mini first, `max_revisions: 0`, longer cache TTL |
| **Provider outage resilience** | Keep at least two research keys configured; add a billing-enabled `GOOGLE_API_KEY` so images survive an OpenAI outage; leave the placeholder at the end of the image chain |
| **Strong long-form voice** | Try `LLM_PROVIDER=anthropic` for the writers; compare drafts in the Content Dashboard |
| **Privacy-sensitive** | Keep moderation on, and keep PII redaction (always on); consider self-hosted models via a custom `create_chat_model` branch |
