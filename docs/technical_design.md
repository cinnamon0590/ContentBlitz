# Technical Design Document

## 1. Goals and constraints

- Six specialized agents with clear separation of concerns and intelligent routing (initial and post-research).
- Platform-optimized, SEO-aware content with **verifiable sources** and a consistent brand voice.
- Keep working through provider outages (LLM, search, images) and keep costs predictable.
- Each capability testable on its own, without network access or API keys.

## 2. Key decisions and trade-offs

| Decision | Alternatives considered | Rationale / trade-off |
|---|---|---|
| **One LangGraph `StateGraph` with a shared `TypedDict` state** | Supervisor library, CrewAI crews, agents calling each other | Explicit edges make routing auditable and testable. Agents stay decoupled because they only talk through state. Cost: more wiring code |
| **Agents as factories over injected `AgentDeps`** | Module-level clients | Each agent can be unit-tested with fakes, and providers can be swapped. Cost: a little more boilerplate |
| **Artifacts stored as `model_dump()` dicts** | Pydantic objects in state | Serializes safely with every checkpointer and renders directly in the UI. Cost: rehydration with `model_validate` |
| **Separate `agent_scratch` channel for tool loops** | Template pattern (tools in `messages`) | Conversation memory holds only user turns and final answers, which means smaller prompts and cleaner "what did I ask?" answers. The template's ToolNode pattern is kept via `messages_key` |
| **Two-step research: tool loop, then structured synthesis over numbered sources** | Let the agent write the report straight from tool output | Citation ids are assigned by code, not by the model. The model never sees URLs, so it can't invent or alter them |
| **Deterministic quality scores; the LLM critique is advisory** | LLM-as-judge scores | Reproducible, explainable, free, and immune to self-grading drift. The LLM still adds qualitative suggestions |
| **Revision loop capped at 2 and budget-aware** | Unlimited "until it passes" | Bounded latency and cost; the best effort plus suggestions is returned |
| **Fallback chains as ordered lists in YAML** | Hard-coded try/except | Reordering providers is a config change. Each provider failure is recorded in warnings and provenance |
| **Model tiering (smart vs fast)** | One model for everything | Routing, tool decisions, prompt enhancement and critique are the most frequent and least demanding calls |
| **Regex-based injection and policy screen before the LLM** | LLM classifier | Zero cost and latency, deterministic, testable. Paired with the moderation API for broader coverage. Trade-off: patterns need maintenance; false negatives are mitigated by grounding rules and output checks |
| **PII redaction before memory and LLM** | Redact only the LLM payload | Prevents PII from being stored in checkpoints or sent to providers |
| **LinkedIn links moved to the first comment** | Inline links | Platform best practice (links reduce reach) while keeping attribution |

## 3. Service integration patterns

```
caller ──▶ Chain.search()/generate()
            ├─ cache hit? ──────────────────────────▶ return (from_cache=True)
            ├─ for provider in configured order:
            │     ├─ no key → record "no API key", continue
            │     ├─ request_json (timeout, 3× exponential backoff on 429/5xx/network)
            │     ├─ ProviderError → record, continue
            │     └─ success → cache.set, tracker.record, return
            ├─ stale cache (expired entry) ─────────▶ return (stale=True)
            └─ explicit offline outcome ────────────▶ agent labels output "model knowledge only"
```

- **LLM**: `LLMService.invoke` tries the primary model, then the fallback models. `structured()` tries native function
  calling, then JSON-mode parsing, then returns `None` so the agent can use its deterministic fallback.
- **Images**: `ImageChain` handles each model's quirks (the gpt-image family returns base64; quality is pinned to
  `medium`; sizes are configured per model) and always ends with a Pillow placeholder.
- **Moderation**: fails *open* with a logged warning, so a moderation outage never takes the product down.
  The regex guardrails still apply.

## 4. Content quality validation and enhancement

1. **Generation-time context engineering**: the brand block, grounding rules, template outline, keyword data
   (related searches, People-also-ask), research findings with ids, Moz/E-E-A-T/HubSpot guidance, and few-shot
   routing examples.
2. **Deterministic post-processing**: meta description fitting, slugs, exactly one H1, author/reviewer
   byline, hashtag normalization, LinkedIn length and paragraph enforcement, link stripping.
3. **Guardrail scrubbing**: invented URLs, invalid citation ids, and model-written source lists are removed.
   Provenance comes from the registry.
4. **Scoring** (`QualityValidator`): weighted categories per content type (see agents.md §5) plus actionable suggestions.
5. **Revision loop**: suggestions are fed back to the same writer with the previous draft.
6. **Human in the loop**: Dashboard edits are re-scored and persisted to the thread state; "Ask AI to refine"
   starts a refine turn.

## 5. Scalability and performance

- **Stateless workers**: all conversation state lives in the checkpointer. Moving from SQLite to the Postgres
  checkpointer lets multiple app replicas share threads.
- **Caching**: the disk TTL cache is per host. The `TTLCache` interface (`get/set/get_stale`) can be swapped for Redis
  when scaling horizontally.
- **Concurrency**: one `ContentBlitz` per Streamlit session isolates cost tracking. The graph itself is re-entrant per
  `thread_id`.
- **Latency**: research searches are I/O-bound, and the agent typically issues 1–4 searches. A future
  optimization is parallel fan-out with LangGraph's `Send` API, since queries are independent. Streaming node
  updates keep the UI responsive.
- **Cost guardrails**: `request_budget_usd`, revision cap, deep-research cap, cheap-tier routing, and caching.

## 6. Security and privacy

- Keys are only read from the environment; `service_status()` exposes presence booleans, never values.
- PII redaction runs before both LLM calls and memory writes.
- Prompt-injection screen, policy screen, moderation in and out, recursion limit.
- Generated content carries an AI-assistance disclosure and grounding labels (E-E-A-T trust).

## 7. Known limitations

- Keyword density and readability are heuristics; they don't guarantee rankings.
- Regex injection detection can miss novel attacks. Defense in depth limits the impact: grounding rules,
  no tool access to secrets, output scrubbing.
- Pricing in `services.yaml` is approximate and must be maintained.
- The SQLite checkpointer is single-host.
