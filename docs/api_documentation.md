# API Documentation

All public interfaces, grouped by layer. Types refer to `src/workflow/state_management.py` unless noted.

---

## 1. Application facade — `src.core.workflow`

### `ContentBlitz(config=None, deps=None, checkpointer=None)`
| Argument | Default | Notes |
|---|---|---|
| `config` | `get_config()` | `AppConfig` from YAML + `.env` |
| `deps` | `AgentDeps.from_config(config)` | inject fakes for testing |
| `checkpointer` | `make_checkpointer(config)` | `InMemorySaver` or `SqliteSaver` |

| Method | Returns | Description |
|---|---|---|
| `run(message, thread_id="default", brand=None)` | `TurnResult` | Process one turn. Never raises: errors come back in `TurnResult.error` |
| `stream(message, thread_id, brand)` | iterator of `(node, update)` | Live progress. The last item is `("__result__", {"result": TurnResult})` |
| `state(thread_id)` | `dict` | Current checkpointed `ContentState` values |
| `history(thread_id)` | `list[(role, text)]` | User and assistant messages |
| `update_artifact(thread_id, kind, data, quality=None)` | `None` | Persist dashboard edits, plus an optional new `QualityReport` |
| `rescore(kind, data, brand=None, thread_id=...)` | `QualityReport` | Deterministic score for `blog` / `linkedin` / `report` |
| `learn_brand_voice(sample_text, base=None)` | `BrandProfile` | Infer tone, audience, values and vocabulary from sample text |
| `service_status()` | `dict` | Model, key presence (booleans only), chains, checkpointer |

### `TurnResult`
`response: str` · `route: str` · `outputs: list[str]` · `artifacts: dict[kind, dict]` ·
`quality: dict[kind, QualityReport dict]` · `warnings: list[str]` · `sources: list[Source dict]` ·
`usage: dict` (`llm`, `searches`, `images`, `cache_hits`, `total_tokens`, `estimated_cost_usd`, `elapsed_s`) · `error: str`

---

## 2. Agents — `src.agents.*`

Each factory returns a LangGraph node `node(state: ContentState) -> dict` (a partial state update).

| Factory | Node(s) | Inputs read | Outputs written |
|---|---|---|---|
| `query_handler.make_query_handler(deps)` | `query_handler` | `user_input`, `messages`, artifacts | `request`, `route`, `pending_formats`, `revision_count`, `revision_feedback` |
| `query_handler.make_post_research_router(deps)` | `post_research_router` | `request`, `pending_formats`, `research` | `next_after_research` |
| `query_handler.make_general_handler(deps)` | `general` | `messages`, `request` | `final_response`, `outputs_this_turn` |
| `research_agent.make_research_agent(deps, tools)` | `research_agent` | `request`, `sources`, `agent_scratch` | `agent_scratch` (loop) **or** `research`, `sources`, `warnings` |
| `research_agent.make_research_tools(deps)` | `research_tools` | — | tools `web_search(query)`, `deep_research(question)` → JSON |
| `blog_writer.make_blog_writer(deps, tools)` | `blog_writer` | `research`, `brand`, `request`, `blog`, `revision_feedback` | `blog`, `sources`, `quality_target` |
| `blog_writer.make_blog_tools(deps)` | `blog_tools` | — | tool `keyword_research(seed)` → JSON |
| `linkedin_writer.make_linkedin_writer(deps, tools)` | `linkedin_writer` | `research`, `brand`, `blog`, `linkedin` | `linkedin`, `sources`, `quality_target` |
| `linkedin_writer.make_linkedin_tools(deps)` | `linkedin_tools` | — | tool `trending_hashtags(topic)` → JSON |
| `image_generator.make_image_generator(deps)` | `image_generator` | `request`, `brand`, `blog`, `linkedin` | `image`, `images`, `pending_formats` |
| `content_strategist.make_content_strategist(deps)` | `content_strategist` | `research`, `brand`, `sources` | `report`, `sources`, `quality_target` |

Shared helpers (`src/agents/__init__.py`): `AgentDeps`, `TurnBudget`, `tool_loop_step`, `should_continue_tools`,
`tool_results`, `scrub_body`, `format_sources`, `advance_pending`, `brand_of`, `request_of`, `topic_of`.

### Tool JSON payloads
- `web_search` / `deep_research` → `{"query", "provider", "from_cache", "stale", "answer", "results": [{"title","url","publisher","snippet","retrieved_at","provider"}], "errors"}`
- `keyword_research` → `{"seed", "related_searches": [...], "people_also_ask": [...]}`
- `trending_hashtags` → `{"topic", "angles": [...], "hashtag_candidates": [...], "provider"}`

---

## 3. Domain models — `src.workflow.state_management`

| Model | Key fields |
|---|---|
| `Source` | `id` (citation number), `title`, `url`, `publisher`, `snippet`, `retrieved_at`, `provider`, `used_in` |
| `Provenance` | `sources`, `generated_by`, `image_model`, `grounding` (`live_research`/`cached_research`/`model_knowledge_only`), `revision_count`, `created_at`, `notes`; `.label`, `.to_markdown()` |
| `BrandProfile` | `name`, `industry`, `location`, `audience`, `tone`, `values`, `banned_words`, `required_phrases`, `cta`, `color_palette`, `visual_style`, `sample_text`, `target_reading_ease` |
| `ContentRequest` | `intent`, `topic`, `target_keyword`, `audience`, `tone`, `template`, `formats`, `needs_research`, `image_prompt`, `refine_target`, `refine_instructions`, `clarifying_question` |
| `ResearchReport` | `topic`, `queries`, `summary`, `key_findings[{claim, source_ids}]`, `statistics`, `trends`, `keywords`, `markdown`, `provenance` |
| `BlogPost` | `title`, `slug`, `meta_description`, `target_keyword`, `secondary_keywords`, `markdown`, `internal_link_suggestions`, `template`, `provenance` |
| `LinkedInPost` | `text`, `hook`, `hashtags`, `first_comment`, `template`, `provenance` |
| `ImageAsset` | `path`, `original_prompt`, `enhanced_prompt`, `provider`, `width`, `height`, `alt_text`, `metadata_path`, `attempts`, `provenance` |
| `StrategyReport` | `title`, `markdown`, `template`, `provenance` |
| `QualityReport` | `content_type`, `overall`, `passed`, `scores`, `metrics`, `suggestions`, `llm_feedback` |
| `SourceRegistry` | `.add(Source) -> Source` (stable id), `.mark_used(ids, artifact)`, `.get`, `.by_ids`, `.urls`, `.dump()` |
| `merge_sources(left, right)` | LangGraph reducer for `ContentState.sources` |

---

## 4. Integrations — `src.integrations.*`

| Class / function | Purpose |
|---|---|
| `openai_client.LLMService(config, tracker, chat_models=None, fallback_models=None)` | `.invoke(messages, tier)`, `.text(system, user, tier)`, `.structured(schema, system, user, tier)` → model or `None`, `.bind_tools(tools, tier)`, `.describe(tier)` |
| `openai_client.create_chat_model(provider, model, temperature, config)` | Provider factory (openai / anthropic / gemini). Raises `ProviderUnavailableError` |
| `openai_client.ModerationClient(api_key, enabled)` | `.check(text) -> (flagged, categories)`; fails open |
| `serp_client.SerpAPIClient(api_key)` | `.search(q, num) -> (sources, answer)`, `.keyword_ideas(q)` |
| `tavily_client.TavilyClient(api_key)` | `.search(q, num, depth="basic"/"advanced")` |
| `perplexity_client.PerplexityClient(api_key, model="sonar-pro")` | `.search(q)` → cited answer plus sources |
| `serp_client.SearchChain(providers, cache, tracker, keyword_provider)` | `.search(q, num, depth, order) -> SearchOutcome`, `.keyword_ideas(q)`, `.available_providers` |
| `image_clients.OpenAIImageClient(api_key)` | `.generate(model, prompt, size) -> bytes` |
| `image_clients.ImageChain(client, chain, sizes, tracker)` | `.generate(prompt, title, palette, brand) -> ImageResult` |
| `image_clients.validate_image(bytes, min_dimension)`, `render_placeholder(...)` | Pillow validation and the branded placeholder |
| `integrations.request_json(provider, method, url, ...)` | HTTP with retry and backoff. Raises `ProviderError` |

---

## 5. Utilities — `src.utils.*`

| Module | Main functions |
|---|---|
| `guardrails` | `check_input(text, max_chars, moderator) -> InputCheck`, `redact_pii`, `detect_injection`, `detect_policy_violation`, `enforce_citation_integrity(text, allowed_urls)`, `strip_links`, `check_output` |
| `content_optimization` | `keyword_density`, `readability`, `heading_structure`, `check_meta_description`, `fit_meta_description`, `check_title`, `slugify`, `uncited_statistics`, `citation_ids`, `has_faq_section`, `has_cta`, `suggest_internal_links`, `linkedin_metrics`, `format_linkedin`, `normalize_hashtag` |
| `quality_validation` | `QualityValidator(pass_threshold, llm, critique_prompt).score_blog / score_linkedin / score_report`, `brand_voice_check`, `learn_brand_voice` |
| `export_tools` | `artifact_markdown(kind, data)`, `to_html`, `to_wordpress_html`, `to_text`, `to_pdf`, `export_artifact(kind, data, fmt) -> (bytes, mime, filename)`, `bundle_zip(artifacts, sources)` |
| `cache` | `TTLCache(dir, ttl_seconds).get / get_stale / set / clear / make_key` |
| `observability` | `setup_logging(level, json)`, `get_logger`, `CostTracker(pricing...)` (LangChain callback) `.summary() / .over_budget(usd)` |

---

## 6. Graph — `src.workflow.langgraph_workflow`

| Function | Description |
|---|---|
| `build_graph(deps, checkpointer=None)` | Compile the full `StateGraph` |
| `make_checkpointer(config)` | `InMemorySaver` or `SqliteSaver` |
| `make_studio_graph()` | Factory for `langgraph dev` (`langgraph.json`) |
| `make_input_guardrail`, `make_quality_gate`, `make_finalize`, `render_response` | Control nodes |

Routing functions (`src.core.router`): `route_input`, `route_initial`, `route_after_research`,
`route_after_quality`, `route_next_pending`, `should_continue_tools`.
