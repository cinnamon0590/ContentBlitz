# Deployment Guide

## 1. Environments

| | development (default) | production |
|---|---|---|
| Select with | `APP_ENV=development` | `APP_ENV=production` |
| Logs | human-readable, DEBUG | JSON lines, INFO |
| Memory | `InMemorySaver` (lost on restart) | `SqliteSaver` at `.cache/checkpoints.sqlite` |
| Quality threshold | 70 | 75 |

Secrets always come from environment variables or `.env`, never from YAML.

## 2. Local

```bash
cd contentalchemy
source ../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add OPENAI_API_KEY + SERPAPI_API_KEY (and optionally TAVILY / PERPLEXITY)
pytest                  # sanity check, offline
streamlit run src/web_app/streamlit_app.py
```

Health check: `curl localhost:8501/_stcore/health` → `ok`.

## 3. Docker

```bash
docker compose up --build        # http://localhost:8501
docker compose logs -f app       # JSON logs (production mode)
docker compose down
```

- `.env` is passed with `env_file` (it's never baked into the image).
- `outputs/` (images and exports) and `.cache/` (research cache and SQLite memory) are mounted volumes, so they survive container restarts.
- The image uses Python 3.12-slim and installs `fonts-dejavu-core` for Unicode PDF export.
- A container health check calls Streamlit's `/_stcore/health`.

## 4. Streamlit Community Cloud

1. Push the repo to GitHub (it must **not** include `.env`).
2. New app → main file `src/web_app/streamlit_app.py`, Python 3.12.
3. In *Secrets*, add `OPENAI_API_KEY = "..."`, `SERPAPI_API_KEY = "..."`, and so on. Streamlit exposes them as environment variables.
4. Optionally set `APP_ENV = "production"`. The SQLite file is ephemeral on Community Cloud, so keep the default memory mode there.

## 5. LangGraph Studio / Platform

```bash
pip install "langgraph-cli[inmem]"
langgraph dev                    # opens Studio with the `contentblitz` graph
```

`langgraph.json` points to `src/workflow/studio.py:graph`, a graph compiled once at import. (`langgraph dev`
calls a graph *factory* on its async event loop for every request and rejects blocking calls there, and building
the graph creates the cache folder, so a factory fails with `BlockingError`.) The platform supplies
persistence. Verified with `langgraph dev`: Studio draws all 16 nodes, and research, general and guardrail runs
complete through its API. Invoke it with input `{"user_input": "Research coffee shop marketing trends"}`, and optionally
`"brand": {...}`.

## 6. Observability

- **Cost and usage**: shown per request in the UI. `TurnResult.usage` gives programmatic access.
- **LangSmith tracing** (optional): set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` in `.env`.
- **Logs**: logger `contentblitz.*` covers routing decisions, fallbacks, blocked inputs, stripped URLs, and revisions.

## 7. Operations checklist

- [ ] `.env` present on the host and **not** committed (`git check-ignore .env` → `.env`)
- [ ] `pytest` passes; `pytest -m live` passes with production keys
- [ ] Budget (`guardrails.request_budget_usd`) and pass threshold set for the deployment
- [ ] Pricing in `config/services.yaml` checked against current provider pricing pages
- [ ] `outputs/` and `.cache/` on persistent storage (Docker volumes)
- [ ] Rotate keys if they were ever pasted into logs or chats
