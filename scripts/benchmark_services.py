"""Benchmark the configured AI services (latency, result counts, source diversity, cost).

Produces the measured numbers for docs/service_comparison.md.

    python scripts/benchmark_services.py                 # research + LLM tiers (cheap: a few cents)
    python scripts/benchmark_services.py --images        # also image models (≈ $0.15–0.25 extra)
    python scripts/benchmark_services.py --queries 5

Writes JSON to outputs/benchmarks/<timestamp>.json and prints Markdown tables to paste into the docs.
Providers without a key are reported as "skipped".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config  # noqa: E402
from src.integrations import ProviderError  # noqa: E402
from src.integrations.image_clients import OpenAIImageClient, validate_image  # noqa: E402
from src.integrations.openai_client import LLMService  # noqa: E402
from src.integrations.perplexity_client import PerplexityClient  # noqa: E402
from src.integrations.serp_client import SerpAPIClient  # noqa: E402
from src.integrations.tavily_client import TavilyClient  # noqa: E402
from src.utils.observability import CostTracker  # noqa: E402

QUERIES = [
    "email marketing statistics for small businesses 2026",
    "sustainable packaging trends e-commerce",
    "LinkedIn engagement best practices B2B",
    "local SEO tips for restaurants",
    "AI tools for small business marketing",
]


def bench_research(config, queries: list[str]) -> list[dict]:
    s = config.secrets
    providers = [SerpAPIClient(s.serpapi_api_key), TavilyClient(s.tavily_api_key),
                 PerplexityClient(s.perplexity_api_key, config.research.get("perplexity_model", "sonar-pro"))]
    rows = []
    for provider in providers:
        if not provider.available:
            rows.append({"provider": provider.name, "status": "skipped (no key)"})
            continue
        latencies, counts, domains, errors = [], [], set(), 0
        for q in queries:
            start = time.perf_counter()
            try:
                sources, _ = provider.search(q, num=5)
            except ProviderError:
                errors += 1
                continue
            latencies.append(time.perf_counter() - start)
            counts.append(len(sources))
            domains.update(src.publisher for src in sources)
        rows.append({"provider": provider.name, "status": "ok", "queries": len(queries), "errors": errors,
                     "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
                     "avg_results": round(statistics.mean(counts), 1) if counts else 0,
                     "unique_domains": len(domains),
                     "cost_per_call_usd": config.search_pricing.get(provider.name, 0.0)})
    return rows


def bench_llm(config) -> list[dict]:
    if not config.secrets.openai_api_key and config.llm_provider == "openai":
        return [{"tier": t, "status": "skipped (no key)"} for t in ("fast", "smart")]
    rows = []
    for tier in ("fast", "smart"):
        tracker = CostTracker(config.pricing)
        llm = LLMService(config, tracker=tracker)
        start = time.perf_counter()
        try:
            llm.text("You write concise marketing copy.", "Write a 40-word LinkedIn hook about fast customer replies.",
                     tier=tier)
        except RuntimeError as exc:
            rows.append({"tier": tier, "model": llm.model_name(tier), "status": f"error: {exc}"})
            continue
        summary = tracker.summary()
        rows.append({"tier": tier, "model": llm.model_name(tier), "status": "ok",
                     "latency_s": round(time.perf_counter() - start, 2), "tokens": summary["total_tokens"],
                     "cost_usd": summary["estimated_cost_usd"]})
    return rows


def bench_images(config) -> list[dict]:
    client = OpenAIImageClient(config.secrets.openai_api_key)
    rows = []
    for model in [m for m in config.images.get("chain", []) if m != "placeholder"]:
        if not client.available:
            rows.append({"model": model, "status": "skipped (no key)"})
            continue
        start = time.perf_counter()
        try:
            _, w, h = validate_image(client.generate(model, "A cozy coffee shop counter at sunrise, no text",
                                                     config.images["sizes"].get(model, "1024x1024")))
            rows.append({"model": model, "status": "ok", "latency_s": round(time.perf_counter() - start, 1),
                         "size": f"{w}x{h}", "cost_usd": config.image_pricing.get(model)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": model, "status": f"error: {str(exc)[:80]}"})
    return rows


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_no data_"
    headers = list(dict.fromkeys(k for row in rows for k in row))
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join("" if row.get(h) is None else str(row.get(h)) for h in headers) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=3, help="number of research queries per provider (max 5)")
    parser.add_argument("--images", action="store_true", help="also benchmark image models (costs money)")
    args = parser.parse_args()

    config = load_config()
    results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "research": bench_research(config, QUERIES[: max(1, min(args.queries, 5))]),
               "llm": bench_llm(config)}
    if args.images:
        results["images"] = bench_images(config)

    out_dir = config.resolve_path(config.output_dir) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(results, indent=2))
    for section in ("research", "llm", "images"):
        if section in results:
            print(f"\n### {section.title()}\n\n{markdown_table(results[section])}")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
