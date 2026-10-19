import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "benchmark_services", Path(__file__).resolve().parents[2] / "scripts" / "benchmark_services.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_markdown_table():
    table = bench.markdown_table([{"a": 1, "b": None}, {"a": 2, "c": "x"}])
    assert table.splitlines()[0] == "| a | b | c |" and "| 2 |  | x |" in table
    assert bench.markdown_table([]) == "_no data_"


def test_benchmarks_skip_without_keys(config):
    assert all(r["status"].startswith("skipped") for r in bench.bench_research(config, ["q"]))
    assert all(r["status"].startswith("skipped") for r in bench.bench_llm(config))
    assert all(r["status"].startswith("skipped") for r in bench.bench_images(config))
