import json
import logging

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.utils.observability import CostTracker, _JsonFormatter, get_logger, setup_logging


def make_tracker():
    return CostTracker(pricing={"gpt-4o": [2.5, 10.0]}, image_pricing={"dall-e-3": 0.08},
                       search_pricing={"perplexity": 0.005})


def test_cost_from_llm_output():
    t = make_tracker()
    t.on_llm_end(LLMResult(generations=[[]], llm_output={"model_name": "gpt-4o-2024-08-06",
                                                         "token_usage": {"prompt_tokens": 1_000_000,
                                                                         "completion_tokens": 100_000}}))
    assert round(t.total_cost, 2) == 3.5  # prefix pricing match


def test_cost_from_usage_metadata_and_extras():
    t = make_tracker()
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    response_metadata={"model_name": "mystery-model"})
    t.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]], llm_output=None))
    t.record_image("dall-e-3")
    t.record_search("perplexity")
    t.record_cache_hit()
    s = t.summary()
    assert s["unpriced_models"] == ["mystery-model"]
    assert s["total_tokens"] == 15
    assert s["cache_hits"] == 1
    assert round(s["estimated_cost_usd"], 3) == 0.085
    assert t.over_budget(0.01) and not t.over_budget(1.0) and not t.over_budget(0)


def test_reset_clears_counts():
    t = make_tracker()
    t.record_llm("gpt-4o", 10, 10)
    t.reset()
    assert t.summary()["total_tokens"] == 0


def test_logging_setup_and_json_formatter():
    logger = setup_logging("DEBUG", json_logs=True)
    assert logger.level == logging.DEBUG
    assert get_logger("x").name == "contentblitz.x"
    record = logging.LogRecord("n", logging.INFO, "f", 1, "hello %s", ("world",), None)
    record.data = {"k": 1}
    payload = json.loads(_JsonFormatter().format(record))
    assert payload["msg"] == "hello world" and payload["data"] == {"k": 1}
