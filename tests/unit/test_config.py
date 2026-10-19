from src.core.config import AppConfig, Secrets, _deep_merge, load_config


def test_load_development_defaults(config):
    assert config.env == "development"
    assert config.model_for("smart") == "gpt-4o"
    assert config.model_for("fast") == "gpt-4o-mini"
    assert config.research["chain"] == ["serpapi", "tavily", "perplexity"]
    assert config.images["chain"][-1] == "placeholder"
    assert config.cache.ttl_seconds == 1800


def test_production_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    cfg = load_config("production")
    assert cfg.checkpointer == "sqlite"
    assert cfg.log_json is True
    assert cfg.quality.pass_threshold == 75


def test_llm_provider_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    cfg = load_config("development")
    assert cfg.llm_provider == "anthropic"
    assert cfg.model_for("smart").startswith("claude")


def test_secrets_status_never_exposes_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    status = Secrets.from_env().status()
    assert status["openai"] is True
    assert "sk-secret" not in str(status)


def test_deep_merge_nested():
    merged = _deep_merge({"a": {"b": 1, "c": 2}, "x": 1}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}, "x": 1}


def test_resolve_path_relative_and_absolute(tmp_path):
    cfg = AppConfig()
    assert cfg.resolve_path(str(tmp_path)) == tmp_path
    assert cfg.resolve_path("outputs").name == "outputs"


def test_missing_config_dir_uses_defaults(tmp_path):
    cfg = load_config("development", config_dir=tmp_path)
    assert cfg.guardrails.max_input_chars == 4000


def test_provider_settings_for_claude(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    cfg = load_config("development")
    assert cfg.temperature_for("smart") is None and cfg.temperature_for("fast") is None  # 400 on Claude 5.5
    assert cfg.structured_output_method() == "json_schema"
    assert cfg.provider_settings()["max_tokens"] == 16000
    assert cfg.pricing["claude-sonnet-5-5"] == [2.0, 10.0] and cfg.pricing["claude-haiku-5-5"] == [0.1, 0.5]


def test_openai_keeps_sampling_and_function_calling(config):
    assert config.temperature_for("smart") == 0.7 and config.temperature_for("fast") == 0.0
    assert config.structured_output_method() == "function_calling"


def test_key_format_warnings_catch_a_key_in_the_wrong_slot(monkeypatch):
    for name in ("OPENAI_API_KEY", "TAVILY_API_KEY", "PERPLEXITY_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-proj-looks-like-openai")
    monkeypatch.setenv("GOOGLE_API_KEY", "AQ.any-google-format")  # not prefix-checked
    warnings = Secrets.from_env().format_warnings()
    assert len(warnings) == 1 and "ANTHROPIC_API_KEY" in warnings[0] and "sk-ant-" in warnings[0]
    assert "sk-proj-looks-like-openai" not in warnings[0]  # never echoes the value
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ok")
    assert Secrets.from_env().format_warnings() == []
