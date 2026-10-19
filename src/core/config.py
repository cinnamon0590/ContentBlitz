"""Configuration loading for ContentBlitz.

Configuration comes from three layers, merged in this order (later wins):

1. ``config/services.yaml``      – provider catalogue, fallback chains, pricing
2. ``config/<APP_ENV>.yaml``     – environment settings (development / production)
3. Environment variables / ``.env`` – secrets and a few overrides (``LLM_PROVIDER``)

Secrets are only ever read from the environment, never from YAML, so the YAML
files are safe to commit.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Secrets(BaseModel):
    """API keys. Missing keys simply disable the matching provider."""

    openai_api_key: str | None = None
    serpapi_api_key: str | None = None
    tavily_api_key: str | None = None
    perplexity_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Secrets:
        def get(name: str) -> str | None:
            value = os.getenv(name, "").strip()
            return value or None

        return cls(
            openai_api_key=get("OPENAI_API_KEY"),
            serpapi_api_key=get("SERPAPI_API_KEY"),
            tavily_api_key=get("TAVILY_API_KEY"),
            perplexity_api_key=get("PERPLEXITY_API_KEY"),
            anthropic_api_key=get("ANTHROPIC_API_KEY"),
            google_api_key=get("GOOGLE_API_KEY"),
        )

    def status(self) -> dict[str, bool]:
        """Which keys are present (never exposes the values)."""
        return {name.removesuffix("_api_key"): bool(value) for name, value in self.model_dump().items()}


class CacheConfig(BaseModel):
    enabled: bool = True
    dir: str = ".cache/research"
    ttl_seconds: int = 1800


class GuardrailConfig(BaseModel):
    max_input_chars: int = 4000
    use_moderation_api: bool = True
    max_tool_iterations: int = 6
    max_deep_research_calls: int = 1
    max_revisions: int = 2
    request_budget_usd: float = 0.50
    recursion_limit: int = 40


class QualityConfig(BaseModel):
    pass_threshold: int = 70


class AppConfig(BaseModel):
    env: str = "development"
    log_level: str = "INFO"
    log_json: bool = False
    output_dir: str = "outputs"
    cache: CacheConfig = Field(default_factory=CacheConfig)
    checkpointer: str = "memory"
    sqlite_path: str = ".cache/checkpoints.sqlite"
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    llm: dict[str, Any] = Field(default_factory=dict)
    research: dict[str, Any] = Field(default_factory=dict)
    images: dict[str, Any] = Field(default_factory=dict)
    pricing: dict[str, list[float]] = Field(default_factory=dict)
    image_pricing: dict[str, float] = Field(default_factory=dict)
    search_pricing: dict[str, float] = Field(default_factory=dict)
    secrets: Secrets = Field(default_factory=Secrets)

    # ---- convenience accessors -------------------------------------------------
    @property
    def llm_provider(self) -> str:
        return self.llm.get("provider", "openai")

    def model_for(self, tier: str) -> str:
        """Model name for a tier ('smart' or 'fast') under the active provider."""
        return self.llm["models"][self.llm_provider][tier]

    def model_fallbacks(self) -> list[str]:
        return list(self.llm["models"][self.llm_provider].get("fallbacks", []))

    def temperature_for(self, tier: str) -> float:
        return float(self.llm.get("temperature", {}).get(tier, 0.0))

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(env: str | None = None, config_dir: Path | None = None) -> AppConfig:
    """Build an :class:`AppConfig` from YAML files plus environment variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config_dir = config_dir or CONFIG_DIR
    env = env or os.getenv("APP_ENV", "development")

    data = _deep_merge(_read_yaml(config_dir / "services.yaml"), _read_yaml(config_dir / f"{env}.yaml"))
    data.setdefault("env", env)

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        data.setdefault("llm", {})["provider"] = provider

    config = AppConfig(**data)
    config.secrets = Secrets.from_env()
    return config


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached configuration."""
    return load_config()
