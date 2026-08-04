"""Environment-backed configuration.

Settings are read once from the process environment (populated from .env by
python-dotenv) and cached. Keeping every tunable here is what makes the
"swap the backend with a config change" claim in docs/ARCHITECTURE.md true
rather than aspirational.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    target_llm_provider: str
    target_llm_model: str
    openai_api_key: str
    openai_base_url: str
    host: str
    port: int
    request_timeout: float


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        target_llm_provider=os.getenv("TARGET_LLM_PROVIDER", "openai").strip().lower(),
        target_llm_model=os.getenv("TARGET_LLM_MODEL", "gpt-4o-mini").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).strip().rstrip("/"),
        host=os.getenv("MOCHI_HOST", "127.0.0.1").strip(),
        port=_get_int("MOCHI_PORT", 8000),
        request_timeout=_get_float("MOCHI_REQUEST_TIMEOUT", 60.0),
    )
