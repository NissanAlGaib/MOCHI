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
    log_path: str
    log_payloads: bool
    enable_stage1: bool
    enable_stage2: bool
    stage2_model_dir: str
    block_severity: str


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


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
        log_path=os.getenv("MOCHI_LOG_PATH", "logs/mochi.jsonl").strip(),
        log_payloads=_get_bool("MOCHI_LOG_PAYLOADS", False),
        enable_stage1=_get_bool("MOCHI_ENABLE_STAGE1", True),
        # Stage II defaults off: it needs torch and a trained model, and the
        # gateway must start without either. Turning it on with no model in
        # place raises ModelUnavailable at startup with instructions, rather
        # than silently degrading to Stage I only.
        enable_stage2=_get_bool("MOCHI_ENABLE_STAGE2", False),
        stage2_model_dir=os.getenv("MOCHI_STAGE2_MODEL_DIR", "").strip(),
        block_severity=os.getenv("MOCHI_BLOCK_SEVERITY", "high").strip().lower(),
    )
