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

from mochi.session import DEFAULT_THRESHOLD, DEFAULT_WINDOW

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
    sanitize_untrusted: bool
    resolve_band_by_trust: bool
    enable_session_risk: bool
    session_window: int
    session_risk_threshold: float
    enable_outbound: bool
    outbound_remove_click_urls: bool
    allow_buffered_streaming: bool


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
        # Redact injections found in untrusted content rather than rejecting the
        # whole request. Off means blunter behaviour: any detection blocks.
        # Kept configurable because it is an ablation arm in Chapter IV.
        sanitize_untrusted=_get_bool("MOCHI_SANITIZE_UNTRUSTED", True),
        # Resolve Stage II's uncertain band by source trust instead of escalating
        # to a Stage III LLM. Deterministic and free; see docs/BUILD_PLAN.md.
        resolve_band_by_trust=_get_bool("MOCHI_RESOLVE_BAND_BY_TRUST", True),
        # Cross-turn risk accumulation. On by default: it costs a dict lookup and
        # is the only defence against multi-step chains. Requires clients to send
        # session_id - untagged requests are simply not tracked.
        enable_session_risk=_get_bool("MOCHI_ENABLE_SESSION_RISK", True),
        session_window=_get_int("MOCHI_SESSION_WINDOW", DEFAULT_WINDOW),
        session_risk_threshold=_get_float(
            "MOCHI_SESSION_RISK_THRESHOLD", DEFAULT_THRESHOLD
        ),
        # Outbound response inspection. On by default: it is pure regex over the
        # response body and it is the only defence against markdown-image
        # exfiltration, which no inbound check can see.
        enable_outbound=_get_bool("MOCHI_ENABLE_OUTBOUND", True),
        # Also strip click-required links carrying data. Off gives the narrower,
        # higher-precision auto-fetch-only arm for the ablation.
        outbound_remove_click_urls=_get_bool("MOCHI_OUTBOUND_REMOVE_LINKS", True),
        # Serve stream=true by buffering the whole response, scanning it, then
        # emitting it as one SSE burst. Off by default because it is not
        # incremental streaming and callers should opt in knowingly - see
        # docs/BUILD_PLAN.md Phase 11.
        allow_buffered_streaming=_get_bool("MOCHI_ALLOW_BUFFERED_STREAMING", False),
    )
