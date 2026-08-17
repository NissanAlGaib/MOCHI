"""Telemetry record schema.

Implements the structure in the thesis "Attack Logging and Telemetry" section
(Figure 9), extended with the fields agreed in docs/ARCHITECTURE.md:
``request_id``, ``session_id``, ``normalization_flags``, and a per-stage
latency breakdown.

One record is emitted per inspected request **regardless of classification** -
benign traffic is logged too. That is what makes false-positive rate
computable in Phase 13; if only attacks were logged, FPR would be unknowable.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MitigationAction(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    SANITIZE = "SANITIZE"
    #: Set when the pipeline did not reach a decision (transport error, 4xx, ...).
    NOT_APPLICABLE = "N/A"


class SeverityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StageOutcome(StrEnum):
    NOT_RUN = "not_run"
    PASS = "pass"
    BLOCK = "block"
    ESCALATE = "escalate"
    SAFE = "safe"
    UNSAFE = "unsafe"
    NOT_APPLICABLE = "N/A"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def estimate_tokens(text: str) -> int:
    """Approximate token count as ``len // 4``.

    Phase 2 has no tokenizer dependency (transformers arrives in Phase 8).
    This heuristic is adequate for the token-budget checks in the thesis
    "Prompt Length and Token Budget Management" section and is replaced by the
    real tokenizer once Stage II lands. ``char_length`` is always exact, so no
    downstream analysis is forced to rely on the estimate.
    """
    return max(1, len(text) // 4) if text else 0


class PayloadCharacteristics(BaseModel):
    """Descriptive stats about the inspected text.

    ``content`` is populated only when ``MOCHI_LOG_PAYLOADS=true``. It is off
    by default so routine operation never persists raw user prompts, which is
    what the thesis Ethical Considerations section commits to under
    "Anonymization and De-identification of Data". ``content_sha256`` is always
    recorded, so identical payloads remain correlatable across runs without
    storing the text itself.
    """

    char_length: int = 0
    token_length: int = 0
    language: str | None = None  # populated in Phase 3 (normalization layer)
    content_sha256: str | None = None
    content: str | None = None

    @classmethod
    def from_text(cls, text: str, *, include_content: bool) -> PayloadCharacteristics:
        return cls(
            char_length=len(text),
            token_length=estimate_tokens(text),
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            content=text if include_content else None,
        )


class DetectionResults(BaseModel):
    stage_1_syntactic: str = StageOutcome.NOT_RUN
    stage_2_semantic: str = StageOutcome.NOT_RUN
    stage_3_arbitration: str = StageOutcome.NOT_APPLICABLE
    #: Stage II maximum window score in [0, 1] (Phase 8). ``None`` when not run.
    semantic_score: float | None = None
    #: Character offsets of the highest-scoring window in the segment text.
    #:
    #: This is the Stage II answer to "which part of the input drove the
    #: prediction". Stage I answers it with an exact regex span; Stage II
    #: answers it with a window plus :attr:`attributed_tokens`. Phase 10's
    #: SANITIZE action needs one of the two to know what to redact.
    semantic_span: tuple[int, int] | None = None
    #: Highest attention-weighted tokens from the winning window, best first.
    attributed_tokens: list[str] = Field(default_factory=list)
    #: Contribution this turn made to the session's rolling risk (Phase 7).
    session_risk_contribution: float | None = None
    #: Session cumulative risk after this turn (Phase 7).
    session_cumulative_risk: float | None = None


class LatencyBreakdown(BaseModel):
    """Per-stage timings in milliseconds.

    Feeds the Response Latency row of the thesis secondary-metrics table and
    the NFR1 targets (<2ms syntactic, <55ms semantic). Stage III is measured
    separately because it makes a network call and cannot meet those targets -
    reporting it separately is what makes the escalation-rate argument work.
    """

    stage_1_ms: float | None = None
    stage_2_ms: float | None = None
    stage_3_ms: float | None = None
    inspection_ms: float | None = None
    upstream_ms: float | None = None
    total_ms: float | None = None


class TelemetryRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    timestamp: str = Field(default_factory=utc_now_iso)
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")
    session_id: str | None = None

    #: Which tagged segment triggered detection (Phase 4). ``None`` until a
    #: detector fires or the request is untagged.
    source_origin: str | None = None
    #: e.g. direct_injection, indirect_injection, jailbreak, data_exfiltration,
    #: role_manipulation, adversarial_prompt, url_exfiltration (Phase 6).
    attack_type: str | None = None
    severity_level: str | None = None
    #: Flags raised by the normalization layer (Phase 3), e.g.
    #: ["zero_width_chars_detected", "base64_decoded"].
    normalization_flags: list[str] = Field(default_factory=list)
    #: Source tags present in this request (Phase 4), in order. Requests with no
    #: untrusted tag cannot be indirect-injection vectors, which makes this the
    #: denominator when reporting indirect ASR separately from direct ASR.
    segments_inspected: list[str] = Field(default_factory=list)
    #: Attack taxonomy for the detection, derived from the offending segment's
    #: trust level: "direct", "indirect", or "n/a" (Phase 6+).
    injection_class: str | None = None

    payload_characteristics: PayloadCharacteristics | None = None
    detection_results: DetectionResults = Field(default_factory=DetectionResults)
    mitigation_action_applied: str = MitigationAction.NOT_APPLICABLE
    #: Why that action was chosen (Phase 10). On a BLOCK this is also the message
    #: returned to the client, so it never quotes the payload.
    mitigation_detail: str | None = None
    #: Segment origins whose content was redacted, e.g. ``["context.web_content"]``.
    redacted_origins: list[str] = Field(default_factory=list)
    #: How many spans were actually removed. Zero on a SANITIZE means redaction
    #: failed and the request was escalated to BLOCK - see mochi.mitigate.
    spans_redacted: int = 0

    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)

    #: HTTP status returned to the client.
    response_status: int | None = None
    #: Target LLM and provider actually used, for multi-LLM comparison
    #: (thesis Table 19).
    target_provider: str | None = None
    target_model: str | None = None

    def to_json_line(self) -> str:
        return self.model_dump_json(exclude_none=False)
