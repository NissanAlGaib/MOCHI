"""MOCHI gateway application.

An OpenAI-compatible reverse proxy. A client changes only its ``base_url`` and
traffic flows client -> MOCHI -> target LLM -> client.

The request path, in order:

1. parse and segment by source trust (Phase 4)
2. normalize and de-obfuscate each segment (Phase 3)
3. Stage I syntactic, then Stage II semantic detection (Phases 6, 8)
4. accumulate session risk across turns (Phase 7)
5. enforce ALLOW / BLOCK / SANITIZE (Phase 10)
6. dispatch upstream
7. inspect the response for exfiltration and disclosure (Phase 11)

Every request emits one telemetry record regardless of outcome (Phase 2).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mochi import __version__
from mochi.detect import InspectionResult, inspect
from mochi.gateway.adapters import UpstreamError, get_adapter
from mochi.gateway.config import get_settings
from mochi.gateway.models import ChatCompletionRequest
from mochi.mitigate import BLOCK_STATUS, enforce, protected_text, scan_completion
from mochi.session import RiskAccumulator
from mochi.telemetry import (
    MitigationAction,
    PayloadCharacteristics,
    TelemetryRecord,
    TelemetryWriter,
    stage_timer,
)

logger = logging.getLogger("mochi.gateway")

#: Only requests under this prefix are inspected and logged; /health and /docs
#: are infrastructure endpoints and would otherwise pollute the evaluation data.
INSPECTED_PREFIX = "/v1/"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.adapter = get_adapter(settings.target_llm_provider)
    app.state.telemetry_writer = TelemetryWriter(settings.log_path)

    # Stage II holds a ~500 MB model, so exactly one instance is built here and
    # reused for the process lifetime. Loading is eager and failure is fatal:
    # a gateway that silently ran Stage I only, while its operator believed
    # Stage II was active, would produce a false sense of coverage.
    # One accumulator for the process; it holds all session history.
    app.state.accumulator = (
        RiskAccumulator(window=settings.session_window,
                        threshold=settings.session_risk_threshold)
        if settings.enable_session_risk else None
    )

    app.state.stage2 = None
    if settings.enable_stage2:
        from mochi.detect.stage2_semantic import get_detector as get_stage2

        app.state.stage2 = get_stage2(settings.stage2_model_dir or None)
        app.state.stage2.scorer.score(["warmup"])  # surface load errors now
        logger.info("Stage II enabled - model=%s",
                    settings.stage2_model_dir or "models/e5-fine-tuned")

    logger.info(
        "MOCHI %s ready - provider=%s default_model=%s log=%s payloads=%s "
        "stage1=%s stage2=%s",
        __version__,
        settings.target_llm_provider,
        settings.target_llm_model,
        settings.log_path,
        settings.log_payloads,
        settings.enable_stage1,
        settings.enable_stage2,
    )
    try:
        yield
    finally:
        await app.state.adapter.aclose()
        app.state.telemetry_writer.close()


app = FastAPI(
    title="MOCHI",
    description=(
        "Middleware for Observing, Classifying, and Handling Prompt Injections. "
        "Transparent security gateway between an LLM application and its target LLM."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    """Attach a telemetry record to the request and emit it on the way out.

    The record is created here and populated incrementally by the route
    handler and (from Phase 6) each detection stage. Writing happens in a
    ``finally`` block so a record is emitted even when the handler raises -
    a request that crashed the pipeline is exactly the kind of event that
    must not vanish from the log.
    """
    if not request.url.path.startswith(INSPECTED_PREFIX):
        return await call_next(request)

    record = TelemetryRecord()
    request.state.telemetry = record
    start = time.perf_counter()

    try:
        response = await call_next(request)
        record.response_status = response.status_code
        return response
    finally:
        record.latency.total_ms = round((time.perf_counter() - start) * 1000, 3)
        writer: TelemetryWriter | None = getattr(
            request.app.state, "telemetry_writer", None
        )
        if writer is not None:
            writer.write(record)


async def inspect_request(payload: ChatCompletionRequest,
                          record: TelemetryRecord,
                          *, app_state: Any = None) -> InspectionResult:
    """Detection seam.

    Segments the payload by source, normalizes each segment, and runs the
    detection cascade. Stages I and II reach a verdict but nothing enforces it
    yet - see Phase 10. Later phases extend
    :func:`mochi.detect.pipeline.inspect` in place:

    * Phase 7  - session risk accumulation (sets ``session_risk_contribution``)
    * Phase 9  - Stage III cognitive arbitration (sets ``stage_3_arbitration``)
    * Phase 10 - ALLOW / BLOCK / SANITIZE enforcement, using each segment's
      trust level to decide between blocking and redacting

    Enforcement will surface as a raised decision exception (BLOCK) or a
    mutated payload (SANITIZE).
    """
    settings = get_settings()
    stage2 = getattr(app_state, "stage2", None) if app_state is not None else None
    accumulator = (
        getattr(app_state, "accumulator", None) if app_state is not None else None
    )
    return inspect(
        payload,
        record,
        block_severity=settings.block_severity,
        enable_stage1=settings.enable_stage1,
        enable_stage2=settings.enable_stage2 and stage2 is not None,
        stage2=stage2,
        accumulator=accumulator,
    )


def _error(status_code: int, message: str, *,
           payload: dict[str, Any] | None = None,
           extra: dict[str, Any] | None = None) -> JSONResponse:
    """Render an error in the OpenAI error envelope clients already parse.

    ``extra`` is merged into the error object - used to return the request id on
    a block, so a user reporting a false positive can name the log entry.
    """
    if payload is not None:
        return JSONResponse(status_code=status_code, content=payload)
    error: dict[str, Any] = {"message": message, "type": "mochi_error"}
    if extra:
        error.update(extra)
    return JSONResponse(status_code=status_code, content={"error": error})


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "provider": settings.target_llm_provider,
        "default_model": settings.target_llm_model,
        "api_key_configured": bool(settings.openai_api_key),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    settings = get_settings()
    record: TelemetryRecord = request.state.telemetry
    record.target_provider = settings.target_llm_provider

    try:
        body = await request.json()
    except ValueError:
        return _error(400, "Request body must be valid JSON.")

    try:
        parsed = ChatCompletionRequest.model_validate(body)
    except Exception as exc:  # pydantic ValidationError
        return _error(422, f"Invalid chat completion request: {exc}")

    record.session_id = parsed.session_id
    record.target_model = parsed.model or settings.target_llm_model
    record.payload_characteristics = PayloadCharacteristics.from_text(
        parsed.inspectable_text(), include_content=settings.log_payloads
    )

    if parsed.stream and not settings.allow_buffered_streaming:
        # Outbound inspection (Phase 11) needs the whole body: an exfiltration
        # URL can straddle two chunks, so scanning chunk-by-chunk would miss the
        # split case. Buffered streaming is available behind
        # MOCHI_ALLOW_BUFFERED_STREAMING, but it is not *incremental* streaming
        # and the caller should choose it knowingly rather than discover it.
        return _error(
            501,
            "Streaming is not supported by default: outbound inspection needs "
            "the complete response body. Set stream=false, or enable "
            "MOCHI_ALLOW_BUFFERED_STREAMING to receive the full response as a "
            "single stream event. Incremental streaming is not implemented.",
        )

    with stage_timer(record.latency, "inspection"):
        inspection = await inspect_request(
            parsed, record, app_state=request.app.state
        )

    # --- enforcement (Phase 10) ---
    # ``enforce`` mutates ``parsed`` in place on SANITIZE, so it must run before
    # ``upstream_payload`` serializes it.
    verdict = enforce(
        parsed,
        inspection,
        sanitize_untrusted=settings.sanitize_untrusted,
        resolve_band_by_trust=settings.resolve_band_by_trust,
    )
    record.mitigation_action_applied = verdict.action
    record.mitigation_detail = verdict.reason
    record.redacted_origins = verdict.redacted_origins
    record.spans_redacted = verdict.spans_removed

    if verdict.blocks:
        logger.info("BLOCK %s - %s", record.request_id, verdict.reason)
        return _error(BLOCK_STATUS, verdict.reason,
                      extra={"request_id": record.request_id})

    # ``stream`` is dropped before dispatch even in buffered mode: MOCHI needs the
    # complete body from the provider, then re-frames it as a stream itself.
    upstream_body = parsed.upstream_payload(default_model=settings.target_llm_model)
    wants_stream = bool(upstream_body.pop("stream", False))

    try:
        with stage_timer(record.latency, "upstream"):
            completion = await request.app.state.adapter.chat_completion(upstream_body)
    except UpstreamError as exc:
        logger.warning("Upstream error: %s", exc)
        record.mitigation_action_applied = MitigationAction.NOT_APPLICABLE
        return _error(exc.status_code, str(exc), payload=exc.payload)

    # --- outbound interception (Phase 11) ---
    if settings.enable_outbound:
        with stage_timer(record.latency, "outbound"):
            outbound = scan_completion(
                completion,
                protected=protected_text(inspection.segments),
                remove_click_urls=settings.outbound_remove_click_urls,
            )
        record.outbound_action = outbound.action
        record.outbound_exfiltration_risk = outbound.exfiltration_risk
        record.outbound_urls_removed = outbound.urls_removed
        record.outbound_leaked_spans = outbound.leaked_spans
        record.outbound_findings = list(outbound.findings)
        if outbound.modified:
            logger.info("OUTBOUND redact %s - %s", record.request_id,
                        "; ".join(outbound.findings))

    if wants_stream:
        return _as_stream(completion)
    return completion


def _as_stream(completion: dict[str, Any]) -> StreamingResponse:
    """Re-frame an inspected completion as a single-event SSE stream.

    This keeps ``stream=true`` clients working without giving up outbound
    inspection. It is *not* incremental streaming - the whole response arrives at
    once - and the 501 path above says so, so no caller can mistake it for
    token-by-token delivery.
    """
    chunk = {
        "id": completion.get("id", ""),
        "object": "chat.completion.chunk",
        "created": completion.get("created", 0),
        "model": completion.get("model", ""),
        "choices": [
            {
                "index": choice.get("index", index),
                "delta": choice.get("message", {}),
                "finish_reason": choice.get("finish_reason"),
            }
            for index, choice in enumerate(completion.get("choices") or [])
        ],
    }

    def emit():
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(emit(), media_type="text/event-stream")


def main() -> None:
    """Entry point for ``python -m mochi.gateway.app``."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
