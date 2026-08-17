"""MOCHI gateway application.

Phase 1 scope: a working OpenAI-compatible reverse proxy. A client changes
only its ``base_url`` and traffic flows client -> MOCHI -> target LLM ->
client. Detection is not wired in yet; :func:`inspect_request` is the single
seam where the Phase 3-10 pipeline attaches.

Phase 2 adds structured JSON telemetry: every inspected request emits one
record regardless of outcome.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mochi import __version__
from mochi.detect import InspectionResult, inspect
from mochi.gateway.adapters import UpstreamError, get_adapter
from mochi.gateway.config import get_settings
from mochi.gateway.models import ChatCompletionRequest
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
    return inspect(
        payload,
        record,
        block_severity=settings.block_severity,
        enable_stage1=settings.enable_stage1,
        enable_stage2=settings.enable_stage2 and stage2 is not None,
        stage2=stage2,
    )


def _error(status_code: int, message: str, *,
           payload: dict[str, Any] | None = None) -> JSONResponse:
    """Render an error in the OpenAI error envelope clients already parse."""
    if payload is not None:
        return JSONResponse(status_code=status_code, content=payload)
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "mochi_error"}},
    )


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

    if parsed.stream:
        # Streaming is deferred: Phase 11 outbound interception needs the full
        # response body to scan for leaked instructions and exfiltration URLs,
        # so a streaming path would have to be buffered anyway. Failing loudly
        # now beats silently skipping outbound checks later.
        return _error(
            501,
            "Streaming responses are not supported yet. Set stream=false. "
            "See docs/BUILD_PLAN.md Phase 11.",
        )

    with stage_timer(record.latency, "inspection"):
        await inspect_request(parsed, record, app_state=request.app.state)

    # Phase 10 will set this from the pipeline decision. Until enforcement
    # exists, everything that reaches dispatch was allowed through.
    record.mitigation_action_applied = MitigationAction.ALLOW

    upstream_body = parsed.upstream_payload(default_model=settings.target_llm_model)

    try:
        with stage_timer(record.latency, "upstream"):
            return await request.app.state.adapter.chat_completion(upstream_body)
    except UpstreamError as exc:
        logger.warning("Upstream error: %s", exc)
        record.mitigation_action_applied = MitigationAction.NOT_APPLICABLE
        return _error(exc.status_code, str(exc), payload=exc.payload)


def main() -> None:
    """Entry point for ``python -m mochi.gateway.app``."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
