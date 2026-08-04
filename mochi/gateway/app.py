"""MOCHI gateway application.

Phase 1 scope: a working OpenAI-compatible reverse proxy. A client changes
only its ``base_url`` and traffic flows client -> MOCHI -> target LLM ->
client. Detection is not wired in yet; :func:`inspect_request` is the single
seam where the Phase 3-10 pipeline attaches.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mochi import __version__
from mochi.gateway.adapters import UpstreamError, get_adapter
from mochi.gateway.config import get_settings
from mochi.gateway.models import ChatCompletionRequest

logger = logging.getLogger("mochi.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.adapter = get_adapter(settings.target_llm_provider)
    logger.info(
        "MOCHI %s ready - provider=%s default_model=%s",
        __version__,
        settings.target_llm_provider,
        settings.target_llm_model,
    )
    try:
        yield
    finally:
        await app.state.adapter.aclose()


app = FastAPI(
    title="MOCHI",
    description=(
        "Middleware for Observing, Classifying, and Handling Prompt Injections. "
        "Transparent security gateway between an LLM application and its target LLM."
    ),
    version=__version__,
    lifespan=lifespan,
)


async def inspect_request(payload: ChatCompletionRequest) -> None:
    """Detection seam.

    Phase 1 is intentionally a no-op so the transport path can be validated in
    isolation. Later phases attach here in order:

    * Phase 3  - normalization / de-obfuscation of every segment
    * Phase 4  - per-source-tag segmentation (``payload.context``)
    * Phase 6  - Stage I syntactic filtering
    * Phase 7  - session risk accumulation (``payload.session_id``)
    * Phase 8  - Stage II semantic detection
    * Phase 9  - Stage III cognitive arbitration
    * Phase 10 - ALLOW / BLOCK / SANITIZE enforcement

    Enforcement will surface as a raised decision exception (BLOCK) or a
    mutated payload (SANITIZE).
    """
    return None


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
    try:
        body = await request.json()
    except ValueError:
        return _error(400, "Request body must be valid JSON.")

    try:
        parsed = ChatCompletionRequest.model_validate(body)
    except Exception as exc:  # pydantic ValidationError
        return _error(422, f"Invalid chat completion request: {exc}")

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

    await inspect_request(parsed)

    settings = get_settings()
    upstream_body = parsed.upstream_payload(default_model=settings.target_llm_model)

    try:
        return await request.app.state.adapter.chat_completion(upstream_body)
    except UpstreamError as exc:
        logger.warning("Upstream error: %s", exc)
        return _error(exc.status_code, str(exc), payload=exc.payload)


def main() -> None:
    """Entry point for ``python -m mochi.gateway.app``."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
