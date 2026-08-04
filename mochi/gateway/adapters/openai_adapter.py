"""OpenAI (and OpenAI-compatible) target-LLM adapter.

Because the canonical internal format *is* the OpenAI shape, this adapter is a
thin authenticated forwarder. Pointing ``OPENAI_BASE_URL`` at any
OpenAI-compatible server (vLLM, Ollama, LM Studio, OpenRouter, Azure OpenAI)
makes those work too, which is useful for running evaluations without paid
API calls.
"""

from __future__ import annotations

from typing import Any

import httpx

from mochi.gateway.adapters.base import LLMAdapter, UpstreamError
from mochi.gateway.config import get_settings


class OpenAIAdapter(LLMAdapter):
    name = "openai"

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout,
        )

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise UpstreamError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key.",
                status_code=503,
            )

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"Target LLM timed out: {exc}", status_code=504
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                f"Could not reach target LLM: {exc}", status_code=502
            ) from exc

        if response.status_code >= 400:
            # Pass the provider's own error through rather than masking it -
            # a 401 from OpenAI should not look like a MOCHI bug.
            try:
                detail = response.json()
            except ValueError:
                detail = {"error": {"message": response.text}}
            raise UpstreamError(
                f"Target LLM returned {response.status_code}",
                status_code=response.status_code,
                payload=detail,
            )

        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
