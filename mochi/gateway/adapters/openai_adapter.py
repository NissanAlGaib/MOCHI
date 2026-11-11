"""OpenAI (and OpenAI-compatible) target-LLM adapter.

Because the canonical internal format *is* the OpenAI shape, this adapter is a
thin authenticated forwarder. Pointing ``OPENAI_BASE_URL`` at any
OpenAI-compatible server (vLLM, Ollama, LM Studio, OpenRouter, Azure OpenAI)
makes those work too, which is useful for running evaluations without paid
API calls.

It also forwards Responses API requests as they are. Whether the server
behind ``OPENAI_BASE_URL`` implements ``/responses`` is up to that server;
one that does not answers 404, and the error is passed back unchanged.
"""

from __future__ import annotations

from typing import Any

import httpx

from mochi.gateway.adapters.base import LLMAdapter, post_json
from mochi.gateway.config import get_settings


class OpenAIAdapter(LLMAdapter):
    name = "openai"
    native_formats = frozenset({"openai_chat", "openai_responses"})

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout,
        )

    def _post(self, path: str, payload: dict[str, Any]):
        return post_json(self._client, path, payload,
                         headers={"Authorization": f"Bearer {self._api_key}"},
                         key_name="OPENAI_API_KEY", api_key=self._api_key)

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/chat/completions", payload)

    async def native(self, fmt: str, payload: dict[str, Any], *,
                     model: str) -> dict[str, Any]:
        if fmt == "openai_chat":
            return await self.chat_completion(payload)
        if fmt == "openai_responses":
            return await self._post("/responses", payload)
        return await super().native(fmt, payload, model=model)

    async def aclose(self) -> None:
        await self._client.aclose()
