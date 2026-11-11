"""Anthropic target-LLM adapter - ``TARGET_LLM_PROVIDER=anthropic``.

Chat requests are translated to the Messages API and the reply back, by the
same :class:`~mochi.formats.anthropic.AnthropicFormat` the gateway uses to
read Anthropic SDK requests, so the two directions cannot drift apart.
Requests that arrive already in the Messages format are forwarded as they
are.
"""

from __future__ import annotations

from typing import Any

import httpx

from mochi.formats import get_format
from mochi.formats.base import FormatError
from mochi.gateway.adapters.base import LLMAdapter, UpstreamError, post_json
from mochi.gateway.config import get_settings


class AnthropicAdapter(LLMAdapter):
    name = "anthropic"
    native_formats = frozenset({"openai_chat", "anthropic"})

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.anthropic_api_key
        self._version = settings.anthropic_version
        self._client = httpx.AsyncClient(base_url=settings.anthropic_base_url,
                                         timeout=settings.request_timeout)
        self._format = get_format("anthropic")

    def _post(self, payload: dict[str, Any]):
        return post_json(self._client, "/v1/messages", payload,
                         headers={"x-api-key": self._api_key,
                                  "anthropic-version": self._version},
                         key_name="ANTHROPIC_API_KEY", api_key=self._api_key)

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            body = self._format.from_chat(payload)
        except FormatError as exc:
            raise UpstreamError(str(exc), status_code=400) from exc
        return self._format.response_to_chat(await self._post(body))

    async def native(self, fmt: str, payload: dict[str, Any], *,
                     model: str) -> dict[str, Any]:
        if fmt == "anthropic":
            return await self._post({**payload, "model": payload.get("model") or model})
        if fmt == "openai_chat":
            return await self.chat_completion(payload)
        return await super().native(fmt, payload, model=model)

    async def aclose(self) -> None:
        await self._client.aclose()
