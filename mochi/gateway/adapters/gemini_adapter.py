"""Google Gemini target-LLM adapter - ``TARGET_LLM_PROVIDER=gemini``.

Chat requests are translated to ``generateContent`` and the reply back, by the
same :class:`~mochi.formats.gemini.GeminiFormat` the gateway uses to read
Gemini SDK requests. Requests that arrive already in that format are
forwarded as they are. The model goes in the URL, not the body.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from mochi.formats import get_format
from mochi.formats.base import FormatError
from mochi.gateway.adapters.base import LLMAdapter, UpstreamError, post_json
from mochi.gateway.config import get_settings


class GeminiAdapter(LLMAdapter):
    name = "gemini"
    native_formats = frozenset({"openai_chat", "gemini"})

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.gemini_api_key
        self._default_model = settings.target_llm_model
        self._client = httpx.AsyncClient(base_url=settings.gemini_base_url,
                                         timeout=settings.request_timeout)
        self._format = get_format("gemini")

    def _post(self, model: str, payload: dict[str, Any]):
        model = (model or self._default_model).removeprefix("models/")
        return post_json(self._client,
                         f"/v1beta/models/{quote(model, safe='-._')}:generateContent",
                         payload, headers={"x-goog-api-key": self._api_key},
                         key_name="GEMINI_API_KEY", api_key=self._api_key)

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            body = self._format.from_chat(payload)
        except FormatError as exc:
            raise UpstreamError(str(exc), status_code=400) from exc
        response = await self._post(payload.get("model") or "", body)
        return self._format.response_to_chat(response)

    async def native(self, fmt: str, payload: dict[str, Any], *,
                     model: str) -> dict[str, Any]:
        if fmt == "gemini":
            return await self._post(model, payload)
        if fmt == "openai_chat":
            return await self.chat_completion(payload)
        return await super().native(fmt, payload, model=model)

    async def aclose(self) -> None:
        await self._client.aclose()
