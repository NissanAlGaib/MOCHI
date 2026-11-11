"""Adapter interface shared by every target-LLM provider.

The canonical wire format inside MOCHI is the OpenAI chat-completions shape.
Each adapter translates that canonical form to and from its provider's native
API, so detection logic never learns which backend is in use.

An adapter can also accept requests already in its provider's own format
(:attr:`LLMAdapter.native_formats`). The gateway uses that when the client's
SDK and the provider match - an Anthropic SDK in front of Anthropic - so the
request goes upstream as the client built it, with MOCHI's edits written in,
instead of making a round trip through the chat shape that could only lose
fields.
"""

from __future__ import annotations

import abc
from typing import Any

import httpx


class UpstreamError(RuntimeError):
    """A target LLM returned an error or was unreachable.

    Carries the upstream status code (when there was one) so the gateway can
    surface a faithful status to the client instead of flattening everything
    to a 500.
    """

    def __init__(self, message: str, *, status_code: int = 502,
                 payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class LLMAdapter(abc.ABC):
    """Translates canonical requests to a specific provider's API."""

    #: Registry key / human-readable provider name.
    name: str = "base"

    #: Wire formats (see :mod:`mochi.formats`) this provider accepts as they
    #: are. ``openai_chat`` is always accepted, through :meth:`chat_completion`.
    native_formats: frozenset[str] = frozenset({"openai_chat"})

    @abc.abstractmethod
    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a canonical chat-completion request and return the response.

        Args:
            payload: Canonical (OpenAI-shaped) request body.

        Returns:
            Canonical (OpenAI-shaped) response body.

        Raises:
            UpstreamError: The provider errored or was unreachable.
        """

    async def native(self, fmt: str, payload: dict[str, Any], *,
                     model: str) -> dict[str, Any]:
        """Send a request already in this provider's ``fmt``, unchanged."""
        raise UpstreamError(f"{self.name} does not accept {fmt} requests natively",
                            status_code=501)

    async def aclose(self) -> None:
        """Release any held resources. Overridden by adapters with clients."""


async def post_json(client: httpx.AsyncClient, url: str, payload: dict[str, Any],
                    *, headers: dict[str, str], key_name: str,
                    api_key: str) -> dict[str, Any]:
    """POST to a provider and return its JSON, as an :class:`UpstreamError` if not.

    The provider's own error body is passed through rather than masked - a
    401 from the provider should not look like a MOCHI bug.
    """
    if not api_key:
        raise UpstreamError(
            f"{key_name} is not set. Copy .env.example to .env and add a key.",
            status_code=503,
        )
    try:
        response = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise UpstreamError(f"Target LLM timed out: {exc}", status_code=504) from exc
    except httpx.HTTPError as exc:
        raise UpstreamError(f"Could not reach target LLM: {exc}", status_code=502) from exc

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = {"error": {"message": response.text}}
        raise UpstreamError(f"Target LLM returned {response.status_code}",
                            status_code=response.status_code, payload=detail)
    return response.json()
