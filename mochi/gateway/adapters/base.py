"""Adapter interface shared by every target-LLM provider.

The canonical wire format inside MOCHI is the OpenAI chat-completions shape.
Each adapter translates that canonical form to and from its provider's native
API, so detection logic never learns which backend is in use.
"""

from __future__ import annotations

import abc
from typing import Any


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

    async def aclose(self) -> None:
        """Release any held resources. Overridden by adapters with clients."""
