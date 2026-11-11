"""Provider adapters - the target LLM behind the gateway.

Selected by ``TARGET_LLM_PROVIDER``. Every adapter accepts the canonical
chat-completions request, and its provider's own format as it is, behind the
same :class:`~mochi.gateway.adapters.base.LLMAdapter` interface. That is what
lets the same MOCHI instance protect different backends without touching
detection code: swapping the provider is a change to ``.env``.
"""

from __future__ import annotations

from mochi.gateway.adapters.anthropic_adapter import AnthropicAdapter
from mochi.gateway.adapters.base import LLMAdapter, UpstreamError
from mochi.gateway.adapters.gemini_adapter import GeminiAdapter
from mochi.gateway.adapters.openai_adapter import OpenAIAdapter

_REGISTRY: dict[str, type[LLMAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


def get_adapter(provider: str) -> LLMAdapter:
    """Instantiate the adapter registered for ``provider``."""
    try:
        adapter_cls = _REGISTRY[provider]
    except KeyError:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unsupported TARGET_LLM_PROVIDER {provider!r}. Supported: {supported}"
        ) from None
    return adapter_cls()


__all__ = ["AnthropicAdapter", "GeminiAdapter", "LLMAdapter", "OpenAIAdapter",
           "UpstreamError", "get_adapter"]
