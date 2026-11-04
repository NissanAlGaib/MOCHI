"""Provider adapters.

Phase 1 ships the OpenAI adapter only. Phase 12 adds Anthropic and Gemini
adapters behind the same :class:`~mochi.gateway.adapters.base.LLMAdapter`
interface, which is what lets the same MOCHI instance protect different
backends without touching detection code.
"""

from __future__ import annotations

from mochi.gateway.adapters.base import LLMAdapter, UpstreamError
from mochi.gateway.adapters.openai_adapter import OpenAIAdapter

_REGISTRY: dict[str, type[LLMAdapter]] = {
    "openai": OpenAIAdapter,
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


__all__ = ["LLMAdapter", "OpenAIAdapter", "UpstreamError", "get_adapter"]
