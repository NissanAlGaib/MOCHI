"""Wire formats: the request and response shapes of each LLM SDK.

MOCHI's detection reads one shape, the OpenAI chat-completions message list.
These modules let it protect applications written against other SDKs, in
both of its surfaces:

=====================  ===============================  ==========================
format                 SDK call (``guard()``)           gateway route
=====================  ===============================  ==========================
``openai_chat``        ``chat.completions.create``      ``/v1/chat/completions``
``openai_responses``   ``responses.create``             ``/v1/responses``
``anthropic``          ``messages.create``              ``/v1/messages``
``gemini``             ``models.generate_content``      ``/v1beta/models/{m}:generateContent``
=====================  ===============================  ==========================

Each format translates its request into chat messages for inspection and
writes enforcement's edits back in its own shape (:mod:`mochi.formats.base`),
so detection, trust and enforcement are the same code whichever SDK sent the
request. A verdict therefore does not depend on the SDK - which is the
property the tests check, by sending one attack through every format.
"""

from __future__ import annotations

from mochi.formats.anthropic import AnthropicFormat
from mochi.formats.base import ChatView, FormatError, WireFormat
from mochi.formats.gemini import GeminiFormat, normalize_contents
from mochi.formats.openai_chat import ChatFormat
from mochi.formats.openai_responses import ResponsesFormat

FORMATS: dict[str, WireFormat] = {
    f.name: f for f in (ChatFormat(), ResponsesFormat(), AnthropicFormat(), GeminiFormat())
}


def get_format(name: str) -> WireFormat:
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(f"Unknown wire format {name!r}. Known: "
                         f"{', '.join(sorted(FORMATS))}") from None


__all__ = ["FORMATS", "ChatView", "FormatError", "WireFormat", "get_format",
           "normalize_contents"]
