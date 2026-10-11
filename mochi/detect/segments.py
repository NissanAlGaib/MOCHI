"""Payload segmentation and the source trust model.

A request is not one blob of text. It is several pieces from different origins
with genuinely different trust properties, and the whole direct/indirect
injection distinction rests on telling them apart.

The trust tiers below are derived from the thesis threat model, not invented:
the attacker is assumed able to alter data the agent *fetches* (emails, web
pages, files, API results) but unable to alter the system prompt or the
execution logic. Trust therefore tracks attacker reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mochi.preprocess import NormalizationResult, preprocess_segment


class SourceTag(StrEnum):
    """Where a segment of the payload came from.

    The first five are the tags defined in the thesis "Source Identification
    and Payload Tracing" section. ``ASSISTANT_OUTPUT`` is an addition: in a
    multi-turn conversation, prior model output is replayed back into context,
    so an injection that succeeded on turn 1 can re-enter on turn 2. It needs a
    tag of its own and should be added to the thesis taxonomy.
    """

    USER_INPUT = "user_input"
    RETRIEVED_DOCUMENT = "retrieved_document"
    WEB_CONTENT = "web_content"
    API_RESPONSE = "api_response"
    SYSTEM_PROMPT = "system_prompt"
    ASSISTANT_OUTPUT = "assistant_output"


class TrustLevel(StrEnum):
    """How much authority a segment's content carries.

    TRUSTED       - defined by the developer, outside attacker reach.
    SEMI_TRUSTED  - supplied by the human principal. May carry a direct
                    injection, but the user is who the system serves, so
                    over-blocking here is a direct utility cost.
    UNTRUSTED     - fetched from somewhere the attacker may control. This is
                    the indirect injection channel and should never be treated
                    as carrying instructions.
    """

    TRUSTED = "trusted"
    SEMI_TRUSTED = "semi_trusted"
    UNTRUSTED = "untrusted"


class InjectionClass(StrEnum):
    """Attack taxonomy a detection in this segment would fall under."""

    DIRECT = "direct"
    INDIRECT = "indirect"
    NOT_APPLICABLE = "n/a"


TRUST_LEVELS: dict[str, TrustLevel] = {
    SourceTag.SYSTEM_PROMPT: TrustLevel.TRUSTED,
    SourceTag.USER_INPUT: TrustLevel.SEMI_TRUSTED,
    SourceTag.RETRIEVED_DOCUMENT: TrustLevel.UNTRUSTED,
    SourceTag.WEB_CONTENT: TrustLevel.UNTRUSTED,
    SourceTag.API_RESPONSE: TrustLevel.UNTRUSTED,
    SourceTag.ASSISTANT_OUTPUT: TrustLevel.UNTRUSTED,
}

#: A detection in a semi-trusted segment is by definition a direct injection -
#: the principal supplied it. A detection in an untrusted segment is indirect.
#: Trusted segments are out of the attacker's reach per the threat model, so a
#: hit there indicates misconfiguration rather than an attack.
INJECTION_CLASSES: dict[TrustLevel, InjectionClass] = {
    TrustLevel.TRUSTED: InjectionClass.NOT_APPLICABLE,
    TrustLevel.SEMI_TRUSTED: InjectionClass.DIRECT,
    TrustLevel.UNTRUSTED: InjectionClass.INDIRECT,
}

#: Chat roles mapped to source tags for untagged requests. ``tool``/``function``
#: results map to API_RESPONSE deliberately: in agentic tool-calling, the tool
#: result message *is* the primary indirect injection vector.
ROLE_TO_SOURCE: dict[str, SourceTag] = {
    "system": SourceTag.SYSTEM_PROMPT,
    "developer": SourceTag.SYSTEM_PROMPT,
    "user": SourceTag.USER_INPUT,
    "assistant": SourceTag.ASSISTANT_OUTPUT,
    "tool": SourceTag.API_RESPONSE,
    "function": SourceTag.API_RESPONSE,
}


@dataclass
class Segment:
    """One attributed piece of an inspected payload."""

    source_tag: str
    origin: str
    """Where in the request this came from, e.g. ``context.web_content`` or
    ``messages[2]``. Logged so an operator can point at the exact JSON node."""
    raw_text: str
    normalized: NormalizationResult = field(default_factory=lambda: NormalizationResult(text=""))

    @property
    def trust(self) -> TrustLevel:
        return TRUST_LEVELS.get(self.source_tag, TrustLevel.UNTRUSTED)

    @property
    def injection_class(self) -> InjectionClass:
        return INJECTION_CLASSES[self.trust]

    @property
    def is_untrusted(self) -> bool:
        return self.trust is TrustLevel.UNTRUSTED

    @property
    def scannable(self) -> list[str]:
        """Every text variant a detector must examine for this segment."""
        return self.normalized.scannable


def _text_from_content(content: Any, *,
                       read_images: bool = False) -> tuple[str, list[str]]:
    """Flatten message content, which may be a string or multimodal blocks.

    Returns ``(text, flags)``. With ``read_images``, image blocks are OCR'd
    and the recovered text joins the rest; without it they are dropped, and a
    payload rendered as pixels is invisible to every downstream stage.

    **This is the function detection actually reads.**
    ``ChatCompletionRequest.inspectable_text`` looks similar and feeds only
    telemetry, so wiring OCR there alone changes what gets logged and nothing
    about what gets caught - which is exactly the mistake that produced a
    run where every image attack sailed through while the extractor worked
    perfectly in isolation.
    """
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []

    parts: list[str] = []
    flags: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif read_images:
            text, found = _image_block(block)
            if text:
                parts.append(text)
            for flag in found:
                if flag not in flags:
                    flags.append(flag)
    return "\n".join(parts), flags


def _image_block(block: dict) -> tuple[str, list[str]]:
    """OCR one image content block. Imported lazily; failures yield nothing.

    A malformed image must not take down a request that may be legitimate,
    and a deployment that never enables OCR should never load the runtime.
    """
    try:
        from mochi.preprocess.image_extract import extract_from_content_block

        result = extract_from_content_block(block)
        return result.text, list(result.flags)
    except Exception:  # noqa: BLE001 - OCR absent, broken, or fed a bad image
        return "", []


def segment_tagged(context: Any) -> list[Segment]:
    """Build segments from an explicit ``context`` object."""
    return [
        Segment(source_tag=tag, origin=f"context.{tag}", raw_text=text)
        for tag, text in context.segments()
    ]


def segment_messages(messages: list[Any], *,
                     read_images: bool = False) -> list[Segment]:
    """Derive segments from chat message roles when no ``context`` is supplied.

    This is the zero-integration fallback. It is coarser than explicit tagging -
    a RAG application that pastes retrieved documents into a user message will
    have them attributed to ``user_input`` - but it still separates system
    prompt, user turns, and tool results without the client changing anything.
    """
    segments: list[Segment] = []
    for index, message in enumerate(messages):
        role = (getattr(message, "role", "") or "").lower()
        text, image_flags = _text_from_content(
            getattr(message, "content", None), read_images=read_images
        )
        if not text.strip():
            continue
        segment = Segment(
            source_tag=ROLE_TO_SOURCE.get(role, SourceTag.USER_INPUT),
            origin=f"messages[{index}]",
            raw_text=text,
        )
        segment.image_flags = image_flags
        segments.append(segment)
    return segments


def build_segments(request: Any, *, read_images: bool = False) -> list[Segment]:
    """Segment a parsed chat-completion request and preprocess each piece.

    An explicit ``context`` takes precedence over message-role inference. That
    is consistent with the threat model: the client application is the
    developer's own code, not the adversary, so its tagging is trustworthy
    metadata. The adversary's reach is over the *content* of fetched data,
    not over how the application labels it.
    """
    context = getattr(request, "context", None)
    if context is not None:
        segments = segment_tagged(context)
    else:
        segments = segment_messages(getattr(request, "messages", []) or [],
                                    read_images=read_images)

    for segment in segments:
        segment.normalized = preprocess_segment(
            segment.raw_text, source_tag=segment.source_tag
        )
        # OCR flags are raised before normalization runs, so merge them in
        # rather than let them be overwritten: an operator needs to see that
        # text came out of an image, and that some of it was concealed.
        for flag in getattr(segment, "image_flags", ()) or ():
            if flag not in segment.normalized.flags:
                segment.normalized.flags.append(flag)
    return segments
