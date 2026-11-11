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

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mochi.preprocess import NormalizationResult, normalize, preprocess_segment

#: Score each hidden attachment channel on its own as well as inside the
#: merged attachment text. See :func:`_attachment_block`. A module switch, not
#: a setting: the evaluation flips it to measure the merged-only behaviour on
#: the same build, and no deployment has a reason to turn it off.
SCORE_HIDDEN_CHANNELS_ALONE = True

#: Words (two or more letters) a hidden channel must hold to be scored alone.
#: An instruction is a sentence; a fragment is OCR noise, and noise alone is
#: what the classifier is least calibrated on - measured, the contrast pass
#: on a benign image read "sbecauseoneof", which scored 0.61 by itself and
#: blocked an exchange-rate question. Every one of the 61 attack payloads
#: holds at least four. A shorter channel is still scanned in the merged text.
MIN_CHANNEL_WORDS = 3
_WORD = re.compile(r"[^\W\d_]{2,}")


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
                       read_images: bool = False) -> tuple[str, list[str], list[str]]:
    """Flatten message content, which may be a string or multimodal blocks.

    Returns ``(text, flags, channels)``. With ``read_images``, attachments
    are read and the recovered text joins the rest; without it they are
    dropped, and a payload rendered as pixels is invisible to every
    downstream stage. ``channels`` are the attachments' hidden and metadata
    texts, each one also to be scored on its own - see
    :func:`_attachment_block`.

    **This is the function detection actually reads.**
    ``ChatCompletionRequest.inspectable_text`` looks similar and feeds only
    telemetry, so wiring OCR there alone changes what gets logged and nothing
    about what gets caught - which is exactly the mistake that produced a
    run where every image attack sailed through while the extractor worked
    perfectly in isolation.
    """
    if isinstance(content, str):
        return content, [], []
    if not isinstance(content, list):
        return "", [], []

    parts: list[str] = []
    flags: list[str] = []
    channels: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif read_images:
            text, found, hidden = _attachment_block(block)
            if text:
                parts.append(text)
            for flag in found:
                if flag not in flags:
                    flags.append(flag)
            channels += [c for c in hidden if c not in channels]
    return "\n".join(parts), flags, channels


def _attachment_block(block: dict) -> tuple[str, list[str], list[str]]:
    """Recover text from one non-text content block, whatever modality it is.

    Returns ``(text, flags, channels)``: everything read, merged; the
    extraction flags; and the hidden and metadata texts on their own.

    **Why the channels come back separately.** The merged text is what the
    model would read, so it is scored as it is. But a short payload inside a
    page of ordinary visible text is scored as that page: measured on attack
    T1-08 hidden in an image's low bits, 0.99 alone and 0.24 with the report
    text it was hidden in. Content placed where a reader cannot see it has
    no innocent reason to be judged in the company of the content they can,
    so each hidden channel is also scored by itself - the same way hidden
    HTML already is (``preprocessor._merge``). Visible text is not split
    out: it is what the user sees, and it is already scored with their words.

    Dispatches through ``mochi.preprocess.extract``, which owns the registry,
    so adding audio or video needs no change here. Imported lazily: a
    deployment that never receives attachments should never load an OCR or
    speech runtime.

    The registry already swallows a failing extractor, returning an empty
    result rather than raising - a corrupt attachment must not take down a
    request that may be entirely legitimate. The try/except stays for the
    import itself, which can fail when the optional runtime is absent.
    """
    try:
        from mochi.preprocess.extract import extract_from_content_block

        result = extract_from_content_block(block)
        channels = [c for c in [*result.hidden_text, *result.metadata_text]
                    if c and c.strip()]
        return result.text, list(result.flags), channels
    except Exception:  # noqa: BLE001 - extraction runtime absent or broken
        return "", [], []


def segment_tagged(context: Any) -> list[Segment]:
    """Build segments from an explicit ``context`` object.

    The tag is coerced to :class:`SourceTag` so both paths yield the same type.
    It used to stay a plain ``str`` here while role inference produced the
    enum, so ``segment.source_tag.value`` worked on one path and raised on the
    other.
    """
    return [
        Segment(source_tag=SourceTag(tag), origin=f"context.{tag}", raw_text=text)
        for tag, text in context.segments()
    ]


#: Tagged text shorter than this is not subtracted from untagged messages.
#: Subtraction removes text from the principal's own turn before it is
#: inspected, so it is limited to spans long enough to be a document. A short
#: tag - "ok", "the" - would otherwise punch holes in unrelated words.
MIN_SUBTRACT_CHARS = 32


def _untagged_message_segments(messages: list[Any], tagged: list[Segment], *,
                               read_images: bool = False) -> list[Segment]:
    """Role-inferred segments for the sources ``context`` did not tag.

    Context used to *replace* role inference outright. That made tagging the
    document alone - the obvious thing a RAG application does - silently stop
    inspecting the user's own turn and the system prompt: one segment in, two
    gone. Measured on a request carrying all three, ``context=
    {"retrieved_document": doc}`` produced exactly one segment.

    So tags stay authoritative for the sources they name, and messages fill in
    the rest. A source that was tagged is not inferred again, which keeps any
    request that tags every source it sends - the evaluation's own requests
    among them - exactly as it was. Text that was tagged is removed from the
    messages it came from first, so a document pasted into the user turn is
    inspected once, as the untrusted content it is, rather than a second time
    as the user's words, which would turn every sanitize into a block.
    """
    covered = {segment.source_tag for segment in tagged}
    pieces = [segment.raw_text for segment in tagged
              if len(segment.raw_text.strip()) >= MIN_SUBTRACT_CHARS]
    extra: list[Segment] = []
    for segment in segment_messages(messages, read_images=read_images):
        if segment.source_tag in covered:
            continue
        text = segment.raw_text
        for piece in pieces:
            text = text.replace(piece, " ")
        if not text.strip():
            continue
        segment.raw_text = text
        extra.append(segment)
    return extra


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
        text, image_flags, channels = _text_from_content(
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
        segment.attachment_channels = channels
        segments.append(segment)
    return segments


def build_segments(request: Any, *, read_images: bool = False) -> list[Segment]:
    """Segment a parsed chat-completion request and preprocess each piece.

    An explicit ``context`` takes precedence over message-role inference for
    every source it tags. That is consistent with the threat model: the client
    application is the developer's own code, not the adversary, so its tagging
    is trustworthy metadata. The adversary's reach is over the *content* of
    fetched data, not over how the application labels it.

    Sources the context leaves untagged are still inferred from message roles
    - see :func:`_untagged_message_segments` for why replacing them was wrong.
    """
    context = getattr(request, "context", None)
    messages = getattr(request, "messages", []) or []
    if context is not None:
        segments = segment_tagged(context)
        segments += _untagged_message_segments(messages, segments,
                                               read_images=read_images)
    else:
        segments = segment_messages(messages, read_images=read_images)

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
        if SCORE_HIDDEN_CHANNELS_ALONE:
            _add_channels(segment, getattr(segment, "attachment_channels", ()) or ())
    return segments


def _add_channels(segment: Segment, channels: Any) -> None:
    """Make each hidden attachment channel a scannable variant of its segment.

    A variant, not a segment of its own: it inherits the segment's source
    and trust, so a payload hidden in a user's upload is still refused and
    one in a tool result still redacted. Stage I and Stage II already score
    every variant separately and keep the worst, which is the whole change.
    Each channel is normalized first, so an encoded or zero-width payload is
    decoded the same as it would be anywhere else, and the word floor
    (:data:`MIN_CHANNEL_WORDS`) is met by any one of those readings.
    """
    variants = segment.normalized.variants
    for channel in channels:
        read = normalize(channel)
        readings = (read.text, *read.variants)
        if max(len(_WORD.findall(text)) for text in readings) < MIN_CHANNEL_WORDS:
            continue
        for text in readings:
            if text and text.strip() and text not in variants:
                variants.append(text)
        for flag in read.flags:
            if flag not in segment.normalized.flags:
                segment.normalized.flags.append(flag)
