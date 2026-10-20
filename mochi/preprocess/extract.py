"""One way in for every kind of attachment, whatever modality it is.

MOCHI inspects text. Anything that is not text has to become text before any
detector can see it, and until now exactly one modality did: images, wired
directly into ``mochi.detect.segments`` by name. Adding audio would have meant
a second hard-coded branch, video a third, and each one a separate chance to
wire the extraction into the wrong function - which has already happened once,
when OCR was wired into the telemetry path and every image attack sailed
through while the extractor worked perfectly in isolation.

This module is the single place a modality registers itself. The pipeline
behind it does not change: recovered text joins the scannable set and flows
through normalization, Stage I, Stage II, the trust rules and enforcement with
nothing downstream needing to know where it came from. No second detector, no
per-modality policy.

**Every extractor returns the same four things**, because each one means
something to a detector and the meaning does not vary by modality:

``visible_text``
    What a person would perceive. Words on the page, speech in the recording.

``hidden_text``
    What only a second, harder pass recovered. Pale text found by histogram
    equalization is the image case. Finding content *only* here is evidence in
    itself - content placed where a human cannot perceive it but a model can
    has no more legitimate purpose than a zero-width character in a prompt.

``metadata_text``
    Free text in the container rather than the content: EXIF and PNG chunks,
    ID3 tags on audio. Not an attack on the model, which never parses them -
    an attack on the pipeline around it, since document processors routinely
    lift captions and descriptions into the prompt.

``ran``
    Whether extraction actually happened. Distinct from "ran and found
    nothing", which is a real result. A caller that cannot tell the two apart
    will report a clean scan for a file it never opened.

Registering a new modality is one decorator and one function::

    @register("input_audio")
    def _audio(block: dict) -> Extraction:
        ...

Nothing else changes.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mochi.preprocess.flags import NormalizationFlag as F


@dataclass
class Extraction:
    """Text recovered from one non-text attachment."""

    visible_text: str = ""
    hidden_text: list[str] = field(default_factory=list)
    metadata_text: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    ran: bool = False

    @property
    def all_segments(self) -> list[str]:
        """Everything worth scanning, visible content first."""
        return [s for s in [self.visible_text, *self.hidden_text,
                            *self.metadata_text] if s and s.strip()]

    @property
    def text(self) -> str:
        return "\n".join(self.all_segments)

    def add_flag(self, flag: F | str) -> None:
        value = flag.value if isinstance(flag, F) else flag
        if value not in self.flags:
            self.flags.append(value)


#: Content-block ``type`` -> extractor. Populated by :func:`register`.
#:
#: Keyed on the block type rather than a MIME type because that is what the
#: OpenAI message format actually carries, and matching the wire format keeps
#: the dispatch honest. A modality with several spellings registers each.
EXTRACTORS: dict[str, Callable[[dict], Extraction]] = {}


def register(*block_types: str):
    """Register an extractor for one or more content-block types."""
    def decorate(fn: Callable[[dict], Extraction]):
        for block_type in block_types:
            EXTRACTORS[block_type] = fn
        return fn
    return decorate


def supported() -> frozenset[str]:
    """Block types something can currently read. Useful in diagnostics."""
    return frozenset(EXTRACTORS)


#: Extractions already computed this process, keyed by a hash of the payload.
#:
#: Extraction runs twice per attachment on a converting deployment: once when
#: detection reads the request, once when the attachment is replaced by its
#: text before dispatch. Images cost seconds, not milliseconds, so doing the
#: work twice is not a rounding error - it doubles the slowest step in the
#: request.
#:
#: Keyed on a digest rather than the payload so the table stays small, and
#: bounded so a long-running gateway cannot accumulate every image it has ever
#: seen. The cache is per process and holds no plaintext key material; the
#: payloads it summarises are already in memory for the length of the request.
_CACHE: OrderedDict[str, Extraction] = OrderedDict()
_CACHE_MAX = 32


def _cache_key(block: dict) -> str | None:
    """A stable digest of this block's payload, or ``None`` if unhashable."""
    source = block.get("image_url") or block.get("image") or block.get("input_audio") \
        or block.get("audio") or block.get("video") or {}
    raw = source.get("url") or source.get("data") if isinstance(source, dict) else source
    if not isinstance(raw, str) or not raw:
        return None
    return f"{block.get('type')}:{hashlib.sha256(raw.encode()).hexdigest()}"


def extract_from_content_block(block: dict) -> Extraction:
    """Recover text from one content block, whatever modality it is.

    Returns an empty, ``ran=False`` result for a block nothing is registered
    for, for a malformed block, and for any extractor that raises. That last
    case is deliberate: a corrupt attachment must not take down a request that
    may be entirely legitimate, and an extractor failing is not evidence the
    attachment was safe. ``ran`` is what tells those apart downstream.
    """
    if not isinstance(block, dict):
        return Extraction()
    extractor = EXTRACTORS.get(block.get("type"))
    if extractor is None:
        return Extraction()

    key = _cache_key(block)
    if key is not None and key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    try:
        found = extractor(block)
    except Exception:  # noqa: BLE001 - absent runtime, broken file, bad input
        return Extraction()

    if key is not None:
        _CACHE[key] = found
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return found


#: How an attachment's recovered text is labelled once it replaces the
#: attachment. The model is told what it is reading, because a transcript
#: dropped in with no marker reads as something the user typed - and a
#: payload inside it would then arrive wearing the user's authority.
CONVERTED_TEMPLATE = "[{kind} transcribed by MOCHI]\n{text}"

#: Block type -> the word used in that marker.
KINDS = {
    "image_url": "image", "image": "image",
    "input_audio": "audio", "audio": "audio",
    "video": "video", "video_url": "video",
}


def convert_attachments(request: Any) -> int:
    """Replace every attachment in ``request`` with the text read from it.

    Returns the number of blocks replaced. Mutates the request in place, and
    is called after inspection so the cache above makes the second extraction
    free.

    **Why a gateway would do this.** A model that accepts only text cannot be
    sent an image, so without conversion an attachment either bypasses the
    model entirely or forces the deployment to run a multimodal one. Convert,
    and any text model can serve a multimodal application while MOCHI remains
    the only component that ever parses an attachment.

    **What it costs.** The model no longer sees the original. For audio and
    video that is no loss worth naming - a transcript is what a text model
    would have received anyway. For images it is a real one: a vision encoder
    reading a chart understands more than OCR text of that chart. A
    deployment running a vision model should leave this off.

    **What it buys beyond convenience.** Nothing unexamined reaches the model.
    With the original forwarded, MOCHI inspects a rendering of the attachment
    while the model reads the attachment itself, and the two can disagree -
    which is exactly the scaling attack in ``eval/probe_scaling_attack.py``.
    Converting closes that gap by construction, because the thing inspected
    and the thing forwarded become the same bytes.
    """
    replaced = 0
    for message in getattr(request, "messages", []) or []:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        rebuilt: list[Any] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") not in EXTRACTORS:
                rebuilt.append(block)
                continue
            found = extract_from_content_block(block)
            if not found.ran:
                # Could not read it. Dropping it would silently discard
                # whatever the user attached; forwarding it would hand the
                # model bytes nothing inspected. Replace it with a marker so
                # the model answers about an attachment it cannot see rather
                # than one nobody checked.
                rebuilt.append({"type": "text", "text":
                                f"[{KINDS.get(block.get('type'), 'attachment')} "
                                f"could not be read by MOCHI]"})
                replaced += 1
                continue
            rebuilt.append({"type": "text", "text": CONVERTED_TEMPLATE.format(
                kind=KINDS.get(block.get("type"), "attachment"),
                text=found.text)})
            replaced += 1
        if replaced:
            message.content = rebuilt
    return replaced


# --- images -----------------------------------------------------------------
#
# Adapted rather than rewritten. ``image_extract`` is working, measured code
# behind 87.5% and 75.0% mitigation on the two image arms, and the value of
# this module is the extension point, not renaming a field inside it. The
# adapter is the whole migration: ``ImageExtraction`` already carries these
# four things, under one different name.


@register("image_url", "image")
def _image(block: dict) -> Extraction:
    from mochi.preprocess.image_extract import extract_from_content_block as read

    found = read(block)
    # ``rescaled_text`` folds into hidden_text here only so it reaches the
    # scanner; the distinction that matters - whether the concealed-text flag
    # is raised - was already decided inside the image extractor, and this
    # type carries no flag of its own.
    return Extraction(
        visible_text=found.visible_text,
        hidden_text=[*found.hidden_text, *found.rescaled_text],
        metadata_text=list(found.metadata_text),
        flags=list(found.flags),
        ran=found.ocr_ran,
    )
