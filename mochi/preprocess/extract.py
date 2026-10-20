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

from collections.abc import Callable
from dataclasses import dataclass, field

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
    try:
        return extractor(block)
    except Exception:  # noqa: BLE001 - absent runtime, broken file, bad input
        return Extraction()


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
