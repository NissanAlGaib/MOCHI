"""Pull text out of images, so an injection in a screenshot is still inspected.

MOCHI reads the text of a request. Until now it discarded image content blocks
outright - ``ChatCompletionRequest.payload_text`` extracted the text parts of a
multimodal message and ignored the rest. A payload rendered as pixels passed
every stage untouched, because every stage was looking at a string that no
longer contained it.

This module closes that by treating an image the way ``html_extract`` treats
markup and ``file_extract`` treats a PDF: as a container to get text out of.
The recovered text then flows through the existing pipeline - normalization,
Stage I, Stage II, trust rules, enforcement - with nothing downstream needing
to know where it came from. No second detector, no separate policy.

Why two OCR passes
------------------
The attack worth catching is the one a human reviewer cannot see: pale text on
a pale ground, legible to a model's vision encoder and invisible at a glance.
Measured on a probe image carrying black text plus a line in ``#FCFCFC`` on
white:

    as-is                 2 lines - hidden line MISSED
    autocontrast          2 lines - hidden line MISSED
    histogram equalize    3 lines - hidden line RECOVERED

``autocontrast`` fails because the image already spans the full range: the
black text pins one end, so there is nothing to stretch. Equalization
redistributes the histogram instead and separates near-white from white.

So every image is read twice, original and equalized, and the union is
returned. This mirrors how the normalization layer handles encoded payloads -
decoded variants are *added* to the scannable set rather than replacing the
original, because either form may be the one a detector matches.

Finding text only in the equalized pass is itself evidence. Low-contrast text
inside an image has no more legitimate purpose than a zero-width character
inside a prompt, and it is flagged the same way.

OCR is optional
---------------
``rapidocr_onnxruntime`` is imported lazily and the whole module degrades to
"no text found" without it, so a deployment that never receives images pays
nothing. It was chosen over pytesseract (needs a system binary) and easyocr
(pulls a second deep-learning runtime) because ``onnxruntime`` is already a
dependency.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
from dataclasses import dataclass, field

from mochi.preprocess.flags import NormalizationFlag as F

#: Below this, a line is more likely OCR noise than text. RapidOCR returns
#: ~0.98 on clean rendered text, so this rejects very little that is real.
MIN_CONFIDENCE = 0.5

#: Images larger than this are rejected rather than resized. A caller sending
#: 50 MB of pixels to a text gateway is not doing OCR, and decoding it first to
#: find that out is the vulnerability.
MAX_IMAGE_BYTES = 12 * 1024 * 1024

#: Two boxes overlapping by at least this much are the same line read twice.
#: Compared by position rather than text: the passes disagree on characters
#: (one line came back as "Quarterly re port page 3" and "Quatle1ly 1epott
#: p0ge 3") but agree on geometry to within a pixel.
SAME_REGION_IOU = 0.50

#: ``data:image/png;base64,...`` - how a multimodal message carries an inline
#: image in the OpenAI format.
DATA_URI = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)


class OCRUnavailable(RuntimeError):
    """Raised only when OCR is explicitly required and cannot be loaded."""


@dataclass
class ImageExtraction:
    """Text recovered from one image."""

    visible_text: str = ""
    """Lines the plain pass read - what a human looking at the image sees."""

    hidden_text: list[str] = field(default_factory=list)
    """Lines only the equalized pass read. Low-contrast content, which is the
    case a reviewer cannot catch by looking."""

    flags: list[str] = field(default_factory=list)
    ocr_ran: bool = False
    """False when OCR was unavailable or the input was not a usable image -
    distinct from 'ran and found nothing', which is a real result."""

    @property
    def all_segments(self) -> list[str]:
        """Everything worth scanning, visible content first."""
        return [s for s in [self.visible_text, *self.hidden_text] if s and s.strip()]

    @property
    def text(self) -> str:
        return "\n".join(self.all_segments)

    def add_flag(self, flag: F) -> None:
        if flag.value not in self.flags:
            self.flags.append(flag.value)


def _reader():
    """The OCR engine, built once. ``None`` when unavailable."""
    global _READER
    if _READER is _UNSET:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _READER = RapidOCR()
        except Exception:  # noqa: BLE001 - absent, broken, or missing models
            _READER = None
    return _READER


_UNSET = object()
_READER = _UNSET


def decode_data_uri(value: str) -> bytes | None:
    """Bytes from a ``data:image/...;base64,...`` URI, or ``None``.

    A plain ``https://`` image URL returns ``None``: fetching it would make the
    gateway issue outbound requests to attacker-chosen hosts on every
    inspection, which is a worse hole than the one being closed.
    """
    match = DATA_URI.match(value.strip())
    if not match:
        return None
    try:
        return base64.b64decode(match.group(1), validate=False)
    except (binascii.Error, ValueError):
        return None


def _ocr_lines(reader, image) -> list[tuple[tuple[float, float, float, float], str]]:
    """``(bounding box, text)`` for each line read, confident ones only."""
    import numpy as np

    try:
        result, _ = reader(np.array(image))
    except Exception:  # noqa: BLE001 - a malformed image must not take the request down
        return []
    lines = []
    for box, text, confidence in (result or []):
        if not text or not text.strip() or float(confidence) < MIN_CONFIDENCE:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        lines.append(((min(xs), min(ys), max(xs), max(ys)), text.strip()))
    return lines


def _overlap(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _is_new(box: tuple[float, float, float, float], plain: list) -> bool:
    """Whether the equalized pass found text where the plain pass found none.

    Compared by **position**, not by text. The two passes rarely agree
    character for character - the same line came back as
    ``'Quarterly re port page 3'`` and ``'Quatle1ly 1epott p0ge 3'`` from
    boxes differing by one pixel. String similarity scored that pair at 0.55
    and reported a concealed payload on an image that had none, which would
    have raised the hidden-text flag on essentially every image.

    Position does not have that problem: a genuinely concealed line occupies
    space nothing was read from before, and a re-read of a visible line sits on
    top of it.
    """
    return not any(_overlap(box, other) >= SAME_REGION_IOU for other, _ in plain)


def extract_image(data: bytes) -> ImageExtraction:
    """Read an image twice - as-is and equalized - and return both findings."""
    out = ImageExtraction()
    if not data or len(data) > MAX_IMAGE_BYTES:
        return out

    reader = _reader()
    if reader is None:
        return out

    try:
        from PIL import Image, ImageOps

        image = Image.open(io.BytesIO(data))
        image.load()
        image = image.convert("RGB")
    except Exception:  # noqa: BLE001 - not an image, or a corrupt one
        return out

    out.ocr_ran = True
    plain = _ocr_lines(reader, image)   # [(box, text), ...]

    # Equalization redistributes the histogram, separating near-white from
    # white. autocontrast cannot: black text already pins the range.
    try:
        equalized = ImageOps.equalize(image.convert("L")).convert("RGB")
    except Exception:  # noqa: BLE001
        equalized = None
    boosted = _ocr_lines(reader, equalized) if equalized is not None else []

    hidden = [text for box, text in boosted if _is_new(box, plain)]

    out.visible_text = "\n".join(text for _box, text in plain)
    out.hidden_text = hidden
    if plain or hidden:
        out.add_flag(F.IMAGE_TEXT_EXTRACTED)
    if hidden:
        # Same reasoning as zero-width characters: text placed where a human
        # cannot read it but a model can is not an accident.
        out.add_flag(F.IMAGE_HIDDEN_TEXT_DETECTED)
    return out


def extract_from_content_block(block: dict) -> ImageExtraction:
    """Extract from one OpenAI-format multimodal content block.

    Accepts the ``{"type": "image_url", "image_url": {"url": "data:..."}}``
    shape and the bare-string variant some clients send. Anything else, or a
    remote URL, returns an empty result with ``ocr_ran`` False.
    """
    if not isinstance(block, dict) or block.get("type") not in {"image_url", "image"}:
        return ImageExtraction()
    source = block.get("image_url") or block.get("image") or {}
    url = source.get("url") if isinstance(source, dict) else source
    if not isinstance(url, str):
        return ImageExtraction()
    data = decode_data_uri(url)
    return extract_image(data) if data else ImageExtraction()
