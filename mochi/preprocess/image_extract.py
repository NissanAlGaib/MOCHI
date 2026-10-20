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

#: Resolutions to re-read the image at, because that is what the model reads.
#:
#: A vision encoder never sees the file that was sent. It resizes every input
#: to a fixed resolution first, and an attacker who knows that resolution can
#: write the payload into exactly the pixels the resampler keeps - leaving a
#: file that reads as a quarterly report at full size and as an injection once
#: resized. Measured with ``eval/probe_scaling_attack.py``: the file as sent
#: scores 0.0247 and is allowed; the same file at 336x336 scores 0.9958.
#:
#: The detector was never the problem there. It was being shown the wrong
#: pixels. These are the common vision-tower input sizes; 336 is LLaVA's,
#: which is what this project's image evaluation runs against.
#:
#: Each entry costs another OCR pass, so this is a short list rather than a
#: sweep. An image already smaller than an entry is skipped for it - upscaling
#: recovers nothing that was not already legible.
SCALING_PASSES: tuple[int, ...] = (336, 224)


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

    rescaled_text: list[str] = field(default_factory=list)
    """Lines read only after resizing to a vision encoder's input resolution.

    Scanned like everything else, but deliberately **not** counted as hidden
    text. Finding a line here is not evidence of concealment: downsampled OCR
    garbles ordinary text, and an invoice reading ``Acme Bank`` at full size
    comes back as ``Acne Bark`` at 336x336. Treating that as a concealed
    payload would raise the hidden-text flag on routine documents - the same
    false-positive trap that made the equalized pass compare bounding boxes
    rather than strings.

    A genuine scaling attack still gets caught, because the recovered text
    reaches the detector either way. Only the flag is withheld."""

    metadata_text: list[str] = field(default_factory=list)
    """Text from PNG text chunks and EXIF, as ``"field: value"``.

    Not an attack on the model: a vision encoder reads pixels and never parses
    metadata. Verified - llava:7b described an image whose Comment field held
    an injection without mentioning it.

    It is an attack on the *pipeline*. Document-processing applications
    routinely lift captions, descriptions and EXIF comments into the prompt,
    and at that point the payload is text the model does obey. ``file_extract``
    already raises FILE_METADATA_EXTRACTED for the same reason; images were
    simply not covered.

    Trailing bytes after the PNG terminator are deliberately not collected.
    Nothing in the chain surfaces them, and scanning arbitrary trailing data
    for text would invent findings rather than report them."""

    flags: list[str] = field(default_factory=list)
    ocr_ran: bool = False
    """False when OCR was unavailable or the input was not a usable image -
    distinct from 'ran and found nothing', which is a real result."""

    @property
    def all_segments(self) -> list[str]:
        """Everything worth scanning, visible content first."""
        return [s for s in [self.visible_text, *self.hidden_text,
                            *self.rescaled_text, *self.metadata_text]
                if s and s.strip()]

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


def _squash(text: str) -> str:
    """Normalise a line for comparison across resolutions.

    Downsampled OCR loses spacing and confuses glyph pairs - the same line
    came back as ``'Ignore all previous'`` at full size and
    ``'lgnoreallprevious'`` at 336x336. Dropping whitespace and case, and
    folding the handful of pairs that differ only by stroke, is enough to
    recognise one as the other without claiming they are identical.
    """
    folded = text.lower()
    for a, b in (("l", "i"), ("0", "o"), ("1", "i"), ("5", "s"), ("8", "b")):
        folded = folded.replace(a, b)
    return "".join(ch for ch in folded if ch.isalnum())


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


#: Metadata fields worth reading: free text an author controls. Dimensions,
#: DPI and colour profiles carry no instruction and are skipped.
TEXT_METADATA_KEYS = frozenset({
    "comment", "description", "title", "author", "artist", "copyright",
    "software", "usercomment", "imagedescription", "xmlcom", "parameters",
})

#: A single field longer than this is a payload, not a caption. Truncated
#: rather than dropped - the opening is what a detector needs.
MAX_METADATA_CHARS = 4000


def _metadata_text(image) -> list[str]:
    """Free-text metadata fields, as ``"field: value"``.

    Reads PNG text chunks (``image.info``) and EXIF. Binary and numeric
    fields are skipped: they carry no instruction and would only give the
    detector noise to score.
    """
    found: list[str] = []

    def take(key, value) -> None:
        if str(key).lower() not in TEXT_METADATA_KEYS:
            return
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if not isinstance(value, str):
            return
        value = value.strip()
        if value:
            found.append(f"{key}: {value[:MAX_METADATA_CHARS]}")

    for key, value in (getattr(image, "info", None) or {}).items():
        take(key, value)

    try:
        from PIL import ExifTags

        for tag, value in (image.getexif() or {}).items():
            take(ExifTags.TAGS.get(tag, tag), value)
    except Exception:  # noqa: BLE001 - no EXIF, or an unreadable block
        pass
    return found


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

    # Third kind of pass: read the image the way the model will receive it.
    #
    # Unlike the equalized pass, these cannot be compared by bounding box -
    # the coordinates are in a different space, and scaling them back lands
    # within a pixel or two of nothing in particular. Compared by text
    # instead, normalised, which is the weaker test the box comparison was
    # introduced to avoid. It is acceptable here only because the failure
    # mode is different: a false "new" line at this stage adds a line of
    # already-legible text to the scannable set, where a false positive in
    # the equalized pass would have raised the concealed-text flag on
    # essentially every image.
    seen = {_squash(text) for _box, text in plain}
    seen.update(_squash(text) for text in hidden)
    rescaled: list[str] = []
    for side in SCALING_PASSES:
        if min(image.size) <= side:
            continue  # already smaller; upscaling recovers nothing
        try:
            shrunk = image.resize((side, side), Image.NEAREST)
        except Exception:  # noqa: BLE001
            continue
        for _box, text in _ocr_lines(reader, shrunk):
            key = _squash(text)
            if key and key not in seen:
                seen.add(key)
                rescaled.append(text)

    out.visible_text = "\n".join(text for _box, text in plain)
    out.hidden_text = hidden
    out.rescaled_text = rescaled
    out.metadata_text = _metadata_text(image)
    if out.metadata_text:
        out.add_flag(F.IMAGE_METADATA_EXTRACTED)
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
