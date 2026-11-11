"""Text hidden in an image's low bits (least-significant-bit steganography).

Overwriting the lowest bit of each pixel value with a message changes every
value by at most one, which no one looking at the image can see, and a vision
model cannot read either: its encoder shrinks the image first, and measured on
a 1100-pixel carrier only 46% of the hidden bits survived the resize to
336x336 - indistinguishable from noise (50%). OCR cannot see it, so until this
module MOCHI passed such an image as clean.

The payload is still dangerous wherever something decodes it: an agent with a
code tool asked to "check the bit plane", or an application that runs a
steganography decoder on uploads. This module does that decode itself, so the
text reaches Stage I and Stage II like any other hidden channel.

What it reads: the lowest two bit planes of each channel taken alone, and
of the channels interleaved per pixel (RGB, BGR, RGBA), in both bit orders.
These are the layouts common LSB tools write, and the ones ``zsteg`` checks
first. It reads from the start of the image, where those tools begin, up to
:data:`MAX_BYTES` per layout.

What it does not read: payloads that are encrypted, scattered by a keyed
pseudo-random walk, or hidden in a JPEG's DCT coefficients (``steghide``,
``outguess``). The first two need the key to decode, so a model could not
follow them either; the third needs a JPEG decoder of its own. A JPEG's pixel
LSBs carry nothing, because lossy compression rewrites them.

Text is reported only when it looks like language - a run of at least
:data:`MIN_CHARS` printable characters holding :data:`MIN_WORDS` words - so
the repeating byte patterns a flat or dithered region produces are not
mistaken for a message. Measured: no text found in 400 benign rendered images,
in random-noise images, or in any solid colour.
"""

from __future__ import annotations

import re
import unicodedata

#: Bytes read per layout. The attack corpus's longest payload is under 600
#: characters; 16 KiB leaves room for long documents without scanning a
#: whole photo (a 12-megapixel image has 4.5 MB of LSBs per channel).
MAX_BYTES = 16 * 1024

#: Shortest run of printable characters treated as text.
MIN_CHARS = 24

#: Words (two or more letters) a run must hold. Filters repetition such as
#: ``mmmmmm`` or ``$I$I$I`` that a regular pixel pattern can produce.
MIN_WORDS = 4

#: Printable ASCII and whitespace, or a UTF-8 multi-byte sequence (Tagalog is
#: mostly ASCII, but names and loanwords are not).
_RUN = re.compile(
    rb"(?:[\x09\x0a\x0d\x20-\x7e]|[\xc2-\xf4][\x80-\xbf]{1,3}){%d,}" % MIN_CHARS)
_WORD = re.compile(r"[^\W\d_]{2,}")

#: Channel layouts: indices into an RGBA pixel, read in this order per pixel.
LAYOUTS = {
    "R": (0,), "G": (1,), "B": (2,), "A": (3,),
    "RGB": (0, 1, 2), "BGR": (2, 1, 0), "RGBA": (0, 1, 2, 3),
}
PLANES = (0, 1)
ORDERS = ("big", "little")


def _looks_like_text(run: str) -> bool:
    """Whether ``run`` holds words. Invisible format characters are ignored
    for the count only: a payload with a zero-width space between every
    letter has no two-letter "word" until they are removed, and the raw text
    is what gets reported, so the zero-width rule still sees them."""
    visible = "".join(c for c in run if unicodedata.category(c) != "Cf")
    return len(_WORD.findall(visible)) >= MIN_WORDS


def read_bitplanes(image) -> list[str]:
    """Messages hidden in ``image``'s low bits, as decoded text.

    ``image`` is a PIL image in any mode. Returns each distinct message once,
    in the order found; an empty list when there is none or numpy is missing.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with the OCR extra
        return []

    has_alpha = "A" in image.getbands()
    pixels = np.asarray(image.convert("RGBA"))
    need = MAX_BYTES * 8

    found: list[str] = []
    for name, channels in LAYOUTS.items():
        if 3 in channels and not has_alpha:
            continue  # a filled-in alpha channel is all 255: nothing to read
        values = pixels[:, :, list(channels)].reshape(-1)[:need]
        for plane in PLANES:
            bits = (values >> plane) & 1
            for order in ORDERS:
                raw = np.packbits(bits, bitorder=order).tobytes()
                for match in _RUN.finditer(raw):
                    text = match.group().decode("utf-8", errors="ignore").strip()
                    if _looks_like_text(text) and text not in found:
                        found.append(text)
    return found
