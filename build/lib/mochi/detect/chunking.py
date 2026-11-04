"""Sliding-window chunking, shared by Stage I and Stage II.

Both stages have the same failure mode if they truncate: a long document with
a short injected payload loses the payload silently. The measured payload share
in this project's corpus has a median of 3.4% of the document, and 72% of
locatable attacks occupy under 5% of their document, so truncation is not a
rare edge case - it is the common case for indirect injection.

Two properties matter and both are easy to get wrong:

* **Overlap.** Adjacent windows must overlap by more than the longest thing a
  detector can match, or a payload straddling a boundary is invisible to both
  windows. Zero-overlap chunking trades silent truncation for silent splitting.
* **Word boundaries.** Cutting mid-token corrupts the text a detector sees.
  Windows are nudged to the nearest whitespace when one is close enough.

Chunking is deliberately character-based rather than token-based. Stage I is
regex over characters, and Stage II is given a character budget derived from
its token limit; a shared primitive keeps one implementation instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

#: Chars per token, English prose. Matches ``eval.audit_datasets.CHARS_PER_TOKEN``.
#: A heuristic - the model tokenizer remains authoritative and truncates as a
#: safety net if a window still overflows.
CHARS_PER_TOKEN = 4

#: How far from a computed boundary to look for whitespace before giving up and
#: cutting mid-word.
BOUNDARY_SEARCH = 64


@dataclass(frozen=True)
class Chunk:
    """One window of a larger text, with its position in the original."""

    text: str
    start: int
    end: int
    index: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _snap(text: str, position: int, *, search: int = BOUNDARY_SEARCH) -> int:
    """Move ``position`` back to the nearest whitespace within ``search`` chars.

    Returns ``position`` unchanged when no whitespace is close enough - a long
    unbroken run (base64, a URL, minified JSON) has no word boundary to snap to
    and must be cut somewhere.
    """
    if position >= len(text):
        return len(text)
    lower = max(0, position - search)
    for index in range(position, lower, -1):
        if text[index - 1].isspace():
            return index
    return position


def chunk_text(
    text: str,
    *,
    size: int,
    overlap: int,
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping windows of at most ``size`` characters.

    Args:
        size: Maximum window length in characters.
        overlap: Characters shared between adjacent windows. Must exceed the
            longest match a consumer can produce, or boundary-straddling
            payloads are lost.
        max_chunks: Stop after this many windows. The caller is responsible for
            reporting that the tail went unscanned - see
            :func:`chunk_count_for` to detect the case in advance.

    Returns a single chunk spanning the whole text when it already fits, so
    callers need no special case for short input.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size}); "
            "otherwise the window never advances"
        )
    if not text:
        return []
    if len(text) <= size:
        return [Chunk(text=text, start=0, end=len(text), index=0)]

    chunks: list[Chunk] = []
    stride = size - overlap
    start = 0
    index = 0

    while start < len(text):
        if max_chunks is not None and index >= max_chunks:
            break
        end = _snap(text, min(start + size, len(text)))
        # A snap that erases the window (whitespace right after ``start``) would
        # stall the loop; fall back to the hard boundary.
        if end <= start:
            end = min(start + size, len(text))
        chunks.append(Chunk(text=text[start:end], start=start, end=end, index=index))
        if end >= len(text):
            break
        start += stride
        index += 1

    return chunks


def chunk_count_for(length: int, *, size: int, overlap: int) -> int:
    """How many windows ``chunk_text`` would produce for a text of ``length``.

    Lets a caller decide whether to scan, sample, or flag before paying for the
    scan itself.
    """
    if length <= 0:
        return 0
    if length <= size:
        return 1
    stride = size - overlap
    return 1 + -(-(length - size) // stride)  # ceil division


def iter_windows(text: str, *, size: int, overlap: int) -> Iterator[Chunk]:
    """Lazy :func:`chunk_text`, for callers that stop early on a match."""
    if len(text) <= size:
        if text:
            yield Chunk(text=text, start=0, end=len(text), index=0)
        return
    for chunk in chunk_text(text, size=size, overlap=overlap):
        yield chunk


def token_budget_to_chars(max_tokens: int) -> int:
    """Character window that approximately fits ``max_tokens`` tokens."""
    return max_tokens * CHARS_PER_TOKEN
