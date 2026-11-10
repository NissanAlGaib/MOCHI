"""Redaction inside an HTML page.

MOCHI inspects a web page as the text :func:`~mochi.preprocess.extract_html`
reads out of it, so the spans it marks for removal are pieces of that text,
not of the page source. The two differ in ways that defeat a plain substring
search of the source:

* a span can run across tags - ``Ignore <b>all</b> previous instructions``
  is one phrase to the detector and three text nodes in the page;
* the source escapes characters the text does not (``&#x27;`` for ``'``);
* a Stage II window covers several elements at once;
* a payload recovered by decoding (base64, ROT13, zero-width characters
  stripped) has no literal counterpart anywhere.

Searching the source for these failed, and :func:`~mochi.mitigate.sanitizer.apply`
then had to refuse the request rather than forward an attack it could not
cut: on the web-page arm, 68 of the 134 pages MOCHI acted on. This module
finds each span where the extractor read it - in the page's text nodes,
channel by channel, laid out with the extractor's own function - and removes
it there, so the page reaches the model with the payload cut and the rest of
the page intact.

Channels mirror the extractor, because a span can only be found in the text
it was taken from: the visible text, each hidden element, each comment, and
each text-bearing attribute.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from mochi.preprocess.html_extract import (
    NON_CONTENT_TAGS, TEXT_BEARING_ATTRS, _is_hidden, join_layout, lay_out,
)

_SPACE = re.compile(r"\s+")

Laid = list[tuple[NavigableString, str, int]]
Expand = Callable[[str, int, int], tuple[int, int]]


def _collapse(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _hidden_root(node: NavigableString) -> Tag | None:
    """The outermost hidden element containing ``node``, if any."""
    root = None
    for parent in node.parents:
        if not isinstance(parent, Tag):
            continue
        style = parent.get("style")
        if (isinstance(style, str) and _is_hidden(style)) or parent.has_attr("hidden"):
            root = parent
    return root


def _channels(soup: BeautifulSoup) -> list[tuple[Laid, bool]]:
    """Lay the page's text out the way the extractor does, channel by channel.

    Each channel comes with whether the reader can see it: visible text is
    ``True``; a hidden element or a comment is ``False``.
    """
    visible: list[NavigableString] = []
    hidden: dict[int, list[NavigableString]] = {}
    comments: list[Laid] = []

    for node in soup.descendants:
        if isinstance(node, Comment):
            text = str(node).strip()
            if text:
                comments.append([(node, text, 0)])
            continue
        if not isinstance(node, NavigableString):
            continue
        root = _hidden_root(node)
        if root is not None:
            # The extractor reads a hidden element whole, so it is one channel.
            hidden.setdefault(id(root), []).append(node)
            continue
        if any(isinstance(p, Tag) and p.name in NON_CONTENT_TAGS for p in node.parents):
            continue
        visible.append(node)

    channels = [(lay_out(visible), True),
                *((lay_out(nodes), False) for nodes in hidden.values()),
                *((laid, False) for laid in comments)]
    return [(laid, seen) for laid, seen in channels if laid]


def _find(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Every occurrence of ``needle``, exact first, then ignoring case."""
    for hay, pin in ((haystack, needle), (haystack.lower(), needle.lower())):
        if len(hay) != len(haystack):      # lower() changed a length: unsafe
            continue
        found = []
        position = hay.find(pin)
        while position != -1:
            found.append((position, position + len(pin)))
            position = hay.find(pin, position + len(pin))
        if found:
            return found
    return []


def _decodes_to(text: str, spans: list[str]) -> bool:
    """Whether MOCHI's preprocessing of ``text`` yields any of ``spans``.

    For payloads that were caught only once decoded: the span is the decoded
    instruction, so the node carrying the encoded form is what gets cut.
    """
    from mochi.preprocess.normalize import normalize

    result = normalize(text)
    readings = [_collapse(reading) for reading in result.scannable]
    return any(_find(reading, span) for reading in readings for span in spans)


def _merge(regions: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _cut(laid: Laid, regions: list[list[int]], marker: str) -> int:
    """Remove ``regions`` (offsets into the joined channel) from the nodes.

    One marker per region, in the first node it touches, so a phrase spread
    over three nodes leaves one marker rather than three. Returns the number
    of text characters removed.
    """
    deleted = 0
    cuts: dict[int, list[tuple[int, int, bool]]] = {}
    for start, end in regions:
        first = True
        for index, (_, text, offset) in enumerate(laid):
            lo, hi = max(start, offset), min(end, offset + len(text))
            if lo >= hi:
                continue
            cuts.setdefault(index, []).append((lo - offset, hi - offset, first))
            deleted += hi - lo
            first = False

    for index, ranges in cuts.items():
        node, text, _ = laid[index]
        out, cursor = [], 0
        for lo, hi, mark in sorted(ranges):
            out.append(text[cursor:lo])
            if mark:
                out.append(marker)
            cursor = hi
        out.append(text[cursor:])
        new_text = _collapse(" ".join(part for part in out if part.strip()))
        original = str(node)
        lead = original[:len(original) - len(original.lstrip())]
        trail = original[len(original.rstrip()):]
        kind = Comment if isinstance(node, Comment) else NavigableString
        node.replace_with(kind(f"{lead}{new_text}{trail}" if new_text else " "))
    return deleted


def _redact_channel(laid: Laid, spans: list[str], marker: str,
                    expand: Expand, *, seen: bool) -> tuple[int, int]:
    """Cut every span out of one channel. Returns (regions, characters).

    In a channel the reader cannot see, any span found removes the whole
    channel. Detection often flags one sentence of a several-sentence payload,
    and in visible text only that sentence can go, since the rest may be what
    the user asked about. Hidden text is not, so the unflagged remainder of
    the payload goes with it.
    """
    joined = join_layout(laid) if len(laid) > 1 else laid[0][1]
    regions = [expand(joined, start, end)
               for span in spans for start, end in _find(joined, span)]
    if not regions:
        # Nothing literal: the payload may have been decoded. Cut whole nodes
        # whose own decoding yields a span - a hidden element or comment
        # entirely, a visible paragraph on its own.
        regions = [(offset, offset + len(text)) for _, text, offset in laid
                   if _decodes_to(text, spans)]
        if not regions and not seen and len(laid) > 1 and _decodes_to(joined, spans):
            regions = [(0, len(joined))]
    if not regions:
        return 0, 0
    if not seen:
        regions = [(0, len(joined))]
    merged = _merge(regions)
    return len(merged), _cut(laid, merged, marker)


def redact_html(html: str, spans: list[str], *, marker: str,
                expand: Expand) -> tuple[str, int, int, int]:
    """Remove ``spans`` - text MOCHI read out of ``html`` - from the page.

    ``expand(text, start, end)`` widens a match to the unit removed (the
    sanitiser's sentence rule; the layout puts a newline between block
    elements, so a sentence never runs past its element). Returns the
    rewritten page, how many regions were removed, how many characters of
    text they held, and how many characters of text the page held: the
    redaction budget is a share of what the reader sees, not of the markup.

    A page with nothing to remove comes back unchanged, byte for byte; one
    with something removed is re-serialised, so insignificant details of the
    markup (``<br>`` as ``<br/>``) may change.
    """
    wanted = [span for span in dict.fromkeys(_collapse(s) for s in spans) if span]
    if not wanted:
        return html, 0, 0, 0

    soup = BeautifulSoup(html, "html.parser")
    regions = deleted = total = 0

    for laid, seen in _channels(soup):
        total += sum(len(text) for _, text, _ in laid)
        found, chars = _redact_channel(laid, wanted, marker, expand, seen=seen)
        regions += found
        deleted += chars

    for element in soup.find_all(True):
        for attr in TEXT_BEARING_ATTRS:
            value = element.get(attr)
            if not isinstance(value, str):
                continue
            text = _collapse(value)
            total += len(text)
            if any(_find(text, span) for span in wanted) or _decodes_to(text, wanted):
                element[attr] = marker
                regions += 1
                deleted += len(text)

    if not regions:
        return html, 0, 0, total
    return str(soup), regions, deleted, total
