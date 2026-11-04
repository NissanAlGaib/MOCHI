"""HTML extraction for ``web_content`` segments.

A naive ``BeautifulSoup(html).get_text()`` throws away exactly the places an
indirect injection hides: ``display:none`` divs, HTML comments, and ``alt`` /
``title`` attributes. The scraper discards them, so a human reviewing the page
never sees them - but if the application feeds raw HTML to the model, the model
reads them.

This module extracts hidden content *separately* and returns it as scannable
variants, so Stage I/II inspect it while the telemetry records where it came
from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Comment

from mochi.preprocess.flags import NormalizationFlag as F

#: CSS patterns that render an element invisible while leaving its text in the
#: DOM (and therefore in anything that serializes the DOM for an LLM).
HIDING_PATTERNS = (
    re.compile(r"display\s*:\s*none", re.I),
    re.compile(r"visibility\s*:\s*hidden", re.I),
    re.compile(r"font-size\s*:\s*0(?:px|em|rem|%)?\b", re.I),
    re.compile(r"opacity\s*:\s*0(?:\.0+)?\b", re.I),
    re.compile(r"color\s*:\s*(?:#fff(?:fff)?|white|rgba?\(\s*255\s*,\s*255\s*,\s*255)", re.I),
    re.compile(r"(?:text-indent|left|top)\s*:\s*-\d{3,}", re.I),  # off-screen
    re.compile(r"clip\s*:\s*rect\(\s*0", re.I),
    re.compile(r"height\s*:\s*0(?:px)?\s*;.*overflow\s*:\s*hidden", re.I | re.S),
)

#: Attributes whose values reach the model but not the reader.
TEXT_BEARING_ATTRS = ("alt", "title", "aria-label", "placeholder", "data-tooltip")

#: Tags whose content is never prose and would only add noise.
NON_CONTENT_TAGS = ("script", "style", "noscript")


@dataclass
class HTMLExtraction:
    visible_text: str
    hidden_text: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    attribute_text: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def all_segments(self) -> list[str]:
        """Everything worth scanning, visible content first."""
        return [
            segment
            for segment in [
                self.visible_text,
                *self.hidden_text,
                *self.comments,
                *self.attribute_text,
            ]
            if segment and segment.strip()
        ]

    def add_flag(self, flag: F) -> None:
        if flag.value not in self.flags:
            self.flags.append(flag.value)


def _is_hidden(style: str) -> bool:
    return any(pattern.search(style) for pattern in HIDING_PATTERNS)


def looks_like_html(text: str) -> bool:
    """Cheap check so plain prose is not run through the parser needlessly."""
    return bool(re.search(r"<[a-zA-Z!/][^>]*>", text))


def extract_html(html: str) -> HTMLExtraction:
    """Split HTML into visible text plus every hidden text channel."""
    result = HTMLExtraction(visible_text="")
    if not html or not html.strip():
        return result

    soup = BeautifulSoup(html, "html.parser")
    result.add_flag(F.HTML_STRIPPED)

    # 1. HTML comments - invisible to the reader, present in the source.
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        text = str(comment).strip()
        if text:
            result.comments.append(text)
        comment.extract()
    if result.comments:
        result.add_flag(F.HTML_COMMENT_EXTRACTED)

    # 2. Text-bearing attributes, collected before elements are unwrapped.
    for element in soup.find_all(True):
        for attr in TEXT_BEARING_ATTRS:
            value = element.get(attr)
            if isinstance(value, str) and value.strip():
                result.attribute_text.append(value.strip())
    if result.attribute_text:
        result.add_flag(F.ATTRIBUTE_TEXT_EXTRACTED)

    # 3. CSS-hidden elements. Removed from the visible tree and kept aside so
    #    they are still scanned - the point is that they are suspicious, not
    #    that they are irrelevant.
    for element in soup.find_all(style=True):
        style = element.get("style", "")
        if isinstance(style, str) and _is_hidden(style):
            text = element.get_text(separator=" ", strip=True)
            if text:
                result.hidden_text.append(text)
            element.decompose()
    # The `hidden` boolean attribute has the same effect as display:none.
    for element in soup.find_all(hidden=True):
        text = element.get_text(separator=" ", strip=True)
        if text:
            result.hidden_text.append(text)
        element.decompose()
    if result.hidden_text:
        result.add_flag(F.HIDDEN_CSS_DETECTED)

    # 4. Drop non-content tags, then take what remains as visible text.
    for tag_name in NON_CONTENT_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    result.visible_text = soup.get_text(separator=" ", strip=True)
    return result
