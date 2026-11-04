"""Source-aware preprocessing entry point.

Ties the Unicode/encoding layer to the markup and file extractors and routes
each payload segment through the right one based on its source tag. This is
the single function the detection pipeline calls in Phase 6.
"""

from __future__ import annotations

from mochi.preprocess.file_extract import extract_file
from mochi.preprocess.html_extract import extract_html, looks_like_html
from mochi.preprocess.normalize import NormalizationResult, normalize

#: Source tags whose content may legitimately be markup.
MARKUP_CAPABLE_TAGS = frozenset({"web_content", "api_response", "retrieved_document"})


def _merge(primary: NormalizationResult,
           others: list[NormalizationResult]) -> NormalizationResult:
    """Fold several normalization results into one, preserving all signal."""
    merged = NormalizationResult(
        text=primary.text,
        variants=list(primary.variants),
        flags=list(primary.flags),
        script=primary.script,
    )
    for other in others:
        if other.text and other.text.strip():
            merged.variants.append(other.text)
        merged.variants.extend(other.variants)
        for flag in other.flags:
            if flag not in merged.flags:
                merged.flags.append(flag)
    return merged


def preprocess_segment(text: str, source_tag: str | None = None) -> NormalizationResult:
    """Preprocess one payload segment.

    ``web_content`` (and other markup-capable sources) containing HTML is split
    into visible text plus hidden channels first; every resulting piece is then
    Unicode-normalized and de-obfuscated independently, because an attacker can
    combine techniques - a base64 payload inside a ``display:none`` div needs
    both layers peeled to be visible to a detector.
    """
    if not text:
        return NormalizationResult(text="")

    if source_tag in MARKUP_CAPABLE_TAGS and looks_like_html(text):
        extraction = extract_html(text)
        segments = extraction.all_segments
        if not segments:
            result = normalize("")
            for flag in extraction.flags:
                if flag not in result.flags:
                    result.flags.append(flag)
            return result

        primary = normalize(segments[0])
        others = [normalize(segment) for segment in segments[1:]]
        merged = _merge(primary, others)
        for flag in extraction.flags:
            if flag not in merged.flags:
                merged.flags.append(flag)
        return merged

    return normalize(text)


def preprocess_file(data: bytes, mime_type: str | None = None,
                    filename: str | None = None) -> NormalizationResult:
    """Extract a file's text and metadata, then normalize every piece."""
    extraction = extract_file(data, mime_type=mime_type, filename=filename)
    segments = extraction.all_segments

    if not segments:
        result = NormalizationResult(text="")
        result.flags.extend(extraction.flags)
        return result

    primary = normalize(segments[0])
    others = [normalize(segment) for segment in segments[1:]]
    merged = _merge(primary, others)
    for flag in extraction.flags:
        if flag not in merged.flags:
            merged.flags.append(flag)
    return merged
