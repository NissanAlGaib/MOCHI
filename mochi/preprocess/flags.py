"""Normalization flag vocabulary.

These strings land in ``TelemetryRecord.normalization_flags`` and are consumed
by the Stage I ``invisible_text`` / ``obfuscation_encoding`` detectors in
Phase 6, so they are defined once here rather than spelled inline.

A flag records *what the preprocessor had to undo*. It is not itself a verdict:
legitimate content occasionally contains a base64 blob. But a flag on otherwise
benign-looking text is a strong prior, and several flags together (say,
``zero_width_chars_detected`` plus ``base64_decoded``) is close to conclusive.
"""

from __future__ import annotations

from enum import StrEnum


class NormalizationFlag(StrEnum):
    # --- Unicode-level tampering ---
    UNICODE_NFKC_APPLIED = "unicode_nfkc_applied"
    ZERO_WIDTH_CHARS_DETECTED = "zero_width_chars_detected"
    BIDI_CONTROL_CHARS_DETECTED = "bidi_control_chars_detected"
    HOMOGLYPHS_NORMALIZED = "homoglyphs_normalized"
    MIXED_SCRIPT_DETECTED = "mixed_script_detected"
    EXCESSIVE_SPECIAL_CHARACTERS = "excessive_special_characters"

    # --- Encoding wrappers ---
    BASE64_DECODED = "base64_decoded"
    HEX_DECODED = "hex_decoded"
    ROT13_DECODED = "rot13_decoded"
    URL_ENCODED_DECODED = "url_encoded_decoded"

    # --- Markup / document structure ---
    HTML_STRIPPED = "html_stripped"
    HIDDEN_CSS_DETECTED = "hidden_css_detected"
    HTML_COMMENT_EXTRACTED = "html_comment_extracted"
    ATTRIBUTE_TEXT_EXTRACTED = "attribute_text_extracted"
    FILE_METADATA_EXTRACTED = "file_metadata_extracted"

    # --- Resource guards ---
    DECODE_DEPTH_EXCEEDED = "decode_depth_exceeded"
    OVERSIZED_AFTER_DECODE = "oversized_after_decode"
    TRUNCATED_FOR_INSPECTION = "truncated_for_inspection"
