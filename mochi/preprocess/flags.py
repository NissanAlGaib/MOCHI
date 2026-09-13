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

    # --- Language ---
    #: Two languages interleaved in one segment. Sits beside
    #: ``MIXED_SCRIPT_DETECTED`` as a signal rather than an undo, because
    #: code-switching is a property of the text, not damage to repair.
    #:
    #: There is deliberately **no ``NON_ENGLISH_DETECTED`` flag.** The corpus
    #: already associates Spanish tokens with the malicious class, so a flag
    #: meaning "not English" would harden a measured bias into a detector rule.
    #: Code-switching is recorded; foreignness is not.
    CODE_SWITCHED_DETECTED = "code_switched_detected"

    #: A Tagalog word was machine-translated to English by
    #: :mod:`mochi.preprocess.code_switch` (opt-in, ``MOCHI_ENABLE_TAGALOG_TRANSLATION``).
    #: Records that a translation happened; carries no verdict about the word.
    TAGALOG_WORD_TRANSLATED = "tagalog_word_translated"

    #: A word matching neither the English nor the Tagalog classifier was
    #: removed from the code-switch filter's translated variant.
    #:
    #: This is the same bias risk the ``NON_ENGLISH_DETECTED`` decision above
    #: already named, one step further: the per-word classifier is a curated
    #: lexicon plus morphology (see ``classify_word_language`` in
    #: ``normalize.py``), not an exhaustive dictionary of either language, so
    #: this flag fires on genuine third-language words *and* on ordinary
    #: English/Tagalog words the lexicon simply does not cover - it cannot
    #: distinguish the two. For that reason this flag must never be added as a
    #: Track A engineered feature or otherwise used as evidence of
    #: maliciousness; it is an artifact of classifier coverage, not a property
    #: of the request.
    NON_ENGLISH_TAGALOG_WORD_STRIPPED = "non_english_tagalog_word_stripped"

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
