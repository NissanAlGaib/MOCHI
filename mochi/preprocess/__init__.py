"""Normalization and de-obfuscation layer (Phase 3).

Runs before any detector sees text. Decodes encoding wrappers, strips
invisible characters, folds homoglyphs, and surfaces hidden HTML and file
metadata - so that Stage I and Stage II inspect what the target model will
actually read, not the disguise wrapped around it.

Guiding rule: **reveal, never discard.** Recovered payloads are added as extra
scannable variants alongside the original text.
"""

from mochi.preprocess.file_extract import FileExtraction, extract_file, extract_pdf
from mochi.preprocess.flags import NormalizationFlag
from mochi.preprocess.html_extract import HTMLExtraction, extract_html, looks_like_html
from mochi.preprocess.normalize import (
    NormalizationResult,
    dominant_script,
    fold_homoglyphs,
    normalize,
    strip_invisible,
    unwrap_encodings,
)
from mochi.preprocess.preprocessor import preprocess_file, preprocess_segment

__all__ = [
    "FileExtraction",
    "HTMLExtraction",
    "NormalizationFlag",
    "NormalizationResult",
    "dominant_script",
    "extract_file",
    "extract_html",
    "extract_pdf",
    "fold_homoglyphs",
    "looks_like_html",
    "normalize",
    "preprocess_file",
    "preprocess_segment",
    "strip_invisible",
    "unwrap_encodings",
]
