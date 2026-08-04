"""Unicode normalization and encoding de-obfuscation.

Runs before any detector sees text. The guiding rule: **reveal, never
discard.** Decoded payloads are added as extra scannable variants rather than
replacing the original, because both forms carry signal - the original tells
you obfuscation was attempted, the decoded form tells you what it was hiding.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import unquote

from mochi.preprocess.flags import NormalizationFlag as F

# --- Resource guards -------------------------------------------------------
# Decoding is attacker-controlled: a nested base64 chain expands geometrically,
# which is a resource-exhaustion vector in its own right. The thesis token
# budget section calls this out; these caps are the enforcement.
MAX_DECODE_DEPTH = 3
MAX_DECODED_CHARS = 100_000
MAX_INSPECTION_CHARS = 200_000

# --- Character classes -----------------------------------------------------
ZERO_WIDTH_CHARS = (
    "​"  # zero width space
    "‌"  # zero width non-joiner
    "‍"  # zero width joiner
    "⁠"  # word joiner
    "﻿"  # zero width no-break space / BOM
    "᠎"  # mongolian vowel separator
    "­"  # soft hyphen
    "͏"  # combining grapheme joiner
)

# Trojan Source (CVE-2021-42574) style directional overrides. These can make
# rendered text read differently from the byte sequence the model receives.
BIDI_CONTROL_CHARS = (
    "‪‫‬‭‮"  # LRE RLE PDF LRO RLO
    "⁦⁧⁨⁩"        # LRI RLI FSI PDI
)

#: Cyrillic/Greek characters that render identically to Latin ones. NFKC does
#: *not* fold these - they are distinct characters, not compatibility variants -
#: so homoglyph substitution survives standard normalization and needs its own
#: mapping.
CONFUSABLES = {
    # Cyrillic -> Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ј": "j", "һ": "h",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "І": "I", "Ѕ": "S",
    # Greek -> Latin
    "ο": "o", "α": "a", "ε": "e", "ρ": "p", "υ": "u",
    "Ο": "O", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z",
    "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ρ": "P", "Τ": "T", "Χ": "X",
}

BASE64_CANDIDATE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
HEX_CANDIDATE = re.compile(r"(?:[0-9a-fA-F]{2}){8,}")
URL_ENCODED_CANDIDATE = re.compile(r"(?:%[0-9a-fA-F]{2}){4,}")

#: Common English words used to decide whether a ROT13 transform *increased*
#: readability. Cheap and dependency-free; ROT13 is symmetric so this is the
#: only practical way to tell an encoded string from ordinary text.
COMMON_WORDS = frozenset(
    """the and you are for not with this that have from your all can will
    ignore instructions system prompt reveal disregard previous above now
    act pretend role assistant user please output print send email password""".split()
)

_WORD_RE = re.compile(r"[a-z]+")


@dataclass
class NormalizationResult:
    """Outcome of preprocessing a single text segment.

    Attributes:
        text: The cleaned primary text. This is what a detector should treat as
            "the message" - homoglyphs folded, invisible characters removed.
        variants: Additional texts recovered during preprocessing (decoded
            payloads, hidden HTML, file metadata). Every one must be scanned;
            hiding a payload in one of these is the whole point of the attack.
        flags: :class:`NormalizationFlag` values describing what was undone.
        script: Dominant Unicode script of the input ("latin", "cyrillic", ...).
    """

    text: str
    variants: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    script: str | None = None

    @property
    def scannable(self) -> list[str]:
        """Primary text plus every recovered variant, deduplicated."""
        seen: set[str] = set()
        out: list[str] = []
        for candidate in [self.text, *self.variants]:
            stripped = candidate.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                out.append(candidate)
        return out

    def combined(self) -> str:
        """All scannable text as one string, for detectors that want a blob."""
        return "\n".join(self.scannable)

    def add_flag(self, flag: F) -> None:
        if flag.value not in self.flags:
            self.flags.append(flag.value)


# --- Individual transforms -------------------------------------------------


def strip_invisible(text: str) -> tuple[str, list[F]]:
    """Remove zero-width and bidi-control characters.

    These split trigger words mid-token (``ig<ZWSP>nore``) so that regex misses
    them, while the model still reads the word normally.
    """
    flags: list[F] = []
    cleaned = text

    if any(char in cleaned for char in ZERO_WIDTH_CHARS):
        flags.append(F.ZERO_WIDTH_CHARS_DETECTED)
        cleaned = cleaned.translate({ord(c): None for c in ZERO_WIDTH_CHARS})

    if any(char in cleaned for char in BIDI_CONTROL_CHARS):
        flags.append(F.BIDI_CONTROL_CHARS_DETECTED)
        cleaned = cleaned.translate({ord(c): None for c in BIDI_CONTROL_CHARS})

    return cleaned, flags


def fold_homoglyphs(text: str) -> tuple[str, list[F]]:
    """Map Cyrillic/Greek lookalikes onto their Latin equivalents."""
    if not any(char in CONFUSABLES for char in text):
        return text, []
    folded = "".join(CONFUSABLES.get(char, char) for char in text)
    return folded, [F.HOMOGLYPHS_NORMALIZED]


def dominant_script(text: str) -> tuple[str | None, bool]:
    """Return ``(dominant_script, is_mixed)`` for the alphabetic characters.

    Mixed script in a single short segment is a homoglyph-attack signal in its
    own right - ordinary text rarely interleaves Latin and Cyrillic letters.

    Note: this is Unicode *script* detection, not language identification.
    """
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        script = name.split()[0].lower()
        counts[script] = counts.get(script, 0) + 1

    if not counts:
        return None, False

    total = sum(counts.values())
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    # "Mixed" means a second script holds a non-trivial share, not a stray char.
    is_mixed = any(
        script != dominant and count / total > 0.05 for script, count in counts.items()
    )
    return dominant, is_mixed


def _looks_like_text(candidate: str) -> bool:
    """Heuristic gate on decoded output.

    Random API keys and hashes decode to binary noise; we only want to surface
    decodes that produce something a language model would read as instructions.
    """
    if len(candidate) < 4:
        return False
    printable = sum(1 for c in candidate if c.isprintable() or c.isspace())
    if printable / len(candidate) < 0.9:
        return False
    return sum(1 for c in candidate if c.isalpha()) >= 3


def _english_word_score(text: str) -> int:
    return sum(1 for word in _WORD_RE.findall(text.lower()) if word in COMMON_WORDS)


def decode_base64(text: str) -> list[str]:
    """Return plausible plaintext decodings of base64 runs in ``text``."""
    out: list[str] = []
    for match in BASE64_CANDIDATE.findall(text):
        chunk = match
        # base64 requires length % 4 == 0; trim rather than reject so that a
        # blob embedded in surrounding prose still decodes.
        chunk = chunk[: len(chunk) - (len(chunk) % 4)] if len(chunk) % 4 else chunk
        if len(chunk) < 16:
            continue
        try:
            decoded = base64.b64decode(chunk, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if _looks_like_text(decoded):
            out.append(decoded)
    return out


def decode_hex(text: str) -> list[str]:
    out: list[str] = []
    for match in HEX_CANDIDATE.findall(text):
        try:
            decoded = bytes.fromhex(match).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if _looks_like_text(decoded):
            out.append(decoded)
    return out


def decode_url_encoding(text: str) -> list[str]:
    out: list[str] = []
    for match in URL_ENCODED_CANDIDATE.findall(text):
        decoded = unquote(match)
        if decoded != match and _looks_like_text(decoded):
            out.append(decoded)
    return out


def decode_rot13(text: str) -> str | None:
    """Return the ROT13 transform only if it looks *more* like English.

    ROT13 is symmetric, so there is no structural marker to detect. Comparing
    common-word density before and after is the practical test.
    """
    if not text.strip():
        return None
    transformed = codecs.encode(text, "rot_13")
    if _english_word_score(transformed) > _english_word_score(text):
        return transformed
    return None


def _decode_layer(text: str) -> list[tuple[str, F]]:
    """One pass of every decoder. Returns ``(decoded_text, flag)`` pairs."""
    found: list[tuple[str, F]] = []
    found += [(d, F.BASE64_DECODED) for d in decode_base64(text)]
    found += [(d, F.HEX_DECODED) for d in decode_hex(text)]
    found += [(d, F.URL_ENCODED_DECODED) for d in decode_url_encoding(text)]
    rot = decode_rot13(text)
    if rot is not None:
        found.append((rot, F.ROT13_DECODED))
    return found


def unwrap_encodings(text: str) -> tuple[list[str], list[F]]:
    """Recursively decode encoding wrappers, depth- and size-capped.

    Nesting is real (base64 of hex of base64), so this iterates - but bounded,
    because unbounded expansion of attacker-supplied input is itself an attack.
    """
    variants: list[str] = []
    flags: list[F] = []
    frontier = [text]
    total_chars = 0

    for depth in range(MAX_DECODE_DEPTH):
        next_frontier: list[str] = []
        for candidate in frontier:
            for decoded, flag in _decode_layer(candidate):
                if decoded in variants or decoded == text:
                    continue
                total_chars += len(decoded)
                if total_chars > MAX_DECODED_CHARS:
                    if F.OVERSIZED_AFTER_DECODE not in flags:
                        flags.append(F.OVERSIZED_AFTER_DECODE)
                    return variants, flags
                variants.append(decoded)
                if flag not in flags:
                    flags.append(flag)
                next_frontier.append(decoded)

        if not next_frontier:
            break
        frontier = next_frontier
    else:
        # Loop completed without breaking: there was still more to decode.
        if frontier:
            flags.append(F.DECODE_DEPTH_EXCEEDED)

    return variants, flags


def has_excessive_special_chars(text: str, threshold: float = 0.35) -> bool:
    """Flag adversarial-suffix style payloads (dense punctuation/symbols)."""
    if len(text) < 20:
        return False
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / len(text) > threshold


# --- Orchestration ---------------------------------------------------------


def normalize(text: str) -> NormalizationResult:
    """Full preprocessing pass over one text segment.

    Order matters: invisible characters are stripped *before* decoding, since
    a zero-width character inserted into a base64 blob would otherwise break
    the decode and let the payload through untouched.
    """
    result = NormalizationResult(text=text)

    if not text:
        return result

    if len(text) > MAX_INSPECTION_CHARS:
        text = text[:MAX_INSPECTION_CHARS]
        result.add_flag(F.TRUNCATED_FOR_INSPECTION)

    script, is_mixed = dominant_script(text)
    result.script = script
    if is_mixed:
        result.add_flag(F.MIXED_SCRIPT_DETECTED)

    cleaned, invisible_flags = strip_invisible(text)
    for flag in invisible_flags:
        result.add_flag(flag)

    normalized = unicodedata.normalize("NFKC", cleaned)
    if normalized != cleaned:
        result.add_flag(F.UNICODE_NFKC_APPLIED)

    folded, homoglyph_flags = fold_homoglyphs(normalized)
    for flag in homoglyph_flags:
        result.add_flag(flag)

    decoded_variants, decode_flags = unwrap_encodings(folded)
    for flag in decode_flags:
        result.add_flag(flag)
    result.variants.extend(decoded_variants)

    if has_excessive_special_chars(folded):
        result.add_flag(F.EXCESSIVE_SPECIAL_CHARACTERS)

    result.text = folded
    return result
