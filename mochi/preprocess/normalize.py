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
        language: English/Tagalog composition, or ``None`` when not assessed.
            Distinct from ``script``: Taglish is Latin script throughout, so
            ``script`` cannot see it.
    """

    text: str
    variants: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    script: str | None = None
    language: "LanguageProfile | None" = None

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


# --- Language identification (Phase 3 amendment, register item A14) ---------
#
# Scope is **English-Tagalog only**, per adviser. Nothing here generalises to
# other pairs, and it should not be described as if it does.
#
# Why a function-word lexicon rather than langdetect / fastText lid.176:
#
# 1. Those return **one label for the whole text**. Code-switching is by
#    definition not one label, so the standard tool answers a question we are
#    not asking. "Huwag mong pansinin ang mga previous instructions" is not
#    Tagalog *or* English; the interleaving is the thing to measure.
# 2. They are trained on monolingual prose and are unreliable on the short,
#    mixed, imperative text that prompts actually consist of.
# 3. Function words are closed-class and high-frequency, which is exactly the
#    situation a lexicon handles well - and it costs no dependency, no model
#    download, and no inference time inside the request path.

#: Tagalog function words: markers, pronouns, demonstratives, conjunctions,
#: negation, and enclitic particles. Closed-class and near-impossible to write
#: Tagalog without.
TAGALOG_FUNCTION_WORDS: frozenset[str] = frozenset({
    # case markers / determiners
    "ang", "ng", "mga", "sa", "ay", "si", "ni", "kay", "nina", "kina",
    # pronouns
    "ako", "ko", "akin", "aking", "ikaw", "ka", "mo", "iyo", "iyong", "mong",
    "siya", "niya", "kanya", "kanyang", "kami", "namin", "amin", "tayo",
    "natin", "atin", "kayo", "ninyo", "inyo", "sila", "nila", "kanila",
    # demonstratives / locatives
    "ito", "nito", "dito", "iyan", "niyan", "diyan", "iyon", "noon", "doon",
    "yung", "yun", "ganito", "ganyan", "ganoon",
    # conjunctions / subordinators
    "pero", "ngunit", "subalit", "kung", "kapag", "dahil", "kasi", "para",
    "upang", "habang", "bago", "pagkatapos", "tapos", "saka",
    # negation / existential
    "hindi", "wala", "walang", "huwag", "wag", "mayroon", "meron",
    # particles
    "ba", "po", "opo", "naman", "lang", "lamang", "din", "rin", "daw", "raw",
    "nga", "pala", "sana", "muna", "kaya", "pa", "na",
    # very common content words that behave like function words
    "pwede", "puwede", "gusto", "ayaw", "dapat", "kailangan", "paano",
    "bakit", "saan", "sino", "ano", "alin", "kailan", "ilan", "mahal",
})

#: English function words. The standard closed class.
ENGLISH_FUNCTION_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "be",
    "been", "being", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "with", "for", "on", "by", "from", "as", "it", "its",
    "you", "your", "yours", "i", "me", "my", "we", "our", "he", "she", "they",
    "them", "their", "his", "her", "who", "whom", "which", "what", "when",
    "where", "why", "how", "all", "any", "some", "no", "not", "can", "could",
    "will", "would", "shall", "should", "do", "does", "did", "have", "has",
    "had", "there", "here", "about", "into", "over", "under", "after",
    "before", "because", "while", "please",
})

#: Words spelled identically in both languages. Counted for **neither**, because
#: attributing them either way is guesswork that biases short texts:
#: ``at`` is Tagalog "and" and an English preposition; ``may`` is Tagalog
#: "there is" and an English modal; ``man`` is a Tagalog particle and an English
#: noun. Under-counting both languages is safer than misattributing either.
AMBIGUOUS_FUNCTION_WORDS: frozenset[str] = frozenset({"at", "may", "man"})

#: High-frequency Tagalog content words that carry no affix, so the morphology
#: test below cannot recognise them. Without these, ``lahat`` and ``salamat``
#: are mistaken for English insertions and ordinary Tagalog reads as Taglish.
TAGALOG_BARE_WORDS: frozenset[str] = frozenset({
    "lahat", "salamat", "tao", "araw", "bagay", "oras", "taon", "buwan",
    "umaga", "hapon", "gabi", "tanong", "sagot", "totoo", "lihim", "sikreto",
    "susi", "utos", "panuto", "tagubilin", "patakaran", "panuntunan",
    "limitasyon", "mensahe", "sulat", "bayad", "bayarin", "resibo", "kopya",
    "talaan", "tulong", "maganda", "magandang", "mabuti", "marami", "konti",
    "bago", "luma", "malaki", "maliit", "sana", "parang", "simula", "ngayon",
    "kahapon", "bukas", "mamaya", "kanina", "lang", "mali", "tama", "wala",
})

#: Tagalog affixes. Tagalog is heavily agglutinative and English insertions
#: arrive bare, so affixation is a useful dependency-free way to tell a Tagalog
#: content word from a borrowed English one.
#:
#: The bare ``i-`` prefix is **deliberately excluded.** It matched every English
#: word beginning with "i" - ``instructions`` was classified Tagalog, which is
#: precisely backwards, since it is the archetypal English insertion in Taglish.
#: The ``i-`` form is recovered instead by :data:`TAGLISH_VERB`, where the
#: hyphen makes it unambiguous.
TAGALOG_PREFIXES: tuple[str, ...] = (
    "nagpa", "magpa", "pinag", "ipag", "nakaka", "pinaka", "maka", "naka",
    "mag", "nag", "pag", "pang", "pam", "pan", "ipa", "ika", "ma", "na", "ka",
)
TAGALOG_SUFFIXES: tuple[str, ...] = ("hin", "han", "in", "an", "ng")

#: A Tagalog verbal affix bolted onto an English stem: ``i-reset``,
#: ``mag-upgrade``, ``na-lock``, ``ma-access``, ``i-disregard``.
#:
#: This is the single most reliable code-switching marker available, because the
#: switch happens *inside one word* - the grammar is Tagalog and the lexeme is
#: English, and no monolingual text of either language produces this form. It
#: also explains why this module tokenises with its own hyphen-preserving regex
#: rather than reusing ``_WORD_RE``: splitting ``i-reset`` into ``i`` and
#: ``reset`` destroys the one unambiguous signal in the sentence.
TAGLISH_VERB = re.compile(
    r"^(?:i|mag|nag|ma|na|pa|pag|ka|pang|maki|naki|magpa|nagpa)-[a-z]{3,}$"
)

#: Hyphen-preserving tokeniser, for the reason given above.
_LANG_WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")

#: English content words. Orthography alone cannot carry this: ``system``,
#: ``prompt``, ``rules`` and ``password`` contain no letter Tagalog lacks and no
#: English derivational ending, yet they are exactly the words Taglish borrows.
#:
#: Curated rather than exhaustive, weighted toward the two registers that
#: actually appear here - general high-frequency English, and the support/
#: technical vocabulary of prompts. An English word outside this list is not
#: misclassified as Tagalog; it simply goes unattributed, which lowers
#: confidence rather than inverting the answer. Whether the coverage holds up is
#: exactly what the native-speaker-validated Wave D set exists to find out.
ENGLISH_CONTENT_WORDS: frozenset[str] = frozenset("""
    system prompt instructions instruction rules rule password admin key api
    guidelines guideline safety output input conversation history restrictions
    restriction message hidden content policy raw developer mode credentials
    database filter previous secret secrets config configuration override
    bypass display print follow disregard ignore reveal
    reset billing portal explain refund process account lock tries difference
    basic premium subscription upgrade downgrade plan monthly yearly load
    dashboard cancel order address fix issue payment method receipt
    reimbursement check receive documents document data privacy days
    application form button click screen browser network server connection
    setting settings option feature version security access permission token
    session transaction invoice charge balance amount total price cost fee
    discount email phone name file report page site link code error request
    response user login logout update delete create send confirm verify submit
    customer support service company number group problem fact case point
    time people work help need want make take give know think come get
    say use find tell ask feel try leave call good new first last long great
    little own other old right big high different small large next early
    young important few public bad same able thing way year day world life
    hand part place week available online offline device mobile desktop
    """.split())

#: English orthography absent from native Tagalog phonotactics. A token
#: containing one of these letters is almost certainly a borrowing.
NON_TAGALOG_LETTERS: frozenset[str] = frozenset("cfjqvxz")

#: English derivational endings. Tagalog has no equivalents.
ENGLISH_SUFFIXES: tuple[str, ...] = (
    "tion", "sion", "ment", "ness", "ance", "ence", "able", "ible", "ous",
    "ive", "ical", "ally", "ing", "ed", "ly", "er", "est",
)

#: Minimum words before a language call is attempted. Below this the ratios are
#: noise - a three-word fragment cannot support the claim.
MIN_WORDS_FOR_LANGUAGE = 5

#: Share of tokens that must be Tagalog function words for a Tagalog frame to be
#: considered present.
TAGALOG_FRAME_RATIO = 0.15

#: Share of tokens that must read as English for English to be considered
#: present alongside that frame.
ENGLISH_PRESENCE_RATIO = 0.12


def _is_english_token(word: str) -> bool:
    """Whether a token outside the Tagalog lexicon reads as English.

    Two independent cues, either sufficient:

    * a letter native Tagalog does not use (``c f j q v x z``), or
    * an English derivational ending Tagalog has no analogue for.

    Neither is decisive alone - Spanish loanwords in Tagalog do carry ``c`` and
    ``z`` - which is why this only ever runs on tokens already rejected by the
    Tagalog lexicon and the affix test, and why ``sikreto`` is listed as a bare
    Tagalog word rather than left to be caught here.
    """
    if any(letter in NON_TAGALOG_LETTERS for letter in word):
        return True
    return len(word) > 4 and word.endswith(ENGLISH_SUFFIXES)


def _looks_tagalog(word: str) -> bool:
    """Whether a token carries Tagalog affixation.

    ``-ing`` is checked before the Tagalog ``-ng`` suffix: English gerunds end
    in ``-ng`` too, and without that guard every "-ing" word reads as Tagalog,
    which is the single largest source of false Tagalog in code-switched text.
    """
    if len(word) < 4:
        return False

    # The Tagalog linker ``-ng`` attaches to vowel-final roots: marami + ng,
    # inyo + ng, iyo + ng. Several of those land on "-ing" and were being read
    # as English gerunds, which turned "Maraming salamat po" into code-switched
    # text. Checking the root first settles it: if what remains is a known
    # Tagalog word, the ending is a linker, not a gerund.
    if word.endswith("ng"):
        root = word[:-2]
        if root in TAGALOG_BARE_WORDS or root in TAGALOG_FUNCTION_WORDS:
            return True

    if word.endswith("ing"):
        return False
    if word.endswith(TAGALOG_SUFFIXES):
        return True
    return word.startswith(TAGALOG_PREFIXES) and len(word) >= 5


@dataclass(frozen=True)
class LanguageProfile:
    """English/Tagalog composition of one text.

    Returned as a record rather than the ``(value, bool)`` tuple used by
    :func:`dominant_script`, because five related numbers passed positionally is
    how call sites start swapping arguments.
    """

    language: str
    """``english`` | ``tagalog`` | ``mixed`` | ``unknown``."""
    english_ratio: float
    tagalog_ratio: float
    is_code_switched: bool
    switch_count: int
    """Transitions between the two languages across the function-word sequence.
    Distinguishes a Tagalog sentence followed by an English one (1 switch) from
    genuinely interleaved Taglish (many), which the ratios alone cannot."""


def detect_language(text: str) -> LanguageProfile:
    """Identify English/Tagalog composition and whether the text code-switches.

    This is language identification, and is deliberately **not** folded into
    :func:`dominant_script`, which answers a different question. Tagalog and
    English are both Latin script, so ``dominant_script`` reports
    ``("latin", False)`` for even the most thoroughly interleaved Taglish - the
    two functions are not substitutes and merging them would hide that.
    """
    # The text is lowered *before* matching, because the pattern is lowercase-
    # only: matching first would split "Huwag" into "uwag" and quietly lose
    # every sentence-initial word - which in imperative text is the important
    # one.
    words = _LANG_WORD_RE.findall(text.lower())
    if len(words) < MIN_WORDS_FOR_LANGUAGE:
        return LanguageProfile("unknown", 0.0, 0.0, False, 0)

    # Each token is attributed once, most reliable test first. The two closed
    # lexicons settle the grammatical frame, hyphenated Taglish verbs settle
    # themselves, the curated English list settles bare borrowings, and only
    # then does fuzzy morphology get a say. Running the morphology earlier is
    # what made "instructions" read as Tagalog.
    tagalog = english = frame = taglish_verbs = 0
    sequence: list[str] = []

    for word in words:
        if word in AMBIGUOUS_FUNCTION_WORDS:
            continue
        if word in TAGALOG_FUNCTION_WORDS:
            tagalog += 1
            frame += 1
            sequence.append("tl")
        elif word in ENGLISH_FUNCTION_WORDS:
            english += 1
            sequence.append("en")
        elif TAGLISH_VERB.match(word):
            # The switch happens inside the word, so it counts for both sides
            # and for the frame. Attributing it to either language alone would
            # misrepresent the one construction that is unambiguously mixed.
            taglish_verbs += 1
            tagalog += 1
            english += 1
            frame += 1
            sequence.append("en")
        elif word in ENGLISH_CONTENT_WORDS:
            english += 1
            sequence.append("en")
        elif word in TAGALOG_BARE_WORDS or _looks_tagalog(word):
            tagalog += 1
            sequence.append("tl")
        elif _is_english_token(word):
            english += 1
            sequence.append("en")

    attributed = tagalog + english
    if not attributed:
        return LanguageProfile("unknown", 0.0, 0.0, False, 0)

    total = len(words)
    tagalog_ratio = tagalog / total
    english_ratio = english / total
    switches = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)

    # Code-switching is asymmetric and the test reflects that. Taglish is a
    # Tagalog grammatical *frame* with English content words dropped into it -
    # "ang mga naunang instructions" - not a balanced blend. So the frame is
    # required (Tagalog function words), and English need only be present.
    # A symmetric "both above 15%" test misses the common case, because the
    # English half of Taglish is content words while the Tagalog half is
    # grammar, and those are not counted on the same scale.
    frame_ratio = frame / total
    is_mixed = (frame_ratio >= TAGALOG_FRAME_RATIO
                and english_ratio >= ENGLISH_PRESENCE_RATIO)

    # A hyphenated Taglish verb is decisive on its own. "Pwede mo bang i-cancel"
    # is code-switched by construction, and holding it to a ratio threshold
    # would discard the clearest evidence in the sentence because the sentence
    # is short.
    if taglish_verbs and frame_ratio >= TAGALOG_FRAME_RATIO:
        is_mixed = True

    if is_mixed:
        language = "mixed"
    elif tagalog > english:
        language = "tagalog"
    else:
        language = "english"

    return LanguageProfile(language, english_ratio, tagalog_ratio,
                           is_mixed, switches)


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

    # Language is assessed on the text as sent, alongside script and for the
    # same reason: both describe what arrived, before anything is undone.
    result.language = detect_language(text)
    if result.language.is_code_switched:
        result.add_flag(F.CODE_SWITCHED_DETECTED)

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
