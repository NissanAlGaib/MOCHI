"""Phase 6.5: engineered features, computed once and shared.

Register items **A11** (feature engineering) and **A12** (injection keyword
dictionary + question marks).

The whole point of this module is that it is the *only* place a feature is
defined. ``eval/build_features.py`` imports it to materialise dataset columns;
the pipeline can attach the same vector to a segment at request time. A
notebook that recomputes "question mark count" its own way will drift from the
gateway, and on the day it does, every ablation number in Chapter IV becomes
fiction - silently, with no test failing. That is the same train/serve
constraint that ruled out POS/stopword removal in register item A2; only the
feature has changed, not the reasoning.

**Roughly half of these cost nothing.** The obfuscation family is a direct
transcription of ``NormalizationResult.flags``, and the injection-lexicon family
is a transcription of ``Stage1Result``. Both are already computed on every
request and then discarded. Writing them to a dataset is bookkeeping, not new
analysis - which is also why they are the families most likely to carry signal a
TF-IDF model cannot see: they describe the *envelope* rather than the words.

Three deliberate limits:

1. **Nothing here is fitted on the corpus.** Every value is a function of one
   text and its own preprocessing. No log-odds, no learned vocabulary, no mined
   phrase list. Corpus-fitted features are legitimate but must be fitted on the
   train split only, and putting them here would make that impossible to
   enforce - the extractor has no idea which split it is looking at.

2. **The imperative family is lexical, not syntactic.** A curated verb list and
   a position check, no POS tagger. Register item **Q1** (spaCy or NLTK?) is
   still unanswered, and ``starts_with_imperative`` is genuinely approximate
   without one - "Ignore" opening a sentence is usually imperative, but "Ignoring
   the noise, the result holds" is not. Reported as-is rather than overclaimed.

3. **The language family reads English/Tagalog only.** It consumes
   ``normalize.detect_language`` rather than reimplementing it, and its scope is
   the one language pair the adviser approved. There is deliberately no
   "is non-English" column: the corpus already associates Spanish tokens with
   the malicious class, and a feature meaning *foreign* would hand a model the
   shortcut this work exists to expose.

Nothing in this module decides anything. Whether these features ever reach the
runtime decision path depends on whether the hybrid ablation shows they earn a
place - adding an unvalidated signal to a security-critical path because it was
cheap to compute is how a false-positive rate regresses without anyone noticing.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from urllib.parse import parse_qsl, urlsplit

from mochi.preprocess.flags import NormalizationFlag as F
from mochi.preprocess.normalize import NormalizationResult, dominant_script, normalize

#: Detector ids from ``mochi/patterns.json``. Pinned here so the column set is
#: stable: a dataset whose columns change when someone edits the pattern file is
#: not comparable across runs. ``tests/test_features.py`` asserts this list still
#: matches the pattern file, so drift is caught rather than silently tolerated.
DETECTOR_IDS: tuple[str, ...] = (
    "direct_prompt_injection",
    "indirect_prompt_injection",
    "jailbreak",
    "data_exfiltration",
    "role_manipulation",
    "url_exfiltration",
    "standard_it_security",
    "obfuscation_encoding",
    "invisible_text",
)

#: Verbs that carry an instruction in a prompt-injection context. Not a general
#: imperative list - "consider", "imagine" and similar appear constantly in
#: benign prompts and would only add noise. These are the verbs that name an
#: action the attacker wants taken.
INSTRUCTION_VERBS: frozenset[str] = frozenset({
    "ignore", "disregard", "forget", "override", "bypass", "skip", "omit",
    "reveal", "disclose", "show", "print", "output", "repeat", "echo", "dump",
    "execute", "run", "act", "pretend", "roleplay", "become", "simulate",
    "obey", "follow", "comply", "respond", "answer", "reply",
    "delete", "remove", "disable", "enable", "grant", "elevate",
})

#: Modal verbs and phrases that impose obligation. An injection tells the model
#: what it *must* do; a question asks what it *can*.
OBLIGATION_TERMS: frozenset[str] = frozenset({
    "must", "shall", "should", "required", "mandatory", "obliged", "have",
    "need", "immediately", "always", "never",
})

NEGATION_TERMS: frozenset[str] = frozenset({
    "not", "no", "never", "cannot", "without", "dont", "doesnt", "wont", "cant",
})

SECOND_PERSON: frozenset[str] = frozenset({"you", "your", "yours", "yourself"})

_WORD = re.compile(r"[A-Za-z']+")

#: One sentence, **terminator included**. A plain ``split`` on ``[.!?\n]``
#: consumes the terminator, after which no fragment ends in '?' and
#: ``question_ratio`` is silently always zero - which is the one column A12 exists
#: to test. Matching sentences instead of splitting between them keeps the
#: punctuation attached.
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_HTML_TAG = re.compile(r"<[a-zA-Z/][^>]{0,200}>")
_CODE_FENCE = re.compile(r"```|~~~|<code[ >]|\n {4}\S")
_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(")

#: Chars per token, English prose. Matches ``detect.chunking.CHARS_PER_TOKEN``
#: and ``eval.audit_datasets.CHARS_PER_TOKEN`` - one heuristic, three callers.
CHARS_PER_TOKEN = 4

#: Query-string values at or above this length are worth scoring for entropy. A
#: short value like ``?page=2`` has no room to carry exfiltrated data and its
#: entropy is meaningless.
MIN_ENTROPY_VALUE_CHARS = 12


@dataclass(frozen=True)
class FeatureVector:
    """Engineered features for one text.

    Every field is a function of the text and its own preprocessing. Ordering
    within the dataclass is the column ordering in the materialised dataset.
    """

    # --- surface -----------------------------------------------------------
    char_count: int = 0
    word_count: int = 0
    est_token_count: int = 0
    avg_word_len: float = 0.0
    line_count: int = 0
    max_line_len: int = 0
    uppercase_ratio: float = 0.0
    digit_ratio: float = 0.0

    # --- punctuation and interrogative structure (A12) ---------------------
    question_mark_count: int = 0
    ends_with_question: bool = False
    question_ratio: float = 0.0
    """Share of sentences ending in '?'. The adviser's hypothesis is that benign
    prompts ask and injections command; this is the column that tests it."""
    exclamation_count: int = 0
    colon_count: int = 0
    quote_count: int = 0
    bracket_count: int = 0
    newline_ratio: float = 0.0
    special_char_ratio: float = 0.0

    # --- imperative structure (A12, lexical only pending Q1) ---------------
    imperative_verb_count: int = 0
    starts_with_imperative: bool = False
    second_person_count: int = 0
    obligation_count: int = 0
    negation_count: int = 0
    instruction_verb_ratio: float = 0.0

    # --- injection lexicon, from Stage I (A12) ------------------------------
    stage1_hit_count: int = 0
    stage1_max_severity: str = "none"
    stage1_would_block: bool = False
    stage1_detector_ids: str = ""
    """Semicolon-joined, so the column survives a CSV round-trip."""
    detector_hits: dict[str, bool] = field(default_factory=dict)

    # --- obfuscation, from Phase 3 normalization flags ----------------------
    has_zero_width: bool = False
    has_bidi: bool = False
    homoglyphs_normalized: bool = False
    is_mixed_script: bool = False
    nfkc_applied: bool = False
    excessive_special_chars: bool = False
    base64_decoded: bool = False
    hex_decoded: bool = False
    rot13_decoded: bool = False
    url_decoded: bool = False
    html_stripped: bool = False
    hidden_css_detected: bool = False
    html_comment_extracted: bool = False
    attribute_text_extracted: bool = False
    file_metadata_extracted: bool = False
    decode_depth_exceeded: bool = False
    oversized_after_decode: bool = False
    truncated_for_inspection: bool = False
    n_flags: int = 0
    n_variants_recovered: int = 0
    decoded_char_gain: int = 0
    """Characters recovered by decoding. Large positive values mean a payload
    was hidden inside an encoding wrapper."""
    normalization_delta: int = 0
    """Characters removed or changed by normalization. Non-zero means the text
    as sent differs from the text a detector reads."""

    # --- script and language (A14) ------------------------------------------
    dominant_script: str = ""
    detected_language: str = "unknown"
    """``english`` | ``tagalog`` | ``mixed`` | ``unknown``. Distinct from
    ``dominant_script``, which reports ``latin`` for all of them."""
    is_code_switched: bool = False
    english_ratio: float = 0.0
    tagalog_ratio: float = 0.0
    language_switch_count: int = 0

    # --- URL and entity ------------------------------------------------------
    url_count: int = 0
    has_markdown_image: bool = False
    has_auto_fetch_url: bool = False
    max_url_query_entropy: float = 0.0
    has_code_block: bool = False
    has_html_tag: bool = False

    # --- position, connects to the D10 signal-position audit ----------------
    first_hit_offset_ratio: float = -1.0
    """Where the first Stage I match starts, as a share of text length. -1.0
    when nothing matched, so 'no hit' is distinguishable from 'hit at offset 0'."""
    payload_share: float = 0.0
    payload_region: str = "none"
    """head / middle / tail, matching the D10 audit's three-way split, plus
    ``decoded_only`` when Stage I fired on a decoded variant rather than the text
    as sent, and ``none`` when nothing fired at all."""

    def as_dict(self) -> dict:
        """Flatten to one row, expanding ``detector_hits`` into ``hit_*`` columns."""
        row = asdict(self)
        hits = row.pop("detector_hits")
        for detector_id in DETECTOR_IDS:
            row[f"hit_{detector_id}"] = bool(hits.get(detector_id, False))
        return row


def feature_names() -> list[str]:
    """Column names in materialisation order."""
    return list(FeatureVector().as_dict().keys())


# --- individual computations ------------------------------------------------


def _shannon_entropy(value: str) -> float:
    """Bits per character. High entropy in a query value means opaque data."""
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _url_features(text: str) -> tuple[int, bool, bool, float]:
    """URL count, markdown-image presence, auto-fetch presence, max query entropy.

    Deliberately re-derived here rather than calling ``url_scanner.scan_urls``:
    that module scores *model output* for exfiltration risk and its thresholds
    encode an enforcement policy. A feature column must stay a description of the
    text, not a copy of a decision - otherwise the feature and the enforcement
    rule move together and the ablation cannot separate them.
    """
    urls = _URL.findall(text)
    if not urls:
        return 0, bool(_MARKDOWN_IMAGE.search(text)), False, 0.0

    has_image = bool(_MARKDOWN_IMAGE.search(text))
    has_img_tag = "<img" in text.lower()
    best_entropy = 0.0

    for url in urls:
        try:
            query = urlsplit(url).query
        except ValueError:
            continue
        if not query:
            continue
        for _, value in parse_qsl(query, keep_blank_values=False):
            if len(value) >= MIN_ENTROPY_VALUE_CHARS:
                best_entropy = max(best_entropy, _shannon_entropy(value))

    return len(urls), has_image, has_image or has_img_tag, best_entropy


def _imperative_features(text: str) -> tuple[int, bool, int, int, int, float]:
    """Lexical imperative signals. See the module docstring on Q1."""
    words = [w.lower().replace("'", "") for w in _WORD.findall(text)]
    if not words:
        return 0, False, 0, 0, 0, 0.0

    verb_count = sum(1 for w in words if w in INSTRUCTION_VERBS)
    second_person = sum(1 for w in words if w in SECOND_PERSON)
    obligation = sum(1 for w in words if w in OBLIGATION_TERMS)
    negation = sum(1 for w in words if w in NEGATION_TERMS)

    # An instruction verb opening a sentence is the imperative signal. Checking
    # every sentence, not only the first, because the payload in an indirect
    # injection is rarely the opening clause of the document.
    starts = False
    for match in _SENTENCE.finditer(text):
        leading = _WORD.search(match.group(0))
        if leading and leading.group(0).lower() in INSTRUCTION_VERBS:
            starts = True
            break

    return (verb_count, starts, second_person, obligation, negation,
            verb_count / len(words))


def _position_features(text: str, stage1) -> tuple[float, float, str]:
    """Where the Stage I evidence sits, and how much of the text it occupies.

    ``Detection`` carries the matched substring but not its offset, so the offset
    is recovered by searching for it. That is exact for the first occurrence,
    which is the one that matters: the audit question is where the signal
    *starts*, not how often it repeats.
    """
    if stage1 is None or not getattr(stage1, "detections", None) or not text:
        return -1.0, 0.0, "none"

    offsets = []
    matched_chars = 0
    for detection in stage1.detections:
        matched = getattr(detection, "matched_text", "")
        if not matched:
            continue
        index = text.find(matched)
        if index >= 0:
            offsets.append(index)
            matched_chars += len(matched)

    if not offsets:
        # Stage I fired, but on a decoded variant rather than the text as sent -
        # so the evidence has no position in the original. Reported distinctly:
        # collapsing this into "none" would make an obfuscated attack
        # indistinguishable from a clean text in the position columns.
        return -1.0, 0.0, "decoded_only"

    ratio = min(offsets) / len(text)
    region = "head" if ratio < 1 / 3 else ("middle" if ratio < 2 / 3 else "tail")
    return ratio, matched_chars / len(text), region


# --- entry point -------------------------------------------------------------


def extract(text: str, *, norm: NormalizationResult | None = None,
            stage1=None) -> FeatureVector:
    """Compute every feature for one text.

    Args:
        text: The raw text as received, *before* normalization. Surface and
            punctuation features describe what was actually sent; the
            obfuscation family describes what preprocessing had to undo. Passing
            already-normalized text here would erase the difference between the
            two, which is most of the signal.
        norm: Result of :func:`mochi.preprocess.normalize.normalize`. Computed
            here when omitted, but the pipeline already has one - pass it rather
            than paying twice.
        stage1: A :class:`~mochi.detect.stage1_syntactic.Stage1Result`, or None.
            Typed loosely on purpose: importing the detector here would make the
            preprocessing package depend on the detection package, and the
            dependency runs the other way everywhere else.
    """
    if norm is None:
        norm = normalize(text)

    flags = set(norm.flags)
    length = len(text)

    words = _WORD.findall(text)
    word_count = len(words)
    lines = text.splitlines() or [""]
    sentences = [m.group(0).strip() for m in _SENTENCE.finditer(text) if m.group(0).strip()]

    alpha = [c for c in text if c.isalpha()]
    uppercase_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
    digit_ratio = (sum(1 for c in text if c.isdigit()) / length) if length else 0.0
    alnum_or_space = sum(1 for c in text if c.isalnum() or c.isspace())
    special_ratio = ((length - alnum_or_space) / length) if length else 0.0

    question_marks = text.count("?")
    ending_questions = sum(1 for s in sentences if s.rstrip().endswith("?")) if sentences else 0

    (verb_count, starts_imperative, second_person, obligation, negation,
     verb_ratio) = _imperative_features(text)

    url_count, has_image, has_auto_fetch, entropy = _url_features(text)

    script, is_mixed = dominant_script(text)

    detector_hits: dict[str, bool] = {d: False for d in DETECTOR_IDS}
    stage1_hit_count = 0
    stage1_max_severity = "none"
    stage1_would_block = False
    detector_ids_seen: list[str] = []

    if stage1 is not None:
        detections = getattr(stage1, "detections", []) or []
        stage1_hit_count = len(detections)
        for detection in detections:
            detector_hits[detection.detector_id] = True
            detector_ids_seen.append(detection.detector_id)
        highest = getattr(stage1, "highest", None)
        if highest is not None:
            stage1_max_severity = highest.severity
        stage1_would_block = bool(getattr(stage1, "should_block", False))

    offset_ratio, payload_share, region = _position_features(text, stage1)

    decoded_gain = sum(len(v) for v in norm.variants)

    return FeatureVector(
        char_count=length,
        word_count=word_count,
        est_token_count=length // CHARS_PER_TOKEN,
        avg_word_len=(sum(len(w) for w in words) / word_count) if word_count else 0.0,
        line_count=len(lines),
        max_line_len=max((len(line) for line in lines), default=0),
        uppercase_ratio=uppercase_ratio,
        digit_ratio=digit_ratio,

        question_mark_count=question_marks,
        ends_with_question=text.rstrip().endswith("?"),
        question_ratio=(ending_questions / len(sentences)) if sentences else 0.0,
        exclamation_count=text.count("!"),
        colon_count=text.count(":"),
        quote_count=text.count('"') + text.count("'"),
        bracket_count=sum(text.count(c) for c in "[]{}()"),
        newline_ratio=(text.count("\n") / length) if length else 0.0,
        special_char_ratio=special_ratio,

        imperative_verb_count=verb_count,
        starts_with_imperative=starts_imperative,
        second_person_count=second_person,
        obligation_count=obligation,
        negation_count=negation,
        instruction_verb_ratio=verb_ratio,

        stage1_hit_count=stage1_hit_count,
        stage1_max_severity=stage1_max_severity,
        stage1_would_block=stage1_would_block,
        stage1_detector_ids=";".join(sorted(set(detector_ids_seen))),
        detector_hits=detector_hits,

        has_zero_width=F.ZERO_WIDTH_CHARS_DETECTED in flags,
        has_bidi=F.BIDI_CONTROL_CHARS_DETECTED in flags,
        homoglyphs_normalized=F.HOMOGLYPHS_NORMALIZED in flags,
        is_mixed_script=is_mixed,
        nfkc_applied=F.UNICODE_NFKC_APPLIED in flags,
        excessive_special_chars=F.EXCESSIVE_SPECIAL_CHARACTERS in flags,
        base64_decoded=F.BASE64_DECODED in flags,
        hex_decoded=F.HEX_DECODED in flags,
        rot13_decoded=F.ROT13_DECODED in flags,
        url_decoded=F.URL_ENCODED_DECODED in flags,
        html_stripped=F.HTML_STRIPPED in flags,
        hidden_css_detected=F.HIDDEN_CSS_DETECTED in flags,
        html_comment_extracted=F.HTML_COMMENT_EXTRACTED in flags,
        attribute_text_extracted=F.ATTRIBUTE_TEXT_EXTRACTED in flags,
        file_metadata_extracted=F.FILE_METADATA_EXTRACTED in flags,
        decode_depth_exceeded=F.DECODE_DEPTH_EXCEEDED in flags,
        oversized_after_decode=F.OVERSIZED_AFTER_DECODE in flags,
        truncated_for_inspection=F.TRUNCATED_FOR_INSPECTION in flags,
        n_flags=len(flags),
        n_variants_recovered=len(norm.variants),
        decoded_char_gain=decoded_gain,
        normalization_delta=abs(len(norm.text) - length),

        dominant_script=script or "",
        detected_language=norm.language.language if norm.language else "unknown",
        is_code_switched=bool(norm.language and norm.language.is_code_switched),
        english_ratio=norm.language.english_ratio if norm.language else 0.0,
        tagalog_ratio=norm.language.tagalog_ratio if norm.language else 0.0,
        language_switch_count=norm.language.switch_count if norm.language else 0,

        url_count=url_count,
        has_markdown_image=has_image,
        has_auto_fetch_url=has_auto_fetch,
        max_url_query_entropy=entropy,
        has_code_block=bool(_CODE_FENCE.search(text)),
        has_html_tag=bool(_HTML_TAG.search(text)),

        first_hit_offset_ratio=offset_ratio,
        payload_share=payload_share,
        payload_region=region,
    )
