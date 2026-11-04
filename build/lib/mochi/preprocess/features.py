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

**The injection-lexicon family costs nothing.** It is a direct transcription of
``Stage1Result``, already computed on every request and then discarded. Writing
it to a dataset is bookkeeping, not new analysis.

**The obfuscation family - what ``NormalizationResult.flags`` recorded having to
be undone - is deliberately not a column here.** Revealing and decoding
obfuscation is the normalization layer's job (Phase 3), and it has already run
by the time any classifier sees the text: a base64-wrapped payload is decoded
into a scannable variant, homoglyphs are folded, zero-width characters are
stripped. What "was something obfuscated" would add on top of that is not a
description of the prompt but a description of the preprocessing step - and it
measured out negligible on this corpus in every case tried, because the corpora
in hand carry very little obfuscation. The flags remain fully computed and
exposed on ``NormalizationResult`` for telemetry and Stage I; they are simply
never promoted to a feature column.

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

#: Fixed-point scale for every float field, applied only in ``as_dict()``.
#: ``0.0342`` becomes ``342`` - four decimal digits of precision survive, far
#: finer than any effect size this project has measured (the smallest reported
#: in ``docs/CLASSIFICATION_PLAN.md`` is two decimal places). Every classical
#: model in Track A is invariant to it: a decision tree's threshold splits do
#: not care about a constant positive rescaling, and the SVM step in
#: ``eval/baseline_models.py`` standardises its inputs anyway, which undoes any
#: fixed multiplier exactly. The scale is named in the column, not left
#: implicit - ``instruction_verb_ratio_x10k`` reads as "this is the ratio times
#: 10,000", not as the ratio itself.
RATIO_SCALE = 10_000

#: Every ``float``-typed field on ``FeatureVector``. ``as_dict()`` scales and
#: renames each of these with an ``_x10k`` suffix rather than emitting a float.
#: ``tests/test_features.py`` asserts this list still matches every ``float``
#: field on the dataclass, so a new float field added later cannot reach the
#: materialised dataset unscaled without the test failing first.
FLOAT_FIELDS: tuple[str, ...] = (
    "avg_word_len",
    "uppercase_ratio",
    "digit_ratio",
    "question_ratio",
    "newline_ratio",
    "special_char_ratio",
    "instruction_verb_ratio",
    "english_ratio",
    "tagalog_ratio",
    "max_url_query_entropy",
    "first_hit_offset_ratio",
    "payload_share",
)


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
        """Flatten to one row of plain ints only - nothing else survives.

        The dataclass itself stays human-readable: ``vector.payload_region ==
        "head"`` and ``vector.ends_with_question is True`` keep working
        everywhere a ``FeatureVector`` is used directly, tests included. Every
        categorical and boolean field is expanded or cast here into an int, the
        same way ``detector_hits`` (a dict) is expanded into ``hit_*`` columns -
        so the materialised dataset and any consumer of ``.as_dict()`` never has
        to special-case a string or a Python ``bool``.

        Booleans matter here specifically because CSV has no boolean type:
        ``pandas.to_csv`` writes a ``bool`` column as the literal text
        ``True``/``False``, which is a string in the file regardless of the
        in-memory dtype. Casting to ``0``/``1`` before the row is built means
        the file on disk is what it claims to be.

        * ``stage1_max_severity`` - ordinal int (none=0 < low=1 < medium=2 <
          high=3). One-hot would discard the real ordering.
        * ``dominant_script`` - ``is_latin_script`` boolean. Only Latin-vs-not
          has ever been read; a script name has no natural numeric form.
        * ``detected_language`` - one-hot ``lang_is_*`` over its four values.
          No ordering exists between english/tagalog/mixed/unknown.
        * ``payload_region`` - one-hot ``region_is_*`` over its five values,
          same reasoning.
        * ``stage1_detector_ids`` - dropped, not encoded. It is a
          semicolon-joined string that duplicates the nine ``hit_*`` booleans
          above exactly.
        * Every ``FLOAT_FIELDS`` entry - fixed-point int, scaled by
          ``RATIO_SCALE`` and suffixed ``_x10k`` (``instruction_verb_ratio``
          becomes ``instruction_verb_ratio_x10k``).
        * Every boolean, including the ``hit_*`` / ``is_latin_script`` /
          ``lang_is_*`` / ``region_is_*`` columns produced by the encodings
          above - cast to plain ``0``/``1`` as the final step, so nothing
          upstream of it needs to already know this rule.
        """
        row = asdict(self)
        hits = row.pop("detector_hits")
        for detector_id in DETECTOR_IDS:
            row[f"hit_{detector_id}"] = bool(hits.get(detector_id, False))

        row.pop("stage1_detector_ids")

        severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
        row["stage1_max_severity"] = severity_rank.get(row.pop("stage1_max_severity"), 0)

        row["is_latin_script"] = row.pop("dominant_script") == "latin"

        language = row.pop("detected_language")
        for name in ("english", "tagalog", "mixed", "unknown"):
            row[f"lang_is_{name}"] = language == name

        region = row.pop("payload_region")
        for name in ("none", "head", "middle", "tail", "decoded_only"):
            row[f"region_is_{name}"] = region == name

        for name in FLOAT_FIELDS:
            row[f"{name}_x10k"] = round(row.pop(name) * RATIO_SCALE)

        for key, value in row.items():
            if isinstance(value, bool):
                row[key] = int(value)

        return row


def feature_names() -> list[str]:
    """Column names in materialisation order."""
    return list(FeatureVector().as_dict().keys())


#: The classification study's Step 1c frozen column set - what Track A (SVM,
#: decision tree, random forest) actually trains on. Chosen from the Step 1b
#: association pass over the full 46-column table (``eval/feature_stats.py``,
#: ``docs/CLASSIFICATION_PLAN.md``), not from every column that measured a real
#: effect: the injection-lexicon family (``stage1_*``, ``hit_*``) is excluded
#: entirely by this list rather than run as an ablation, which is what settles
#: the circularity concern raised for Step 2 - a model trained on this set
#: cannot be rediscovering Stage I's own regexes, because none of Stage I's
#: output is in its input. Likewise the URL/entity, position, and script
#: families are excluded regardless of their individual effect sizes.
#:
#: ``is_code_switched`` is kept for construct validity on the A14 language
#: dimension even though it measured not-significant on this corpus - the
#: corpus's Taglish share is small, and the Step 5 Taglish evaluation set is
#: where this column is expected to start doing work. ``question_mark_count``
#: is kept alongside the stronger ``ends_with_question`` because A12 names
#: question marks specifically; the two test the hypothesis at different
#: grains rather than one making the other redundant.
#:
#: Every entry here is numeric or boolean already - ``engineered_transformer()``
#: in ``eval/baseline_models.py`` needs no categorical encoding for this set,
#: which the previous, larger column set did. ``instruction_verb_ratio`` and
#: ``special_char_ratio`` are named with the ``_x10k`` suffix ``as_dict()``
#: gives every float field - see ``FLOAT_FIELDS`` - since the two ratios in
#: this frozen set are int columns in the materialised row, not floats.
#:
#: ``tests/test_features.py`` pins this list the same way ``DETECTOR_IDS`` is
#: pinned: a name that no longer resolves on ``FeatureVector``'s ``as_dict()``
#: output fails loudly, and changing the set at all requires touching this
#: comment.
TRACK_A_FEATURES: tuple[str, ...] = (
    "char_count",
    "word_count",
    "is_code_switched",
    "second_person_count",
    "obligation_count",
    "negation_count",
    "imperative_verb_count",
    "starts_with_imperative",
    "instruction_verb_ratio_x10k",
    "question_mark_count",
    "ends_with_question",
    "colon_count",
    "quote_count",
    "bracket_count",
    "special_char_ratio_x10k",
    "line_count",
    "max_line_len",
)


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
            punctuation features describe what was actually sent, and that is
            only meaningful measured against the pre-normalization original.
        norm: Result of :func:`mochi.preprocess.normalize.normalize`. Computed
            here when omitted, but the pipeline already has one - pass it rather
            than paying twice. Used for the language columns and to hand Stage I
            its scannable text; its ``flags`` are not otherwise read here - see
            the module docstring on why the obfuscation family is not a column.
        stage1: A :class:`~mochi.detect.stage1_syntactic.Stage1Result`, or None.
            Typed loosely on purpose: importing the detector here would make the
            preprocessing package depend on the detection package, and the
            dependency runs the other way everywhere else.
    """
    if norm is None:
        norm = normalize(text)

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

    script, _is_mixed = dominant_script(text)

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
