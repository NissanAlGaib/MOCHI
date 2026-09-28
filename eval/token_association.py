"""Which tokens are associated with the malicious class, and is it significant?

    python eval/token_association.py --data data/clean
    python eval/token_association.py --data data/clean --top 40 --json reports/tokens.json

Answers the adviser question "relationship of the tokens to the target variable"
with the test that actually fits the data shape.

**Choosing the tool.** Both variables are categorical and binary - a token is
present or absent, a sample is benign or malicious - so the relationship lives in
a 2x2 contingency table:

                     malicious   benign
        token present    a          b
        token absent     c          d

That makes the **chi-square test of independence** the first appropriate tool.
It is not a correlation problem: Pearson's r assumes two continuous variables,
and the point-biserial variant still needs one continuous side. Both would be
the wrong instrument here.

Three refinements matter and are applied:

* **Yates' continuity correction is not used.** With tens of thousands of rows
  the expected counts sit far above 5, where the correction is known to be
  over-conservative. Fisher's exact test is substituted instead whenever any
  expected cell count falls below 5.
* **Multiple comparisons.** Testing ~10,000 tokens at alpha = 0.05 produces ~500
  false positives by construction. Benjamini-Hochberg FDR control is applied
  across the whole token set, and the reported q-values are what should be cited
  rather than the raw p-values.
* **Effect size.** Significance is not importance. At n = 82,765 almost any token
  clears p < 0.05, so Cramer's V and the log-odds ratio carry the actual finding.

Mutual information is reported alongside as a ranking statistic. It is not a
significance test and has no p-value; it measures how much knowing whether the
token is present reduces uncertainty about the label.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import DATA_DIR, DatasetError, load_directory  # noqa: E402

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Synonyms folded to one canonical token before association testing, opt-in
#: via ``tokenize(..., fold_synonyms=True)``.
#:
#: Two things already cover verb synonymy elsewhere in this project, and this
#: dict is deliberately *not* a third, redundant copy of either:
#:
#: * ``patterns.json``'s Stage I regexes already alternate over close synonyms
#:   directly - ``\b(?:print|reveal|show|display|output|repeat)\s+...`` - so a
#:   canonicalisation pass would mostly just re-derive what detection-time
#:   matching already does.
#: * ``mochi.preprocess.features.INSTRUCTION_VERBS`` counts any of its 34 verbs
#:   equally toward ``imperative_verb_count`` - the feature does not care
#:   *which* synonym appeared, only that one did, so folding would not change
#:   that count either.
#:
#: What neither of those does is what this project's own A9/A10 corpus-bias
#: work (``docs/BUILD_PLAN.md``) already found unigrams do badly: without
#: folding, "ignore", "disregard", and "forget" each collect their own,
#: individually noisier evidence and their own individually-corrected q-value,
#: fragmenting one concept's support across three rows of the results table.
#: Folding pools that evidence into one canonical token before the contingency
#: table is built, the same reasoning ``tokenize``'s own n-gram mode already
#: applies to *phrase* fragmentation, applied here to *lexical* fragmentation
#: instead.
#:
#: Every canonical form (the dict's values) is a member of ``INSTRUCTION_VERBS``
#: - ``tests/test_token_association.py`` pins this, so the two vocabularies
#: cannot silently diverge. Only close synonyms of that existing, already-
#: adopted list are added here; common, low-specificity verbs
#: ("show", "tell", "run", "send") are deliberately excluded even where a
#: near-synonym relationship exists, because folding a word that carries almost
#: no signal on its own into one that does would manufacture significance
#: rather than reveal it.
HAND_BUILT_SYNONYMS: dict[str, str] = {
    # ignore - "disregard" and "forget" are themselves existing
    # INSTRUCTION_VERBS members, not new additions: they are the exact
    # motivating example above, folded here rather than left to fragment
    # their own evidence the way the module docstring on tokenize() describes.
    "disregard": "ignore", "forget": "ignore",
    "discard": "ignore", "dismiss": "ignore", "neglect": "ignore",
    "overlook": "ignore", "abandon": "ignore",
    # reveal - "disclose" is likewise an existing INSTRUCTION_VERBS member
    # folded here as a near-perfect synonym; "show"/"print"/"output"/"repeat"/
    # "echo"/"dump" are left independent even though related, because each
    # carries a distinct connotation (rendering vs restating vs bulk export)
    # that patterns.json's own separate alternations already treat differently.
    "disclose": "reveal",
    "expose": "reveal", "divulge": "reveal", "uncover": "reveal",
    "leak": "reveal", "unveil": "reveal",
    # bypass - "override" is left independent: patterns.json's own regexes
    # show it targeting a different object ("override ... instruction") than
    # bypass/disable ("bypass ... filter/restriction"), so treating them as
    # one concept would blur a distinction the detector itself preserves.
    "circumvent": "bypass", "evade": "bypass", "sidestep": "bypass",
    "subvert": "bypass", "workaround": "bypass",
    # delete - "remove" is left independent: too common and low-specificity
    # in ordinary text on its own to fold safely (the same reasoning that
    # excludes "show"/"tell"/"run").
    "erase": "delete", "wipe": "delete", "purge": "delete",
    # disable
    "deactivate": "disable", "unplug": "disable",
    # pretend
    "impersonate": "pretend", "masquerade": "pretend", "imitate": "pretend",
    # execute - "run" is left independent for the same common-word reason as
    # "remove" above.
    "invoke": "execute", "trigger": "execute", "launch": "execute",
    # obey - "comply" is an existing INSTRUCTION_VERBS member, folded for the
    # same reason as "disregard"/"forget"/"disclose" above. "follow" is left
    # independent: too common and low-specificity on its own to fold safely.
    "comply": "obey", "adhere": "obey", "conform": "obey",
}


def _load_synonyms() -> dict[str, str]:
    """The WordNet-derived map when it has been generated, else the hand list.

    ``eval/wordnet_synonyms.py`` writes ``data/features/synonym_map.json`` from
    WordNet verb synsets of ``INSTRUCTION_VERBS``. That file is what should be
    in force: it is eight times larger, it carries phrasal verbs the hand list
    could not express, and "WordNet 3.0 verb synsets" is a method a panellist
    can check while a hand-written dictionary is not.

    The hand-built list survives as the fallback so a fresh checkout - where
    ``data/`` is gitignored and nltk may not be installed - still folds
    something sensible rather than silently folding nothing. Which one is in
    force is reported by :data:`SYNONYM_SOURCE`.
    """
    path = (Path(__file__).resolve().parents[1] / "data" / "features"
            / "synonym_map.json")
    if not path.exists():
        return dict(HAND_BUILT_SYNONYMS)
    mapping = json.loads(path.read_text(encoding="utf-8"))["mapping"]
    # The hand list wins on conflict: its entries were chosen against this
    # corpus, WordNet's against English in general.
    return _resolve_chains({**mapping, **HAND_BUILT_SYNONYMS})


def _resolve_chains(mapping: dict[str, str]) -> dict[str, str]:
    """Collapse every mapping onto its terminal base form.

    Merging the two sources creates chains the hand list alone never had:
    WordNet gives ``abide by -> comply`` while the hand list gives
    ``comply -> obey``, so "abide by" would land on "comply" or "obey" depending
    on which entry a lookup happened to hit first. Folding has to be
    order-independent or two sentences using the same concept still fragment
    into different tokens - the exact failure folding exists to prevent.

    Following each chain to its end makes ``abide by -> obey`` directly, and
    restores the invariant ``tests/test_token_association.py`` pins: no term is
    both a key and a value.

    A cycle would loop forever, so the walk is bounded and leaves any term it
    cannot terminate pointing at its immediate base - wrong, but bounded, and
    visible as a failing disjointness test rather than a hang.
    """
    resolved: dict[str, str] = {}
    for term, base in mapping.items():
        seen = {term}
        while base in mapping and base not in seen:
            seen.add(base)
            base = mapping[base]
        resolved[term] = base
    return {term: base for term, base in resolved.items() if term != base}


SYNONYM_TO_BASE: dict[str, str] = _load_synonyms()

#: Which map is in force, for the fit script's header and the thesis method.
SYNONYM_SOURCE = ("wordnet" if len(SYNONYM_TO_BASE) > len(HAND_BUILT_SYNONYMS)
                  else "hand-built")

#: Multi-word entries, longest first, so ``pay no attention`` is tried before
#: any two-word prefix of it. Built once at import: the scan runs per row over
#: tens of thousands of rows.
SYNONYM_PHRASES: list[tuple[tuple[str, ...], str]] = sorted(
    ((tuple(term.split()), base)
     for term, base in SYNONYM_TO_BASE.items() if " " in term),
    key=lambda item: -len(item[0]),
)

#: Words that turn a bare instruction verb into an actual instruction-
#: manipulation phrase, rather than an ordinary use of the same verb. "ignore"
#: alone appears in narrative prose all the time ("she decided to ignore the
#: previous warnings about the cursed necklace") - a bare-word weight cannot
#: tell that apart from "ignore all previous instructions", and 234 benign
#: training rows containing the word "ignore" prove it is a real, not
#: theoretical, false-positive source.
#:
#: This is the same distinction ``patterns.json``'s own regexes already
#: enforce with wildcards - ``\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|...)
#: \s+(?:instruction|rule|...)`` - generalised into one shared word list rather
#: than re-derived per verb family. Drawn directly from the object-noun and
#: qualifier alternations already used across ``patterns.json``'s direct/
#: indirect-injection and jailbreak detectors, so this does not invent a new
#: notion of "instruction-like" - it reuses the one Stage I already encodes.
INSTRUCTION_CONTEXT_WORDS: frozenset[str] = frozenset({
    # the object being targeted
    "instruction", "instructions", "direction", "directions", "directive",
    "directives", "command", "commands", "prompt", "prompts", "rule", "rules",
    "guideline", "guidelines", "guidance", "restriction", "restrictions", "policy",
    "policies", "filter", "filters", "constraint", "constraints", "setting",
    "settings", "safety", "protocol", "protocols", "system", "training",
    "programming", "message", "guardrail", "guardrails",
    # which one - "previous"/"prior" alone carry most of the signal even when
    # the object noun itself sits outside the window ("ignore what I told you
    # before" has no object noun at all, but "before" still marks the phrase)
    "previous", "prior", "above", "earlier", "preceding", "foregoing",
    "original", "initial", "before",
})

#: Words checked on either side of a candidate verb. 5 covers "ignore all of
#: the previous instructions" (verb to object noun across three intervening
#: words) without growing wide enough to catch an unrelated mention two
#: sentences away - the regex patterns this generalises stay within a
#: similarly short span themselves.
INSTRUCTION_CONTEXT_WINDOW = 5

#: Tokens rarer than this are dropped before testing. Rare tokens give unstable
#: odds ratios and inflate the multiple-comparison burden without contributing
#: usable evidence.
MIN_DOC_FREQ = 30

#: Expected cell count below which chi-square is unreliable and Fisher's exact
#: test is substituted.
MIN_EXPECTED = 5

#: Benjamini-Hochberg target false discovery rate.
FDR = 0.05


@dataclass
class TokenResult:
    token: str
    n_malicious: int
    n_benign: int
    chi2: float
    p_value: float
    q_value: float
    cramers_v: float
    log_odds: float
    mutual_info: float
    test: str

    @property
    def direction(self) -> str:
        return "malicious" if self.log_odds > 0 else "benign"


def tokenize(text: str, *, ngram_max: int = 1,
             fold_synonyms: bool = False) -> set[str]:
    """Presence set of n-grams, not counts - the table is about occurrence.

    ``ngram_max=1, fold_synonyms=False`` reproduces the original unigram
    behaviour exactly, so any figure already cited from a previous run stays
    reproducible - folding is opt-in specifically so it never silently changes
    a number someone already quoted.

    Unigrams alone mislead here, and the existing results show how. Four of the
    strongest associations - ``instructions`` (V=0.291), ``reveal`` (0.259),
    ``ignore`` (0.240), ``previous`` (0.211) - are fragments of the *same phrase*,
    counted as four independent findings. And no unigram model can separate
    "ignore previous instructions" from "ignore the previous email": the
    distinction lives entirely in the adjacency, which is exactly what Stage I
    hand-codes as ``ignore ... previous <instruction|rule|...>``.

    ``fold_synonyms`` addresses the same fragmentation from the other side:
    not a phrase split into unigrams, but one concept split across near-
    synonymous unigrams ("ignore" / "disregard" / "forget"). Applied to words
    before n-grams are built, so a folded word inside a multi-word gram is
    canonicalised too, e.g. ``ngram_max=2`` sees "ignore previous" from either
    "ignore previous" or "disregard previous" alike. See :data:`SYNONYM_TO_BASE`.
    """
    words = [match.group(0).lower() for match in TOKEN.finditer(text)]
    if fold_synonyms:
        words = _fold(words)
    if ngram_max <= 1:
        return set(words)

    grams: set[str] = set(words)
    for size in range(2, ngram_max + 1):
        for start in range(len(words) - size + 1):
            grams.add(" ".join(words[start:start + size]))
    return grams


def _fold(words: list[str]) -> list[str]:
    """Fold synonyms to canonical verbs, **phrases before single words**.

    Returns a list the same length as ``words`` so the proximity window's
    indices still line up with the original text. A matched phrase puts the
    canonical verb at its first position and leaves the remaining positions as
    they were - those trailing words are ordinary context either way, and
    dropping them would shift every later index and silently move the window.

    Phrase-first ordering is the point. ``brush aside`` appears 185 times in the
    training corpus and is an exact phrasal synonym of ``ignore``; matching
    ``brush`` and ``aside`` separately finds neither.
    """
    folded = [SYNONYM_TO_BASE.get(word, word) for word in words]

    for phrase, base in SYNONYM_PHRASES:
        width = len(phrase)
        for index in range(len(words) - width + 1):
            if tuple(words[index:index + width]) == phrase:
                folded[index] = base
    return folded


def instruction_verbs_in_context(text: str) -> set[str]:
    """Canonical instruction verbs that appear near an instruction-object word.

    Distinct from ``tokenize(text, fold_synonyms=True)`` in exactly one way -
    the one that matters here: a bare verb no longer counts on its own.
    "ignore" in "she decided to ignore the previous warnings" and "ignore" in
    "ignore all previous instructions" are the same token to ``tokenize()``,
    but only the second is what :data:`SYNONYM_TO_BASE` and
    ``INSTRUCTION_VERBS`` exist to catch - see :data:`INSTRUCTION_CONTEXT_WORDS`.

    This is a **proximity window, not an n-gram**. An n-gram needs the verb and
    its object adjacent ("ignore previous"), which "ignore *the* previous"
    already breaks; a window catches both without needing ``ngram_max`` pushed
    high enough to absorb every stopword variant Stage I's own regexes
    already tolerate with a wildcard.

    Used identically by :func:`fit_instruction_verb_weights` (via ``analyse``'s
    ``token_fn``) and :func:`score_malicious_word_weight`, so a verb's fit
    weight and its applied weight are always measured under the same rule.
    """
    from mochi.preprocess.features import INSTRUCTION_VERBS

    words = [match.group(0).lower() for match in TOKEN.finditer(text)]
    canonical = _fold(words)

    found: set[str] = set()
    for i, word in enumerate(canonical):
        if word not in INSTRUCTION_VERBS or word in found:
            continue
        lo = max(0, i - INSTRUCTION_CONTEXT_WINDOW)
        hi = min(len(words), i + INSTRUCTION_CONTEXT_WINDOW + 1)
        window = words[lo:i] + words[i + 1:hi]
        if any(w in INSTRUCTION_CONTEXT_WORDS for w in window):
            found.add(word)
    return found


def contingency(a: int, b: int, c: int, d: int):
    """Chi-square (or Fisher when sparse) plus effect sizes for one 2x2 table."""
    from scipy.stats import chi2_contingency, fisher_exact

    table = [[a, b], [c, d]]
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    expected_min = min(row1 * col1, row1 * col2, row2 * col1, row2 * col2) / n

    if expected_min < MIN_EXPECTED:
        _, p = fisher_exact(table)
        chi2 = float("nan")
        test = "fisher"
    else:
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        test = "chi2"

    # For a 2x2 table Cramer's V reduces to phi = sqrt(chi2 / n).
    v = math.sqrt(chi2 / n) if chi2 == chi2 else float("nan")

    # Haldane-Anscombe correction: add 0.5 to every cell so an empty one does
    # not send the odds ratio to infinity.
    log_odds = math.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
    return chi2, p, v, log_odds, test


def mutual_information(a: int, b: int, c: int, d: int) -> float:
    """Mutual information in bits between token presence and the label."""
    n = a + b + c + d
    total = 0.0
    for observed, row, col in ((a, a + b, a + c), (b, a + b, b + d),
                               (c, c + d, a + c), (d, c + d, b + d)):
        if observed:
            total += (observed / n) * math.log2((observed * n) / (row * col))
    return total


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH-adjusted q-values, returned in the input order."""
    m = len(p_values)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    previous = 1.0
    for rank, index in enumerate(reversed(order), start=1):
        position = m - rank + 1
        value = min(previous, p_values[index] * m / position)
        q[index] = value
        previous = value
    return q


def analyse(samples, *, min_freq: int = MIN_DOC_FREQ,
            ngram_max: int = 1, fold_synonyms: bool = False,
            vocabulary: set[str] | None = None,
            token_fn=None,
            ) -> tuple[list[TokenResult], dict]:
    """Test every token (or only ``vocabulary``, if given) against the label.

    ``vocabulary`` restricts *which hypotheses are tested at all* - not a
    post-hoc filter applied to a full run. That distinction matters for
    Benjamini-Hochberg: it corrects across whatever ``results`` ends up
    holding, so testing the full ~10,000-token corpus and filtering the
    output down to a handful of words of interest afterward would correct
    across all 10,000 hypotheses and understate how significant those few
    words actually are. Passing a small ``vocabulary`` up front corrects only
    across that vocabulary instead - see
    :func:`fit_instruction_verb_weights`, the caller this exists for.

    ``token_fn``, when given, replaces the ``tokenize(text, ngram_max=...,
    fold_synonyms=...)`` call entirely - it takes the raw text and returns
    whatever set of tokens should be counted as present, e.g.
    :func:`instruction_verbs_in_context`. ``ngram_max``/``fold_synonyms`` are
    ignored when ``token_fn`` is supplied, since the replacement function
    decides tokenisation on its own terms.
    """
    malicious_docs = benign_docs = 0
    malicious_count: Counter = Counter()
    benign_count: Counter = Counter()

    for sample in samples:
        tokens = (token_fn(sample.text) if token_fn is not None else
                 tokenize(sample.text, ngram_max=ngram_max, fold_synonyms=fold_synonyms))
        if vocabulary is not None:
            tokens &= vocabulary
        if sample.label == 1:
            malicious_docs += 1
            malicious_count.update(tokens)
        else:
            benign_docs += 1
            benign_count.update(tokens)

    candidates = vocabulary if vocabulary is not None else (
        set(malicious_count) | set(benign_count)
    )
    vocabulary_tested = [
        token for token in candidates
        if malicious_count[token] + benign_count[token] >= min_freq
    ]

    results: list[TokenResult] = []
    for token in vocabulary_tested:
        a = malicious_count[token]
        b = benign_count[token]
        c = malicious_docs - a
        d = benign_docs - b
        chi2, p, v, log_odds, test = contingency(a, b, c, d)
        results.append(TokenResult(
            token=token, n_malicious=a, n_benign=b, chi2=chi2, p_value=p,
            q_value=0.0, cramers_v=v, log_odds=log_odds,
            mutual_info=mutual_information(a, b, c, d), test=test,
        ))

    for result, q in zip(results, benjamini_hochberg([r.p_value for r in results])):
        result.q_value = q

    meta = {
        "n_samples": malicious_docs + benign_docs,
        "n_malicious": malicious_docs,
        "n_benign": benign_docs,
        "vocabulary_tested": len(vocabulary_tested),
        "min_doc_freq": min_freq,
        "ngram_max": ngram_max,
        "fold_synonyms": fold_synonyms,
        "significant_at_fdr": sum(1 for r in results if r.q_value < FDR),
        "fdr": FDR,
        "fisher_substitutions": sum(1 for r in results if r.test == "fisher"),
    }
    return results, meta


#: Every significant verb's weight is linearly rescaled to fall in
#: ``[WEIGHT_MIN, WEIGHT_MAX]`` - a small, directly-readable severity score,
#: not a fixed-point stand-in for the underlying log-odds the way
#: ``mochi.preprocess.features.RATIO_SCALE`` is for a 0-1 fraction. A verb with
#: no measured malicious association still gets ``0``, outside this range on
#: purpose - "no evidence" and "the weakest evidence we found" are different
#: claims, and only the second should read as ``WEIGHT_MIN``.
#:
#: **This scale is relative to whichever verbs are currently significant, not
#: an absolute unit.** Rescaling depends on the current min and max log-odds
#: across the significant verbs, so refitting on an updated corpus - a
#: different verb becoming the strongest, or the weakest dropping out - shifts
#: what every other weight means, not just the one that changed. Accepted
#: trade-off for a score simple enough to read at a glance; not a property to
#: rely on across two different fits of this table.
WEIGHT_MIN = 1
WEIGHT_MAX = 10


def fit_instruction_verb_weights(train_samples, *, min_freq: int = 5,
                                 ) -> tuple[dict[str, int], list[TokenResult]]:
    """Corpus-measured weight for each canonical instruction verb.

    **``train_samples`` must already be the train split, and nothing else.**
    This function does not check that itself - it has no way to, since a list
    of ``Sample`` carries no split information - so the caller is the only
    thing standing between this and leakage. It exists specifically because
    ``mochi/preprocess/features.py``'s own docstring rules this kind of value
    out of ``FeatureVector``: "nothing here is fitted on the corpus... putting
    [a corpus-fitted feature] here would make [fitting on train only]
    impossible to enforce - the extractor has no idea which split it is
    looking at." A weight derived from log-odds is exactly the corpus-fitted
    feature that warning describes, so it is fit here, once, by a caller that
    *does* know which split it is looking at, and then applied - never
    refit - to every row a later step scores.

    Counted via :func:`instruction_verbs_in_context`, not a bare
    ``tokenize(..., fold_synonyms=True)`` presence check: a verb only counts as
    "present" when an instruction-object word sits within
    :data:`INSTRUCTION_CONTEXT_WINDOW` words of it. Without that, "ignore"
    counts identically in "ignore all previous instructions" and in "she
    decided to ignore the previous warnings" - 234 benign training rows
    contain the bare word for exactly that reason. Folding still applies
    inside that function, so a synonym's evidence (``disregard``, ``expose``,
    ``comply``, ...) pools into its canonical verb's weight the same as
    before - only the presence *test* changed, not the vocabulary.

    Vocabulary is additionally restricted to
    :data:`mochi.preprocess.features.INSTRUCTION_VERBS` (belt and braces -
    ``instruction_verbs_in_context`` only ever returns members of that set
    already, but restricting explicitly here is what keeps the Benjamini-
    Hochberg correction honest regardless - see :func:`analyse`).

    A verb's weight is *only* nonzero if that verb is significantly malicious-
    associated (``log_odds > 0`` and ``q_value < FDR``); otherwise its weight
    is ``0``. A verb present in a prompt with no measured malicious
    association contributes nothing to the sum a weight feeds - "present but
    unproven" is not the same claim as "present and evidenced", and only the
    second should move a score.

    Every significant verb's raw log-odds is then linearly rescaled into
    ``[WEIGHT_MIN, WEIGHT_MAX]`` - the weakest significant verb becomes
    ``WEIGHT_MIN``, the strongest becomes ``WEIGHT_MAX``, everything else maps
    proportionally between. Rescaling (not just rounding) happens here, once,
    at fit time - see :data:`WEIGHT_MAX` for what that trades away - so every
    later consumer of this weight table sums plain, small ints and never has
    to know a raw log-odds value or a scale factor exists.
    """
    from mochi.preprocess.features import INSTRUCTION_VERBS

    results, _meta = analyse(train_samples, min_freq=min_freq,
                             token_fn=instruction_verbs_in_context,
                             vocabulary=set(INSTRUCTION_VERBS))

    significant = {
        result.token: result.log_odds for result in results
        if result.log_odds > 0 and result.q_value < FDR
    }

    weights: dict[str, int] = {}
    if significant:
        lo_min, lo_max = min(significant.values()), max(significant.values())
        span = lo_max - lo_min
        for verb, log_odds in significant.items():
            if span == 0:
                # Every significant verb tied exactly - nothing distinguishes
                # "strongest" from "weakest" here, so none of them earns more
                # than the floor of the significant range.
                weights[verb] = WEIGHT_MIN
            else:
                fraction = (log_odds - lo_min) / span
                weights[verb] = round(WEIGHT_MIN + fraction * (WEIGHT_MAX - WEIGHT_MIN))

    # A verb too rare to test at all, or not significant, still gets an
    # explicit 0 rather than being silently absent - a scoring function doing
    # weights.get(word, 0) would behave the same either way, but an explicit
    # entry is reviewable in the written-out file and an accidental absence
    # is not.
    for verb in INSTRUCTION_VERBS:
        weights.setdefault(verb, 0)

    return weights, results


#: The Track A columns that need a corpus-fitted weight table as an extra
#: input, on top of the text. Deliberately **not** added to
#: ``mochi.preprocess.features.TRACK_A_FEATURES`` - every entry in that tuple
#: is a pure function of one text, by that module's own explicit rule
#: ("nothing here is fitted on the corpus"), and a weight from
#: ``fit_instruction_verb_weights`` is exactly the corpus-fitted value that
#: rule exists to keep out.
#:
#: The full Track A input a model actually trains on is
#: ``mochi.preprocess.features.TRACK_A_FEATURES + TRACK_A_FITTED_FEATURES``
#: (17 + 1 = 18 columns) - two lists in two layers, not one list pretending
#: both kinds of column are the same kind of thing. ``eval/build_features.py``
#: and ``eval/baseline_models.py``'s ``engineered_transformer()`` are the two
#: places that actually join them.
#:
#: No ``_x100``/``_x10k``-style suffix, unlike ``mochi.preprocess.features``'s
#: scaled float fields: this value is not a fixed-point stand-in you would
#: ever divide back down to recover a "real" number - the 1-10 rescale in
#: :func:`fit_instruction_verb_weights` produces the final severity score
#: directly, so the materialised column already holds exactly what it means.
TRACK_A_FITTED_FEATURES: tuple[str, ...] = (
    "malicious_word_weight_sum",
)


def score_malicious_word_weight(text: str, weights: dict[str, int]) -> int:
    """Sum of fit weights for every canonical instruction verb found in ``text``.

    Applies weights already fit by :func:`fit_instruction_verb_weights` - this
    function fits nothing itself and is safe to call on train or test rows
    alike, which is the entire point: the leakage rule is about *fitting*, not
    about *applying*, and a fixed lookup applied identically to every row
    leaks nothing back about which split a row belongs to.

    A **plain sum of small ints**, deliberately: each weight is already
    rescaled into ``[WEIGHT_MIN, WEIGHT_MAX]`` at fit time, not scaled here.
    Doing the rescale once at fit time, rather than combining raw log-odds
    floats here and rescaling the total, means no caller of this function -
    not ``eval/build_features.py``, not ``engineered_transformer()`` - needs
    to know the rescale exists at all; the weight table is already in its
    final, materialisable form the moment it is fit.

    Counts presence via :func:`instruction_verbs_in_context` - the same
    proximity-window rule the weights were fit under - not a bare
    ``tokenize(..., fold_synonyms=True)`` presence check. Scoring "ignore" by
    bare presence when its weight was fit by windowed presence would apply a
    number to a claim ("this verb is being used to manipulate instructions")
    the text was never actually checked for.
    """
    verbs = instruction_verbs_in_context(text)
    return sum(weights.get(verb, 0) for verb in verbs)


def ablation(samples, *, min_freq: int, sizes=(1, 2, 3)) -> None:
    """Does adding n-grams actually find better indicators?

    **Cramer's V is the wrong lens for this question and is reported anyway, so
    the reader can see why.** V is a symmetric association measure: it penalises
    rarity, and every bigram is rarer than its parts. So V does not rise with n,
    and read alone it says n-grams add nothing.

    What actually changes is *contamination* - the share of a term's occurrences
    that sit on benign text. That is the quantity a detector cares about, because
    it is the false-positive rate the term would produce if used as a rule.

        previous              4,184 malicious /  746 benign   (15.1% contaminated)
        previous instructions 2,439 malicious /    9 benign   ( 0.4% contaminated)

    Same phrase family, a 40x cleaner indicator, and a *lower* V. Reporting V
    alone would have hidden the entire finding.
    """
    print()
    print("=" * 100)
    print("  N-gram ablation  -  does context find better indicators?")
    print("=" * 100)
    header = (f"  {'n-gram range':<14}{'tested':>9}{'significant':>13}"
              f"{'max V':>8}{'multiword':>12}{'benign contamination':>23}")
    print(header)
    print("  " + "-" * 96)

    for size in sizes:
        results, meta = analyse(samples, min_freq=min_freq, ngram_max=size)
        significant = [r for r in results if r.q_value < FDR]
        strongest = sorted(
            (r for r in significant if r.cramers_v == r.cramers_v),
            key=lambda r: r.cramers_v, reverse=True,
        )[:50]
        multiword = sum(1 for r in strongest if " " in r.token)
        peak = strongest[0].cramers_v if strongest else float("nan")

        # Mean share of each strong term's occurrences that are benign. This is
        # the false-positive rate the term would produce as a standalone rule.
        contamination = [
            r.n_benign / (r.n_malicious + r.n_benign)
            for r in strongest if r.log_odds > 0 and (r.n_malicious + r.n_benign)
        ]
        mean_contamination = sum(contamination) / len(contamination) if contamination else 0.0

        label = "1 (unigram)" if size == 1 else f"1-{size}"
        print(f"  {label:<14}{meta['vocabulary_tested']:>9,}"
              f"{len(significant):>13,}{peak:>8.3f}"
              f"{multiword:>9}/50{mean_contamination:>22.1%}")

    print("  " + "-" * 96)
    print("  Read the last column, not max V. V penalises rarity, so it cannot")
    print("  rise with n; contamination is the false-positive rate a term would")
    print("  produce if promoted to a Stage I rule, and that is what improves.")
    print("=" * 100)
    print()


def report(results: list[TokenResult], meta: dict, *, top: int) -> None:
    significant = [r for r in results if r.q_value < FDR]
    by_effect = sorted(significant, key=lambda r: abs(r.log_odds), reverse=True)
    malicious = [r for r in by_effect if r.log_odds > 0][:top]
    benign = [r for r in by_effect if r.log_odds < 0][:top]

    print()
    print("=" * 94)
    print("  Token / Class Association  -  chi-square test of independence")
    print("=" * 94)
    print(f"  samples {meta['n_samples']:,}    malicious {meta['n_malicious']:,}"
          f"    benign {meta['n_benign']:,}")
    print(f"  tokens tested (doc freq >= {meta['min_doc_freq']}): "
          f"{meta['vocabulary_tested']:,}"
          f"    Fisher substitutions: {meta['fisher_substitutions']:,}")
    if meta.get("fold_synonyms"):
        print(f"  synonyms folded to canonical form: "
              f"{len(SYNONYM_TO_BASE)} mappings (see SYNONYM_TO_BASE)")
    print(f"  significant after Benjamini-Hochberg at FDR {meta['fdr']}: "
          f"{meta['significant_at_fdr']:,} "
          f"({meta['significant_at_fdr'] / max(meta['vocabulary_tested'], 1):.1%})")
    print()

    header = (f"  {'token':<20}{'mal':>8}{'ben':>8}{'chi2':>11}"
              f"{'q':>11}{'V':>8}{'logOR':>9}{'MI':>9}")
    for title, rows in (("Most associated with MALICIOUS", malicious),
                        ("Most associated with BENIGN", benign)):
        print(f"  {title}")
        print(header)
        print("  " + "-" * 90)
        for r in rows:
            chi = f"{r.chi2:,.0f}" if r.chi2 == r.chi2 else "fisher"
            print(f"  {r.token:<20}{r.n_malicious:>8,}{r.n_benign:>8,}{chi:>11}"
                  f"{r.q_value:>11.2e}{r.cramers_v:>8.3f}"
                  f"{r.log_odds:>+9.2f}{r.mutual_info:>9.4f}")
        print()

    strong = [r for r in significant
              if r.cramers_v == r.cramers_v and r.cramers_v >= 0.1]
    print(f"  Effect size: {len(strong):,} of {len(significant):,} significant tokens "
          f"reach Cramer's V >= 0.10 (a small effect).")
    print("  At this n, significance is cheap - effect size carries the finding.")
    print("=" * 94)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-freq", type=int, default=MIN_DOC_FREQ)
    parser.add_argument("--ngram-max", type=int, default=1,
                        help="largest n-gram to test; 1 (default) reproduces "
                             "the original unigram report exactly")
    parser.add_argument("--ablation", action="store_true",
                        help="compare 1 / 1-2 / 1-3 instead of a single report")
    parser.add_argument("--fold-synonyms", action="store_true",
                        help="pool close synonyms (disregard/forget -> ignore) "
                             "into one canonical token before testing; off by "
                             "default so existing cited figures stay reproducible "
                             "- see SYNONYM_TO_BASE")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    try:
        samples = load_directory(args.data)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.ablation:
        ablation(samples, min_freq=args.min_freq)
        return 0

    results, meta = analyse(samples, min_freq=args.min_freq,
                            ngram_max=args.ngram_max,
                            fold_synonyms=args.fold_synonyms)
    report(results, meta, top=args.top)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        ranked = sorted(results, key=lambda r: abs(r.log_odds), reverse=True)
        args.json.write_text(
            json.dumps({"meta": meta, "tokens": [asdict(r) for r in ranked[:500]]},
                       indent=2),
            encoding="utf-8",
        )
        print(f"  Wrote {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
