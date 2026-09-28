"""English/Tagalog content filter: Tagalog translated, everything else stripped.

Opt-in (``MOCHI_ENABLE_TAGALOG_TRANSLATION``), unlike every other function in
:mod:`mochi.preprocess`. Two things distinguish this module from the rest of
Phase 3, both deliberate:

**It rewrites, rather than reveals.** ``normalize.py``'s rule everywhere else is
*reveal, never discard* - a decoded payload is appended as a variant, the
original is kept. A content filter that removes third-language words is a
different kind of operation: nothing is hidden here that needs revealing, there
is a request to translate one language and drop everything that is neither
English nor Tagalog, and that is inherently destructive to whatever it removes.
The output of this module is therefore treated as an *additional scannable
variant* on top of the untouched original - see the module docstring on
:mod:`mochi.detect.pipeline` for where it attaches - never a replacement of the
text forwarded upstream. The forwarded request stays exactly what the caller
sent; only what a detector sees gains this extra, filtered view.

**It carries a real, unavoidable dependency cost.** The only maintained Python
library with a ``tl<->en`` model is Argos Translate, and Argos Translate
declares ``stanza`` as a hard dependency, which in turn requires ``torch``. That
means enabling this feature pulls in the same ~2.5 GB stack Stage II keeps
optional - a Stage-I-only deployment that never trains or loads a semantic
model still doesn't want it, so this module is imported lazily, exactly the way
:class:`mochi.detect.stage2_semantic.E5Scorer` imports torch and transformers
only inside its own ``_load``.

**The curated per-word classifier is not enough on its own, and a first version
of this module proved it.** ``classify_word_language`` in ``normalize.py`` was
built to estimate aggregate Taglish ratios over a corpus, where an unattributed
word merely lowers confidence - it was never meant to gate every single word
for keep-or-strip. Tried directly, it classifies ordinary words like "hello",
"now", and "thanks" as ``"unknown"`` (stripped as if foreign) simply because
they carry no letter or suffix its heuristics look for, while nonsense like
"xyzabc" reads as ``"english"`` because it happens to contain a letter Tagalog
orthography avoids. That is a real, measured precision failure, not a
theoretical one - see ``tests/test_code_switch.py`` for the case that caught it.

The fix is a second, broad-coverage opinion for exactly the words the curated
cascade cannot place: `wordfreq <https://pypi.org/project/wordfreq/>`_, a
frequency dictionary covering English and Filipino (the nearest wordfreq has to
Tagalog) with no torch dependency of its own. A word the cascade calls
``"unknown"`` is looked up in both languages; whichever has the higher
Zipf frequency wins, and a word absent from both is finally, genuinely,
stripped. ``classify_word_language`` still runs first and settles the fast,
already-tested cases - the two Tagalog closed lexicons, the hyphenated
Taglish-verb construction, the curated English list - and only escalates to
``wordfreq`` for what it cannot place. Coverage is real but not exhaustive
either way: the Step 5 Taglish evaluation set (``docs/CLASSIFICATION_PLAN.md``)
is where any residual false-strip rate should be measured before a claim is
made about this filter's behaviour on live traffic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from mochi.preprocess.flags import NormalizationFlag as F
from mochi.preprocess.normalize import (
    _LANG_WORD_RE,
    TAGALOG_PREFIXES,
    TAGALOG_SUFFIXES,
    classify_word_language,
)

#: Cached Tagalog -> English word list, written by
#: ``eval/build_tagalog_lexicon.py``. Absent on a fresh checkout, which is not
#: an error - the neural fallback still works, just slowly.
TAGALOG_LEXICON_PATH = (Path(__file__).resolve().parents[2] / "data"
                        / "features" / "tagalog_lexicon.json")

#: Categories from classify_word_language() that keep a word as English -
#: either it already is, or it is the English half of a hyphenated Taglish verb
#: (mag-upgrade -> keep "upgrade" untranslated, drop the "mag-" grammar marker
#: rather than mistranslate a construction that has no monolingual reading).
_KEEP_AS_ENGLISH = frozenset({"english", "taglish_verb"})

#: Categories that mean "translate this word to English".
_TRANSLATE_TO_ENGLISH = frozenset({"tagalog_frame", "tagalog"})

#: A word must clear this Zipf frequency in at least one language to count as
#: "a real word" rather than noise. 1.0 is wordfreq's own floor for a rare-but-
#: attested word; below it is typically a typo, a proper noun, or gibberish.
WORDFREQ_MIN_ZIPF = 1.0

#: wordfreq has no "tl" code; "fil" (Filipino) is its nearest match and is what
#: it silently substitutes if asked for "tl" directly. Named explicitly here so
#: that substitution is a documented decision, not a warning buried in a log.
_WORDFREQ_TAGALOG_CODE = "fil"


class TranslatorUnavailable(RuntimeError):
    """Argos Translate isn't installed, or its tl<->en package isn't downloaded."""


@dataclass
class CodeSwitchResult:
    """Outcome of filtering one text to English-only content.

    Attributes:
        text: English-only reconstruction - Tagalog words translated, anything
            neither English nor Tagalog removed. A scannable variant, never a
            replacement of what gets forwarded upstream.
        translated: ``(original, translated)`` pairs, one per distinct Tagalog
            word found. Kept for telemetry - being able to show *what* was
            translated is what makes a wrong translation reviewable instead of
            an invisible failure mode.
        stripped: Distinct words removed for matching neither language.
        flags: :class:`NormalizationFlag` values, mirroring the rest of Phase 3.
    """

    text: str
    translated: list[tuple[str, str]] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def strip_affixes(word: str) -> str:
    """Reduce a Tagalog surface form toward its root, longest affix first.

    Tagalog is agglutinative, so ``balewalain``, ``binalewala`` and
    ``magbalewala`` are surface forms of one root. A table keyed on surface
    forms misses all three unless every inflection is enumerated, which is why
    the lookup strips instead of enumerating.

    Longest-first matters: ``nagpa`` must be tried before ``nag``, or the
    shorter prefix matches and leaves ``pa`` fused to the root.

    Best-effort, not a morphological analyser - it can over-strip a word that
    merely starts with the same letters. Acceptable because the result is only
    ever a *second* dictionary key: an over-stripped root misses, and the
    caller falls through to the neural path.
    """
    root = word
    for suffix in sorted(TAGALOG_SUFFIXES, key=len, reverse=True):
        if root.endswith(suffix) and len(root) - len(suffix) >= 3:
            root = root[: -len(suffix)]
            break
    for prefix in sorted(TAGALOG_PREFIXES, key=len, reverse=True):
        if root.startswith(prefix) and len(root) - len(prefix) >= 3:
            root = root[len(prefix):]
            break
    return root


class CodeSwitchTranslator:
    """Lazily-loaded Argos Translate ``tl -> en`` model.

    Mirrors :class:`mochi.detect.stage2_semantic.E5Scorer`: the heavy import
    happens inside :meth:`_load`, not at module import time, so constructing
    this object - which ``mochi/gateway/app.py`` does once at startup, only
    when ``MOCHI_ENABLE_TAGALOG_TRANSLATION`` is set - is what pays the torch
    cost, never importing :mod:`mochi.preprocess`.
    """

    def __init__(self) -> None:
        self._translation = None
        self._wordfreq = None
        self._lexicon: dict[str, str] | None = None

    def _load_wordfreq(self) -> None:
        """Load the frequency dictionary only - no torch, unlike :meth:`_load`.

        Kept separate so a prompt that turns out to need no translation at all
        (every word settles via the fast cascade, or the frequency check itself
        resolves everything to English) never forces the argostranslate/stanza/
        torch stack to load just to answer a lookup that does not need it.
        """
        if self._wordfreq is not None:
            return
        try:
            import wordfreq
        except ImportError as exc:
            raise TranslatorUnavailable(
                "The content filter needs wordfreq (no torch, small install) "
                "to resolve words classify_word_language cannot place on its "
                "own - see the module docstring on why the curated cascade "
                "alone is not enough:\n"
                "  pip install wordfreq\n"
            ) from exc
        self._wordfreq = wordfreq

    def _load(self) -> None:
        """Load the translation model. Heavy - see the module docstring."""
        if self._translation is not None:
            return
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError as exc:
            raise TranslatorUnavailable(
                "Tagalog translation needs argostranslate:\n"
                "  pip install argostranslate\n"
                "This pulls in stanza, which requires torch (~2.5 GB) - the "
                "same cost Stage II already carries. Or run with "
                "MOCHI_ENABLE_TAGALOG_TRANSLATION=false to skip this filter.\n"
            ) from exc

        installed = argostranslate.translate.get_installed_languages()
        tagalog = next((lang for lang in installed if lang.code == "tl"), None)
        english = next((lang for lang in installed if lang.code == "en"), None)
        if tagalog is None or english is None:
            raise TranslatorUnavailable(
                "argostranslate is installed but the tl<->en package is not.\n"
                "Download it once:\n"
                "  python -c \"import argostranslate.package as p; "
                "p.update_package_index(); "
                "pkg = next(x for x in p.get_available_packages() "
                "if x.from_code=='tl' and x.to_code=='en'); "
                "p.install_from_path(pkg.download())\"\n"
            )

        translation = tagalog.get_translation(english)
        if translation is None:
            raise TranslatorUnavailable("No installed tl -> en translation path.")

        # argostranslate splits input into sentences before translating, and
        # its sentencizer for this pair is stanza - which ships no Tagalog
        # model and raises "Language tl is currently unsupported" on the first
        # call. Every input here is one word, so there is nothing to split.
        # Without this the fallback path crashes on the first word the lexicon
        # does not cover, which in a gateway means a failed request.
        class _SingleSentence:
            def split_sentences(self, text: str) -> list[str]:
                return [text]

        try:
            translation.underlying.sentencizer = _SingleSentence()
        except AttributeError:  # pragma: no cover - argos internals moved
            pass
        self._translation = translation

    def _load_lexicon(self) -> None:
        """Read the cached word list; an absent file means an empty lexicon.

        Deliberately not an error - a fresh checkout has no ``data/`` at all,
        and the neural fallback still works.
        """
        if self._lexicon is not None:
            return
        if not TAGALOG_LEXICON_PATH.exists():
            self._lexicon = {}
            return
        self._lexicon = json.loads(
            TAGALOG_LEXICON_PATH.read_text(encoding="utf-8"))["mapping"]

    def lexicon_lookup(self, word: str) -> str | None:
        """Cached translation for ``word``, or None if the table lacks it.

        Exact match first, then the affix-stripped root, so an inflected form
        resolves through its root rather than needing its own entry.
        """
        self._load_lexicon()
        hit = self._lexicon.get(word)
        if hit is not None:
            return hit
        root = strip_affixes(word)
        return self._lexicon.get(root) if root != word else None

    def translate_word(self, word: str) -> str:
        """Translate one Tagalog word to English.

        **The cached lexicon is tried first, and usually answers.** That is what
        makes this callable from the request path: a dict lookup costs
        microseconds, while the fallback below loads argostranslate, which pulls
        stanza and torch and costs tens of milliseconds per call. The A14
        amendment declined translation-before-prediction on exactly that latency
        argument - it applies to the fallback, not to the lookup.

        Grammar is not preserved, by design (see
        ``eval/build_tagalog_lexicon.py``): the consumer is a classifier that
        counts words, not a reader.
        """
        cached = self.lexicon_lookup(word)
        if cached is not None:
            return cached

        # Anything the lexicon does not know gets a frequency check before the
        # translator sees it. ``classify_word_language`` calls a word Tagalog if
        # it merely *starts* with a Tagalog prefix, and English is full of those
        # - "navigate", "national", "material" all begin "na-". Handed one,
        # argostranslate does not decline; it mangles it, turning "navigate"
        # into "vigate" and corrupting text that was never Tagalog.
        #
        # Genuine Tagalog separates cleanly on frequency: "balewalain" is 0.00
        # in English against 3.68 in Filipino, while "navigate" is 3.68 against
        # 3.14. Ties go to leaving the word alone, for the same reason the rest
        # of this module prefers a false keep to a false strip.
        if self._looks_english(word):
            return word

        self._load()
        try:
            return self._translation.translate(word)
        except Exception:
            # The word stays in Tagalog rather than the request failing. This
            # filter runs inline on live traffic and its output is an *extra*
            # scannable variant, never a replacement - so a word it cannot
            # translate costs the detectors nothing they had before, while an
            # exception here would cost the caller their request. Translation
            # is best-effort by construction; only the lookup is guaranteed.
            return word

    def _looks_english(self, word: str) -> bool:
        """Whether frequency data says this word is English, not Tagalog.

        Used as a guard before translation, never for the language *call* -
        ``detect_language`` keeps its own cascade. Returns False when wordfreq
        is unavailable or has no opinion, so a missing dependency loses the
        guard rather than blocking every translation.
        """
        try:
            self._load_wordfreq()
        except Exception:  # pragma: no cover - wordfreq is a light dependency
            return False
        english = self._wordfreq.zipf_frequency(word, "en")
        tagalog = self._wordfreq.zipf_frequency(word, _WORDFREQ_TAGALOG_CODE)
        if english == 0.0 and tagalog == 0.0:
            return False
        return english >= tagalog

    def _resolve_uncertain(self, word: str) -> str:
        """Second opinion for a word ``classify_word_language`` called "unknown".

        Compares Zipf frequency in English against Filipino (wordfreq's nearest
        match for Tagalog) and returns ``"english"``, ``"tagalog"``, or
        ``"unknown"`` if neither clears :data:`WORDFREQ_MIN_ZIPF`. Ties go to
        English, on the grounds that this filter's own false-strip risk (an
        ordinary word wrongly removed) is more disruptive to a legitimate
        request than its false-keep risk (an ordinary word wrongly left
        untranslated, which changes nothing a downstream detector could not
        already see in the original text).
        """
        self._load_wordfreq()
        en_freq = self._wordfreq.zipf_frequency(word, "en")
        tl_freq = self._wordfreq.zipf_frequency(word, _WORDFREQ_TAGALOG_CODE)
        if en_freq < WORDFREQ_MIN_ZIPF and tl_freq < WORDFREQ_MIN_ZIPF:
            return "unknown"
        return "tagalog" if tl_freq > en_freq else "english"

    def filter_and_translate(self, text: str) -> CodeSwitchResult:
        """Rebuild ``text`` as English-only: Tagalog translated, else stripped.

        Runs on the text as sent (matching every other Phase 3 function -
        surface structure describes what arrived), and reuses
        ``classify_word_language`` word-for-word so this filter's decisions
        can never disagree with :func:`mochi.preprocess.normalize.detect_language`
        about which language a given token belongs to.

        Distinct words are translated at most once and cached for the call -
        a prompt repeating the same Tagalog word many times pays for one
        translation, not one per occurrence.
        """
        if not text:
            return CodeSwitchResult(text=text)

        translated_cache: dict[str, str] = {}
        translated_pairs: list[tuple[str, str]] = []
        stripped_words: list[str] = []
        flags: list[str] = []

        pieces: list[str] = []
        cursor = 0

        for match in _LANG_WORD_RE.finditer(text.lower()):
            start, end = match.span()
            original = text[start:end]
            lowered = match.group(0)
            category = classify_word_language(lowered)

            pieces.append(text[cursor:start])
            cursor = end

            # "ambiguous" (at/may/man) is spelled identically in both
            # languages and is ordinary English on its own - detect_language
            # under-counts it for ratio purposes, but there is nothing to
            # strip or translate here, so it is kept exactly like "english".
            if category == "ambiguous":
                category = "english"
            elif category == "unknown":
                category = self._resolve_uncertain(lowered)

            if category in _KEEP_AS_ENGLISH or category == "english":
                pieces.append(original)
            elif category in _TRANSLATE_TO_ENGLISH or category == "tagalog":
                if lowered not in translated_cache:
                    translated_cache[lowered] = self.translate_word(original)
                    translated_pairs.append((original, translated_cache[lowered]))
                pieces.append(translated_cache[lowered])
                if F.TAGALOG_WORD_TRANSLATED.value not in flags:
                    flags.append(F.TAGALOG_WORD_TRANSLATED.value)
            else:  # "unknown", surviving wordfreq's second opinion too
                stripped_words.append(original)
                if F.NON_ENGLISH_TAGALOG_WORD_STRIPPED.value not in flags:
                    flags.append(F.NON_ENGLISH_TAGALOG_WORD_STRIPPED.value)
                # Nothing appended for the word itself. The gap before it was
                # already appended above (every match's leading gap is kept
                # regardless of category); the gap after it is appended by
                # whichever match or trailing slice comes next. A stripped
                # word between two ordinary single spaces therefore leaves
                # two adjacent horizontal-whitespace runs in `pieces`, which
                # the collapse below resolves in one pass rather than by
                # special-casing each stripped word here.

        pieces.append(text[cursor:])
        rebuilt = "".join(pieces)
        # Collapse the double space (or worse) a stripped word leaves behind.
        # `[ \t]+`, not `\s+`, so intentional newlines survive - this variant
        # is a scannable canonical form, not a byte-exact copy, so normalising
        # horizontal whitespace here is harmless.
        rebuilt = re.sub(r"[ \t]+", " ", rebuilt).strip()

        return CodeSwitchResult(
            text=rebuilt,
            translated=translated_pairs,
            stripped=stripped_words,
            flags=flags,
        )
