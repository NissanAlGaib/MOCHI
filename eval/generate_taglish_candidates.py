"""Generate candidate Taglish samples for the Step 5 evaluation set.

    python eval/generate_taglish_candidates.py --data data/clean --n 250
    python eval/generate_taglish_candidates.py --data data/clean --n 250 --switch-rate 0.4

Answers a question the Step 5 requirement in ``docs/CLASSIFICATION_PLAN.md``
raises but does not solve on its own: where do ~200 attack + ~200 benign
code-switched English-Tagalog samples come from? Hand-writing them does not
scale and a static English-to-Tagalog dictionary was already declined
(``docs/BUILD_PLAN.md``'s A14 amendment) - the corpus itself is generated here
instead, by machine-translating a *subset* of each English prompt's content
words into Tagalog with Argos Translate's ``en -> tl`` model.

**This produces candidates, not the evaluation set.** Step 5 explicitly
requires native-speaker validation before any of this is trusted for a
reported number - a machine translation can be grammatically wrong, and this
script has no way to tell. Output goes to ``data/taglish_candidates/``, kept
out of ``data/clean/`` on purpose, so nothing downstream can accidentally treat
an unreviewed row as ground truth. A native speaker reviews the CSV, keeps or
edits each row, and drops it into ``data/clean/`` as its own file only once
that review is done - the same file layout every other corpus in this project
already uses.

**Why partial, word-level switching, not full-sentence translation.** A fully
Tagalog sentence is a Tagalog sample, not a Taglish one - it exercises a
different capability than the code-switching detection this set is for. Real
Taglish keeps an English or Tagalog grammatical frame and drops content words
from the other language into it (see ``normalize.py``'s own docstring on this).
This script approximates that by translating a random subset of a prompt's
*content* words - the ones ``classify_word_language`` would call ``"english"``
- and leaving function words, punctuation, and any word it cannot confidently
place untouched. It is a rougher approximation than a native speaker's own
code-switching, which is exactly why review is not optional.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import DATA_DIR, Sample, load_file  # noqa: E402
from mochi.preprocess.normalize import (  # noqa: E402
    _LANG_WORD_RE,
    ENGLISH_FUNCTION_WORDS,
    classify_word_language,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "taglish_candidates"

#: Share of eligible English function words replaced by Tagalog particles.
#: Set from the threshold it has to clear: ``normalize.TAGALOG_FRAME_RATIO``
#: requires 15% of a prompt's tokens to be Tagalog function words before it will
#: call the text code-switched, and function words are roughly a third of an
#: English prompt. Switching half of them lands comfortably above that without
#: translating the sentence outright.
FRAME_SWITCH_RATE = 0.5


def _is_switchable(word: str) -> bool:
    """Whether ``word`` is an English *content* word eligible for switching.

    ``classify_word_language`` returns the single category ``"english"`` for
    both function words (the/a/is/of) and content words (password/account) -
    a distinction its only other caller, ``detect_language``, has no need for.
    This script does need it: switching a function word would replace the
    grammatical frame a real Taglish sentence keeps, rather than the content
    dropped into it (see the module docstring, and ``normalize.py``'s own
    docstring on the same point). Function words are therefore excluded
    explicitly here rather than by extending the shared classifier for a
    distinction only this one caller needs. Words the classifier could not
    confidently place at all are excluded too - switching an uncertain word
    would make the candidate's English half unreliable on top of the
    translation risk already present in the half being switched on purpose.
    """
    return (classify_word_language(word) == "english"
            and word not in ENGLISH_FUNCTION_WORDS)


def _load_translator():
    """Argos Translate's en -> tl model. Raises with install instructions if
    the optional dependency (see requirements.txt) is not present - this
    script needs the same heavy stack code_switch.py needs, for the same
    reason: it is the only maintained library with this language pair.
    """
    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError as exc:
        raise SystemExit(
            "This script needs argostranslate:\n"
            "  pip install argostranslate\n"
            "See requirements.txt's Tagalog content filter section for the "
            "full install, including the one-time en<->tl package download."
        ) from exc

    installed = argostranslate.translate.get_installed_languages()
    english = next((lang for lang in installed if lang.code == "en"), None)
    tagalog = next((lang for lang in installed if lang.code == "tl"), None)
    if english is None or tagalog is None:
        raise SystemExit(
            "argostranslate is installed but the en<->tl package is not.\n"
            "Download it once:\n"
            "  python -c \"import argostranslate.package as p; "
            "p.update_package_index(); "
            "pkg = next(x for x in p.get_available_packages() "
            "if x.from_code=='en' and x.to_code=='tl'); "
            "p.install_from_path(pkg.download())\""
        )
    translation = english.get_translation(tagalog)
    if translation is None:
        raise SystemExit("No installed en -> tl translation path.")
    return translation


#: English function words mapped to the Tagalog particles that carry a Tagalog
#: grammatical frame. Every value is a member of
#: ``normalize.TAGALOG_FUNCTION_WORDS`` - the set ``detect_language`` counts -
#: because a "frame" the detector does not recognise is not a frame.
#:
#: **This mapping is what makes the generated corpus usable at all.** An earlier
#: version switched only content words and deliberately left function words in
#: English, reasoning that the grammatical frame should be preserved. It was
#: preserving the *English* frame, while ``detect_language`` looks for a
#: *Tagalog* one: real Taglish is Tagalog grammar with English content dropped
#: in ("balewalain mo ang previous instructions"), not the reverse. The result
#: was 2,389 rows at a median 7.5% Tagalog against a 15% threshold - a corpus
#: none of which the feature it was built for could see.
#:
#: Translated by table rather than by the model: these are closed-class words
#: with stable renderings, and Argos returns inconsistent results for them out
#: of context ("the" alone has no Tagalog equivalent at all).
FUNCTION_WORD_TO_TAGALOG: dict[str, str] = {
    "the": "ang", "a": "ang", "an": "ang",
    "all": "lahat ng", "of": "ng", "to": "sa", "in": "sa", "at": "sa",
    # "and"/"or" are deliberately absent. Their Tagalog renderings are "at" and
    # "o", which are also ordinary English words (Zipf 6.70 and 5.12) - adding
    # them to the detector's Tagalog list would make plain English prose read as
    # code-switched, and leaving them out of the detector would make them
    # substitutions that carry no frame. Either way they cannot help here.
    "but": "pero", "if": "kung", "because": "dahil",
    "your": "iyong", "you": "mo", "my": "aking", "me": "akin", "i": "ako",
    "we": "tayo", "they": "sila", "he": "siya", "she": "siya", "it": "ito",
    "this": "ito", "that": "iyon", "these": "ang mga", "those": "ang mga",
    "is": "ay", "are": "ay", "was": "ay", "were": "ay", "be": "ay",
    "not": "hindi", "no": "wala", "never": "hindi",
    # Each of these pairs a natural rendering with a counted particle: on its
    # own "pakiusap" is real Tagalog the frame test does not count, and a
    # substitution that adds no frame is a substitution that does nothing here.
    "please": "pakiusap po", "now": "ngayon na",
    "then": "tapos", "after": "pagkatapos",
    "before": "bago", "while": "habang", "when": "kapag", "why": "bakit",
    "how": "paano", "what": "ano", "who": "sino", "where": "saan",
    "can": "puwede", "must": "dapat", "should": "dapat", "need": "kailangan",
    "want": "gusto", "for": "para", "with": "kasama ng", "from": "mula sa",
    "so": "kaya", "also": "din", "only": "lamang", "just": "lang",
    "more": "pa", "some": "ilan", "many": "mga",
}


def switch_function_words(text: str, *, rate: float, rng: random.Random
                          ) -> tuple[str, list[str]]:
    """Replace a share of English function words with Tagalog particles.

    This is what builds the Tagalog *frame*. Applied before content switching,
    so the two together produce "balewalain mo ang previous instructions" -
    Tagalog grammar carrying English content - which is the construction
    ``detect_language`` recognises and the one Filipinos actually write.
    """
    matches = list(_LANG_WORD_RE.finditer(text))
    eligible = [m for m in matches
                if m.group(0).lower() in FUNCTION_WORD_TO_TAGALOG]
    if not eligible:
        return text, []

    n_switch = max(1, round(len(eligible) * rate))
    chosen = set(rng.sample(eligible, min(n_switch, len(eligible))))

    pieces: list[str] = []
    cursor = 0
    switched: list[str] = []
    for match in matches:
        start, end = match.span()
        pieces.append(text[cursor:start])
        cursor = end
        if match in chosen:
            pieces.append(FUNCTION_WORD_TO_TAGALOG[match.group(0).lower()])
            switched.append(match.group(0))
        else:
            pieces.append(match.group(0))
    pieces.append(text[cursor:])
    return "".join(pieces), switched


def make_taglish(text: str, translation, *, switch_rate: float,
                  rng: random.Random) -> tuple[str, list[str]]:
    """Switch a random subset of ``text``'s English content words to Tagalog.

    Returns the Taglish candidate and the list of words actually switched, so
    a reviewer can see at a glance what to check first rather than diffing the
    whole sentence.
    """
    # The Tagalog frame first, then English content dropped into it. Order
    # matters only for readability of the result - the frame words are a
    # disjoint closed class, so neither pass can consume the other's tokens.
    text, frame_switched = switch_function_words(
        text, rate=FRAME_SWITCH_RATE, rng=rng)

    matches = list(_LANG_WORD_RE.finditer(text.lower()))
    eligible = [
        m for m in matches
        if _is_switchable(m.group(0)) and len(m.group(0)) > 2
    ]
    if not eligible:
        return text, frame_switched

    n_switch = max(1, round(len(eligible) * switch_rate))
    chosen = set(rng.sample(eligible, min(n_switch, len(eligible))))

    pieces: list[str] = []
    cursor = 0
    switched: list[str] = []
    translated_cache: dict[str, str] = {}

    for match in matches:
        start, end = match.span()
        pieces.append(text[cursor:start])
        cursor = end
        original = text[start:end]

        if match in chosen:
            lowered = match.group(0)
            if lowered not in translated_cache:
                translated_cache[lowered] = translation.translate(original)
            pieces.append(translated_cache[lowered])
            switched.append(original)
        else:
            pieces.append(original)

    pieces.append(text[cursor:])
    return "".join(pieces), frame_switched + switched


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n", type=int, default=250,
                        help="candidates per class (attack, benign) - default "
                             "matches the ~200-per-class Step 5 target with headroom "
                             "for rows a reviewer rejects")
    parser.add_argument("--switch-rate", type=float, default=0.35,
                        help="share of eligible English content words switched "
                             "to Tagalog per sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    translation = _load_translator()
    rng = random.Random(args.seed)

    pooled: list[Sample] = []
    for path in sorted(args.data.glob("*.csv")):
        pooled.extend(load_file(path))

    by_label: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in pooled:
        by_label.setdefault(sample.label, []).append(sample)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "candidates.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "source_dataset", "original_text",
                          "taglish_candidate", "words_switched", "reviewed"])

        for label, pool in by_label.items():
            if not pool:
                continue
            chosen = rng.sample(pool, min(args.n, len(pool)))
            for sample in chosen:
                candidate, switched = make_taglish(
                    sample.text, translation,
                    switch_rate=args.switch_rate, rng=rng,
                )
                if not switched:
                    continue  # nothing eligible to switch - not a useful candidate
                writer.writerow([
                    sample.label, sample.dataset, sample.text,
                    candidate, ";".join(switched), "",
                ])

    print(f"Wrote candidates to {out_path}")
    print("Next: a native Tagalog speaker reviews each row (the 'reviewed' "
          "column), corrects or discards bad machine translations, and only "
          "the reviewed rows move into data/clean/ as the Step 5 evaluation set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
