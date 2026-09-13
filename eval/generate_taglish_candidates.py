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


def make_taglish(text: str, translation, *, switch_rate: float,
                  rng: random.Random) -> tuple[str, list[str]]:
    """Switch a random subset of ``text``'s English content words to Tagalog.

    Returns the Taglish candidate and the list of words actually switched, so
    a reviewer can see at a glance what to check first rather than diffing the
    whole sentence.
    """
    matches = list(_LANG_WORD_RE.finditer(text.lower()))
    eligible = [
        m for m in matches
        if _is_switchable(m.group(0)) and len(m.group(0)) > 2
    ]
    if not eligible:
        return text, []

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
    return "".join(pieces), switched


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
