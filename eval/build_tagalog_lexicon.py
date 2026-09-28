"""Build the Tagalog to English word list the sanitization layer looks up.

    python eval/build_tagalog_lexicon.py
    python eval/build_tagalog_lexicon.py --taglish data/taglish_candidates

Translates a Tagalog vocabulary once, offline, and writes
``data/features/tagalog_lexicon.json``. ``mochi/preprocess/code_switch.py``
reads that file at runtime and never calls a translation model for a word it
covers.

**Why a lookup table and not translation in the request path.** The A14
amendment in ``docs/BUILD_PLAN.md`` declined translation-before-prediction
because a translation step inline "would undermine both the latency claim and
the threat model". That objection is aimed at a *neural* pass: argostranslate
pulls stanza, which pulls torch, and a model call costs tens of milliseconds on
every request. A dict lookup costs microseconds and pulls in nothing, so the
objection does not apply to this form. The neural model still runs - once, here,
offline - and the request path only ever sees its cached output.

**Grammar is not preserved, deliberately.** The consumer is a classifier, not a
reader: Track A counts feature occurrences and Track B tokenises subwords, and
neither depends on Tagalog word order or affix agreement surviving. Word-for-word
substitution keeps the lexical signal a detector needs while staying auditable -
the whole mapping is a JSON file a panellist can read, which a neural model is
not.

**Affixes are handled at lookup, not here.** Tagalog is agglutinative:
``balewalain``, ``binalewala`` and ``magbalewala`` share a root that a
surface-form table would miss three times over. ``code_switch`` strips the
prefixes and suffixes ``normalize.py`` already defines and retries, so this file
stores roots and inflected forms only where the translator disagreed about them.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mochi.preprocess.normalize import (  # noqa: E402
    _LANG_WORD_RE,
    TAGALOG_BARE_WORDS,
    TAGALOG_FUNCTION_WORDS,
    classify_word_language,
)

OUTPUT_PATH = REPO / "data" / "features" / "tagalog_lexicon.json"
TAGLISH_DIR = REPO / "data" / "taglish_candidates"

#: Tagalog renderings of the instruction verbs an injection actually uses.
#: Seeded by hand because the curated word lists in ``normalize.py`` exist to
#: *detect* Tagalog, not to cover attack vocabulary - they carry function words
#: and common bare words, and none of the verbs that matter here.
SEED_ATTACK_WORDS: tuple[str, ...] = (
    "balewalain", "balewala", "kalimutan", "limutin", "huwag", "wag",
    "sundin", "sundan", "ipakita", "ipakita mo", "sabihin", "sabihan",
    "ibunyag", "ilabas", "buksan", "laktawan", "palitan", "baguhin",
    "tanggalin", "burahin", "isantabi", "pansinin", "tagubilin", "panuto",
    "utos", "patakaran", "alituntunin", "dati", "nakaraan", "naunang",
    "sistema", "lihim", "sikreto", "password", "kumilos", "magpanggap",
    "gumanap", "subukan", "gawin", "payagan", "pigilan", "protektahan",
)


def collect_vocabulary(taglish_dir: Path) -> Counter:
    """Tagalog words worth translating, with how often each was seen.

    Three sources, in order of how much they matter:

    1. Words the generated Taglish corpus actually contains - the empirical
       vocabulary, and the only one sized to this project's data.
    2. :data:`SEED_ATTACK_WORDS`, so injection vocabulary is covered even
       before any Taglish corpus exists.
    3. ``normalize.py``'s curated lists, which give the everyday Tagalog a
       code-switched prompt carries around its payload.
    """
    counts: Counter = Counter()

    if taglish_dir.exists():
        for path in sorted(taglish_dir.glob("*.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    text = row.get("text") or row.get("prompt") or ""
                    for match in _LANG_WORD_RE.finditer(text):
                        word = match.group(0).lower()
                        if classify_word_language(word) in ("tagalog",
                                                            "tagalog_frame",
                                                            "taglish_verb"):
                            counts[word] += 1

    for word in SEED_ATTACK_WORDS:
        counts.setdefault(word, 0)
    for word in (*TAGALOG_BARE_WORDS, *TAGALOG_FUNCTION_WORDS):
        counts.setdefault(word, 0)
    return counts


def load_translator():
    """argostranslate's tl -> en model, with install instructions on failure."""
    try:
        import argostranslate.translate
    except ImportError as exc:
        raise SystemExit(
            "This script needs argostranslate:\n"
            "  pip install argostranslate\n"
            "then download the tl<->en package (see requirements.txt)."
        ) from exc

    installed = argostranslate.translate.get_installed_languages()
    tagalog = next((l for l in installed if l.code == "tl"), None)
    english = next((l for l in installed if l.code == "en"), None)
    if tagalog is None or english is None:
        raise SystemExit(
            "argostranslate is installed but the tl<->en package is not.\n"
            '  python -c "import argostranslate.package as p; '
            "p.update_package_index(); "
            "pkg = next(x for x in p.get_available_packages() "
            "if x.from_code=='tl' and x.to_code=='en'); "
            'p.install_from_path(pkg.download())"'
        )
    translation = tagalog.get_translation(english)
    if translation is None:
        raise SystemExit("No installed tl -> en translation path.")

    # argostranslate splits input into sentences before translating, and its
    # default sentencizer for this pair is stanza - which has no Tagalog model
    # and raises "Language tl is currently unsupported". Every input here is a
    # single word, so there is nothing to split: a sentencizer that returns the
    # input unchanged is both correct and faster than the one it replaces.
    class _SingleSentence:
        def split_sentences(self, text: str) -> list[str]:
            return [text]

    translation.underlying.sentencizer = _SingleSentence()
    return translation


#: Anything outside this set is dropped from a translation. The model returns
#: sentence punctuation ("aking" -> "me.") and occasional typographic symbols
#: picked up in training; both are noise to a consumer that tokenises words.
_KEEP = re.compile(r"[^a-z0-9' -]+")


def _clean(text: str) -> str:
    """Lowercase, strip punctuation and symbols, collapse whitespace."""
    return re.sub(r"\s+", " ", _KEEP.sub(" ", text.lower())).strip()


def build(vocabulary: Counter, translation) -> tuple[dict[str, str], dict]:
    """Translate each word, keeping only entries that actually changed it.

    A word the translator returns unchanged is either already English or
    something it could not handle; either way storing it would make the lookup
    claim a translation it did not perform. Multi-word output is kept - Tagalog
    often needs an English phrase ("huwag" -> "do not") - because the consumer
    tokenises afterwards anyway.
    """
    mapping: dict[str, str] = {}
    unchanged: list[str] = []

    for index, word in enumerate(sorted(vocabulary), start=1):
        english = _clean(translation.translate(word))
        if not english or english == word:
            unchanged.append(word)
            continue
        mapping[word] = english
        if index % 50 == 0:
            print(f"    {index}/{len(vocabulary)} ...", flush=True)

    meta = {
        "source": "argostranslate tl->en, cached offline",
        "vocabulary_considered": len(vocabulary),
        "entries": len(mapping),
        "returned_unchanged": len(unchanged),
        "multiword_outputs": sum(1 for v in mapping.values() if " " in v),
        "note": "grammar is not preserved; word-for-word substitution only",
    }
    return mapping, meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taglish", type=Path, default=TAGLISH_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the vocabulary, for a fast smoke run")
    args = parser.parse_args()

    vocabulary = collect_vocabulary(args.taglish)
    if args.limit:
        vocabulary = Counter(dict(vocabulary.most_common(args.limit)))

    print(f"\n  {len(vocabulary):,} Tagalog words to translate "
          f"({sum(1 for c in vocabulary.values() if c) } seen in the corpus)")
    translation = load_translator()
    mapping, meta = build(vocabulary, translation)

    print()
    print("=" * 76)
    print("  Tagalog -> English lexicon")
    print("=" * 76)
    for key in ("vocabulary_considered", "entries", "returned_unchanged",
                "multiword_outputs"):
        print(f"  {key:<26}{meta[key]:>8,}")
    # Written before anything is printed. A console that cannot encode a
    # character in the output would otherwise raise here and discard a table
    # that took minutes to translate.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "mapping": mapping},
                                   indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print()
    print("  Sample (attack vocabulary first):")
    for word in [w for w in SEED_ATTACK_WORDS if w in mapping][:14]:
        line = f"    {word:<20}-> {mapping[word]}"
        print(line.encode("ascii", "replace").decode("ascii"))
    print("=" * 76)
    print(f"\n  Wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
