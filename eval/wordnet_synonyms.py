"""Derive the synonym-folding map from WordNet instead of hand-curation.

    python eval/wordnet_synonyms.py            # write data/features/synonym_map.json
    python eval/wordnet_synonyms.py --preview  # show what would change, write nothing

Replaces a 32-entry dictionary written by hand. Two things were wrong with that
dictionary, and both matter at a defence:

* **It was arbitrary.** "A list we wrote" is not a method a panellist can check.
  "WordNet 3.0 verb synsets of the adopted instruction-verb lexicon" is.
* **It was single-word only**, so phrasal verbs were invisible. The corpus
  contains ``act as`` 980 times, ``push aside`` 189, ``brush aside`` 185 -
  ``brush aside`` being an exact phrasal synonym of ``ignore``, the very case
  the folding exists to catch.

**Why a wider vocabulary is safe here.** WordNet expansion also drags in common
verbs - ``take``, ``make``, ``get``, ``run`` - which as a *detector* would be
catastrophic. This lexicon is not a detector. It is the candidate vocabulary for
a statistical test, and two filters stand behind it:

1. :func:`~eval.token_association.instruction_verbs_in_context` requires an
   instruction-object word within five tokens, so "take the file" never counts.
2. The log-odds fit keeps only tokens significant at FDR 0.05, so a verb spread
   evenly across both classes earns weight zero regardless of membership. This
   already happens - ``execute`` is in the lexicon and scored 0.

Restricting the candidate set by hand was therefore constraining the hypothesis
space for no methodological reason. Benjamini-Hochberg is what controls false
discovery; the dictionary should not be trying to do that job too.

The output is cached to JSON so ``token_association`` never imports nltk, and so
the map is a versioned artifact rather than something regenerated differently on
each machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mochi.preprocess.features import INSTRUCTION_VERBS  # noqa: E402

OUTPUT_PATH = REPO / "data" / "features" / "synonym_map.json"

#: Phrasal synonyms WordNet does not carry but the corpus does. Kept explicitly
#: and separately from the WordNet-derived entries, so the generated portion
#: stays a pure function of WordNet and this short list is auditable on its own.
EXTRA_PHRASES: dict[str, str] = {
    "set aside": "ignore",
    "put aside": "ignore",
    "lay aside": "ignore",
    "pay no attention": "ignore",
    "pay no heed": "ignore",
    "take no notice": "ignore",
}

#: Longest phrase, in words, that folding will attempt to match. Bounds the
#: n-gram scan in ``instruction_verbs_in_context``.
MAX_PHRASE_WORDS = 3


def build_synonym_map() -> tuple[dict[str, str], dict]:
    """Map every WordNet verb synonym of the lexicon back to its canonical verb.

    Ambiguity is resolved by **synset order**: WordNet lists a word's synsets
    most-frequent sense first, so the earliest synset a term appears in wins.
    Deterministic, and it prefers the common reading over an obscure one - ``cut``
    reaches ``skip`` through a common sense and ``disregard`` through a rarer
    one, and the common reading is the one a prompt is likely using.

    A term that is *itself* a canonical verb is never folded away: it is its own
    base form, and mapping it elsewhere would make folding order-dependent.
    """
    from nltk.corpus import wordnet as wn

    best: dict[str, tuple[int, str]] = {}
    for verb in sorted(INSTRUCTION_VERBS):
        for rank, synset in enumerate(wn.synsets(verb, pos=wn.VERB)):
            for lemma in synset.lemmas():
                term = lemma.name().lower().replace("_", " ")
                if term == verb or term in INSTRUCTION_VERBS:
                    continue
                if len(term.split()) > MAX_PHRASE_WORDS:
                    continue
                if term not in best or rank < best[term][0]:
                    best[term] = (rank, verb)

    mapping = {term: verb for term, (_rank, verb) in sorted(best.items())}
    mapping.update(EXTRA_PHRASES)

    meta = {
        "source": "WordNet 3.0 verb synsets via nltk",
        "canonical_verbs": len(INSTRUCTION_VERBS),
        "entries": len(mapping),
        "phrases": sum(1 for t in mapping if " " in t),
        "hand_added_phrases": len(EXTRA_PHRASES),
        "max_phrase_words": MAX_PHRASE_WORDS,
        "ambiguity_rule": "earliest synset wins (most frequent sense)",
    }
    return mapping, meta


def load_synonym_map(path: Path = OUTPUT_PATH) -> dict[str, str] | None:
    """Read the cached map, or None when it has not been generated."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["mapping"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--preview", action="store_true",
                        help="print the comparison and write nothing")
    args = parser.parse_args()

    try:
        import nltk
        from nltk.corpus import wordnet as wn
        wn.synsets("ignore")
    except ImportError:
        print("ERROR: nltk is required.\n  pip install nltk")
        return 1
    except LookupError:
        print("  downloading the WordNet corpus (one time) ...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    from eval.token_association import HAND_BUILT_SYNONYMS

    mapping, meta = build_synonym_map()

    print()
    print("=" * 84)
    print("  Synonym folding  -  WordNet-derived vs hand-built")
    print("=" * 84)
    print(f"  hand-built entries      {len(HAND_BUILT_SYNONYMS):>6}"
          f"   phrases {sum(1 for t in HAND_BUILT_SYNONYMS if ' ' in t):>4}")
    print(f"  WordNet-derived         {meta['entries']:>6}"
          f"   phrases {meta['phrases']:>4}"
          f"  ({meta['hand_added_phrases']} added by hand)")
    print()

    added = sorted(set(mapping) - set(HAND_BUILT_SYNONYMS))
    dropped = sorted(set(HAND_BUILT_SYNONYMS) - set(mapping))
    print(f"  {len(added)} terms gained, {len(dropped)} lost")
    if dropped:
        print(f"    lost: {', '.join(dropped)}")
    print()
    print("  Phrases now covered:")
    for term in sorted(t for t in mapping if " " in t)[:18]:
        print(f"    {term:<24}-> {mapping[term]}")
    print("=" * 84)

    if args.preview:
        print("\n  --preview: nothing written\n")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "mapping": mapping}, indent=2),
                        encoding="utf-8")
    print(f"\n  Wrote {args.out}")
    print("  Next: python eval/fit_malicious_word_weights.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
