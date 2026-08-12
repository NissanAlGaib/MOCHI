"""Dataset cleaning for training and evaluation.

    python eval/clean_datasets.py --data data/ --out data/clean/
    python eval/clean_datasets.py --data data/ --out data/clean/ --dry-run

Guiding rule: **remove artifacts, never remove signal.**

Applied (artifact removal - safe):
  * collapse whitespace runs, strip leading/trailing
  * decode HTML entities (``&amp;`` -> ``&``)
  * strip C0/C1 control characters (keep \\n and \\t)
  * drop empty or near-empty rows
  * drop exact and near duplicates
  * drop rows whose normalized text carries conflicting labels

NOT applied, deliberately:
  * stopword removal, lemmatization, POS filtering - prompt injection lives in
    imperative verb + object structure ("ignore" + "previous instructions").
    Stripping function words destroys the exact signal being detected, and
    creates a train/serve mismatch because runtime text is never stripped.
  * lowercasing - "DAN" vs "Dan" is a real distinction in the pattern set.
  * punctuation removal - "## System:" and "' OR '1'='1" are punctuation-bearing
    attack signatures.
  * Unicode folding - the Phase 3 runtime layer handles homoglyphs and
    zero-width characters and *flags* them; folding here would erase evidence
    that obfuscation was present.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.audit_datasets import CHARS_PER_TOKEN, digest, normalized_key  # noqa: E402
from eval.data_loading import DATA_DIR, DatasetError, Sample, load_directory  # noqa: E402

# Keep newline and tab; strip the rest of C0 plus DEL and C1.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
BLANK_LINES = re.compile(r"\n{3,}")
SPACES = re.compile(r"[ \t]{2,}")
TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)

MIN_LENGTH = 10


@dataclass
class CleanStats:
    read: int = 0
    kept: int = 0
    dropped_tiny: int = 0
    dropped_exact_dupe: int = 0
    dropped_near_dupe: int = 0
    dropped_conflict: int = 0
    chars_before: int = 0
    chars_after: int = 0

    @property
    def tokens_before(self) -> int:
        return self.chars_before // CHARS_PER_TOKEN

    @property
    def tokens_after(self) -> int:
        return self.chars_after // CHARS_PER_TOKEN

    @property
    def token_saving(self) -> float:
        if not self.chars_before:
            return 0.0
        return 1 - (self.chars_after / self.chars_before)


def clean_text(text: str) -> str:
    """Artifact removal only. Casing, punctuation, and word order are preserved."""
    cleaned = html.unescape(text)
    cleaned = unicodedata.normalize("NFC", cleaned)  # canonical only, not NFKC
    cleaned = CONTROL_CHARS.sub("", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = TRAILING_WS.sub("", cleaned)
    cleaned = SPACES.sub(" ", cleaned)
    cleaned = BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def clean(samples: Sequence[Sample], *, drop_conflicts: bool = True
          ) -> tuple[list[Sample], CleanStats]:
    """Clean and deduplicate, preserving first occurrence."""
    stats = CleanStats()

    # Pass 1: clean text, index labels by normalized key to find conflicts.
    staged: list[Sample] = []
    labels_by_key: dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        stats.read += 1
        stats.chars_before += len(sample.text)
        text = clean_text(sample.text)
        if len(text) < MIN_LENGTH:
            stats.dropped_tiny += 1
            continue
        labels_by_key[normalized_key(text)].add(sample.label)
        staged.append(
            Sample(
                text=text,
                label=sample.label,
                dataset=sample.dataset,
                source_tag=sample.source_tag,
                attack_type=sample.attack_type,
                metadata=sample.metadata,
            )
        )

    conflicting = {k for k, labels in labels_by_key.items() if len(labels) > 1}

    # Pass 2: drop conflicts and duplicates.
    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    kept: list[Sample] = []
    for sample in staged:
        key = normalized_key(sample.text)
        if drop_conflicts and key in conflicting:
            stats.dropped_conflict += 1
            continue

        exact = digest(sample.text)
        if exact in seen_exact:
            stats.dropped_exact_dupe += 1
            continue
        if key in seen_near:
            stats.dropped_near_dupe += 1
            continue

        seen_exact.add(exact)
        seen_near.add(key)
        kept.append(sample)
        stats.chars_after += len(sample.text)

    stats.kept = len(kept)
    return kept, stats


def write_csv(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        for sample in samples:
            writer.writerow([sample.text, sample.label])


def print_stats(stats: CleanStats, per_dataset: dict[str, int]) -> None:
    print()
    print("=" * 72)
    print("  Cleaning Summary")
    print("=" * 72)
    print(f"  read                    {stats.read:>10,}")
    print(f"  kept                    {stats.kept:>10,}  ({stats.kept / max(stats.read,1):.1%})")
    print()
    print(f"  dropped: empty/tiny     {stats.dropped_tiny:>10,}")
    print(f"  dropped: exact dupes    {stats.dropped_exact_dupe:>10,}")
    print(f"  dropped: near dupes     {stats.dropped_near_dupe:>10,}")
    print(f"  dropped: label conflict {stats.dropped_conflict:>10,}")
    print()
    print(f"  est. tokens before      {stats.tokens_before:>10,}")
    print(f"  est. tokens after       {stats.tokens_after:>10,}")
    print(f"  reduction               {stats.token_saving:>10.1%}"
          "   (whitespace/entity cleanup + dedup)")
    print()
    print("  Surviving rows by dataset")
    for name, count in sorted(per_dataset.items()):
        print(f"    {name:<28}{count:>10,}")
    print("=" * 72)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only, write nothing")
    parser.add_argument("--keep-conflicts", action="store_true",
                        help="keep rows whose normalized text has contradictory labels")
    parser.add_argument("--per-dataset", action="store_true",
                        help="write one cleaned file per source dataset "
                             "(default: also writes a combined file)")
    args = parser.parse_args()

    try:
        samples = load_directory(args.data)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    kept, stats = clean(samples, drop_conflicts=not args.keep_conflicts)

    by_dataset: dict[str, list[Sample]] = defaultdict(list)
    for sample in kept:
        by_dataset[sample.dataset].append(sample)

    print_stats(stats, {k: len(v) for k, v in by_dataset.items()})

    if args.dry_run:
        print("  (dry run - nothing written)\n")
        return 0

    for name, group in by_dataset.items():
        write_csv(group, args.out / f"{name}.csv")
        print(f"  wrote {len(group):>8,} -> {args.out / (name + '.csv')}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
