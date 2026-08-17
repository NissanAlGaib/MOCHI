"""Dataset cleaning and corpus balancing for training and evaluation.

    python eval/clean_datasets.py --data data/ --out data/clean/
    python eval/clean_datasets.py --data data/ --out data/clean/ --dry-run
    python eval/clean_datasets.py --cap jayavibhav=25000    # override a cap
    python eval/clean_datasets.py --no-cap                  # keep everything

Guiding rule: **remove artifacts, never remove signal.**

Applied (artifact removal - safe):
  * collapse whitespace runs, strip leading/trailing
  * decode HTML entities (``&amp;`` -> ``&``)
  * strip C0/C1 control characters (keep \\n and \\t)
  * drop empty or near-empty rows
  * drop exact and near duplicates
  * drop rows whose normalized text carries conflicting labels

Applied (corpus balancing - see :data:`DATASET_CAPS`):
  * cap any single source so it cannot dominate the corpus, stratified by
    label and seeded. Runs *after* deduplication so the cap describes the
    final corpus rather than a pre-dedup count that shrinks unpredictably.

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
import random
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

#: Seed for the subsample draw. Fixed so the corpus is reproducible, which the
#: thesis Reliability section commits to.
SUBSAMPLE_SEED = 42

#: dataset name -> maximum rows to keep after deduplication.
#:
#: ``jayavibhav`` is 261,738 rows against PromptShield's 43,425 combined, i.e.
#: 86% of the corpus. Left uncapped, a Stage II classifier optimises for that
#: one distribution and PromptShield - the application-representative source
#: with a peer-reviewed paper behind it - becomes rounding error. The cap puts
#: the two sources at rough parity so neither dictates the decision boundary.
#: It is not a token-cost measure; it is a representativeness measure.
DATASET_CAPS: dict[str, int] = {
    "jayavibhav": 40_000,
}


@dataclass
class SubsampleStat:
    """Per-dataset record of what the cap removed, for the methodology table."""

    before: int
    after: int
    benign_before: int
    benign_after: int

    @property
    def malicious_before(self) -> int:
        return self.before - self.benign_before

    @property
    def malicious_after(self) -> int:
        return self.after - self.benign_after

    @property
    def capped(self) -> bool:
        return self.after < self.before


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


def subsample(
    samples: Sequence[Sample],
    caps: dict[str, int] | None = None,
    *,
    seed: int = SUBSAMPLE_SEED,
) -> tuple[list[Sample], dict[str, SubsampleStat]]:
    """Cap over-represented datasets, preserving each source's own class ratio.

    Preserving the source ratio rather than forcing 50/50 is deliberate: a cap
    is a size decision, and silently rebalancing classes at the same time would
    hide a second, separate methodological choice inside it.

    Relative row order within each dataset is preserved so the written CSV is
    stable across runs and reviewable with a plain diff.
    """
    caps = DATASET_CAPS if caps is None else caps
    stats: dict[str, SubsampleStat] = {}

    # Index positions per dataset so unaffected sources pass through untouched.
    positions: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        positions[sample.dataset].append(index)

    keep: set[int] = set()
    for name, indices in positions.items():
        benign = [i for i in indices if samples[i].label == 0]
        malicious = [i for i in indices if samples[i].label != 0]
        cap = caps.get(name)

        if cap is None or cap >= len(indices):
            keep.update(indices)
            stats[name] = SubsampleStat(len(indices), len(indices),
                                        len(benign), len(benign))
            continue

        # Allocate the cap in proportion to the source's existing balance,
        # then hand any rounding remainder to the larger class.
        n_benign = round(cap * len(benign) / len(indices))
        n_benign = min(n_benign, len(benign))
        n_malicious = min(cap - n_benign, len(malicious))
        n_benign = min(cap - n_malicious, len(benign))

        rng = random.Random(f"{seed}:{name}")
        chosen = rng.sample(benign, n_benign) + rng.sample(malicious, n_malicious)
        keep.update(chosen)
        stats[name] = SubsampleStat(len(indices), n_benign + n_malicious,
                                    len(benign), n_benign)

    return [s for i, s in enumerate(samples) if i in keep], stats


def parse_caps(entries: Sequence[str] | None) -> dict[str, int]:
    """Turn ``["jayavibhav=25000"]`` into ``{"jayavibhav": 25000}``.

    Overrides are merged onto :data:`DATASET_CAPS`; a value of ``0`` removes
    the cap for that dataset.
    """
    caps = dict(DATASET_CAPS)
    for entry in entries or []:
        name, separator, raw = entry.partition("=")
        name = name.strip()
        if not separator or not name or not raw.strip().isdigit():
            raise ValueError(f"--cap expects NAME=INTEGER, got {entry!r}")
        limit = int(raw)
        if limit:
            caps[name] = limit
        else:
            caps.pop(name, None)
    return caps


def write_csv(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        for sample in samples:
            writer.writerow([sample.text, sample.label])


def print_subsample(stats: dict[str, SubsampleStat]) -> None:
    affected = {k: v for k, v in stats.items() if v.capped}
    if not affected:
        return
    print()
    print("  Corpus balancing (cap applied after dedup, seed "
          f"{SUBSAMPLE_SEED}, class ratio preserved)")
    print(f"    {'Dataset':<24}{'before':>10}{'after':>10}"
          f"{'benign':>10}{'malicious':>11}")
    for name, s in sorted(affected.items()):
        print(f"    {name:<24}{s.before:>10,}{s.after:>10,}"
              f"{s.benign_after:>10,}{s.malicious_after:>11,}")


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
    parser.add_argument("--cap", action="append", metavar="NAME=N",
                        help="override a per-dataset cap; N=0 removes it. "
                             f"Defaults: {DATASET_CAPS}")
    parser.add_argument("--no-cap", action="store_true",
                        help="keep every deduplicated row, ignoring DATASET_CAPS")
    args = parser.parse_args()

    try:
        caps = {} if args.no_cap else parse_caps(args.cap)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        samples = load_directory(args.data)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    cleaned, stats = clean(samples, drop_conflicts=not args.keep_conflicts)
    kept, cap_stats = subsample(cleaned, caps)

    by_dataset: dict[str, list[Sample]] = defaultdict(list)
    for sample in kept:
        by_dataset[sample.dataset].append(sample)

    # Cleaning stats stay pre-cap: artifact removal and corpus balancing are
    # separate decisions and the thesis has to be able to report them apart.
    print_stats(stats, {k: len(v) for k, v in by_dataset.items()})
    print_subsample(cap_stats)
    print(f"  final corpus            {len(kept):>10,}")
    print()

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
