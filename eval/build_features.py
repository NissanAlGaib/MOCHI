"""Materialise the Phase 6.5 feature table.

    python eval/build_features.py --data data/clean
    python eval/build_features.py --data data/clean --format csv --limit 5000

Writes one row per sample to ``data/features/``, with every column defined by
``mochi.preprocess.features``. This script computes **nothing** itself - it loads
samples, calls the runtime extractor, and writes the result. That is the whole
design: if a feature needs changing, it changes in one place and both the dataset
and the gateway follow.

**Split assignment mirrors ``training/finetune_e5.build_splits`` exactly**, using
the same seed and the same rule (PromptShield's official splits are honoured;
everything else is split with the seeded stratified splitter). It is duplicated
rather than imported because ``build_splits`` returns bare text/label lists and
discards the ``Sample`` objects, so it cannot say which dataset a row came from -
and the per-dataset breakdown is exactly what the length-confound check needs.
``tests/test_build_features.py`` pins the two against each other so the
duplication cannot drift.

Why the split column matters: several features one might add later - log-odds,
learned vocabularies, mined phrase lists - are legitimate only if fitted on the
train split alone. Without a split column recorded at build time, that rule
cannot be checked afterwards, and a leaked ablation looks exactly like a good
one.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import (  # noqa: E402
    DATA_DIR,
    DatasetError,
    Sample,
    load_file,
    stratified_split,
)
from mochi.detect.stage1_syntactic import get_detector  # noqa: E402
from mochi.preprocess import normalize  # noqa: E402
from mochi.preprocess.features import extract, feature_names  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "features"

#: Split names recognised as official when they suffix a filename.
OFFICIAL_SPLITS = ("train", "validation", "test")


def assign_splits(data_dir: Path) -> list[tuple[Sample, str]]:
    """Pair every sample with its split, honouring official splits.

    Mirrors ``training/finetune_e5.build_splits``. Keep the two in step - the
    test suite fails if they diverge.
    """
    tagged: list[tuple[Sample, str]] = []
    pooled: list[Sample] = []

    for path in sorted(data_dir.glob("*.csv")):
        samples = load_file(path)
        official = next(
            (name for name in OFFICIAL_SPLITS if path.stem.endswith(f"_{name}")), None
        )
        if official:
            tagged.extend((sample, official) for sample in samples)
        else:
            pooled.extend(samples)

    if pooled:
        extra_train, extra_validation, extra_test = stratified_split(pooled)
        for split, group in (("train", extra_train),
                             ("validation", extra_validation),
                             ("test", extra_test)):
            tagged.extend((sample, split) for sample in group)

    return tagged


def build_rows(tagged: list[tuple[Sample, str]], *, progress_every: int = 5_000):
    """Extract features for every sample, yielding flat row dicts."""
    detector = get_detector()
    started = time.perf_counter()

    for index, (sample, split) in enumerate(tagged, start=1):
        result = normalize(sample.text)
        stage1 = detector.scan(result.scannable, result.flags)
        vector = extract(sample.text, norm=result, stage1=stage1)

        row = {
            "text_hash": hashlib.sha256(sample.text.encode("utf-8")).hexdigest()[:16],
            "dataset": sample.dataset,
            "split": split,
            "source_tag": sample.source_tag,
            "label": sample.label,
        }
        row.update(vector.as_dict())
        yield row

        if progress_every and index % progress_every == 0:
            rate = index / (time.perf_counter() - started)
            print(f"    {index:,} rows  ({rate:,.0f}/s)", flush=True)


def summarise(frame) -> None:
    """Print the checks worth running before anyone trusts a column."""
    print()
    print("=" * 92)
    print("  Feature table  -  composition and sanity checks")
    print("=" * 92)

    print(f"  rows {len(frame):,}    columns {len(frame.columns)}")
    print()
    print("  Split x label")
    counts = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(counts.to_string().replace("\n", "\n    ").rjust(4))
    print()

    # The length confound. Median tokens run 16 to 106 across sources, so a
    # length feature can look predictive for reasons unrelated to injection.
    print("  Length by dataset (the confound to rule out before trusting it)")
    by_dataset = frame.groupby("dataset")["est_token_count"].median()
    for name, value in by_dataset.items():
        print(f"    {name:<28}{value:>8,.0f} median tokens")
    print()

    # A column that never varies is not a measurement. In a dataframe it looks
    # exactly like one, which is why this is printed rather than assumed.
    constant = [
        column for column in frame.columns
        if column not in ("text_hash",) and frame[column].nunique(dropna=False) <= 1
    ]
    if constant:
        print(f"  CONSTANT COLUMNS ({len(constant)}) - carry no information here:")
        for column in constant:
            print(f"    {column}")
    else:
        print("  No constant columns.")
    print()

    # The A12 hypothesis, previewed. Not a test - just the first look.
    for column in ("question_ratio", "ends_with_question", "imperative_verb_count",
                   "starts_with_imperative", "n_flags"):
        if column not in frame.columns:
            continue
        benign = frame.loc[frame["label"] == 0, column].mean()
        malicious = frame.loc[frame["label"] == 1, column].mean()
        arrow = "malicious" if malicious > benign else "benign"
        print(f"  {column:<26} benign {benign:>8.4f}   malicious {malicious:>8.4f}"
              f"   leans {arrow}")
    print("=" * 92)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows, for a fast smoke run")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. pip install pandas")
        return 1

    if not args.data.exists():
        print(f"ERROR: {args.data} not found. Run eval/clean_datasets.py first.")
        return 1

    try:
        tagged = assign_splits(args.data)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not tagged:
        print(f"ERROR: no samples loaded from {args.data}")
        return 1

    if args.limit and args.limit < len(tagged):
        # Sample rather than head-slice. The files load in sorted order, so a
        # head slice returns one dataset and one split, and every per-dataset
        # column in the summary reads as constant - which looks like a bug in
        # the extractor rather than a bug in the sampling.
        import random

        tagged = random.Random(42).sample(tagged, args.limit)

    print(f"\n  Extracting {len(tagged):,} rows x {len(feature_names()) + 5} columns ...")
    started = time.perf_counter()
    frame = pd.DataFrame(list(build_rows(tagged)))
    elapsed = time.perf_counter() - started
    print(f"  Done in {elapsed:,.1f}s ({len(tagged) / elapsed:,.0f} rows/s)")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        try:
            path = args.out / "features.parquet"
            frame.to_parquet(path, index=False)
        except ImportError:
            path = args.out / "features.csv"
            frame.to_csv(path, index=False)
            print("  (pyarrow missing - wrote CSV instead)")
    else:
        path = args.out / "features.csv"
        frame.to_csv(path, index=False)

    summarise(frame)
    print(f"  Wrote {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
