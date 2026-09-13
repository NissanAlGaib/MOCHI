"""Materialise the Track A feature table.

    python eval/fit_malicious_word_weights.py     # run once first, see below
    python eval/build_features.py --data data/clean
    python eval/build_features.py --data data/clean --format csv --limit 5000

Writes one row per sample to ``data/features/``: the identity/tracking columns
below, ``mochi.preprocess.features.TRACK_A_FEATURES`` (17 columns, pure
functions of the text), and ``eval.token_association.TRACK_A_FITTED_FEATURES``
(1 column, ``malicious_word_weight_sum``) - 18 Track A columns total, split
across two layers because they are two different kinds of thing. This script
computes nothing about the *first* seventeen itself - it loads samples, calls
the runtime extractor, and keeps only the columns Track A actually trains on.
If the frozen set changes, it changes in ``TRACK_A_FEATURES`` and this script
follows automatically.

**The 18th column needs a prerequisite step.** ``malicious_word_weight_sum``
is scored from a weight table fit once on the train split
(``eval/fit_malicious_word_weights.py`` writes
``data/features/instruction_verb_weights.json``) - fitting must happen before
this script can read that file, and must never happen again inside this
script, or the weights would silently be refit on data this script also treats
as held-out test rows. This script only *applies* the already-fit weights,
identically to every row regardless of split - see
``eval.token_association.fit_instruction_verb_weights`` for why fitting lives
outside ``mochi/preprocess/features.py`` entirely.

**No Stage I scan runs here.** None of `TRACK_A_FEATURES` is Stage I-derived -
that family was excluded from the frozen set specifically so a Track A model
cannot be rediscovering Stage I's own regexes (see
``docs/CLASSIFICATION_PLAN.md``'s Step 1c). ``extract()`` is called with
``stage1=None`` accordingly, which is also most of why this script is fast:
a per-row detector scan over ~83,000 rows is the expensive part it no longer
pays for.

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
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import (  # noqa: E402
    DATA_DIR,
    TEST_SHARE,
    VALIDATION_SHARE,
    DatasetError,
    Sample,
    load_file,
    stratified_split,
)
from eval.token_association import score_malicious_word_weight  # noqa: E402
from mochi.preprocess import normalize  # noqa: E402
from mochi.preprocess.features import TRACK_A_FEATURES, extract  # noqa: E402

INSTRUCTION_VERB_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "features"
    / "instruction_verb_weights.json"
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "features"


def assign_splits(data_dir: Path) -> list[tuple[Sample, str]]:
    """Pair every sample with its split, pooling every source file first.

    Mirrors ``training/finetune_e5.build_splits``. Keep the two in step -
    ``tests/test_build_features.py`` fails if they diverge.

    **PromptShield's own train/validation/test files are deliberately not
    honoured.** They are 103,785 / 5,500 / 132,199 rows - a test set larger than
    the train set, or roughly 45/2/51 once jayavibhav is folded in. No weighting
    of those files produces the 70/30 the method calls for, so the only way to
    get it is to pool all four and split from scratch. The cost is comparability
    with numbers published against PromptShield's own test file, which this
    thesis does not claim.

    **Three tiers, from a double 70/30 split** (see ``TEST_SHARE`` /
    ``VALIDATION_SHARE``): 30% of the corpus is sealed as test, and the
    remaining 70% is split 70/30 again into train (49% of the corpus) and
    validation (21%). Ten-fold cross-validation then runs *inside* the train
    tier for hyperparameter selection, and the validation tier is reserved for
    decisions the folds cannot make - the decision threshold above all.

    An earlier revision returned train and test only, on the reasoning that
    k-fold makes a validation tier redundant. It does for hyperparameters; it
    does not for the threshold, which has to be chosen on rows the *final*
    fitted model never saw. Both tracks now take the identical three-way split,
    so a row is in the same tier for Track A and Track B alike.
    """
    pooled: list[Sample] = []
    for path in sorted(data_dir.glob("*.csv")):
        pooled.extend(load_file(path))

    train, validation, test = stratified_split(
        pooled, test=TEST_SHARE, validation=VALIDATION_SHARE)

    tagged: list[tuple[Sample, str]] = []
    for split, group in (("train", train), ("validation", validation),
                         ("test", test)):
        tagged.extend((sample, split) for sample in group)
    return tagged


def build_rows(tagged: list[tuple[Sample, str]], *, weights: dict[str, float],
               progress_every: int = 5_000):
    """Extract features for every sample, yielding flat row dicts.

    ``TRACK_A_FEATURES`` is kept from the extractor's output - the frozen
    Step 1c column set, not every column ``FeatureVector`` can compute. Filter
    happens here, after calling the one real ``extract()``, rather than by
    asking the extractor to compute less: the extractor stays the single place
    a feature is defined, and this script stays the place that decides which of
    its outputs get written down.

    ``malicious_word_weight_sum`` is computed separately, via
    ``score_malicious_word_weight`` and the already-fit ``weights`` table
    (never refit here) - it is not a ``FeatureVector`` output at all, so it
    cannot come from the same ``full[name]`` lookup as the other 17.
    """
    started = time.perf_counter()

    for index, (sample, split) in enumerate(tagged, start=1):
        result = normalize(sample.text)
        vector = extract(sample.text, norm=result)
        full = vector.as_dict()

        row = {
            "text_hash": hashlib.sha256(sample.text.encode("utf-8")).hexdigest()[:16],
            "dataset": sample.dataset,
            "split": split,
            "source_tag": sample.source_tag,
            "label": sample.label,
        }
        row.update({name: full[name] for name in TRACK_A_FEATURES})
        row["malicious_word_weight_sum"] = score_malicious_word_weight(
            sample.text, weights
        )
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

    # The length confound. Median length varies widely across sources, so a
    # length feature can look predictive for reasons unrelated to injection.
    # char_count, not est_token_count - the latter was dropped in Step 1c as a
    # deterministic derivative (char_count // 4) and is no longer materialised.
    print("  Length by dataset (the confound to rule out before trusting it)")
    by_dataset = frame.groupby("dataset")["char_count"].median()
    for name, value in by_dataset.items():
        print(f"    {name:<28}{value:>8,.0f} median chars")
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
    # question_ratio was not kept in TRACK_A_FEATURES (Step 1c); ends_with_question
    # is the column that actually carries the signal, per the Step 1b findings.
    for column in ("question_mark_count", "ends_with_question", "imperative_verb_count",
                   "starts_with_imperative", "negation_count"):
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
    parser.add_argument("--weights", type=Path, default=INSTRUCTION_VERB_WEIGHTS_PATH,
                        help="instruction-verb weight table from "
                             "eval/fit_malicious_word_weights.py")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. pip install pandas")
        return 1

    if not args.data.exists():
        print(f"ERROR: {args.data} not found. Run eval/clean_datasets.py first.")
        return 1

    if not args.weights.exists():
        print(f"ERROR: {args.weights} not found.\n"
              f"Run eval/fit_malicious_word_weights.py first - "
              f"malicious_word_weight_sum needs a weight table fit on the "
              f"train split before this script can apply it.")
        return 1
    weights = json.loads(args.weights.read_text(encoding="utf-8"))["weights"]

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

    print(f"\n  Extracting {len(tagged):,} rows x "
          f"{len(TRACK_A_FEATURES) + 1 + 5} columns ...")
    started = time.perf_counter()
    frame = pd.DataFrame(list(build_rows(tagged, weights=weights)))
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
