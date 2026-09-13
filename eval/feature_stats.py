"""Which engineered features actually separate the classes.

    python eval/feature_stats.py
    python eval/feature_stats.py --top 25 --by-dataset

Built to run the Step 1b association pass that decided
``mochi.preprocess.features.TRACK_A_FEATURES`` - the full 46-column table it was
run against then no longer exists as a materialised file, since
``eval/build_features.py`` now writes only the frozen 17 columns it chose.
Kept, and still useful, as the way to re-verify that frozen set on a rebuilt
table: re-run this script whenever the corpus or the frozen columns change, to
confirm nothing that used to separate the classes has stopped doing so.

Answers three questions, in the order they have to be answered:

1. **Which columns carry signal?** Every column is tested against the label and
   ranked by *effect size*, not by p-value. At tens of thousands of training
   rows almost everything is "significant"; the p-value only says a difference
   is real, and what matters is whether it is large enough to build on.
2. **Which columns are copies of each other?** A correlation matrix over the
   numeric columns, with deterministic derivatives called out separately via
   ``DERIVED_COLUMNS`` - empty against the frozen set (the one derivative found
   in Step 1b, ``est_token_count = char_count // 4``, was dropped from
   ``TRACK_A_FEATURES`` rather than kept and flagged), but still checked here in
   case a future column reintroduces one.
3. **Which apparent signals are really dataset artefacts?** ``--by-dataset``
   recomputes the top effects within each source corpus. A feature that only
   separates classes because one dataset happens to hold most of the attacks is
   the trap ``BUILD_PLAN.md`` names for length, and it fails this check.

**Everything is computed on the training split only.** The 30% test set is not
read by this script at all. Choosing which columns to keep is a modelling
decision, and making it while looking at test rows contaminates every number
that follows - the reason ``build_features.py`` records a ``split`` column in the
first place.

Tests are Benjamini-Hochberg corrected across all columns tested: uncorrected
tests at alpha 0.05 hand back a "finding" or two that is really noise, and the
correction is cheap regardless of how many columns are in play.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.token_association import benjamini_hochberg  # noqa: E402

FEATURES_DIR = Path(__file__).resolve().parents[1] / "data" / "features"

#: Columns that identify a row rather than describe it.
ID_COLUMNS = frozenset({"text_hash", "dataset", "split", "source_tag", "label"})

#: Pairs where the second column is *computed from* the first in
#: ``mochi/preprocess/features.py``. Their correlation is a property of the
#: extractor, not a finding, and reporting them beside real correlations invites
#: a reader to treat both the same way. Empty against the current
#: ``TRACK_A_FEATURES`` - ``est_token_count`` was the one entry and it is no
#: longer materialised - but left as a live check, not deleted: a future
#: addition to the frozen set could reintroduce a derivative, and this is where
#: it gets declared rather than silently mistaken for a genuine correlation.
DERIVED_COLUMNS: dict[str, str] = {}

#: Effect-size bands for Cliff's delta, the standard thresholds.
CLIFF_BANDS = ((0.474, "large"), (0.330, "medium"), (0.147, "small"))


def cliffs_delta(a, b) -> float:
    """Nonparametric effect size: P(x > y) - P(x < y) for x in a, y in b.

    Chosen over Cohen's d because most of these columns are counts and ratios
    with heavy right tails - ``url_count`` is zero for most prompts and 40 for a
    few - and d assumes a normality none of them have. Cliff's delta only reads
    the ordering, so a long tail cannot inflate it.

    Computed from the Mann-Whitney U statistic rather than by comparing all
    pairs, which would be O(n*m) and hopeless at 57,936 rows.
    """
    import numpy as np
    from scipy.stats import mannwhitneyu

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    try:
        u, _ = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:      # every value identical - no ordering to read
        return 0.0
    return (2.0 * u) / (len(a) * len(b)) - 1.0


def interpret_delta(delta: float) -> str:
    magnitude = abs(delta)
    for threshold, name in CLIFF_BANDS:
        if magnitude >= threshold:
            return name
    return "negligible"


def cramers_v_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Association between a boolean column and the label. Returns (v, p).

    For a 2x2 table Cramer's V equals the phi coefficient, and both are bounded
    in [0, 1], which makes booleans and numerics comparable on one scale -
    otherwise there is no way to rank a flag against a ratio.
    """
    from scipy.stats import chi2_contingency

    table = [[a, b], [c, d]]
    n = a + b + c + d
    if n == 0 or min(a + b, c + d, a + c, b + d) == 0:
        return 0.0, 1.0
    chi2, p, _, _ = chi2_contingency(table, correction=False)
    return (chi2 / n) ** 0.5, p


def analyse(frame, columns: list[str]) -> list[dict]:
    """Test every column against the label. One row per column."""
    import numpy as np

    malicious = frame[frame["label"] == 1]
    benign = frame[frame["label"] == 0]
    results: list[dict] = []

    for column in columns:
        series = frame[column]
        is_boolean = series.dtype == bool or set(series.dropna().unique()) <= {0, 1}

        if is_boolean:
            m = malicious[column].astype(bool)
            b = benign[column].astype(bool)
            effect, p = cramers_v_2x2(int(m.sum()), int((~m).sum()),
                                      int(b.sum()), int((~b).sum()))
            # Sign it the same way delta is signed: positive means "more common
            # in malicious", so the two families can be read in one ranking.
            rate_m, rate_b = m.mean(), b.mean()
            effect = effect if rate_m >= rate_b else -effect
            results.append({
                "column": column, "kind": "bool", "effect": effect, "p": p,
                "mean_malicious": float(rate_m), "mean_benign": float(rate_b),
                "measure": "cramers_v",
            })
        else:
            m = malicious[column].to_numpy(dtype=float)
            b = benign[column].to_numpy(dtype=float)
            if np.nanstd(np.concatenate([m, b])) == 0:
                results.append({
                    "column": column, "kind": "num", "effect": 0.0, "p": 1.0,
                    "mean_malicious": float(np.nanmean(m)),
                    "mean_benign": float(np.nanmean(b)),
                    "measure": "constant",
                })
                continue
            from scipy.stats import mannwhitneyu
            try:
                _, p = mannwhitneyu(m, b, alternative="two-sided")
            except ValueError:
                p = 1.0
            results.append({
                "column": column, "kind": "num", "effect": cliffs_delta(m, b),
                "p": float(p),
                "mean_malicious": float(np.nanmean(m)),
                "mean_benign": float(np.nanmean(b)),
                "measure": "cliffs_delta",
            })

    for result, q in zip(results, benjamini_hochberg([r["p"] for r in results])):
        result["q"] = q

    results.sort(key=lambda r: abs(r["effect"]), reverse=True)
    return results


def report_associations(results: list[dict], top: int) -> None:
    print()
    print("=" * 92)
    print(f"  FEATURE -> LABEL ASSOCIATION   (train split only, top {top} by effect size)")
    print("=" * 92)
    print(f"  {'column':<32}{'effect':>9}{'band':>12}"
          f"{'mean(mal)':>12}{'mean(ben)':>12}{'q':>11}")
    print("  " + "-" * 88)

    for result in results[:top]:
        band = (interpret_delta(result["effect"]) if result["kind"] == "num"
                else _v_band(abs(result["effect"])))
        flag = "" if result["q"] < 0.05 else "  (n.s.)"
        note = "  [derived]" if result["column"] in DERIVED_COLUMNS else ""
        print(f"  {result['column']:<32}{result['effect']:>+9.3f}{band:>12}"
              f"{result['mean_malicious']:>12.3f}{result['mean_benign']:>12.3f}"
              f"{result['q']:>11.2e}{flag}{note}")

    print()
    print("  effect: Cliff's delta for numeric columns, signed Cramer's V for booleans.")
    print("  Positive means the value runs higher in malicious prompts.")
    print("  q: Benjamini-Hochberg corrected p-value across all columns tested.")

    negligible = [r for r in results if interpret_delta(r["effect"]) == "negligible"]
    print(f"\n  {len(negligible)} of {len(results)} columns land in the negligible band.")
    if negligible:
        print("  Weakest: " + ", ".join(r["column"] for r in negligible[-8:]))


def _v_band(v: float) -> str:
    if v >= 0.35:
        return "large"
    if v >= 0.20:
        return "medium"
    if v >= 0.10:
        return "small"
    return "negligible"


def report_redundancy(frame, columns: list[str], threshold: float) -> None:
    """Correlated pairs, with deterministic derivatives separated out."""
    numeric = [c for c in columns
               if frame[c].dtype != bool and frame[c].nunique() > 2]
    matrix = frame[numeric].corr(numeric_only=True).abs()

    pairs = []
    for i, left in enumerate(numeric):
        for right in numeric[i + 1:]:
            r = matrix.loc[left, right]
            if r >= threshold:
                pairs.append((r, left, right))
    pairs.sort(reverse=True)

    derived = [p for p in pairs if p[1] in DERIVED_COLUMNS or p[2] in DERIVED_COLUMNS]
    genuine = [p for p in pairs if p not in derived]

    print()
    print("=" * 92)
    print(f"  REDUNDANCY   (|r| >= {threshold}, numeric columns, train split)")
    print("=" * 92)

    if derived:
        print("\n  Deterministic derivatives - drop before presenting any correlation figure.")
        print("  These report the arithmetic in features.py, not a property of prompts.\n")
        for r, left, right in derived:
            source = DERIVED_COLUMNS.get(left) or DERIVED_COLUMNS.get(right)
            print(f"    r={r:5.3f}   {left:<28} ~ {right:<28}  ({source})")

    print(f"\n  Genuinely correlated pairs ({len(genuine)}):\n")
    for r, left, right in genuine[:25]:
        print(f"    r={r:5.3f}   {left:<28} ~ {right}")
    if len(genuine) > 25:
        print(f"    ... and {len(genuine) - 25} more")

    print("\n  Correlated-but-independent columns may stay - they measure different")
    print("  things. Deterministic derivatives may not.")


def report_by_dataset(frame, results: list[dict], top: int) -> None:
    """Re-test the strongest effects inside each source corpus.

    The confound this exists to catch: median prompt length runs 16 to 106
    tokens across these corpora, so any column correlated with length will look
    predictive on the pooled data purely because of how the corpora were
    combined. An effect that survives *within* every dataset is real; one that
    collapses inside each is an artefact of the merge.
    """
    datasets = sorted(frame["dataset"].unique())
    columns = [r["column"] for r in results[:top]]

    print()
    print("=" * 92)
    print(f"  PER-DATASET CHECK   (top {top} effects, recomputed within each corpus)")
    print("=" * 92)
    header = "".join(f"{d[:13]:>15}" for d in datasets)
    print(f"  {'column':<30}{'pooled':>10}{header}")
    print("  " + "-" * (40 + 15 * len(datasets)))

    for column in columns:
        pooled = next(r["effect"] for r in results if r["column"] == column)
        cells = ""
        signs = []
        for dataset in datasets:
            subset = frame[frame["dataset"] == dataset]
            if subset["label"].nunique() < 2:
                cells += f"{'-':>15}"
                continue
            sub = analyse(subset, [column])[0]
            cells += f"{sub['effect']:>+15.3f}"
            signs.append(sub["effect"])
        warn = ""
        if signs and any(abs(s) < 0.147 for s in signs) and abs(pooled) >= 0.147:
            warn = "  <- collapses"
        # A sign flip is only meaningful between two effects large enough to
        # have a direction. Reporting -0.003 as "flipped" from +0.5 dresses up
        # noise as a contradiction; that column has simply vanished, which the
        # collapse check above already says.
        directional = [s for s in signs if abs(s) >= 0.147]
        if len({s > 0 for s in directional}) > 1:
            warn = "  <- sign flips"
        if signs and all(abs(s) > abs(pooled) for s in signs):
            warn = "  <- stronger within every corpus"
        print(f"  {column:<30}{pooled:>+10.3f}{cells}{warn}")

    print("\n  An effect that holds in the pooled data but collapses or flips sign inside")
    print("  each corpus is a property of how the corpora were merged, not of prompts.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=None,
                        help="feature table (default: newest in data/features/)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--by-dataset", action="store_true",
                        help="recompute the strongest effects within each corpus")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. pip install pandas scipy")
        return 1

    path = args.features
    if path is None:
        candidates = sorted(FEATURES_DIR.glob("features.*"))
        if not candidates:
            print(f"ERROR: no feature table in {FEATURES_DIR}. "
                  f"Run eval/build_features.py first.")
            return 1
        path = candidates[0]

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    if "split" not in frame.columns:
        print("ERROR: feature table has no 'split' column - rebuild it.")
        return 1

    train = frame[frame["split"] == "train"]
    if train.empty:
        print("ERROR: no training rows. Was the table built before the 70/30 change?")
        return 1

    print(f"\n  source        {path}")
    print(f"  rows          {len(frame):,} total, {len(train):,} train "
          f"({len(frame[frame['split'] == 'test']):,} test held out and untouched)")
    print(f"  balance       {train['label'].mean():.2%} malicious in train")

    columns = [c for c in train.columns
               if c not in ID_COLUMNS and train[c].dtype.kind in "bifu"]
    dropped = [c for c in train.columns if c not in ID_COLUMNS and c not in columns]
    print(f"  columns       {len(columns)} numeric/boolean tested"
          + (f", {len(dropped)} non-numeric skipped ({', '.join(dropped)})" if dropped else ""))

    results = analyse(train, columns)
    report_associations(results, args.top)
    report_redundancy(train, columns, args.corr_threshold)
    if args.by_dataset:
        report_by_dataset(train, results, min(args.top, 10))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
