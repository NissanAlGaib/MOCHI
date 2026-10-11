"""Per-source detection breakdown on the sealed test split.

    python eval/build_sealed_test.py
    python eval/per_source_detection.py --config stage12

``run_detection.py`` reports one confusion matrix for everything it loads,
which answers "does the gateway work" and cannot answer "on what does it
fail". Those are different questions here because the sealed split is a blend
of populations rather than one: ``jayavibhav.csv`` supplies 47% of its rows,
and register item D6 already records its benign half as not
application-representative.

The aggregate turns out to be carrying that source. It supplies 104 of 167
false positives and 51 of 78 misses, and excluding it moves FPR from 1.15% to
0.75% - from missing the <1% target to meeting it. The aggregate stays the
headline; this is the explanation that has to travel with it, rather than a
licence to quietly drop a hard source.

The breakdown also isolates the Taglish arm, which is the premise of the system
and its weakest case: FPR 3.24% against PromptShield's 0.62%.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.data_loading import Sample  # noqa: E402
from eval.predictors import get_predictor  # noqa: E402

DEFAULT_IN = REPO / "data" / "splits" / "stage12_test.csv"
DEFAULT_OUT = REPO / "reports" / "detection_stage12_per_source.json"

#: Rows of each kind to keep for inspection. Enough to read for a pattern,
#: not so many that the report becomes a copy of the corpus.
KEEP_EXAMPLES = 400


def rates(counts: dict[str, int]) -> dict[str, float]:
    """Recall, FPR, precision and F1, with NaN where undefined.

    A source with no benign rows has no false-positive rate, and reporting 0.0
    for it would read as a perfect score rather than an absent measurement.
    """
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    nan = float("nan")
    recall = tp / (tp + fn) if tp + fn else nan
    fpr = fp / (fp + tn) if fp + tn else nan
    precision = tp / (tp + fp) if tp + fp else nan
    f1 = nan
    if precision == precision and recall == recall and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {"recall": recall, "fpr": fpr, "precision": precision, "f1": f1}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_IN)
    parser.add_argument("--config", default="stage12")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"\nERROR: {args.data} not found."
              f"\nRun: python eval/build_sealed_test.py\n")
        return 1

    with args.data.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"\n{len(rows):,} rows from {args.data.name}")

    predictor = get_predictor(args.config)
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    misses: list[dict] = []
    false_alarms: list[dict] = []

    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        label = int(row["label"])
        prediction = predictor.predict(
            Sample(text=row["text"], label=label, dataset=row["source"])).label
        bucket = counts[row["source"]]
        if label == 1:
            bucket["tp" if prediction == 1 else "fn"] += 1
            if prediction == 0:
                misses.append(row)
        else:
            bucket["fp" if prediction == 1 else "tn"] += 1
            if prediction == 1:
                false_alarms.append(row)
        if index % 5000 == 0:
            print(f"  {index:,}/{len(rows):,}  "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    def show(name: str, bucket: dict[str, int]) -> None:
        total = sum(bucket.values())
        malicious = bucket["tp"] + bucket["fn"]
        r = rates(bucket)
        cell = lambda v: f"{v:>8.4f}" if v == v else f"{'-':>8}"  # noqa: E731
        print(f"{name:<30} {total:>6,} {malicious:>6,} "
              f"{cell(r['recall'])} {cell(r['fpr'])} {cell(r['f1'])}")

    print(f"\n{'source':<30} {'n':>6} {'mal':>6} {'recall':>8} {'FPR':>8} {'F1':>8}")
    print("-" * 72)
    for source in sorted(counts, key=lambda s: -sum(counts[s].values())):
        show(source, counts[source])

    def combine(keys) -> dict[str, int]:
        return {k: sum(counts[s][k] for s in keys) for k in ("tp", "fp", "tn", "fn")}

    print("-" * 72)
    show("TOTAL", combine(counts))

    print("\nsubsets")
    print("-" * 72)
    for label, keys in [
        ("excluding jayavibhav", [s for s in counts if "jayavibhav" not in s]),
        ("promptshield only", [s for s in counts if "promptshield" in s]),
        ("taglish only", [s for s in counts if "taglish" in s]),
    ]:
        if keys:
            show(label, combine(keys))

    print("\nerrors by source")
    print(f"  misses        {dict(Counter(m['source'] for m in misses))}")
    print(f"  false alarms  {dict(Counter(f['source'] for f in false_alarms))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "n_samples": len(rows),
        "per_source": {k: dict(v) for k, v in counts.items()},
        "total": combine(counts),
        "misses": misses[:KEEP_EXAMPLES],
        "false_alarms": false_alarms[:KEEP_EXAMPLES],
    }
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
