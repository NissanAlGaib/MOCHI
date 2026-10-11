"""What was the Taglish contamination actually worth?

    python eval/taglish_contamination.py

``data/clean/taglish_heldout.csv`` was pooled into the corpus that trained
``e5-fine-tuned``: 209 of its 300 rows reached train or validation, leaving 91
valid for evaluation. The procedural consequence is clear - a score measured on
rows a model fitted on measures memorisation. The empirical consequence is a
separate question, and it is answerable rather than assumable, by scoring the
same model on the rows it saw and the rows it did not.

``e5-banking`` settles it from the other side. Its corpus omits
``taglish_heldout.csv`` entirely, so all 300 rows are unseen and give a clean
measurement with no subsetting.

The answer, measured: the contamination was worth almost nothing. Recall is
identical on seen and unseen rows, and the false-positive rate is slightly
*better* on the unseen ones. The likely reason is that
``taglish_generated.csv`` contributes 2,400 rows of similar synthetic Taglish
to training regardless, so an individual held-out row carries little the model
has not already generalised from.

That is a weaker claim than it sounds, and the limitation it leaves is the one
worth writing down. It shows the model is not memorising
``taglish_heldout.csv``; it does not show the model handles Taglish as native
speakers write it, because the held-out set may be distributionally close to
the generated set it was drawn alongside. Native-speaker validation remains
unaddressed and no amount of retraining substitutes for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.data_loading import Sample, load_file  # noqa: E402
from eval.predictors import MochiPredictor  # noqa: E402
from training.split_manifest import row_hash  # noqa: E402

HELDOUT = REPO / "data" / "sealed" / "taglish_heldout.csv"


def seen_hashes(model: str) -> frozenset[str]:
    """Rows the model fitted on - train plus validation. Validation counts:
    it selects the checkpoint, so a score on it is not independent."""
    path = REPO / "models" / model / "split_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(manifest["train"]) | frozenset(manifest["validation"])


def score(model: str, samples: list[Sample], *, device: str) -> dict:
    from mochi.detect.stage2_semantic import get_detector

    detector = get_detector(str(REPO / "models" / model), device=device)
    predictor = MochiPredictor(stages="stage12", stage2=detector)

    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for sample in samples:
        prediction = predictor.predict(sample).label
        if sample.label == 1:
            counts["tp" if prediction == 1 else "fn"] += 1
        else:
            counts["fp" if prediction == 1 else "tn"] += 1

    nan = float("nan")
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    recall = tp / (tp + fn) if tp + fn else nan
    fpr = fp / (fp + tn) if fp + tn else nan
    precision = tp / (tp + fp) if tp + fp else nan
    f1 = nan
    if precision == precision and recall == recall and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {**counts, "recall": recall, "fpr": fpr, "f1": f1}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path,
                        default=REPO / "reports" / "taglish_contamination.json")
    args = parser.parse_args()

    if not HELDOUT.exists():
        print(f"\nERROR: {HELDOUT} not found.\n")
        return 1

    held = load_file(HELDOUT, dataset_name="taglish_heldout")
    malicious = sum(1 for s in held if s.label == 1)
    print(f"\n{HELDOUT.name}: {len(held)} rows "
          f"({malicious} malicious / {len(held) - malicious} benign)")

    fitted = seen_hashes("e5-fine-tuned")
    unseen = [s for s in held if row_hash(s.text) not in fitted]
    memorised = [s for s in held if row_hash(s.text) in fitted]
    print(f"  e5-fine-tuned saw {len(memorised)}, did not see {len(unseen)}")
    print(f"  e5-banking saw 0 (taglish_heldout.csv is not in its corpus)")

    arms = [
        ("e5-fine-tuned", "rows it did NOT see", unseen),
        ("e5-fine-tuned", "rows it DID see", memorised),
        ("e5-banking", "all rows, none seen", held),
    ]

    print(f"\n{'model':<16}{'subset':<24}{'n':>5}{'mal':>5}"
          f"{'recall':>9}{'FPR':>9}{'F1':>9}")
    print("-" * 77)
    results = []
    for model, label, samples in arms:
        if not samples:
            continue
        r = score(model, samples, device=args.device)
        cell = lambda v: f"{v:>9.4f}" if v == v else f"{'-':>9}"  # noqa: E731
        print(f"{model:<16}{label:<24}{len(samples):>5}{r['tp'] + r['fn']:>5}"
              f"{cell(r['recall'])}{cell(r['fpr'])}{cell(r['f1'])}")
        results.append({"model": model, "subset": label,
                        "n": len(samples), **r})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
