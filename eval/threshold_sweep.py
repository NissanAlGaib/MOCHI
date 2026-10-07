"""Resolve Stage II's decision thresholds from the score distribution.

``reports/comparison.json`` records E5 at four operating points, which is enough
to rank models and not enough to site a threshold: it says what happens *at*
0.5036 and 0.8539 and nothing about the 0.35 of scale between them. Two numbers
in ``mochi.detect.stage2_semantic`` depend on what is in that gap -
``BENIGN_THRESHOLD`` and ``MALICIOUS_THRESHOLD`` - and both were chosen as a
symmetric band around 0.5 before any distribution existed.

This module dumps the per-sample scores the comparison threw away, then answers
the three questions the constants rest on:

1. **Where should the block threshold sit?** Selected on *validation* under a
   false-positive budget, reported once on test. Same protocol as
   ``eval.baseline_models.select_threshold``, for the same reason: a cutoff
   chosen on the test set is a cutoff fitted to the test set.

2. **Is the uncertain band populated?** The band exists so ambiguous content can
   be resolved by source trust instead of a Stage III LLM. If almost nothing
   scores inside it, that path is dead code and the honest design is a single
   boundary - a result worth stating rather than a band worth keeping.

3. **Can session risk accumulate?** ``mochi.session.risk_accumulator`` sums raw
   per-turn scores and escalates at 1.0, justified as "three turns that each
   looked slightly wrong" at ~0.35 apiece. That premise needs scores near 0.35
   to exist. A sharply bimodal model never emits them, and the priming chain the
   accumulator was built for would never accumulate.

Scores are cached in ``reports/e5_scores.json``; re-runs read the cache unless
``--rescore`` is passed. The cache is keyed by nothing - it is the caller's job
not to mix models - because the only producer is this script and the only
consumer is this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "clean"
SCORE_CACHE = REPO / "reports" / "e5_scores.json"

#: False-positive budgets to site candidate block thresholds under. A gateway
#: that blocks 1 in 100 legitimate requests is already intrusive; 0.005 and
#: 0.002 are included because the plateau may make them nearly free.
FPR_BUDGETS = (0.02, 0.01, 0.005, 0.002)

#: The band as currently shipped, for the "is it populated" question.
CURRENT_BAND = (0.45, 0.55)

#: Where risk_accumulator's premise lives: turns that look "slightly wrong"
#: without tripping the single-turn threshold.
ACCUMULATOR_WINDOW = (0.20, 0.45)


def score_splits(data_dir: Path, models_dir: Path, *,
                 device: str | None = None, batch_size: int = 64) -> dict:
    """Score validation and test with the fine-tuned E5, batched.

    Batched deliberately, unlike ``eval.compare_tracks._latency`` which scores
    one at a time on purpose. Nothing here is a latency claim - these are the
    same scores either way, and batching turns 20 minutes into about one.

    ``data_dir`` is pooled exactly as ``build_splits`` finds it, with no file
    excluded. That is not an oversight: the split must reproduce the one the
    weights were trained under, and dropping a file reshuffles every row, which
    would quietly move training rows into this test set. Whether a file *should*
    be in the pool is a separate question from whether it was.
    """
    import torch
    from transformers import AutoTokenizer

    from training.finetune_e5 import build_splits
    from training.finetune_e5 import encode as e5_encode
    from training.model import InjectionClassifier

    directory = models_dir / "e5-fine-tuned"
    if not (directory / "mochi_head.pt").exists():
        raise SystemExit(f"no E5 model at {directory}")

    _, validation, test = build_splits(data_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  validation {len(validation.texts):,}   test {len(test.texts):,}")
    print(f"  scoring on {device} (batch {batch_size})")

    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    model = InjectionClassifier.load(directory, device=device).eval()

    def score(texts: list[str]) -> list[float]:
        values: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = e5_encode(texts[start:start + batch_size], tokenizer, 512)
                logits, _ = model(input_ids=batch["input_ids"].to(device),
                                  attention_mask=batch["attention_mask"].to(device))
                values.extend(torch.sigmoid(logits.squeeze(-1)).cpu().tolist())
                if start and start % 6400 == 0:
                    print(f"    {start:,}/{len(texts):,}", flush=True)
        return values

    print("  validation ...", flush=True)
    validation_scores = score(validation.texts)
    print("  test ...", flush=True)
    test_scores = score(test.texts)

    return {
        "validation": {"scores": validation_scores,
                       "labels": [int(v) for v in validation.labels]},
        "test": {"scores": test_scores, "labels": [int(v) for v in test.labels]},
    }


def histogram(scores, labels, *, edges) -> list[dict]:
    """Count malicious and benign samples per score bin."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    rows = []
    for low, high in zip(edges, edges[1:]):
        # Final bin closes on the right so a score of exactly 1.0 is counted.
        inside = ((scores >= low) & (scores < high) if high < 1.0
                  else (scores >= low) & (scores <= high))
        rows.append({
            "low": round(low, 4),
            "high": round(high, 4),
            "malicious": int((inside & (labels == 1)).sum()),
            "benign": int((inside & (labels == 0)).sum()),
        })
    return rows


def band_population(scores, labels, low: float, high: float) -> dict:
    """How many samples fall in ``[low, high)``, and what share of the split."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    inside = (scores >= low) & (scores < high)
    total = len(scores)
    return {
        "low": low,
        "high": high,
        "malicious": int((inside & (labels == 1)).sum()),
        "benign": int((inside & (labels == 0)).sum()),
        "total": int(inside.sum()),
        "share": float(inside.sum() / total) if total else 0.0,
    }


def analyse(cache: dict) -> dict:
    """Select thresholds on validation, report them once on test."""
    from eval.baseline_models import metrics_at, select_threshold

    validation = cache["validation"]
    test = cache["test"]

    edges = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    candidates = {}
    for budget in FPR_BUDGETS:
        threshold = select_threshold(validation["scores"], validation["labels"],
                                     max_fpr=budget)
        candidates[f"FPR<={budget:g}"] = {
            "threshold": float(threshold),
            "test": metrics_at(test["scores"], test["labels"], threshold),
        }
    threshold = select_threshold(validation["scores"], validation["labels"])
    candidates["max F1"] = {
        "threshold": float(threshold),
        "test": metrics_at(test["scores"], test["labels"], threshold),
    }
    # The shipped constant, for a like-for-like comparison against the above.
    candidates["current (0.55)"] = {
        "threshold": 0.55,
        "test": metrics_at(test["scores"], test["labels"], 0.55),
    }

    return {
        "candidates": candidates,
        "histogram": {
            "validation": histogram(validation["scores"], validation["labels"],
                                    edges=edges),
            "test": histogram(test["scores"], test["labels"], edges=edges),
        },
        "bands": {
            "current_uncertain": band_population(
                test["scores"], test["labels"], *CURRENT_BAND),
            "accumulator_window": band_population(
                test["scores"], test["labels"], *ACCUMULATOR_WINDOW),
        },
    }


def report(analysis: dict) -> None:
    print()
    print("=" * 86)
    print("  Stage II threshold selection - chosen on validation, scored on test")
    print("=" * 86)
    print(f"  {'objective':<18}{'threshold':>11}{'F1':>9}{'FPR':>9}"
          f"{'recall':>9}{'FP':>8}{'FN':>8}")
    print("  " + "-" * 82)
    for name, entry in analysis["candidates"].items():
        m = entry["test"]
        print(f"  {name:<18}{entry['threshold']:>11.4f}{m['f1']:>9.4f}"
              f"{m['false_positive_rate']:>9.4f}{m['recall']:>9.4f}"
              f"{m['fp']:>8}{m['fn']:>8}")
    print("=" * 86)

    print()
    print("  Test-set score distribution")
    print("  " + "-" * 82)
    print(f"  {'bin':>16}{'malicious':>12}{'benign':>10}   distribution")
    rows = analysis["histogram"]["test"]
    peak = max(max(r["malicious"], r["benign"]) for r in rows) or 1
    for r in rows:
        bar_m = "#" * int(36 * r["malicious"] / peak)
        bar_b = "." * int(36 * r["benign"] / peak)
        label = f"[{r['low']:.2f},{r['high']:.2f})"
        print(f"  {label:>16}{r['malicious']:>12,}{r['benign']:>10,}   "
              f"{bar_m or bar_b}")
    print("  legend: # malicious-dominated bin, . benign-dominated")
    print("=" * 86)

    print()
    print("  Design questions")
    print("  " + "-" * 82)
    band = analysis["bands"]["current_uncertain"]
    print(f"  uncertain band [{band['low']}, {band['high']}): "
          f"{band['total']:,} samples ({band['share']:.4%} of test) "
          f"- {band['malicious']:,} malicious, {band['benign']:,} benign")
    window = analysis["bands"]["accumulator_window"]
    print(f"  accumulator window [{window['low']}, {window['high']}): "
          f"{window['total']:,} samples ({window['share']:.4%} of test) "
          f"- {window['malicious']:,} malicious, {window['benign']:,} benign")
    print("=" * 86)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--models", type=Path, default=REPO / "models")
    parser.add_argument("--scores", type=Path, default=SCORE_CACHE,
                        help="per-sample score cache")
    parser.add_argument("--rescore", action="store_true",
                        help="re-run the model instead of reading the cache")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda when available)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--json", type=Path,
                        default=REPO / "reports" / "thresholds.json")
    args = parser.parse_args()

    if args.rescore or not args.scores.exists():
        cache = score_splits(args.data, args.models,
                             device=args.device, batch_size=args.batch_size)
        args.scores.parent.mkdir(parents=True, exist_ok=True)
        args.scores.write_text(json.dumps(cache), encoding="utf-8")
        print(f"  scores cached -> {args.scores}")
    else:
        print(f"  reading cached scores from {args.scores} (--rescore to refresh)")
        cache = json.loads(args.scores.read_text(encoding="utf-8"))

    analysis = analyse(cache)
    report(analysis)
    args.json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"  written -> {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
