"""Step 4: the two-track comparison table.

    python eval/compare_tracks.py
    python eval/compare_tracks.py --json reports/comparison.json

Scores every trained model from both tracks on the identical sealed test rows,
under identical rules, and writes one table.

**Threshold parity is the reason this script exists.** Track A's models had
their cutoffs selected on validation across several FPR budgets; Track B's
trainers report at a fixed threshold. Comparing a tuned model against an untuned
one measures the tuning as much as the model, so every model here - both tracks -
gets its threshold selected by the same ``select_threshold`` call on the same
validation split, and is then scored once on test.

**Two devices, on purpose.** Scoring all 42,209 validation and test rows runs
batched on the GPU when one is present, because those probabilities are just an
intermediate - nothing about the comparison depends on where they were computed.
The **latency probe is always CPU, always batch size 1**, because that is the
number the thesis actually reports: what an inline gateway pays per request on
commodity hardware. Batched GPU throughput would flatter the transformer most,
and no real request arrives in a batch of 32.

Requires the models to exist: run ``eval/baseline_models.py --track-a`` for
Track A's hyperparameters, and both trainers in ``training/`` for Track B.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.baseline_models import (  # noqa: E402
    FPR_BUDGETS,
    build_track_a_models,
    decision_scores,
    metrics_at,
    select_threshold,
)
from eval.data_loading import DATA_DIR  # noqa: E402

TRACK_A_CV = REPO / "reports" / "track_a.json"
LATENCY_SAMPLES = 200


def track_a_scores(train, validation, test, *, cv_path: Path = TRACK_A_CV):
    """Fit Track A's models and return their validation/test scores.

    Hyperparameters come from the cross-validated search rather than the
    hand-picked defaults; without them this table would report models the CV
    was run to replace.
    """
    params = {}
    if cv_path.exists():
        saved = json.loads(cv_path.read_text(encoding="utf-8"))
        params = {name: row["params"]
                  for name, row in saved.get("cross_validation", {}).items()}
        print(f"  using cross-validated hyperparameters from {cv_path.name}")
    else:
        print(f"  WARNING: {cv_path} missing - falling back to hand-picked "
              f"defaults. Run eval/baseline_models.py --track-a first.")

    out = {}
    for name, model in build_track_a_models(params=params).items():
        print(f"  fitting {name} ...", flush=True)
        started = time.perf_counter()
        model.fit(train.texts, train.labels)
        train_seconds = time.perf_counter() - started

        out[name] = {
            "track": "A",
            "family": "classical",
            "input": "18 engineered columns",
            "validation": decision_scores(model, validation.texts),
            "test": decision_scores(model, test.texts),
            "train_seconds": train_seconds,
            "parameters": _sklearn_parameters(model),
            "latency_us": _track_a_latency(model, test.texts),
        }
    return out


def _track_a_latency(model, texts) -> float:
    """Per-request cost for a Track A model, measured honestly.

    Two corrections over a naive ``predict([text])`` timing, both of which
    change the number by more than the difference between models:

    **``n_jobs=1`` during the probe.** A forest configured ``n_jobs=-1``
    spawns joblib workers per call; at batch size 1 that dispatch costs more
    than traversing the trees, and the measured 52 ms was mostly scheduling.
    Parallelism helps throughput and hurts single-request latency.

    **The extractor cache is cleared first.** Every model shares one
    ``engineered_transformer``, and by this point it holds every validation and
    test row, so a naive probe would time a dictionary lookup instead of
    ``normalize()`` plus ``extract()``. A gateway pays that extraction on every
    request - it is part of Track A's cost, not an aside.
    """
    classifier = model.named_steps["clf"]
    original = getattr(classifier, "n_jobs", None)
    if original is not None:
        classifier.n_jobs = 1

    step = model.named_steps["vec"]
    extractor = step.named_steps["extract"] if hasattr(step, "named_steps") else step
    cache = getattr(extractor, "_cache", None)
    saved = dict(cache) if cache is not None else None
    if cache is not None:
        cache.clear()

    try:
        return _latency(lambda text: model.predict([text]), texts)
    finally:
        if cache is not None:
            cache.update(saved)
        if original is not None:
            classifier.n_jobs = original


def _sklearn_parameters(model) -> int:
    """A rough capacity figure, so the table's parameter column is not blank.

    Not comparable to a neural parameter count and not presented as one: a
    forest's number is its total node count, a linear model's is its
    coefficients. Both describe stored numbers, which is the only sense in
    which the columns align.
    """
    classifier = model.named_steps["clf"]
    if hasattr(classifier, "estimators_"):
        return int(sum(e.tree_.node_count for e in classifier.estimators_))
    if hasattr(classifier, "tree_"):
        return int(classifier.tree_.node_count)
    if hasattr(classifier, "coef_"):
        return int(classifier.coef_.size)
    inner = getattr(classifier, "estimator_", None)
    if inner is not None and hasattr(inner, "support_vectors_"):
        return int(inner.support_vectors_.size)
    return 0


def _latency(predict_one, texts, *, samples: int = LATENCY_SAMPLES) -> float:
    """Microseconds per single-request prediction.

    One text at a time, deliberately. A gateway scores one prompt per request,
    and batching is the single biggest source of flattering transformer
    benchmarks.
    """
    # Warm on rows *outside* the probe set, so lazy init is paid but the timed
    # rows are still cold - warming on the probe itself would time a cache hit.
    for text in texts[samples:samples + 3]:
        predict_one(text)
    probe = texts[:samples]
    started = time.perf_counter()
    for text in probe:
        predict_one(text)
    return (time.perf_counter() - started) / max(len(probe), 1) * 1e6


def track_b_scores(validation, test, *, models_dir: Path, device: str = "cpu"):
    """Load each trained Track B model and score both splits.

    Scores are ``sigmoid(logit)`` so they share Track A's [0, 1] probability
    scale. Thresholds are still selected per model - a shared scale does not
    make one model's cutoff meaningful for another - but it keeps the reported
    numbers readable against one another.

    Bulk scoring honours ``device``; the latency probe re-runs each model on CPU
    regardless, so the reported per-request cost is hardware the thesis can
    claim without assuming a GPU in production.
    """
    import torch

    out = {}

    for cell in ("lstm", "gru"):
        directory = models_dir / f"bi{cell}"
        if not (directory / "model.pt").exists():
            print(f"  skipping bi{cell} - no model at {directory}")
            continue
        print(f"  loading bi{cell} ...", flush=True)

        from training.rnn_models import BiRNNClassifier, encode

        vocabulary = json.loads(
            (directory / "vocabulary.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        model = BiRNNClassifier(len(vocabulary), cell=cell,
                                hidden_size=report["config"]["hidden_size"])
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu"))
        model.eval()

        def score(texts, _m=model, _v=vocabulary, _d=device):
            _m.to(_d)
            values = []
            with torch.no_grad():
                for start in range(0, len(texts), 128):
                    chunk = texts[start:start + 128]
                    ids, mask = encode(chunk, _v, 512)
                    logits, _ = _m(input_ids=ids.to(_d), attention_mask=mask.to(_d))
                    values.extend(torch.sigmoid(logits.squeeze(-1)).cpu().tolist())
            return values

        def score_cpu(texts, _m=model, _v=vocabulary):
            _m.to("cpu")
            with torch.no_grad():
                ids, mask = encode(texts, _v, 512)
                logits, _ = _m(input_ids=ids, attention_mask=mask)
                return torch.sigmoid(logits.squeeze(-1)).tolist()

        out[f"bi{cell.upper()}"] = {
            "track": "B",
            "family": "recurrent",
            "input": "raw text (fastText)",
            "validation": score(validation.texts),
            "test": score(test.texts),
            "train_seconds": sum(h["seconds"] for h in report["history"]),
            "parameters": report["parameters"]["total"],
            "latency_us": _latency(lambda t, _s=score_cpu: _s([t]), test.texts),
        }

    directory = models_dir / "e5-fine-tuned"
    if (directory / "mochi_head.pt").exists():
        print("  loading e5 ...", flush=True)
        from transformers import AutoTokenizer

        from training.finetune_e5 import encode as e5_encode
        from training.model import InjectionClassifier

        tokenizer = AutoTokenizer.from_pretrained(str(directory))
        model = InjectionClassifier.load(directory, device="cpu").eval()
        report = json.loads(
            (directory / "training_report.json").read_text(encoding="utf-8"))

        def score(texts, _m=model, _t=tokenizer, _d=device):
            _m.to(_d)
            values = []
            with torch.no_grad():
                for start in range(0, len(texts), 64):
                    batch = e5_encode(texts[start:start + 64], _t, 512)
                    logits, _ = _m(input_ids=batch["input_ids"].to(_d),
                                   attention_mask=batch["attention_mask"].to(_d))
                    values.extend(torch.sigmoid(logits.squeeze(-1)).cpu().tolist())
            return values

        def score_cpu(texts, _m=model, _t=tokenizer):
            _m.to("cpu")
            with torch.no_grad():
                batch = e5_encode(texts, _t, 512)
                logits, _ = _m(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"])
                return torch.sigmoid(logits.squeeze(-1)).tolist()

        out["E5 (transformer)"] = {
            "track": "B",
            "family": "transformer",
            "input": "raw text (subword)",
            "validation": score(validation.texts),
            "test": score(test.texts),
            "train_seconds": sum(h["seconds"] for h in report["history"]),
            "parameters": sum(p.numel() for p in model.parameters()),
            "latency_us": _latency(lambda t, _s=score_cpu: _s([t]), test.texts,
                                   samples=60),
        }
    else:
        print(f"  skipping e5 - no model at {directory}")

    return out


def build_rows(scored: dict, validation, test) -> list[dict]:
    """Tune every model's threshold on validation, then score once on test."""
    rows = []
    for name, entry in scored.items():
        objectives = {
            "max F1": select_threshold(entry["validation"], validation.labels),
        }
        for budget in FPR_BUDGETS:
            objectives[f"FPR<={budget:g}"] = select_threshold(
                entry["validation"], validation.labels, max_fpr=budget)

        results = {}
        for objective, threshold in objectives.items():
            results[objective] = metrics_at(entry["test"], test.labels, threshold)

        rows.append({
            "model": name,
            "track": entry["track"],
            "family": entry["family"],
            "input": entry["input"],
            "parameters": entry["parameters"],
            "train_seconds": entry["train_seconds"],
            "latency_us": entry["latency_us"],
            "results": results,
        })
    rows.sort(key=lambda r: -r["results"]["max F1"]["f1"])
    return rows


def print_table(rows: list[dict], n_validation: int, n_test: int) -> None:
    print()
    print("=" * 108)
    print(f"  Two-track comparison  -  thresholds tuned on validation "
          f"({n_validation:,}), scored on test ({n_test:,})")
    print("=" * 108)
    print(f"  {'model':<20}{'tr':>3}{'family':>13}{'F1':>9}{'recall':>9}"
          f"{'FPR':>9}{'prec':>9}{'params':>13}{'train':>9}{'cpu us/req':>12}")
    print("  " + "-" * 104)
    for row in rows:
        best = row["results"]["max F1"]
        print(f"  {row['model']:<20}{row['track']:>3}{row['family']:>13}"
              f"{best['f1']:>9.4f}{best['recall']:>9.4f}{best['fpr']:>9.4f}"
              f"{best['precision']:>9.4f}{row['parameters']:>13,}"
              f"{row['train_seconds']:>8.0f}s{row['latency_us']:>12,.0f}")
    print("=" * 108)

    print()
    print("  Recall at a fixed false-positive budget")
    print("  " + "-" * 104)
    budgets = [f"FPR<={b:g}" for b in FPR_BUDGETS]
    print(f"  {'model':<20}" + "".join(f"{b:>16}" for b in budgets))
    for row in rows:
        cells = "".join(f"{row['results'][b]['recall']:>16.4f}" for b in budgets)
        print(f"  {row['model']:<20}{cells}")
    print("=" * 108)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--models", type=Path, default=REPO / "models")
    parser.add_argument("--json", type=Path, default=REPO / "reports" / "comparison.json")
    parser.add_argument("--skip-track-b", action="store_true")
    args = parser.parse_args()

    from training.finetune_e5 import build_splits

    train, validation, test = build_splits(args.data)
    print(f"\n  train {len(train):,}   validation {len(validation):,}   "
          f"test {len(test):,}\n")

    scored = track_a_scores(train, validation, test)
    if not args.skip_track_b:
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
        print(f"  scoring Track B on {device} (latency probe stays on CPU)")
        scored.update(track_b_scores(validation, test, models_dir=args.models,
                                     device=device))

    rows = build_rows(scored, validation, test)
    print_table(rows, len(validation), len(test))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  Wrote {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
