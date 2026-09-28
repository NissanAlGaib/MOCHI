"""How well does detection survive a switch into Taglish?

    python eval/taglish_report.py
    python eval/taglish_report.py --skip-track-b      # classical models only
    python eval/taglish_report.py --examples 10       # show missed Taglish rows

Reports recall on malicious **English** rows against recall on malicious
**Taglish** rows, per model, on the same sealed test split.

**This is the multilingual result; the headline F1 cannot be.** Taglish is 3.1%
of the test set (797 of 25,639 rows, 396 of them malicious), so a model that
failed completely on code-switched text would move the overall F1 by under a
point. Only a split by language shows whether detection actually transfers.

Thresholds come from the same ``select_threshold`` call the comparison uses, on
the same validation split, so a model is measured at the operating point it
would actually run at - not at whichever cutoff happens to flatter it on the
smaller subset.

**Read the Taglish column with its sample size in mind.** 396 rows carries
roughly +/-2 points of sampling error at these rates, which is enough to
support "detection degrades by a few points" and not enough to rank two models
whose Taglish recall differs by one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.baseline_models import (  # noqa: E402
    build_track_a_models,
    decision_scores,
    select_threshold,
)
from eval.data_loading import (  # noqa: E402
    DATA_DIR,
    TEST_SHARE,
    VALIDATION_SHARE,
    load_file,
    stratified_split,
)

#: Source files whose rows are code-switched. Matched on the dataset name that
#: ``load_file`` derives from the filename.
TAGLISH_MARKER = "taglish"


class _Split:
    """Minimal stand-in for ``finetune_e5.Split`` that keeps the Sample rows.

    ``build_splits`` discards them, and the per-language breakdown needs to know
    which source file each row came from - which is the whole point here.
    """

    def __init__(self, samples):
        self.samples = samples
        self.texts = [s.text for s in samples]
        self.labels = [s.label for s in samples]

    def __len__(self):
        return len(self.samples)


def load_splits(data_dir: Path):
    pooled = []
    for path in sorted(data_dir.glob("*.csv")):
        pooled.extend(load_file(path))
    train, validation, test = stratified_split(
        pooled, test=TEST_SHARE, validation=VALIDATION_SHARE)
    return _Split(train), _Split(validation), _Split(test)


def recall_by_language(scores, split: _Split, threshold: float) -> dict:
    """Recall on malicious rows, split by whether the row is code-switched."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(split.labels)
    taglish = np.array([TAGLISH_MARKER in s.dataset for s in split.samples])

    out = {}
    for name, mask in (("english", ~taglish), ("taglish", taglish)):
        selected = mask & (labels == 1)
        n = int(selected.sum())
        caught = int((scores[selected] >= threshold).sum())
        out[name] = {"recall": caught / n if n else 0.0, "n": n,
                     "missed": n - caught}
    out["delta"] = out["taglish"]["recall"] - out["english"]["recall"]

    # Reported beside recall because recall alone is the one metric a model can
    # max out by saying yes to everything. The linear SVM reaches the highest
    # recall in this study at FPR 0.55 - it flags over half of all legitimate
    # traffic - and without this column its Taglish number reads as multilingual
    # skill rather than as indiscriminate flagging.
    benign = labels == 0
    out["fpr"] = float((scores[benign] >= threshold).mean()) if benign.any() else 0.0
    return out


def score_track_a(train, validation, test, cv_path: Path):
    import json

    params = {}
    if cv_path.exists():
        saved = json.loads(cv_path.read_text(encoding="utf-8"))
        params = {name: row["params"]
                  for name, row in saved.get("cross_validation", {}).items()}

    for name, model in build_track_a_models(params=params).items():
        print(f"  fitting {name} ...", flush=True)
        model.fit(train.texts, train.labels)
        threshold = select_threshold(
            decision_scores(model, validation.texts), validation.labels)
        yield name, decision_scores(model, test.texts), threshold


def score_track_b(validation, test, models_dir: Path):
    """Load each saved Track B model and score both splits.

    Skips any model that is not on disk rather than failing: the trainers are
    slow and this report is useful with whichever subset has finished.
    """
    import json

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cell in ("lstm", "gru"):
        directory = models_dir / f"bi{cell}"
        if not (directory / "model.pt").exists():
            print(f"  skipping bi{cell} (not trained yet)")
            continue
        print(f"  loading bi{cell} ...", flush=True)
        from training.rnn_models import BiRNNClassifier, encode

        vocabulary = json.loads(
            (directory / "vocabulary.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        model = BiRNNClassifier(len(vocabulary), cell=cell,
                                hidden_size=report["config"]["hidden_size"])
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu"))
        model.to(device).eval()

        def score(texts, _m=model, _v=vocabulary):
            values = []
            with torch.no_grad():
                for start in range(0, len(texts), 128):
                    ids, mask = encode(texts[start:start + 128], _v, 512)
                    logits, _ = _m(input_ids=ids.to(device),
                                   attention_mask=mask.to(device))
                    values.extend(torch.sigmoid(logits.squeeze(-1)).cpu().tolist())
            return values

        threshold = select_threshold(score(validation.texts), validation.labels)
        yield f"bi{cell.upper()}", score(test.texts), threshold

    directory = models_dir / "e5-fine-tuned"
    if not (directory / "mochi_head.pt").exists():
        print("  skipping e5 (not trained yet)")
        return

    print("  loading e5 ...", flush=True)
    from transformers import AutoTokenizer

    from training.finetune_e5 import encode as e5_encode
    from training.model import InjectionClassifier

    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    model = InjectionClassifier.load(directory, device=device).eval()

    def score(texts, _m=model, _t=tokenizer):
        values = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                batch = e5_encode(texts[start:start + 64], _t, 512)
                logits, _ = _m(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device))
                values.extend(torch.sigmoid(logits.squeeze(-1)).cpu().tolist())
        return values

    threshold = select_threshold(score(validation.texts), validation.labels)
    yield "E5", score(test.texts), threshold


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--models", type=Path, default=REPO / "models")
    parser.add_argument("--cv", type=Path, default=REPO / "reports" / "track_a.json")
    parser.add_argument("--skip-track-b", action="store_true")
    parser.add_argument("--examples", type=int, default=0,
                        help="print this many missed Taglish rows per model")
    args = parser.parse_args()

    train, validation, test = load_splits(args.data)
    taglish = sum(1 for s in test.samples if TAGLISH_MARKER in s.dataset)
    print(f"\n  test {len(test):,} rows  ({taglish:,} taglish, "
          f"{len(test) - taglish:,} english)\n")

    rows = []
    scored = list(score_track_a(train, validation, test, args.cv))
    if not args.skip_track_b:
        try:
            scored += list(score_track_b(validation, test, args.models))
        except ImportError:
            print("  torch not installed - Track B skipped")

    for name, scores, threshold in scored:
        rows.append((name, recall_by_language(scores, test, threshold), scores,
                     threshold))

    print()
    print("=" * 86)
    print("  Recall on malicious rows, by language of the prompt")
    print("=" * 86)
    print(f"  {'model':<18}{'english':>10}{'taglish':>10}{'delta':>10}"
          f"{'missed':>9}{'FPR':>10}")
    print("  " + "-" * 82)
    for name, split, _scores, _t in sorted(rows, key=lambda r: -r[1]["taglish"]["recall"]):
        print(f"  {name:<18}{split['english']['recall']:>10.4f}"
              f"{split['taglish']['recall']:>10.4f}{split['delta']:>+10.4f}"
              f"{split['taglish']['missed']:>9,}{split['fpr']:>10.4f}")
    print("=" * 86)
    print("  delta is taglish minus english: negative means detection got worse")
    print("  when the same kind of attack was written in code-switched text.")
    print(f"  n = {rows[0][1]['taglish']['n']:,} malicious taglish rows (+/-2 points "
          f"of sampling error at these rates).")
    print("  Read recall WITH the FPR column: a model that flags everything scores")
    print("  high recall on every subset without detecting anything.")
    print()

    if args.examples:
        import numpy as np

        labels = np.asarray(test.labels)
        is_taglish = np.array([TAGLISH_MARKER in s.dataset for s in test.samples])
        for name, _split, scores, threshold in rows:
            missed = np.where((np.asarray(scores) < threshold)
                              & (labels == 1) & is_taglish)[0]
            print(f"  {name} missed {len(missed)} taglish injections; first "
                  f"{min(args.examples, len(missed))}:")
            for index in missed[: args.examples]:
                text = test.samples[index].text.replace("\n", " ")[:88]
                print(f"    score {scores[index]:.3f}  {text}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
