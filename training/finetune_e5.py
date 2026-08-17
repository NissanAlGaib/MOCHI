"""Fine-tune the Stage II classifier. Runs on Colab GPU in ~20 minutes.

    python training/finetune_e5.py --data data/clean --out models/e5-fine-tuned
    python training/finetune_e5.py --data data/clean --epochs 1 --limit 2000  # smoke test

Hyperparameters follow the thesis Chapter III specification: BCE loss, AdamW at
lr 2e-5, batch 32, 3 epochs.

Two ordering rules are enforced here rather than left to the operator, because
getting either wrong invalidates every downstream number:

1. **Deduplicate and split before augmenting.** Augmenting first leaks a
   paraphrase of a training example into the test set, and the resulting
   accuracy is fiction. ``eval/clean_datasets.py`` handles dedup; this script
   refuses to augment anything but the train split.

2. **PromptShield's official splits are respected.** Its train/validation/test
   files are used as-is so results stay comparable to the published numbers.
   Other sources are split with the seeded stratified splitter. Re-splitting
   everything together would discard that comparability and leak across
   PromptShield's own boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Split:
    texts: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def positive_rate(self) -> float:
        return sum(self.labels) / len(self.labels) if self.labels else 0.0


def build_splits(data_dir: Path, *, limit: int | None = None
                 ) -> tuple[Split, Split, Split]:
    """Assemble train/validation/test, honouring PromptShield's own splits."""
    from eval.data_loading import load_file, stratified_split

    official: dict[str, Split] = {"train": Split(), "validation": Split(),
                                  "test": Split()}
    other = []

    for path in sorted(data_dir.glob("*.csv")):
        stem = path.stem
        samples = load_file(path)
        matched = next(
            (name for name in official if stem.endswith(f"_{name}")), None
        )
        if matched:
            official[matched].texts.extend(s.text for s in samples)
            official[matched].labels.extend(s.label for s in samples)
        else:
            other.extend(samples)

    if other:
        extra_train, extra_val, extra_test = stratified_split(other)
        for split, extra in (("train", extra_train), ("validation", extra_val),
                             ("test", extra_test)):
            official[split].texts.extend(s.text for s in extra)
            official[split].labels.extend(s.label for s in extra)

    train, validation, test = (official["train"], official["validation"],
                               official["test"])
    if limit:
        for split in (train, validation, test):
            del split.texts[limit:]
            del split.labels[limit:]

    if not len(train) or not len(validation):
        raise SystemExit(
            f"Not enough data in {data_dir}. Run eval/clean_datasets.py first."
        )
    return train, validation, test


def encode(texts: list[str], tokenizer, max_length: int):
    # E5 was pretrained with "query: " / "passage: " prefixes and degrades
    # without one. Inspected content is the thing being classified, so it is a
    # passage.
    return tokenizer(
        [f"passage: {t}" for t in texts],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def evaluate(model, loader, device, torch) -> dict[str, float]:
    """Confusion matrix at the Stage II blocking threshold."""
    from mochi.detect.stage2_semantic import MALICIOUS_THRESHOLD

    model.eval()
    tp = fp = tn = fn = 0
    loss_total = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = logits.squeeze(-1)
            loss_total += criterion(logits, labels.float()).item() * len(labels)
            predicted = (torch.sigmoid(logits) >= MALICIOUS_THRESHOLD).long()
            tp += int(((predicted == 1) & (labels == 1)).sum())
            fp += int(((predicted == 1) & (labels == 0)).sum())
            tn += int(((predicted == 0) & (labels == 0)).sum())
            fn += int(((predicted == 0) & (labels == 1)).sum())

    total = tp + fp + tn + fn or 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": loss_total / total,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=REPO / "data" / "clean")
    parser.add_argument("--out", type=Path, default=REPO / "models" / "e5-fine-tuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows per split, for a fast smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer
    except ImportError:
        print("Stage II training needs torch and transformers:\n"
              "  pip install torch transformers\n"
              "On Colab both are preinstalled - see training/README.md.")
        return 1

    from training.model import DEFAULT_ENCODER, InjectionClassifier

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. Training on CPU will take hours.\n"
              "         Use Colab (training/README.md) or pass --limit for a smoke test.\n")

    print(f"Loading from {args.data}")
    train, validation, test = build_splits(args.data, limit=args.limit)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        print(f"  {name:<12}{len(split):>8,} rows, {split.positive_rate:.1%} malicious")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_ENCODER)
    model = InjectionClassifier(DEFAULT_ENCODER).to(device)

    def loader(split: Split, *, shuffle: bool) -> "DataLoader":
        batch = encode(split.texts, tokenizer, args.max_length)
        dataset = TensorDataset(batch["input_ids"], batch["attention_mask"],
                                torch.tensor(split.labels))
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = loader(train, shuffle=True)
    validation_loader = loader(validation, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        running = 0.0
        for step, (input_ids, attention_mask, labels) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).float()

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.squeeze(-1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f}")

        metrics = evaluate(model, validation_loader, device, torch)
        metrics.update(epoch=epoch, train_loss=running / max(len(train_loader), 1),
                       seconds=round(time.perf_counter() - started, 1))
        history.append(metrics)
        print(f"  epoch {epoch}: val f1 {metrics['f1']:.4f} "
              f"recall {metrics['recall']:.4f} fpr {metrics['fpr']:.4f} "
              f"({metrics['seconds']}s)")

        # Select on validation F1, not the last epoch - E5 overfits this corpus
        # within 3 epochs and the final checkpoint is usually not the best one.
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"  saved (best f1 so far) -> {args.out}")

    report = {"history": history, "best_val_f1": best_f1,
              "config": vars(args) | {"device": device}}
    if len(test):
        report["test"] = evaluate(model, loader(test, shuffle=False), device, torch)
        print(f"\ntest f1 {report['test']['f1']:.4f} "
              f"recall {report['test']['recall']:.4f} "
              f"fpr {report['test']['fpr']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {args.out / 'training_report.json'}")
    print("\nNext:\n  MOCHI_ENABLE_STAGE2=true python -m uvicorn mochi.gateway.app:app\n"
          "  python eval/run_detection.py --data data/clean --config stage1_stage2\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
