"""Train Track B's recurrent baselines - BiLSTM and BiGRU.

    python training/fetch_fasttext.py                      # once, downloads vectors
    python training/finetune_rnn.py --cell lstm --out models/bilstm
    python training/finetune_rnn.py --cell gru  --out models/bigru

Answers adviser comment A13. Shares everything it can with
``training/finetune_e5.py`` - the same ``build_splits``, so all three Track B
models and both Track A models land on the identical sealed test rows, and the
same ``evaluate``, so a number in one row of the comparison table means what it
means in every other row.

**Dynamic padding, unlike the E5 trainer's fixed 512.** Prompts in this corpus
tokenize to a median of ~88 words; padding every batch to a fixed 512 does
roughly 5.8x the necessary work. Each batch is padded to its own longest row
instead, which is free to implement here because the vocabulary is ours and
costs nothing in comparability - the model sees the same tokens either way.

Selection is on validation F1 at each epoch, matching the E5 trainer. The
threshold is *not* chosen here: Track A's models had their cutoffs selected on
the validation split by ``eval/baseline_models.py``, and Track B's are selected
the same way, afterwards, by the same code. Reporting a tuned Track A model
against an untuned Track B one would rig the comparison in Track A's favour.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.finetune_e5 import Split, build_splits, evaluate  # noqa: E402

DEFAULT_VECTORS = REPO / "data" / "vectors"


def make_loader(split: Split, vocabulary, *, batch_size: int, max_length: int,
                shuffle: bool, torch):
    """A DataLoader that tokenises inside ``collate_fn``.

    Encoding per batch rather than up front is what makes dynamic padding
    possible: the batch's own longest row sets the width, so a batch of short
    prompts stays narrow instead of being stretched to the corpus maximum.
    """
    from torch.utils.data import DataLoader, Dataset

    from training.rnn_models import encode

    class TextDataset(Dataset):
        def __len__(self):
            return len(split.texts)

        def __getitem__(self, index):
            return split.texts[index], split.labels[index]

    def collate(batch):
        texts = [text for text, _ in batch]
        labels = [label for _, label in batch]
        input_ids, attention_mask = encode(texts, vocabulary, max_length)
        return input_ids, attention_mask, torch.tensor(labels)

    return DataLoader(TextDataset(), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "clean")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: models/bi<cell>")
    parser.add_argument("--cell", choices=("lstm", "gru"), default="lstm")
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS,
                        help="directory holding fastText .vec files")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-vocab", type=int, default=40_000)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows, for a fast smoke run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out or REPO / "models" / f"bi{args.cell}"

    try:
        import torch
    except ImportError:
        print("ERROR: torch is required.\n"
              "  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return 1

    from training.rnn_models import (
        BiRNNClassifier,
        build_vocabulary,
        load_fasttext_matrix,
    )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: no CUDA device found - this will be slow.")

    train, validation, test = build_splits(args.data, limit=args.limit)
    print(f"\n  train {len(train):,}   validation {len(validation):,}   "
          f"test {len(test):,}   ({test.positive_rate:.1%} malicious in test)")

    # Train split only - see build_vocabulary on why this is not negotiable.
    vocabulary = build_vocabulary(train.texts, max_size=args.max_vocab)
    print(f"  vocabulary {len(vocabulary):,} words (fitted on train only)")

    vector_files = sorted(Path(args.vectors).glob("*.vec")) if args.vectors else []
    if not vector_files:
        print(f"\nERROR: no .vec files in {args.vectors}.\n"
              f"Run python training/fetch_fasttext.py first - the RNN track is "
              f"specified to use pretrained fastText embeddings, and falling "
              f"back to random initialisation silently would make the "
              f"comparison measure something else.")
        return 1

    # English before Tagalog: earlier files win on overlapping words, and this
    # corpus is overwhelmingly English.
    vector_files.sort(key=lambda p: (".tl." in p.name or p.name.startswith("tl"), p.name))
    print(f"  loading vectors from {', '.join(p.name for p in vector_files)} ...")
    matrix, coverage = load_fasttext_matrix(vocabulary, vector_files)
    print(f"  fastText coverage {coverage['coverage']:.1%} "
          f"({coverage['covered']:,}/{coverage['vocabulary_size']:,} words)")
    for name, hits in coverage["per_file"].items():
        print(f"    {name:<28}{hits:>8,} vectors used")

    model = BiRNNClassifier(
        len(vocabulary), cell=args.cell, hidden_size=args.hidden_size,
        num_layers=args.num_layers, embedding_matrix=matrix,
        freeze_embeddings=args.freeze_embeddings,
    ).to(device)

    counts = model.parameter_count()
    print(f"  bi{args.cell.upper()}  {counts['total']:,} parameters "
          f"({counts['embedding']:,} embedding, "
          f"{counts['recurrent_and_head']:,} recurrent + head)")

    loaders = {
        name: make_loader(split, vocabulary, batch_size=args.batch_size,
                          max_length=args.max_length, shuffle=(name == "train"),
                          torch=torch)
        for name, split in (("train", train), ("validation", validation),
                            ("test", test))
    }

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    history: list[dict] = []
    best_f1 = -1.0
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        running = 0.0
        for step, (input_ids, attention_mask, labels) in enumerate(loaders["train"], 1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).float()

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.squeeze(-1), labels)
            optimizer.zero_grad()
            loss.backward()
            # Recurrence over hundreds of steps is where gradients explode;
            # clipping is cheap insurance and standard for this architecture.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss {running / step:.4f}", flush=True)

        metrics = evaluate(model, loaders["validation"], device, torch)
        metrics.update(epoch=epoch, train_loss=running / max(len(loaders["train"]), 1),
                       seconds=round(time.perf_counter() - started, 1))
        history.append(metrics)
        print(f"  epoch {epoch}: val f1 {metrics['f1']:.4f} "
              f"recall {metrics['recall']:.4f} fpr {metrics['fpr']:.4f} "
              f"({metrics['seconds']}s)")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out / "model.pt")
            (out / "vocabulary.json").write_text(
                json.dumps(vocabulary), encoding="utf-8")
            print(f"  saved (best f1 so far) -> {out}")

    model.load_state_dict(torch.load(out / "model.pt"))
    report = {
        "model": f"bi{args.cell}",
        "history": history,
        "best_val_f1": best_f1,
        "parameters": counts,
        "fasttext_coverage": coverage,
        "config": vars(args) | {"device": device},
        "test": evaluate(model, loaders["test"], device, torch),
    }
    report["config"] = {k: str(v) if isinstance(v, Path) else v
                        for k, v in report["config"].items()}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  test f1 {report['test']['f1']:.4f} "
          f"recall {report['test']['recall']:.4f} "
          f"fpr {report['test']['fpr']:.4f}")
    print(f"  Wrote {out / 'report.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
