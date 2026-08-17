# Stage II training

Fine-tunes `intfloat/multilingual-e5-small` into the Stage II injection
classifier. Needs a GPU — on CPU this takes hours, on a free Colab T4 about
20 minutes.

## Before you train

The corpus must be cleaned and balanced first. Training on raw `data/` leaks
duplicates across the train/test boundary and lets `jayavibhav` dominate:

```bash
python eval/clean_datasets.py --data data/ --out data/clean/
```

Order matters and the script enforces it: **deduplicate, then split, then
augment** — and only the train split. Augmenting before splitting puts a
paraphrase of a training row into the test set, and the resulting accuracy is
fiction.

## On Colab

```python
!git clone <your-repo-url> mochi && cd mochi
!pip install -q transformers
# upload data/clean/ or mount it from Drive
!python training/finetune_e5.py --data data/clean --out models/e5-fine-tuned
```

Then download `models/e5-fine-tuned/` and drop it in the repo root.

## Locally (smoke test only)

```bash
pip install torch transformers
python training/finetune_e5.py --data data/clean --epochs 1 --limit 2000
```

This verifies the plumbing. It will not produce a usable model.

## Running with the trained model

```bash
MOCHI_ENABLE_STAGE2=true python -m uvicorn mochi.gateway.app:app
python eval/run_detection.py --data data/clean --config stage12
```

Stage II is **off by default** (`MOCHI_ENABLE_STAGE2=false`). Enabling it with
no model present fails at startup with instructions rather than silently
falling back to Stage I — a gateway that ran one stage while its operator
believed it ran two would report false coverage.

## Architecture notes

Three choices differ from the library defaults, each because a measurement said
the default was wrong. All three are documented at length in
[`model.py`](model.py) and [`../mochi/detect/stage2_semantic.py`](../mochi/detect/stage2_semantic.py).

| Choice | Default | What MOCHI does | Why |
|---|---|---|---|
| Pooling | mean | **gated attention** | Median malicious span is 3.4% of its document; mean pooling averages it into noise |
| Long input | `truncation=True` | **chunk + take max** | 14.8% of attack signal sits in the document tail |
| Explanation | none | **attention weights** | Answers "which token contributed"; Phase 10 SANITIZE needs a span to redact |

Attention pooling follows Ilse, Tomczak & Welling (2018), *Attention-based Deep
Multiple Instance Learning* (ICML) — the standard framing for "the bag carries
the label, a few instances carry the evidence," which is exactly a document with
an injected sentence.

## Hyperparameters

Per thesis Chapter III: BCE loss, AdamW, lr 2e-5, batch 32, 3 epochs, max 512
tokens, seed 42.

The best checkpoint is selected on **validation F1**, not the final epoch — E5
overfits this corpus within 3 epochs, so the last checkpoint is usually not the
best one. `training_report.json` records per-epoch metrics for the thesis.
