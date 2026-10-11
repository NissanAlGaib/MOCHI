"""Rebuild the sealed test split from the training manifest.

    python eval/build_sealed_test.py                      # -> data/splits/stage12_test.csv
    python eval/build_sealed_test.py --out somewhere.csv

Why this exists rather than a fresh split: the corpus directory the splitter
used (``data/_corpus_e5finetuned``) no longer exists, so the split cannot be
re-drawn. It can be *recovered*, because the manifest records a hash of every
row and names the six source files, all of which are still on disk and
unchanged since before training.

The recovery is verifiable, which is the point. The six files hold exactly
85,465 rows, matching the manifest's 41,879 train + 17,947 validation + 25,639
test, and the 11,156 malicious rows recovered here match the threshold sweep's
``tp + fn`` exactly. A reconstruction that agreed on neither count would be a
different split wearing the same name.

Rows the model has seen are excluded, not merely reported. One text -
``"Explain Schrodinger's wave equation."`` - appears in both
``promptshield_train.csv`` and ``taglish_generated.csv``, and because the
splitter split **rows rather than unique texts**, one copy landed in test and
another in train. It is a single benign row in 25,639 and moves no metric, but
a test set is only sealed if it is actually sealed. The mechanism deserves more
attention than the magnitude: it would bite far harder in a corpus with heavier
duplication, and ``jayavibhav.csv`` alone contributes 40,000 of the 85,465 rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.data_loading import load_file  # noqa: E402
from training.split_manifest import row_hash  # noqa: E402

MODEL_DIR = REPO / "models" / "e5-fine-tuned"
#: Deliberately NOT data/sealed/. That directory carries a contract - nothing
#: in it may appear in data/clean/, because it is the set that must never be
#: trained on, and tests/test_sealed_data.py enforces it. This file is the
#: opposite kind of thing: a *subset of the corpus*, recovered by hash, so
#: every row of it appears in data/clean/ by definition. Writing it to
#: data/sealed/ fails those tests with 25,547 overlapping rows - correctly.
DEFAULT_OUT = REPO / "data" / "splits" / "stage12_test.csv"

#: Where each file named in the manifest now lives. The manifest records bare
#: filenames from a corpus directory that was assembled for training and then
#: removed; these are the originals it was built from.
LOCATIONS = {
    "jayavibhav.csv": "data/clean/jayavibhav.csv",
    "promptshield_test.csv": "data/clean/promptshield_test.csv",
    "promptshield_train.csv": "data/clean/promptshield_train.csv",
    "promptshield_validation.csv": "data/clean/promptshield_validation.csv",
    "taglish_generated.csv": "data/clean/taglish_generated.csv",
    "taglish_heldout.csv": "data/sealed/taglish_heldout.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest_path = args.model_dir / "split_manifest.json"
    if not manifest_path.exists():
        print(f"\nERROR: no split manifest at {manifest_path}")
        print("Without it there is no record of which rows were held out, and")
        print("a split drawn now would not be the one the model was trained")
        print("against. Train with training/finetune_e5.py, which writes it.\n")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_hashes = frozenset(manifest["test"])
    seen_hashes = frozenset(manifest["train"]) | frozenset(manifest["validation"])

    print(f"\nmanifest: train {len(manifest['train']):,}  "
          f"validation {len(manifest['validation']):,}  "
          f"test {len(manifest['test']):,}")
    if manifest.get("backfilled"):
        print("  (backfilled - reconstructed after training, count-verified)")

    selected: list[tuple[str, int, str, str]] = []
    loaded = 0
    for name in manifest["corpus_files"]:
        path = REPO / LOCATIONS[name]
        samples = load_file(path, dataset_name=name)
        loaded += len(samples)
        hit = 0
        for sample in samples:
            digest = row_hash(sample.text)
            if digest in test_hashes:
                selected.append((sample.text, sample.label, name, digest))
                hit += 1
        print(f"  {name:<32} {len(samples):>7,} rows -> {hit:>7,} in test")

    expected = sum(manifest["counts"].values())
    print(f"\nloaded {loaded:,} corpus rows (manifest expects {expected:,})"
          f"{'  OK' if loaded == expected else '  MISMATCH - sources have changed'}")

    kept: list[tuple[str, int, str]] = []
    unique: set[str] = set()
    contaminated: list[tuple[str, int, str]] = []
    for text, label, dataset, digest in selected:
        if digest in seen_hashes:
            contaminated.append((text, label, dataset))
            continue
        if digest in unique:
            continue
        unique.add(digest)
        kept.append((text, label, dataset))

    print("\nleakage check")
    print(f"  text also in train/validation : {len(contaminated)}  (excluded)")
    for text, label, dataset in contaminated:
        print(f"      [{dataset}] label={label} {text[:70]!r}")
    duplicates = len(selected) - len(contaminated) - len(kept)
    print(f"  duplicate rows in selection   : {duplicates}  (excluded)")

    malicious = sum(1 for _t, label, _d in kept if label == 1)
    print(f"  sealed test rows              : {len(kept):,}")
    print(f"  label balance                 : {malicious:,} malicious / "
          f"{len(kept) - malicious:,} benign")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label", "source"])
        writer.writerows(kept)
    print(f"\n  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
