"""Assemble the out-of-distribution evaluation set: Gandalf plus benign traffic.

    python eval/build_ood_set.py        # -> data/splits/ood_test.csv

The sealed split in ``build_sealed_test.py`` is the corpus the detector was
trained on, held-out rows of it. It measures the model at its best and it is
where Stage II sits at 99.29%, leaving nothing for a second detector to add.
That is why the in-distribution ablation shows Stage I buying +1 catch for +17
false positives - the conditions are close to the least favourable that could
be constructed for it.

This set exists to ask the same question where the detector is weak:

* **Gandalf** (1,000 rows, label 1). Human adversarial pressure against a
  defended system, independent of everything the detector trained on - zero
  overlap with any split of ``data/clean``, the whole source unseen. Detection
  here is 85.3%, not 99.3%, so there is room for Stage I to matter.
* **BANKING77 test and CLINC150 test** (8,584 rows, label 0). Gandalf carries
  no benign rows, so on its own it yields recall with no false-positive cost -
  which is exactly the half of the trade-off the in-distribution run said was
  the problem. Pairing them makes the arms comparable on both axes.

Neither benign source appears in ``e5-fine-tuned``'s corpus files, so all of
this is unseen for the primary model. ``banking77_train`` does feed
``e5-banking``; ``banking77_test`` is sealed for both, and CLINC150 is
untouched by either.

The class balance here (10.4% malicious) is deliberately not the sealed split's
43.5%. It is closer to what a gateway sees, and it means the FPR column
dominates accuracy - so read recall and FPR, never accuracy, off this set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GANDALF = REPO / "data" / "attacks" / "gandalf.jsonl"
BENIGN = [
    ("banking77_test", REPO / "data" / "external" / "banking77_test.csv", "text"),
    ("clinc150_test", REPO / "data" / "external" / "clinc150_test.csv", "text"),
]
DEFAULT_OUT = REPO / "data" / "splits" / "ood_test.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows: list[tuple[str, int, str]] = []

    if not GANDALF.exists():
        print(f"\nERROR: {GANDALF} not found. Run eval/gandalf_corpus.py first.\n")
        return 1

    attacks = 0
    with GANDALF.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = (item.get("user_input") or "").strip()
            if text:
                rows.append((text, 1, "gandalf"))
                attacks += 1
    print(f"\n  gandalf                {attacks:>6,} attacks")

    for name, path, column in BENIGN:
        if not path.exists():
            print(f"\nERROR: {path} not found.\n")
            return 1
        kept = 0
        with path.open(encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                text = (record.get(column) or "").strip()
                if text:
                    rows.append((text, 0, name))
                    kept += 1
        print(f"  {name:<22} {kept:>6,} benign")

    # Exact duplicates would weight some rows twice. Benign sources overlap
    # very little, but CLINC150 and BANKING77 both contain short phrasings
    # that can collide.
    unique: set[str] = set()
    deduped: list[tuple[str, int, str]] = []
    for text, label, source in rows:
        if text in unique:
            continue
        unique.add(text)
        deduped.append((text, label, source))
    if len(deduped) != len(rows):
        print(f"\n  dropped {len(rows) - len(deduped)} duplicate texts")

    malicious = sum(1 for _t, label, _s in deduped if label == 1)
    print(f"\n  total {len(deduped):,} rows - {malicious:,} malicious "
          f"({malicious / len(deduped):.1%}) / {len(deduped) - malicious:,} benign")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label", "source"])
        writer.writerows(deduped)
    print(f"\n  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
