"""Record which rows a model was trained on, beside its weights.

``build_splits`` does not look up a stored split - it recomputes one from
whatever CSVs are in the data directory at the moment it runs. Seeded, so it is
deterministic for a *fixed* corpus; but add or remove a single file and every
row can land somewhere different, because it shuffles and then cuts by
proportion.

``data/clean`` changed four times in three days: Taglish added, the held-out
set sealed, BANKING77 folded in. Each change silently invalidated every earlier
model's notion of what was held out. Measured after the last one, **60.3% of
the rows ``build_splits`` now calls "test" are rows the shipped model was
trained on.** Nothing reports this. An evaluation against that split returns a
flattering score and no warning.

Four incidents in one session traced to the same cause:

* ``taglish_heldout.csv`` pooled into training - 141 of its 300 rows
* 35% of the "test" split already seen, found while sourcing Taglish samples
* ``features.csv`` carrying split labels from a corpus three revisions old
* a threshold sweep scoring the wrong corpus, caught only because the row
  counts looked wrong

Each was found by hand. This module makes finding them automatic: the manifest
is written at training time, and :func:`check_unseen` turns "did this model see
these rows?" into a lookup instead of a reconstruction.

Hashes, not text - a manifest for 85k rows is a few megabytes, and the rows
themselves already live in ``data/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_NAME = "split_manifest.json"


def row_hash(text: str) -> str:
    """Identify one corpus row.

    Hashes the text exactly, with no normalization: two rows differing only in
    whitespace are different rows to the splitter, so they must be different
    rows here too. A manifest that normalized more aggressively than the thing
    it describes would report a match that the splitter would not.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build(train, validation, test, *, corpus_dir: Path, seed: int) -> dict:
    """Assemble the manifest for one training run."""
    return {
        "seed": seed,
        "corpus_dir": str(corpus_dir),
        "corpus_files": sorted(p.name for p in corpus_dir.glob("*.csv")),
        "counts": {
            "train": len(train.texts),
            "validation": len(validation.texts),
            "test": len(test.texts),
        },
        "train": [row_hash(t) for t in train.texts],
        "validation": [row_hash(t) for t in validation.texts],
        "test": [row_hash(t) for t in test.texts],
    }


def write(manifest: dict, out_dir: Path) -> Path:
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def load(model_dir: Path) -> dict | None:
    """The manifest for a model, or ``None`` for one trained before this existed."""
    path = Path(model_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def seen_by(model_dir: Path) -> frozenset[str] | None:
    """Hashes of every row the model fitted on - train plus validation.

    Validation counts as seen. It never updates weights directly, but it
    selects the checkpoint and the early-stopping point, so a score measured on
    it is not independent.
    """
    manifest = load(model_dir)
    if manifest is None:
        return None
    return frozenset(manifest["train"]) | frozenset(manifest["validation"])


def check_unseen(texts: list[str], model_dir: Path) -> dict:
    """How much of ``texts`` this model has already seen.

    Returns ``{"known": bool, "total": int, "seen": int, "share": float}``.
    ``known`` is False for a model with no manifest - which is not the same as
    "nothing was seen", and callers should treat it as "cannot tell" rather
    than as a pass.
    """
    seen = seen_by(model_dir)
    if seen is None:
        return {"known": False, "total": len(texts), "seen": 0, "share": 0.0}
    hits = sum(1 for t in texts if row_hash(t) in seen)
    return {
        "known": True,
        "total": len(texts),
        "seen": hits,
        "share": hits / len(texts) if texts else 0.0,
    }


def assert_unseen(texts: list[str], model_dir: Path, *, label: str = "evaluation set",
                  tolerance: float = 0.0) -> None:
    """Raise if an evaluation set overlaps what the model was trained on.

    ``tolerance`` exists for corpora with genuine internal duplicates, where a
    handful of collisions are unavoidable. It defaults to zero because the
    failures worth catching were 35% and 60%, not rounding.

    A model without a manifest raises too. Silence there would reproduce the
    exact failure this module exists to prevent: an unverifiable claim that
    reads like a verified one.
    """
    result = check_unseen(texts, model_dir)
    if not result["known"]:
        raise RuntimeError(
            f"{Path(model_dir).name} has no {MANIFEST_NAME}, so it cannot be "
            f"shown that it never saw this {label}. Retrain to generate one, "
            f"or reconstruct the corpus it was trained on and verify by hand."
        )
    if result["share"] > tolerance:
        raise RuntimeError(
            f"{result['seen']:,} of {result['total']:,} rows "
            f"({result['share']:.1%}) in this {label} were in "
            f"{Path(model_dir).name}'s training or validation data. "
            f"Any score measured here is partly measuring memorisation."
        )
