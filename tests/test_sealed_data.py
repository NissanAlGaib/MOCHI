"""Guard the sealed evaluation sets against re-entering the training pool.

``data/sealed/`` holds evaluation data that must never be trained on. Nothing
enforces that structurally - ``build_splits`` and ``build_features`` both pool
their data directory with ``glob("*.csv")``, so a file in the wrong folder is
silently absorbed with no error and no warning.

That is exactly what happened to ``taglish_heldout.csv``. Created on 14
September to be held out entirely from both tracks, it was written to
``data/clean/`` alongside the training corpora. The pooling found it, and 141
of its 300 rows went into the training split with 68 more into validation -
leaving the model having seen 47% of its own held-out test set. Nothing failed.
No report said anything. It surfaced only when the splits were reconstructed by
hand and checked row by row.

A set that silently stops being held out is worse than no set at all, because
the resulting score looks valid. These tests make the failure loud.

Skipped when the data is absent, so a fresh clone without the gitignored
corpora still runs green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEALED_DIR = REPO / "data" / "sealed"
CLEAN_DIR = REPO / "data" / "clean"


def _texts(path: Path) -> set[str]:
    from eval.data_loading import load_file

    return {sample.text for sample in load_file(path)}


def sealed_files() -> list[Path]:
    return sorted(SEALED_DIR.glob("*.csv")) if SEALED_DIR.exists() else []


@pytest.mark.skipif(not sealed_files(), reason="no sealed data present")
def test_sealed_sets_are_not_in_the_pooled_corpus() -> None:
    """No sealed row may appear in any file the training pool globs."""
    pooled: set[str] = set()
    for path in sorted(CLEAN_DIR.glob("*.csv")):
        pooled |= _texts(path)

    for path in sealed_files():
        overlap = _texts(path) & pooled
        assert not overlap, (
            f"{len(overlap)} rows of {path.name} appear in data/clean/ and will "
            f"be pooled into training. Move them out: a sealed set that is "
            f"trained on produces a score that measures memorisation.\n"
            f"First offender: {sorted(overlap)[0][:120]!r}"
        )


@pytest.mark.skipif(not sealed_files(), reason="no sealed data present")
def test_sealed_sets_do_not_reach_the_training_splits() -> None:
    """The end-to-end check: reconstruct the splits and look for sealed rows.

    Stronger than the file-level test above, because it would also catch a
    sealed row that reached the pool by some route other than sitting in
    ``data/clean/`` - a symlink, a second data directory, a loader change.
    """
    from training.finetune_e5 import build_splits

    if not list(CLEAN_DIR.glob("*.csv")):
        pytest.skip("no cleaned corpus present")

    train, validation, test = build_splits(CLEAN_DIR)
    everywhere = set(train.texts) | set(validation.texts) | set(test.texts)

    for path in sealed_files():
        overlap = _texts(path) & everywhere
        assert not overlap, (
            f"{len(overlap)} rows of {path.name} reached the train/validation/"
            f"test splits. The set is not held out."
        )


@pytest.mark.skipif(not sealed_files(), reason="no sealed data present")
def test_sealed_sets_are_not_empty() -> None:
    """A sealed file that fails to load would pass every check above.

    An empty set trivially satisfies "no overlap", so the guard would report
    success for a set that cannot evaluate anything.
    """
    for path in sealed_files():
        assert _texts(path), f"{path.name} loaded no rows"
