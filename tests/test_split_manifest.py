"""The split manifest must catch a model being evaluated on what it trained on.

``build_splits`` recomputes the split from whatever CSVs are in the data
directory, so a corpus change silently moves rows between train and test while
the weights stay where they are. Measured on the shipped model after three such
changes: **60.3% of the rows the splitter now calls "test" were in its training
data**, with nothing anywhere reporting it.

These tests pin the behaviour that makes that detectable:

* a manifest round-trips
* overlap is counted, not approximated
* ``assert_unseen`` raises on overlap
* a model with **no** manifest raises too, rather than passing silently -
  "cannot tell" must not read as "verified clean"

The last one is the point. Every other failure this session was caught by
someone looking; this makes looking unnecessary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.split_manifest import (
    MANIFEST_NAME,
    assert_unseen,
    build,
    check_unseen,
    load,
    row_hash,
    seen_by,
    write,
)


class FakeSplit:
    """Stands in for ``training.finetune_e5.Split`` - only ``texts`` is read."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A model directory whose manifest records six known rows."""
    out = tmp_path / "model"
    out.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.csv").write_text("text,label\n", encoding="utf-8")
    write(
        build(
            FakeSplit(["train one", "train two"]),
            FakeSplit(["val one"]),
            FakeSplit(["test one", "test two", "test three"]),
            corpus_dir=corpus,
            seed=42,
        ),
        out,
    )
    return out


def test_manifest_round_trips(model_dir: Path) -> None:
    manifest = load(model_dir)
    assert manifest is not None
    assert manifest["counts"] == {"train": 2, "validation": 1, "test": 3}
    assert manifest["seed"] == 42
    assert manifest["corpus_files"] == ["a.csv"]


def test_validation_rows_count_as_seen(model_dir: Path) -> None:
    """Validation never updates weights, but it picks the checkpoint.

    A score measured on data that selected the model is not independent of it,
    so the manifest treats validation as seen.
    """
    seen = seen_by(model_dir)
    assert row_hash("train one") in seen
    assert row_hash("val one") in seen
    assert row_hash("test one") not in seen


def test_overlap_is_counted_exactly(model_dir: Path) -> None:
    result = check_unseen(["test one", "train two", "brand new"], model_dir)
    assert result == {"known": True, "total": 3, "seen": 1, "share": 1 / 3}


def test_clean_evaluation_set_passes(model_dir: Path) -> None:
    assert_unseen(["brand new", "also new"], model_dir)  # must not raise


def test_overlap_raises_and_names_the_number(model_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="were in"):
        assert_unseen(["train one", "val one", "fresh"], model_dir)


def test_missing_manifest_raises_rather_than_passing(tmp_path: Path) -> None:
    """The failure mode this module exists to prevent.

    A model trained before manifests existed cannot show it never saw a given
    set. Returning "clean" there would turn an unverifiable claim into one that
    reads as verified - which is how 60.3% contamination went unnoticed.
    """
    bare = tmp_path / "no-manifest"
    bare.mkdir()
    assert load(bare) is None
    assert check_unseen(["anything"], bare)["known"] is False
    with pytest.raises(RuntimeError, match=MANIFEST_NAME):
        assert_unseen(["anything"], bare)


def test_hashing_does_not_normalise(model_dir: Path) -> None:
    """Two rows the splitter treats as different must hash differently.

    Normalising more aggressively than ``stratified_split`` does would report a
    match the splitter would not make.
    """
    assert row_hash("train one") != row_hash("Train One")
    assert row_hash("train one") != row_hash("train  one")
    assert check_unseen(["TRAIN ONE"], model_dir)["seen"] == 0


SHIPPED = Path(__file__).resolve().parents[1] / "models" / "e5-fine-tuned"


@pytest.mark.skipif(not (SHIPPED / MANIFEST_NAME).exists(),
                    reason="shipped model has no manifest here")
def test_shipped_model_manifest_matches_its_training_report() -> None:
    """The backfilled manifest must agree with what the model itself recorded.

    Backfilled manifests are reconstructed rather than observed, so this is the
    only evidence they describe the right corpus.
    """
    import json

    manifest = load(SHIPPED)
    report = json.loads((SHIPPED / "training_report.json").read_text(encoding="utf-8"))
    test = report["test"]
    assert manifest["counts"]["test"] == (
        test["tp"] + test["fp"] + test["tn"] + test["fn"]
    )
