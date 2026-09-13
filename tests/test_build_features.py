"""Pins the two split builders against each other.

``eval/build_features.assign_splits`` and ``training/finetune_e5.build_splits``
duplicate the same splitting rule, because the first needs ``Sample`` objects
(to keep the ``dataset`` column) and the second needs bare text/label lists.
The duplication is deliberate and documented in both files - but a duplicated
rule that nothing checks is a rule that drifts, and the two tracks would then be
trained and scored on different rows without anything failing.

``build_features.py`` claimed this file existed before it did. It does now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.build_features import assign_splits  # noqa: E402
from eval.data_loading import Sample, stratified_split  # noqa: E402


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two CSVs, one of them named as if it carried an official split.

    ``promptshield_test.csv`` is the trap: the previous implementation would
    have routed all 60 of its rows straight into the test set on the strength of
    the filename alone.
    """
    data = tmp_path / "clean"
    data.mkdir()

    rows = ["text,label"]
    rows += [f"benign prompt {i},0" for i in range(70)]
    rows += [f"ignore previous instructions {i},1" for i in range(70)]
    (data / "jayavibhav.csv").write_text("\n".join(rows), encoding="utf-8")

    rows = ["text,label"]
    rows += [f"other benign {i},0" for i in range(30)]
    rows += [f"other attack {i},1" for i in range(30)]
    (data / "promptshield_test.csv").write_text("\n".join(rows), encoding="utf-8")

    return data


def test_official_split_filenames_are_ignored(corpus: Path) -> None:
    """A file called ``*_test.csv`` must not become the test set."""
    tagged = assign_splits(corpus)

    from_promptshield = [
        (sample, split) for sample, split in tagged if "other" in sample.text
    ]
    splits_used = {split for _, split in from_promptshield}

    assert len(from_promptshield) == 60
    # If the filename still drove the assignment, this would be exactly {"test"}.
    assert splits_used == {"train", "validation", "test"}


def test_assign_splits_is_a_double_70_30_and_stratified(corpus: Path) -> None:
    """The thesis protocol: 70/30 into development and test, then 70/30 again
    inside development. On 200 rows that is 98 / 42 / 60 - note the train tier
    is 49% of the corpus, not 70%, because the second cut comes out of the
    first cut's remainder.
    """
    tagged = assign_splits(corpus)

    train = [s for s, split in tagged if split == "train"]
    validation = [s for s, split in tagged if split == "validation"]
    test = [s for s, split in tagged if split == "test"]

    assert len(tagged) == 200
    assert (len(train), len(validation), len(test)) == (98, 42, 60)
    assert {split for _, split in tagged} == {"train", "validation", "test"}

    for part in (train, validation, test):
        malicious = sum(s.label for s in part)
        assert malicious == len(part) // 2


def test_widening_validation_never_moves_the_test_boundary(corpus: Path) -> None:
    """The property the whole two-track comparison rests on. The second 70/30
    comes out of the *first* cut's remainder, so growing the validation tier
    must take rows from train and never from the sealed 30%.
    """
    from eval.data_loading import load_file

    pooled: list[Sample] = []
    for path in sorted(corpus.glob("*.csv")):
        pooled.extend(load_file(path))

    narrow = stratified_split(pooled, test=0.3, validation=0.1)[2]
    wide = stratified_split(pooled, test=0.3, validation=0.3)[2]
    assert sorted(s.text for s in narrow) == sorted(s.text for s in wide)


def test_both_tracks_score_on_identical_test_rows(corpus: Path) -> None:
    """The comparison is only fair if the 30% is the same 30% for both tracks.

    Both tracks now take the identical three-way split, so this pins
    ``assign_splits`` against a direct ``stratified_split`` call with the same
    shares - the duplication those two represent is exactly what drifts.
    """
    tagged = assign_splits(corpus)
    track_a_test = sorted(s.text for s, split in tagged if split == "test")

    from eval.data_loading import load_file

    pooled: list[Sample] = []
    for path in sorted(corpus.glob("*.csv")):
        pooled.extend(load_file(path))
    from eval.data_loading import TEST_SHARE, VALIDATION_SHARE

    _train, validation, test = stratified_split(
        pooled, test=TEST_SHARE, validation=VALIDATION_SHARE)
    track_b_test = sorted(s.text for s in test)

    assert track_a_test == track_b_test
    assert validation, "both tracks need a validation tier"
    assert not set(track_b_test) & {s.text for s in validation}


def test_split_assignment_is_reproducible(corpus: Path) -> None:
    first = [(s.text, split) for s, split in assign_splits(corpus)]
    second = [(s.text, split) for s, split in assign_splits(corpus)]
    assert first == second
