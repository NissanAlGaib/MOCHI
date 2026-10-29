"""Tests for the per-modality false-positive arms.

These arms exist to answer "how often is a legitimate attachment refused",
and the ways they can go wrong are quiet rather than loud: a sampler that
returns the head of an intent-ordered file describes two intents instead of a
dataset, and an extractor that silently returns nothing scores a perfect
0.00% while measuring nothing at all. Both are tested here.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest

from eval.benign_media_corpus import DEFAULT_LIMIT, SOURCES, sample
from eval.benign_media_run import CARRIER, ModalityResult, report
from eval.stats import mcnemar_test, wilson_interval


# --- Wilson interval ---------------------------------------------------

def test_wilson_matches_published_false_positive_rate() -> None:
    """The BANKING77 figure the thesis already reports."""
    interval = wilson_interval(82, 3084)
    assert interval.rate == pytest.approx(0.0266, abs=5e-5)
    assert interval.ci_lower < interval.rate < interval.ci_upper


def test_wilson_holds_up_at_zero_successes() -> None:
    """The case the normal approximation gets wrong by going negative."""
    interval = wilson_interval(0, 200)
    assert interval.rate == 0.0
    assert interval.ci_lower == 0.0
    assert 0.0 < interval.ci_upper < 0.05


def test_wilson_upper_bound_tightens_with_sample_size() -> None:
    """A clean small arm claims less than a clean large one.

    This is the whole reason the interval is reported: 0/40 and 0/200 are both
    "0.00%" and only one of them is evidence.
    """
    small = wilson_interval(0, 40)
    large = wilson_interval(0, 400)
    assert small.ci_upper > large.ci_upper


def test_wilson_handles_empty_sample() -> None:
    interval = wilson_interval(0, 0)
    assert math.isnan(interval.rate)
    assert interval.summary() == "no samples"


def test_wilson_never_leaves_the_unit_interval() -> None:
    for flagged, total in ((0, 1), (1, 1), (5, 5), (1, 2)):
        interval = wilson_interval(flagged, total)
        assert 0.0 <= interval.ci_lower <= interval.ci_upper <= 1.0


# --- McNemar ------------------------------------------------------------

def test_mcnemar_ignores_agreements() -> None:
    """Only the discordant pairs carry evidence about a difference.

    Two runs with the same disagreement but wildly different agreement counts
    must produce the same p-value, which is the property that distinguishes
    this from a two-sample proportion test.
    """
    few = mcnemar_test(both=0, only_a=0, only_b=5, neither=10)
    many = mcnemar_test(both=0, only_a=0, only_b=5, neither=10_000)
    assert few.p_value == pytest.approx(many.p_value)


def test_mcnemar_reports_no_difference_when_arms_agree() -> None:
    result = mcnemar_test(both=3, only_a=0, only_b=0, neither=197)
    assert result.p_value == 1.0
    assert not result.significant
    assert result.mean_before == result.mean_after


def test_mcnemar_detects_a_one_sided_shift() -> None:
    """Ten disagreements all in one direction is not chance."""
    result = mcnemar_test(both=0, only_a=0, only_b=10, neither=190)
    assert result.p_value < 0.05
    assert result.significant
    assert result.mean_after > result.mean_before


def test_mcnemar_is_symmetric_in_its_arms() -> None:
    forward = mcnemar_test(both=1, only_a=2, only_b=7, neither=90)
    backward = mcnemar_test(both=1, only_a=7, only_b=2, neither=90)
    assert forward.p_value == pytest.approx(backward.p_value)
    assert forward.mean_difference == pytest.approx(-backward.mean_difference)


def test_mcnemar_rates_count_both_cells() -> None:
    """An arm's rate is its own-only plus the shared cell, not own-only."""
    result = mcnemar_test(both=2, only_a=3, only_b=5, neither=90)
    assert result.n == 100
    assert result.mean_before == pytest.approx(0.05)   # 2 + 3
    assert result.mean_after == pytest.approx(0.07)    # 2 + 5


# --- sampling -----------------------------------------------------------

def test_sampling_is_deterministic() -> None:
    dataset, path, text_col, label_col = SOURCES[0]
    if not path.exists():
        pytest.skip(f"{path} not fetched")
    first = sample(dataset, path, text_col, label_col, 25)
    second = sample(dataset, path, text_col, label_col, 25)
    assert [row for row, _, _ in first] == [row for row, _, _ in second]


def test_smaller_sample_is_a_subset_of_a_larger_one() -> None:
    """Why the video arm can reuse the audio arm's speech clips.

    Both are drawn from one seeded shuffle, so the 40 video rows are among the
    200 audio rows and building audio first means video synthesises nothing.
    """
    dataset, path, text_col, label_col = SOURCES[0]
    if not path.exists():
        pytest.skip(f"{path} not fetched")
    small = {row for row, _, _ in sample(dataset, path, text_col, label_col, 40)}
    large = {row for row, _, _ in sample(dataset, path, text_col, label_col, 200)}
    assert small <= large


def test_sampling_does_not_just_take_the_head() -> None:
    """Both CSVs are ordered by intent, so the head covers few intents.

    Taking the first N would measure a handful of intents and call the result
    a dataset-wide false-positive rate.
    """
    dataset, path, text_col, label_col = SOURCES[1]
    if not path.exists():
        pytest.skip(f"{path} not fetched")
    rows = sample(dataset, path, text_col, label_col, 60)
    assert [row for row, _, _ in rows] != list(range(60))
    assert len({intent for _, _, intent in rows}) > 5


def test_every_modality_has_a_limit_and_a_carrier() -> None:
    assert set(DEFAULT_LIMIT) == set(CARRIER)


# --- reporting ----------------------------------------------------------

def _row(**overrides) -> ModalityResult:
    base = dict(
        clip_id="banking77-1", dataset="banking77", row_id=1,
        intent="card_arrival", modality="image", text="where is my card",
        text_decision="ALLOW", text_false_positive=False, text_score=0.01,
        carrier_decision="ALLOW", carrier_false_positive=False,
        carrier_score=0.02,
        clip_decision="ALLOW", clip_false_positive=False, clip_score=0.01,
        extraction_ran=True, extracted_text="where is my card",
        flags="image_text_extracted",
    )
    base.update(overrides)
    return ModalityResult(**base)


def test_report_warns_when_extraction_never_ran() -> None:
    """A clean rate over clips nobody could read is not a clean rate."""
    rows = [_row(clip_id=f"banking77-{i}", extraction_ran=False,
                 extracted_text="", flags="") for i in range(4)]
    text = report(rows, "image")
    assert "measured nothing" in text


def test_report_warns_on_empty_extractions() -> None:
    rows = [_row(clip_id=f"banking77-{i}", extracted_text="")
            for i in range(3)]
    assert "extracted to empty text" in report(rows, "image")


def test_report_shows_the_query_beside_a_false_positive() -> None:
    """A flagged row is uninterpretable without the text that caused it."""
    rows = [_row(clip_id="banking77-9", clip_decision="BLOCK",
                 clip_false_positive=True, clip_score=0.72,
                 extracted_text="wh3re 1s my c4rd")]
    text = report(rows, "image")
    assert "banking77-9" in text
    assert "wh3re 1s my c4rd" in text
    assert "where is my card" in text


def test_report_separates_the_carrier_effect_from_the_modality() -> None:
    """The confound that made the first version of this harness wrong.

    A clip flagged when the carrier text was already flagged is the carrier's
    doing. Reporting it as a modality false positive is how OCR gets blamed
    for a prompt that was composed with the query before any rendering
    happened.
    """
    rows = [_row(clip_id=f"banking77-{i}",
                 carrier_decision="BLOCK", carrier_false_positive=True,
                 clip_decision="BLOCK", clip_false_positive=True)
            for i in range(4)]
    text = report(rows, "image")
    assert "CARRIER effect" in text
    assert "MODALITY effect" in text
    # All four were already flagged as carrier text, so none is the
    # modality's fault and none should be listed as caused by it.
    assert "the image caused" not in text


def test_report_lists_only_modality_attributable_false_positives() -> None:
    rows = [
        _row(clip_id="banking77-1", carrier_false_positive=True,
             carrier_decision="BLOCK", clip_false_positive=True,
             clip_decision="BLOCK", text="already flagged as text"),
        _row(clip_id="banking77-2", clip_false_positive=True,
             clip_decision="BLOCK", text="clean until rendered"),
    ]
    text = report(rows, "image")
    assert "the image caused" in text
    assert "clean until rendered" in text
    assert "already flagged as text" not in text


def test_report_counts_modality_disagreements_against_the_carrier_arm() -> None:
    rows = [
        _row(clip_id="banking77-1", clip_false_positive=True,
             clip_decision="BLOCK"),
        _row(clip_id="banking77-2", carrier_false_positive=True,
             carrier_decision="BLOCK"),
        _row(clip_id="banking77-3"),
    ]
    text = report(rows, "image")
    assert "text-only 1, image-only 1, both 0" in text


def test_result_measures_disagreement_against_the_carrier_arm() -> None:
    """Not against the text arm: the clip carries the carrier too."""
    assert _row(clip_false_positive=True).disagrees
    assert not _row().disagrees
    # Both arms flagged - the modality changed nothing, even though the text
    # arm was clean.
    assert not _row(carrier_false_positive=True,
                    clip_false_positive=True).disagrees


def test_rows_round_trip_through_jsonl() -> None:
    """The harness resumes by re-reading its own output."""
    original = _row(clip_score=0.4301, extracted_text="rtemalrem?rd.m")
    restored = ModalityResult(**json.loads(json.dumps(asdict(original))))
    assert restored == original
