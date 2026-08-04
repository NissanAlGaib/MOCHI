"""Phase 5 evaluation harness tests.

The harness produces every Chapter IV number, so a bug here silently corrupts
results rather than crashing. Metrics are checked against hand-computed values
and, where available, cross-checked against scikit-learn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.data_loading import (
    DatasetError,
    Sample,
    describe,
    load_file,
    normalize_label,
    stratified_split,
)
from eval.metrics import ConfusionMatrix, compare, evaluate, format_report
from eval.predictors import NoDefensePredictor, get_predictor, sample_to_request
from eval.run_detection import DEMO_SAMPLES, run
from eval.stats import (
    cohens_d_paired,
    interpret_cohens_d,
    paired_ttest,
    run_all_hypotheses,
)


# --- label normalization ---------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, 0), (1, 1), ("0", 0), ("1", 1),
        ("benign", 0), ("malicious", 1),
        ("safe", 0), ("unsafe", 1),
        ("jailbreak", 1), ("injection", 1),
        (True, 1), (False, 0),
        ("BENIGN", 0), ("  Malicious  ", 1),
        ("nonsense", None), (None, None), (7, None),
    ],
)
def test_label_normalization(raw, expected) -> None:
    assert normalize_label(raw) == expected


# --- dataset loading -------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_csv_with_standard_columns(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "d.csv",
        [{"text": "hello", "label": 0}, {"text": "ignore previous", "label": 1}],
    )
    samples = load_file(path)

    assert len(samples) == 2
    assert samples[0].label == 0 and samples[1].label == 1


def test_column_names_are_auto_detected(tmp_path: Path) -> None:
    """Public datasets disagree on column naming; the loader sniffs them."""
    path = write_csv(
        tmp_path / "d.csv",
        [{"prompt": "hello", "is_injection": "benign"},
         {"prompt": "ignore all", "is_injection": "malicious"}],
    )
    samples = load_file(path)
    assert [s.label for s in samples] == [0, 1]


def test_jsonl_loading(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"text": "a", "label": 0}\n{"text": "b", "label": 1}\n', encoding="utf-8"
    )
    assert len(load_file(path)) == 2


def test_unusable_rows_are_skipped_not_guessed(tmp_path: Path) -> None:
    """Mislabeling evaluation data would corrupt every downstream metric."""
    path = write_csv(
        tmp_path / "d.csv",
        [
            {"text": "good", "label": 0},
            {"text": "", "label": 1},            # empty text
            {"text": "x", "label": "???"},        # unknown label
            {"text": "bad", "label": 1},
        ],
    )
    samples = load_file(path)
    assert len(samples) == 2


def test_invert_labels(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "d.csv", [{"text": "a", "label": 1}])
    assert load_file(path, invert_labels=True)[0].label == 0


def test_missing_text_column_raises_helpful_error(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "d.csv", [{"foo": "a", "label": 1}])
    with pytest.raises(DatasetError, match="text column"):
        load_file(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="No such dataset file"):
        load_file(tmp_path / "nope.csv")


def test_source_tag_is_preserved(tmp_path: Path) -> None:
    """Indirect fixtures must exercise the untrusted path, not user_input."""
    path = write_csv(tmp_path / "d.csv", [{"text": "a", "label": 1}])
    samples = load_file(path, source_tag="web_content")
    assert samples[0].source_tag == "web_content"


def test_describe_reports_composition() -> None:
    output = describe(DEMO_SAMPLES)
    assert "Total" in output
    assert "50.0% malicious" in output


def test_describe_warns_on_imbalance() -> None:
    skewed = [Sample(text="x", label=0, dataset="d") for _ in range(95)]
    skewed += [Sample(text="y", label=1, dataset="d") for _ in range(5)]
    assert "WARNING" in describe(skewed)


def test_stratified_split_preserves_balance_and_is_seeded() -> None:
    samples = [Sample(text=f"b{i}", label=0, dataset="d") for i in range(100)]
    samples += [Sample(text=f"m{i}", label=1, dataset="d") for i in range(100)]

    train, val, evaluation = stratified_split(samples, train=0.6, validation=0.1)

    assert len(train) == 120 and len(val) == 20 and len(evaluation) == 60
    for part in (train, val, evaluation):
        malicious = sum(s.label for s in part)
        assert malicious == len(part) // 2  # balance preserved

    again = stratified_split(samples, train=0.6, validation=0.1)
    assert [s.text for s in again[0]] == [s.text for s in train]  # reproducible


# --- metrics ---------------------------------------------------------------


def test_confusion_matrix_formulas_match_thesis_table() -> None:
    # 8 attacks: 6 caught, 2 missed. 10 benign: 9 allowed, 1 wrongly blocked.
    matrix = ConfusionMatrix(tp=6, fn=2, tn=9, fp=1)

    assert matrix.total == 18
    assert matrix.accuracy == pytest.approx(15 / 18)
    assert matrix.precision == pytest.approx(6 / 7)
    assert matrix.recall == pytest.approx(6 / 8)
    assert matrix.f1 == pytest.approx(2 * (6 / 7) * (6 / 8) / ((6 / 7) + (6 / 8)))
    assert matrix.false_positive_rate == pytest.approx(1 / 10)
    assert matrix.specificity == pytest.approx(9 / 10)


def test_asr_is_complement_of_mitigation_rate() -> None:
    matrix = ConfusionMatrix(tp=7, fn=3, tn=10, fp=0)
    assert matrix.attack_success_rate == pytest.approx(0.3)
    assert matrix.mitigation_rate == pytest.approx(0.7)
    assert matrix.attack_success_rate + matrix.mitigation_rate == pytest.approx(1.0)


def test_metrics_do_not_divide_by_zero() -> None:
    empty = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
    assert empty.accuracy == 0.0
    assert empty.precision == 0.0
    assert empty.recall == 0.0
    assert empty.f1 == 0.0


def test_perfect_classifier() -> None:
    matrix = evaluate([1, 1, 0, 0], [1, 1, 0, 0])
    assert matrix.accuracy == 1.0 and matrix.f1 == 1.0
    assert matrix.attack_success_rate == 0.0


def test_evaluate_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate([1, 0], [1])


def test_metrics_agree_with_sklearn_if_available() -> None:
    matrix = ConfusionMatrix(tp=6, fn=2, tn=9, fp=1)
    reference = matrix.cross_check()
    if reference is None:
        pytest.skip("scikit-learn not installed")
    assert matrix.precision == pytest.approx(reference["precision"])
    assert matrix.recall == pytest.approx(reference["recall"])
    assert matrix.f1 == pytest.approx(reference["f1"])


def test_report_and_compare_render() -> None:
    before = ConfusionMatrix(tp=0, fp=0, tn=10, fn=10)
    after = ConfusionMatrix(tp=9, fp=1, tn=9, fn=1)
    assert "Confusion Matrix" in format_report(before)
    output = compare(before, after)
    assert "improved" in output


# --- predictors ------------------------------------------------------------


def test_baseline_predicts_everything_benign() -> None:
    """The control condition: no defense means ASR is 1.0 by construction."""
    predictor = NoDefensePredictor()
    assert all(predictor.predict(s).label == 0 for s in DEMO_SAMPLES)


def test_baseline_run_yields_asr_of_one() -> None:
    result = run(DEMO_SAMPLES, "baseline")
    assert result.matrix.attack_success_rate == 1.0
    assert result.matrix.recall == 0.0


def test_source_tag_routes_through_context() -> None:
    sample = Sample(text="payload", label=1, dataset="d", source_tag="web_content")
    request = sample_to_request(sample)
    assert request.context is not None
    assert request.context.web_content == "payload"


def test_user_input_sample_needs_no_context() -> None:
    request = sample_to_request(Sample(text="hi", label=0, dataset="d"))
    assert request.context is None


def test_mochi_predictor_runs_and_times() -> None:
    predictor = get_predictor("full")
    prediction = predictor.predict(DEMO_SAMPLES[0])
    assert prediction.label in (0, 1)
    assert prediction.latency_ms >= 0
    assert prediction.record.segments_inspected  # Phase 4 wiring is exercised


def test_unknown_config_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown configuration"):
        get_predictor("nonsense")


def test_cross_validation_produces_per_fold_matrices() -> None:
    result = run(DEMO_SAMPLES, "baseline", folds=4)
    assert len(result.fold_matrices) == 4
    assert sum(m.total for m in result.fold_matrices) == len(DEMO_SAMPLES)


# --- statistics ------------------------------------------------------------


def test_paired_ttest_detects_real_improvement() -> None:
    before = [0.50, 0.52, 0.49, 0.51, 0.50]
    after = [0.95, 0.96, 0.94, 0.97, 0.95]
    result = paired_ttest(before, after, name="H01: accuracy")

    assert result.significant
    assert result.decision == "REJECT H0"
    assert result.mean_difference > 0
    assert result.effect_label == "large"
    assert result.ci_lower > 0  # CI excludes zero


def test_paired_ttest_finds_nothing_when_unchanged() -> None:
    before = [0.80, 0.81, 0.79, 0.82, 0.80]
    after = [0.80, 0.82, 0.78, 0.81, 0.81]
    result = paired_ttest(before, after)

    assert not result.significant
    assert result.decision == "FAIL TO REJECT H0"


def test_ttest_requires_equal_lengths() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_ttest([1.0, 2.0], [1.0])


def test_ttest_requires_multiple_pairs() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        paired_ttest([1.0], [2.0])


@pytest.mark.parametrize(
    "d,label",
    [(0.1, "negligible"), (0.3, "small"), (0.6, "medium"), (1.2, "large")],
)
def test_cohens_d_interpretation(d: float, label: str) -> None:
    assert interpret_cohens_d(d) == label


def test_cohens_d_sign_follows_direction() -> None:
    assert cohens_d_paired([0.5, 0.5, 0.5], [0.9, 0.9, 0.9]) > 0
    assert cohens_d_paired([0.9, 0.9, 0.9], [0.5, 0.5, 0.5]) < 0


def test_all_six_hypotheses_run() -> None:
    folds = 5
    before = {
        "accuracy": [0.50] * folds, "precision": [0.0] * folds,
        "recall": [0.0] * folds, "f1": [0.0] * folds,
        "attack_success_rate": [1.0] * folds, "mitigation_rate": [0.0] * folds,
    }
    after = {
        "accuracy": [0.96, 0.95, 0.97, 0.96, 0.94],
        "precision": [0.95, 0.94, 0.96, 0.95, 0.93],
        "recall": [0.97, 0.96, 0.98, 0.97, 0.95],
        "f1": [0.96, 0.95, 0.97, 0.96, 0.94],
        "attack_success_rate": [0.03, 0.04, 0.02, 0.03, 0.05],
        "mitigation_rate": [0.97, 0.96, 0.98, 0.97, 0.95],
    }
    results = run_all_hypotheses(before, after)

    assert set(results) == set(before)
    assert all(r.significant for r in results.values())
    assert results["attack_success_rate"].mean_difference < 0  # ASR should fall


def test_normality_check_is_reported() -> None:
    """The thesis asserts normality rather than testing it; we report the check."""
    result = paired_ttest(
        [0.51, 0.48, 0.55, 0.47, 0.52, 0.49, 0.53],
        [0.88, 0.94, 0.85, 0.97, 0.83, 0.92, 0.90],
    )
    assert result.normality_p is not None
    assert result.normality_ok is not None


def test_summary_renders() -> None:
    result = paired_ttest([0.5] * 5, [0.9, 0.91, 0.89, 0.92, 0.90])
    assert "Cohen's d" in result.summary()
    assert "p =" in result.summary()
