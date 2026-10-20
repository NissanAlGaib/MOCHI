"""Detection metrics.

Formulas are implemented directly rather than delegated to scikit-learn, for
two reasons: the thesis Derived Metrics table states each formula explicitly
and the code should visibly match it, and it keeps the evaluation path free of
a heavyweight dependency. :func:`ConfusionMatrix.cross_check` validates against
scikit-learn when it happens to be installed.

Convention: **positive class = malicious (1)**. "Recall" therefore means the
proportion of attacks caught, which is the security-relevant reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts, plus every metric derived from them."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        """(TP + TN) / (TP + TN + FP + FN)"""
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """TP / (TP + FP). Undefined with no positive predictions; reported as 0."""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN). Also the Detection Rate in the secondary metrics table."""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        """2 x (P x R) / (P + R)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        """FP / (FP + TN). Target < 1.0% - benign prompts wrongly blocked."""
        denominator = self.fp + self.tn
        return self.fp / denominator if denominator else 0.0

    @property
    def false_negative_rate(self) -> float:
        """FN / (FN + TP). Attacks that got through."""
        denominator = self.fn + self.tp
        return self.fn / denominator if denominator else 0.0

    @property
    def specificity(self) -> float:
        """TN / (TN + FP). Proportion of benign traffic correctly allowed."""
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator else 0.0

    @property
    def attack_success_rate(self) -> float:
        """Share of attacks that evaded detection - equals the false negative rate.

        Reported separately because the thesis treats ASR as a mitigation
        metric with its own hypothesis (H05), while FNR sits with detection.
        """
        return self.false_negative_rate

    @property
    def mitigation_rate(self) -> float:
        """Share of attacks detected and stopped. Complement of ASR."""
        return self.recall

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "total": self.total,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "specificity": self.specificity,
            "attack_success_rate": self.attack_success_rate,
            "mitigation_rate": self.mitigation_rate,
        }

    def cross_check(self) -> dict[str, float] | None:
        """Recompute with scikit-learn if available, for verification."""
        try:
            from sklearn.metrics import precision_recall_fscore_support
        except ImportError:
            return None
        y_true = [1] * (self.tp + self.fn) + [0] * (self.tn + self.fp)
        y_pred = [1] * self.tp + [0] * self.fn + [0] * self.tn + [1] * self.fp
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return {"precision": float(p), "recall": float(r), "f1": float(f)}


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionMatrix:
    """Build a confusion matrix from parallel label sequences."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(y_true)} true labels vs {len(y_pred)} predictions"
        )
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def format_report(matrix: ConfusionMatrix, *, title: str = "Detection Performance") -> str:
    """Render a confusion matrix and its metrics for the console or an appendix."""
    lines = [
        "",
        "=" * 64,
        f"  {title}",
        "=" * 64,
        "",
        "  Confusion Matrix",
        f"  {'':<20}{'Predicted Benign':>18}{'Predicted Malicious':>22}",
        f"  {'Actual Benign':<20}{matrix.tn:>18,}{matrix.fp:>22,}",
        f"  {'Actual Malicious':<20}{matrix.fn:>18,}{matrix.tp:>22,}",
        "",
        "  Derived Metrics",
        f"  {'Accuracy':<24}{matrix.accuracy:>10.4f}   (TP+TN)/Total",
        f"  {'Precision':<24}{matrix.precision:>10.4f}   TP/(TP+FP)",
        f"  {'Recall / Detection Rate':<24}{matrix.recall:>10.4f}   TP/(TP+FN)",
        f"  {'F1-Score':<24}{matrix.f1:>10.4f}   2PR/(P+R)",
        "",
        f"  {'False Positive Rate':<24}{matrix.false_positive_rate:>10.4f}   target < 0.0100",
        f"  {'Attack Success Rate':<24}{matrix.attack_success_rate:>10.4f}   lower is better",
        f"  {'Mitigation Rate':<24}{matrix.mitigation_rate:>10.4f}   higher is better",
        "",
        f"  Samples evaluated: {matrix.total:,}",
        "=" * 64,
    ]
    return "\n".join(lines)


def compare(before: ConfusionMatrix, after: ConfusionMatrix, *,
            before_label: str = "Baseline", after_label: str = "MOCHI") -> str:
    """Side-by-side before/after table - the Performance Delta Analysis figure."""
    rows = [
        ("Accuracy", before.accuracy, after.accuracy, True),
        ("Precision", before.precision, after.precision, True),
        ("Recall", before.recall, after.recall, True),
        ("F1-Score", before.f1, after.f1, True),
        ("False Positive Rate", before.false_positive_rate, after.false_positive_rate, False),
        ("Attack Success Rate", before.attack_success_rate, after.attack_success_rate, False),
        ("Mitigation Rate", before.mitigation_rate, after.mitigation_rate, True),
    ]
    lines = [
        "",
        "=" * 74,
        "  Performance Delta",
        "=" * 74,
        f"  {'Metric':<24}{before_label:>12}{after_label:>12}{'Delta':>12}   Direction",
        "-" * 74,
    ]
    for name, b, a, higher_is_better in rows:
        delta = a - b
        improved = delta > 0 if higher_is_better else delta < 0
        marker = "improved" if delta and improved else ("worse" if delta else "-")
        lines.append(f"  {name:<24}{b:>12.4f}{a:>12.4f}{delta:>+12.4f}   {marker}")
    lines.append("=" * 74)
    return "\n".join(lines)


# --- enforcement leakage ---------------------------------------------------
#
# Added 2026-10-20 after adviser review. The question it answers is distinct
# from every metric above: of the attacks MOCHI *acted on and still forwarded*,
# how many worked anyway?
#
# **This is not a false-positive rate, and the distinction matters.** A false
# positive is a benign item wrongly flagged. These are malicious items
# correctly flagged - true positives - whose mitigation then failed. Pooling
# the two would mean reporting a mitigation rate with no benign measurement
# beside it, and a gateway that refuses every request scores a perfect
# mitigation rate. The two numbers guard different things and both are
# reported.
#
# The denominator is deliberately narrow. Attacks MOCHI never flagged are
# detection misses and belong in the detection rate; attacks it refused
# outright never reached the model and cannot have leaked. What is left is the
# SANITIZE path: a payload was found, cut out, and the remainder forwarded.
# Measured on the generated corpus, 4 of 33 such requests still succeeded,
# against 48 attacks that were never flagged at all - which is the finding
# this metric exists to expose. Redaction mostly holds when it fires; the
# failures are overwhelmingly cases where nothing fired.


@dataclass(frozen=True)
class EnforcementLeakage:
    """Attacks that survived mitigation, as distinct from ones never detected."""

    acted: int
    """Requests where MOCHI returned a verdict other than ALLOW."""
    forwarded: int
    """Of those, the ones still sent upstream - the SANITIZE path."""
    leaked: int
    """Of those, the ones whose attack still succeeded."""
    undetected: int
    """Attacks MOCHI never acted on at all. Not part of the rate; carried so
    the two failure modes can be reported side by side, which is the only way
    the rate can be read correctly."""

    @property
    def leak_rate(self) -> float:
        """leaked / forwarded. ``nan`` when nothing was forwarded.

        NaN rather than 0.0 on an empty denominator: a corpus where nothing
        was ever sanitised has no leak rate, and reporting 0% would read as a
        perfect score rather than an absent measurement. Gandalf is exactly
        this case - every attack sits in the user turn and is refused
        outright, so nothing is ever forwarded.
        """
        return self.leaked / self.forwarded if self.forwarded else float("nan")

    @property
    def undetected_share(self) -> float:
        """Of all attacks that still worked, how many were never flagged.

        The context line for ``leak_rate``: a low leak rate beside a high
        value here means mitigation is sound and detection is the weak point.
        """
        total_failures = self.leaked + self.undetected
        return self.undetected / total_failures if total_failures else float("nan")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "acted": self.acted,
            "forwarded": self.forwarded,
            "leaked": self.leaked,
            "undetected": self.undetected,
            "leak_rate": self.leak_rate,
            "undetected_share": self.undetected_share,
        }


#: Verdicts that mean MOCHI acted. Anything else is "not flagged".
ACTED = frozenset({"BLOCK", "SANITIZE"})
#: Of those, the verdict that still sends the request upstream.
FORWARDED = frozenset({"SANITIZE"})


def enforcement_leakage(rows: Sequence[dict]) -> EnforcementLeakage:
    """Compute leakage from defended simulation rows.

    Each row needs ``mitigation_action`` and ``attack_succeeded``. Rows whose
    verdict is missing are skipped rather than guessed: the image harness
    wrote no verdict before 2026-10-20, and silently treating those as ALLOW
    would report a leak rate for a corpus that cannot support one.
    """
    acted = forwarded = leaked = undetected = 0
    for row in rows:
        verdict = row.get("mitigation_action")
        if verdict is None:
            continue
        succeeded = bool(row.get("attack_succeeded"))
        if verdict in ACTED:
            acted += 1
            if verdict in FORWARDED:
                forwarded += 1
                leaked += succeeded
        elif succeeded:
            undetected += 1
    return EnforcementLeakage(acted=acted, forwarded=forwarded,
                              leaked=leaked, undetected=undetected)
