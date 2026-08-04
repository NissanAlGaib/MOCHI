"""Statistical significance testing (thesis Phase 4 / H01-H06).

Provides the paired t-test, Cohen's d, and confidence intervals the thesis
commits to, plus a normality check - because the paired t-test's validity
depends on an assumption the thesis asserts ("essentially normally
distributed") rather than tests. Reporting the check is the honest move, and
a non-parametric fallback is provided for when it fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

ALPHA = 0.05


@dataclass(frozen=True)
class TestResult:
    """Outcome of a paired significance test."""

    name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int | None
    effect_size: float
    effect_label: str
    mean_before: float
    mean_after: float
    mean_difference: float
    ci_lower: float
    ci_upper: float
    n: int
    normality_p: float | None = None
    normality_ok: bool | None = None

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA

    @property
    def decision(self) -> str:
        return "REJECT H0" if self.significant else "FAIL TO REJECT H0"

    def summary(self) -> str:
        lines = [
            f"  {self.name}",
            f"    n = {self.n}   mean before = {self.mean_before:.4f}   "
            f"mean after = {self.mean_after:.4f}",
            f"    mean difference = {self.mean_difference:+.4f}  "
            f"95% CI [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]",
            f"    {self.name.split()[0]} statistic = {self.statistic:.4f}"
            + (f", df = {self.degrees_of_freedom}" if self.degrees_of_freedom is not None else ""),
            f"    p = {self.p_value:.6f}  ->  {self.decision} (alpha = {ALPHA})",
            f"    Cohen's d = {self.effect_size:.4f} ({self.effect_label} effect)",
        ]
        if self.normality_ok is False:
            lines.append(
                "    WARNING: paired differences failed the normality check "
                f"(Shapiro-Wilk p = {self.normality_p:.4f}). "
                "Report the Wilcoxon result instead - see wilcoxon_test()."
            )
        return "\n".join(lines)


def interpret_cohens_d(d: float) -> str:
    """Conventional Cohen thresholds."""
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def cohens_d_paired(before: Sequence[float], after: Sequence[float]) -> float:
    """Cohen's d for paired samples: mean difference over its standard deviation."""
    differences = [a - b for b, a in zip(before, after)]
    n = len(differences)
    if n < 2:
        return 0.0
    mean_diff = sum(differences) / n
    variance = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
    sd = math.sqrt(variance)
    return mean_diff / sd if sd else 0.0


def check_normality(differences: Sequence[float]) -> tuple[float | None, bool | None]:
    """Shapiro-Wilk on the paired differences.

    Returns ``(p_value, passed)``. Below n=3 the test is not defined, and above
    n=5000 SciPy's approximation degrades, so both return ``(None, None)``.
    """
    n = len(differences)
    if n < 3 or n > 5000:
        return None, None
    try:
        from scipy import stats
    except ImportError:
        return None, None
    if len(set(differences)) == 1:  # zero variance - Shapiro would error
        return None, None
    result = stats.shapiro(list(differences))
    return float(result.pvalue), bool(result.pvalue >= ALPHA)


def paired_ttest(before: Sequence[float], after: Sequence[float], *,
                 name: str = "t-test") -> TestResult:
    """Paired two-tailed t-test with effect size, CI, and a normality check.

    Use when the same evaluation samples are scored under both conditions -
    which is exactly the before/after MOCHI design.
    """
    if len(before) != len(after):
        raise ValueError("paired_ttest requires equal-length sequences")
    n = len(before)
    if n < 2:
        raise ValueError("paired_ttest requires at least 2 pairs")

    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("scipy is required for significance testing") from exc

    differences = [a - b for b, a in zip(before, after)]
    mean_diff = sum(differences) / n

    result = stats.ttest_rel(list(after), list(before))
    statistic, p_value = float(result.statistic), float(result.pvalue)

    variance = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
    standard_error = math.sqrt(variance / n)
    critical = float(stats.t.ppf(1 - ALPHA / 2, n - 1))
    margin = critical * standard_error

    normality_p, normality_ok = check_normality(differences)
    d = cohens_d_paired(before, after)

    return TestResult(
        name=name,
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=n - 1,
        effect_size=d,
        effect_label=interpret_cohens_d(d),
        mean_before=sum(before) / n,
        mean_after=sum(after) / n,
        mean_difference=mean_diff,
        ci_lower=mean_diff - margin,
        ci_upper=mean_diff + margin,
        n=n,
        normality_p=normality_p,
        normality_ok=normality_ok,
    )


def wilcoxon_test(before: Sequence[float], after: Sequence[float], *,
                  name: str = "Wilcoxon signed-rank") -> TestResult:
    """Non-parametric paired alternative, for when normality fails."""
    from scipy import stats

    n = len(before)
    differences = [a - b for b, a in zip(before, after)]
    mean_diff = sum(differences) / n

    result = stats.wilcoxon(list(after), list(before))
    d = cohens_d_paired(before, after)

    return TestResult(
        name=name,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        degrees_of_freedom=None,
        effect_size=d,
        effect_label=interpret_cohens_d(d),
        mean_before=sum(before) / n,
        mean_after=sum(after) / n,
        mean_difference=mean_diff,
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        n=n,
    )


#: The six null hypotheses from the thesis, keyed by the metric they concern.
HYPOTHESES = {
    "accuracy": ("H01", "No significant difference in detection accuracy"),
    "precision": ("H02", "No significant difference in precision"),
    "recall": ("H03", "No significant difference in recall"),
    "f1": ("H04", "No significant difference in F1-score"),
    "attack_success_rate": ("H05", "No significant difference in ASR"),
    "mitigation_rate": ("H06", "No significant difference in Mitigation Rate"),
}


def run_all_hypotheses(
    before_folds: dict[str, Sequence[float]],
    after_folds: dict[str, Sequence[float]],
) -> dict[str, TestResult]:
    """Run H01-H06 across per-fold metric values.

    Each metric needs several paired observations to test, which is what
    5-fold cross-validation provides: one value per fold, per condition.
    """
    results: dict[str, TestResult] = {}
    for metric, (label, statement) in HYPOTHESES.items():
        if metric not in before_folds or metric not in after_folds:
            continue
        results[metric] = paired_ttest(
            before_folds[metric], after_folds[metric], name=f"{label}: {statement}"
        )
    return results


def format_hypothesis_report(results: dict[str, TestResult]) -> str:
    lines = ["", "=" * 74, "  Statistical Significance (paired t-test, alpha = 0.05)", "=" * 74, ""]
    for result in results.values():
        lines.append(result.summary())
        lines.append("")
    lines.append("=" * 74)
    return "\n".join(lines)
