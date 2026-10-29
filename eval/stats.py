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


@dataclass(frozen=True)
class Proportion:
    """A rate, with the interval that says how much to trust it.

    A false-positive rate measured on 40 clips and one measured on 3,080
    queries are both "a percentage", and reporting them side by side without
    intervals invites a comparison neither supports. Wilson rather than the
    normal approximation because these rates sit near zero, where the normal
    interval runs below 0 and stops meaning anything.
    """

    flagged: int
    total: int
    ci_lower: float
    ci_upper: float

    @property
    def rate(self) -> float:
        return self.flagged / self.total if self.total else float("nan")

    def summary(self) -> str:
        if not self.total:
            return "no samples"
        return (f"{self.rate:.2%}  ({self.flagged}/{self.total})  "
                f"95% CI [{self.ci_lower:.2%}, {self.ci_upper:.2%}]")


def wilson_interval(flagged: int, total: int, *,
                    z: float = 1.959963984540054) -> Proportion:
    """Wilson score interval for a binomial proportion.

    Holds up at zero successes, which matters here: an arm that flags nothing
    has a rate of 0.00% and an upper bound that depends entirely on how many
    samples were taken, and the upper bound is the honest part of that claim.
    """
    if total <= 0:
        return Proportion(flagged, 0, float("nan"), float("nan"))
    p = flagged / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    lower, upper = centre - spread, centre + spread

    # At the boundaries the interval is exactly [0, u] or [l, 1], but the
    # arithmetic above leaves float residue there - 1.7e-18 rather than 0 for
    # 0/200. Printed as a percentage that is harmless; compared against 0 it
    # is not, and these arms are expected to report zero often.
    if flagged == 0:
        lower = 0.0
    if flagged == total:
        upper = 1.0
    return Proportion(flagged, total, max(0.0, lower), min(1.0, upper))


def mcnemar_test(both: int, only_a: int, only_b: int, neither: int, *,
                 name: str = "McNemar (paired proportions)") -> TestResult:
    """Whether two paired binary measurements disagree more than by chance.

    The test for "does rendering a query as an image change whether it is
    flagged, on the same queries". A two-sample proportion test would be wrong
    here because the samples are not independent - they are the same queries
    twice - and McNemar uses exactly the information that pairing provides:
    the discordant cells, ``only_a`` and ``only_b``. The agreements carry no
    evidence about a difference and drop out.

    Uses the exact binomial test rather than the chi-square approximation.
    With few discordant pairs - the expected case for rates near zero - the
    approximation is unreliable in the direction that manufactures
    significance.
    """
    discordant = only_a + only_b
    n = both + only_a + only_b + neither

    if discordant == 0:
        p_value = 1.0
    else:
        # Two-sided exact binomial against p = 0.5 on the discordant pairs.
        smaller = min(only_a, only_b)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
        p_value = min(1.0, 2 * tail / (2 ** discordant))

    rate_a = (both + only_a) / n if n else float("nan")
    rate_b = (both + only_b) / n if n else float("nan")
    # Wilson interval on the difference is not standard; the discordant counts
    # are what the test speaks to, so the CI is left to wilson_interval on each
    # arm and this records the shift.
    return TestResult(
        name=name,
        statistic=float(discordant),
        p_value=p_value,
        degrees_of_freedom=None,
        effect_size=0.0 if not n else (only_b - only_a) / n,
        effect_label=f"{only_a} vs {only_b} discordant",
        mean_before=rate_a,
        mean_after=rate_b,
        mean_difference=rate_b - rate_a,
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        n=n,
    )
