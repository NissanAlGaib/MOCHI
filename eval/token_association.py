"""Which tokens are associated with the malicious class, and is it significant?

    python eval/token_association.py --data data/clean
    python eval/token_association.py --data data/clean --top 40 --json reports/tokens.json

Answers the adviser question "relationship of the tokens to the target variable"
with the test that actually fits the data shape.

**Choosing the tool.** Both variables are categorical and binary - a token is
present or absent, a sample is benign or malicious - so the relationship lives in
a 2x2 contingency table:

                     malicious   benign
        token present    a          b
        token absent     c          d

That makes the **chi-square test of independence** the first appropriate tool.
It is not a correlation problem: Pearson's r assumes two continuous variables,
and the point-biserial variant still needs one continuous side. Both would be
the wrong instrument here.

Three refinements matter and are applied:

* **Yates' continuity correction is not used.** With tens of thousands of rows
  the expected counts sit far above 5, where the correction is known to be
  over-conservative. Fisher's exact test is substituted instead whenever any
  expected cell count falls below 5.
* **Multiple comparisons.** Testing ~10,000 tokens at alpha = 0.05 produces ~500
  false positives by construction. Benjamini-Hochberg FDR control is applied
  across the whole token set, and the reported q-values are what should be cited
  rather than the raw p-values.
* **Effect size.** Significance is not importance. At n = 82,765 almost any token
  clears p < 0.05, so Cramer's V and the log-odds ratio carry the actual finding.

Mutual information is reported alongside as a ranking statistic. It is not a
significance test and has no p-value; it measures how much knowing whether the
token is present reduces uncertainty about the label.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import DATA_DIR, DatasetError, load_directory  # noqa: E402

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Tokens rarer than this are dropped before testing. Rare tokens give unstable
#: odds ratios and inflate the multiple-comparison burden without contributing
#: usable evidence.
MIN_DOC_FREQ = 30

#: Expected cell count below which chi-square is unreliable and Fisher's exact
#: test is substituted.
MIN_EXPECTED = 5

#: Benjamini-Hochberg target false discovery rate.
FDR = 0.05


@dataclass
class TokenResult:
    token: str
    n_malicious: int
    n_benign: int
    chi2: float
    p_value: float
    q_value: float
    cramers_v: float
    log_odds: float
    mutual_info: float
    test: str

    @property
    def direction(self) -> str:
        return "malicious" if self.log_odds > 0 else "benign"


def tokenize(text: str, *, ngram_max: int = 1) -> set[str]:
    """Presence set of n-grams, not counts - the table is about occurrence.

    ``ngram_max=1`` reproduces the original unigram behaviour exactly, so any
    figure already cited from a previous run stays reproducible.

    Unigrams alone mislead here, and the existing results show how. Four of the
    strongest associations - ``instructions`` (V=0.291), ``reveal`` (0.259),
    ``ignore`` (0.240), ``previous`` (0.211) - are fragments of the *same phrase*,
    counted as four independent findings. And no unigram model can separate
    "ignore previous instructions" from "ignore the previous email": the
    distinction lives entirely in the adjacency, which is exactly what Stage I
    hand-codes as ``ignore ... previous <instruction|rule|...>``.
    """
    words = [match.group(0).lower() for match in TOKEN.finditer(text)]
    if ngram_max <= 1:
        return set(words)

    grams: set[str] = set(words)
    for size in range(2, ngram_max + 1):
        for start in range(len(words) - size + 1):
            grams.add(" ".join(words[start:start + size]))
    return grams


def contingency(a: int, b: int, c: int, d: int):
    """Chi-square (or Fisher when sparse) plus effect sizes for one 2x2 table."""
    from scipy.stats import chi2_contingency, fisher_exact

    table = [[a, b], [c, d]]
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    expected_min = min(row1 * col1, row1 * col2, row2 * col1, row2 * col2) / n

    if expected_min < MIN_EXPECTED:
        _, p = fisher_exact(table)
        chi2 = float("nan")
        test = "fisher"
    else:
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        test = "chi2"

    # For a 2x2 table Cramer's V reduces to phi = sqrt(chi2 / n).
    v = math.sqrt(chi2 / n) if chi2 == chi2 else float("nan")

    # Haldane-Anscombe correction: add 0.5 to every cell so an empty one does
    # not send the odds ratio to infinity.
    log_odds = math.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
    return chi2, p, v, log_odds, test


def mutual_information(a: int, b: int, c: int, d: int) -> float:
    """Mutual information in bits between token presence and the label."""
    n = a + b + c + d
    total = 0.0
    for observed, row, col in ((a, a + b, a + c), (b, a + b, b + d),
                               (c, c + d, a + c), (d, c + d, b + d)):
        if observed:
            total += (observed / n) * math.log2((observed * n) / (row * col))
    return total


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH-adjusted q-values, returned in the input order."""
    m = len(p_values)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    previous = 1.0
    for rank, index in enumerate(reversed(order), start=1):
        position = m - rank + 1
        value = min(previous, p_values[index] * m / position)
        q[index] = value
        previous = value
    return q


def analyse(samples, *, min_freq: int = MIN_DOC_FREQ,
            ngram_max: int = 1) -> tuple[list[TokenResult], dict]:
    malicious_docs = benign_docs = 0
    malicious_count: Counter = Counter()
    benign_count: Counter = Counter()

    for sample in samples:
        tokens = tokenize(sample.text, ngram_max=ngram_max)
        if sample.label == 1:
            malicious_docs += 1
            malicious_count.update(tokens)
        else:
            benign_docs += 1
            benign_count.update(tokens)

    vocabulary = [
        token for token in set(malicious_count) | set(benign_count)
        if malicious_count[token] + benign_count[token] >= min_freq
    ]

    results: list[TokenResult] = []
    for token in vocabulary:
        a = malicious_count[token]
        b = benign_count[token]
        c = malicious_docs - a
        d = benign_docs - b
        chi2, p, v, log_odds, test = contingency(a, b, c, d)
        results.append(TokenResult(
            token=token, n_malicious=a, n_benign=b, chi2=chi2, p_value=p,
            q_value=0.0, cramers_v=v, log_odds=log_odds,
            mutual_info=mutual_information(a, b, c, d), test=test,
        ))

    for result, q in zip(results, benjamini_hochberg([r.p_value for r in results])):
        result.q_value = q

    meta = {
        "n_samples": malicious_docs + benign_docs,
        "n_malicious": malicious_docs,
        "n_benign": benign_docs,
        "vocabulary_tested": len(vocabulary),
        "min_doc_freq": min_freq,
        "ngram_max": ngram_max,
        "significant_at_fdr": sum(1 for r in results if r.q_value < FDR),
        "fdr": FDR,
        "fisher_substitutions": sum(1 for r in results if r.test == "fisher"),
    }
    return results, meta


def ablation(samples, *, min_freq: int, sizes=(1, 2, 3)) -> None:
    """Does adding n-grams actually find better indicators?

    **Cramer's V is the wrong lens for this question and is reported anyway, so
    the reader can see why.** V is a symmetric association measure: it penalises
    rarity, and every bigram is rarer than its parts. So V does not rise with n,
    and read alone it says n-grams add nothing.

    What actually changes is *contamination* - the share of a term's occurrences
    that sit on benign text. That is the quantity a detector cares about, because
    it is the false-positive rate the term would produce if used as a rule.

        previous              4,184 malicious /  746 benign   (15.1% contaminated)
        previous instructions 2,439 malicious /    9 benign   ( 0.4% contaminated)

    Same phrase family, a 40x cleaner indicator, and a *lower* V. Reporting V
    alone would have hidden the entire finding.
    """
    print()
    print("=" * 100)
    print("  N-gram ablation  -  does context find better indicators?")
    print("=" * 100)
    header = (f"  {'n-gram range':<14}{'tested':>9}{'significant':>13}"
              f"{'max V':>8}{'multiword':>12}{'benign contamination':>23}")
    print(header)
    print("  " + "-" * 96)

    for size in sizes:
        results, meta = analyse(samples, min_freq=min_freq, ngram_max=size)
        significant = [r for r in results if r.q_value < FDR]
        strongest = sorted(
            (r for r in significant if r.cramers_v == r.cramers_v),
            key=lambda r: r.cramers_v, reverse=True,
        )[:50]
        multiword = sum(1 for r in strongest if " " in r.token)
        peak = strongest[0].cramers_v if strongest else float("nan")

        # Mean share of each strong term's occurrences that are benign. This is
        # the false-positive rate the term would produce as a standalone rule.
        contamination = [
            r.n_benign / (r.n_malicious + r.n_benign)
            for r in strongest if r.log_odds > 0 and (r.n_malicious + r.n_benign)
        ]
        mean_contamination = sum(contamination) / len(contamination) if contamination else 0.0

        label = "1 (unigram)" if size == 1 else f"1-{size}"
        print(f"  {label:<14}{meta['vocabulary_tested']:>9,}"
              f"{len(significant):>13,}{peak:>8.3f}"
              f"{multiword:>9}/50{mean_contamination:>22.1%}")

    print("  " + "-" * 96)
    print("  Read the last column, not max V. V penalises rarity, so it cannot")
    print("  rise with n; contamination is the false-positive rate a term would")
    print("  produce if promoted to a Stage I rule, and that is what improves.")
    print("=" * 100)
    print()


def report(results: list[TokenResult], meta: dict, *, top: int) -> None:
    significant = [r for r in results if r.q_value < FDR]
    by_effect = sorted(significant, key=lambda r: abs(r.log_odds), reverse=True)
    malicious = [r for r in by_effect if r.log_odds > 0][:top]
    benign = [r for r in by_effect if r.log_odds < 0][:top]

    print()
    print("=" * 94)
    print("  Token / Class Association  -  chi-square test of independence")
    print("=" * 94)
    print(f"  samples {meta['n_samples']:,}    malicious {meta['n_malicious']:,}"
          f"    benign {meta['n_benign']:,}")
    print(f"  tokens tested (doc freq >= {meta['min_doc_freq']}): "
          f"{meta['vocabulary_tested']:,}"
          f"    Fisher substitutions: {meta['fisher_substitutions']:,}")
    print(f"  significant after Benjamini-Hochberg at FDR {meta['fdr']}: "
          f"{meta['significant_at_fdr']:,} "
          f"({meta['significant_at_fdr'] / max(meta['vocabulary_tested'], 1):.1%})")
    print()

    header = (f"  {'token':<20}{'mal':>8}{'ben':>8}{'chi2':>11}"
              f"{'q':>11}{'V':>8}{'logOR':>9}{'MI':>9}")
    for title, rows in (("Most associated with MALICIOUS", malicious),
                        ("Most associated with BENIGN", benign)):
        print(f"  {title}")
        print(header)
        print("  " + "-" * 90)
        for r in rows:
            chi = f"{r.chi2:,.0f}" if r.chi2 == r.chi2 else "fisher"
            print(f"  {r.token:<20}{r.n_malicious:>8,}{r.n_benign:>8,}{chi:>11}"
                  f"{r.q_value:>11.2e}{r.cramers_v:>8.3f}"
                  f"{r.log_odds:>+9.2f}{r.mutual_info:>9.4f}")
        print()

    strong = [r for r in significant
              if r.cramers_v == r.cramers_v and r.cramers_v >= 0.1]
    print(f"  Effect size: {len(strong):,} of {len(significant):,} significant tokens "
          f"reach Cramer's V >= 0.10 (a small effect).")
    print("  At this n, significance is cheap - effect size carries the finding.")
    print("=" * 94)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-freq", type=int, default=MIN_DOC_FREQ)
    parser.add_argument("--ngram-max", type=int, default=1,
                        help="largest n-gram to test; 1 (default) reproduces "
                             "the original unigram report exactly")
    parser.add_argument("--ablation", action="store_true",
                        help="compare 1 / 1-2 / 1-3 instead of a single report")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    try:
        samples = load_directory(args.data)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.ablation:
        ablation(samples, min_freq=args.min_freq)
        return 0

    results, meta = analyse(samples, min_freq=args.min_freq,
                            ngram_max=args.ngram_max)
    report(results, meta, top=args.top)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        ranked = sorted(results, key=lambda r: abs(r.log_odds), reverse=True)
        args.json.write_text(
            json.dumps({"meta": meta, "tokens": [asdict(r) for r in ranked[:500]]},
                       indent=2),
            encoding="utf-8",
        )
        print(f"  Wrote {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
