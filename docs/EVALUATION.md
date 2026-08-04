# Evaluation Harness

How to produce the Chapter IV numbers. This is research instrumentation, kept
in `eval/` and deliberately outside the `mochi` package.

## Quick start

```bash
python eval/run_detection.py --demo                 # runnable now, no downloads
python eval/run_detection.py --demo --config baseline --folds 5
```

Once datasets are in `data/`:

```bash
python eval/run_detection.py --data data/ --config baseline --folds 5 --json reports/baseline.json
python eval/run_detection.py --data data/ --config full     --folds 5 --json reports/full.json
```

## Getting the datasets

`data/` is gitignored — datasets are large and separately licensed. Download
them yourself and drop the files in.

| Dataset | Source | Notes |
|---|---|---|
| **PromptShield** | You already have it | Place the CSV/JSONL in `data/` |
| **deepset/prompt-injections** | `huggingface.co/datasets/deepset/prompt-injections` | ~660 samples, `text` + `label` |
| **jayavibhav/prompt-injection** | `huggingface.co/datasets/jayavibhav/prompt-injection` | ~66k samples, the bulk of your corpus |
| ~~MCP-ATTACKBENCH~~ | Not publicly released | Verified: the paper promises a future release with no link. Substituted by the three above |

Download via the hub UI, or:

```python
from datasets import load_dataset          # pip install datasets
load_dataset("deepset/prompt-injections", split="train").to_csv("data/deepset.csv")
load_dataset("jayavibhav/prompt-injection", split="train").to_csv("data/jayavibhav.csv")
```

Downloading to `data/` is preferable to loading from the hub at runtime: the
snapshot is preserved, which is what the Reliability section's reproducibility
commitment requires.

### Format handling

The loader auto-detects columns, so you rarely need to configure anything.

- **Text column** — tries `text`, `prompt`, `input`, `content`, `question`, `instruction`, …
- **Label column** — tries `label`, `is_injection`, `malicious`, `target`, `class`, …
- **Label values** — understands `0/1`, `benign/malicious`, `safe/unsafe`, `true/false`, `jailbreak`, …

Convention throughout: **0 = benign, 1 = malicious**. Rows whose label cannot
be interpreted are *skipped and counted*, never guessed — a silently
mislabeled row corrupts every downstream metric.

Override when detection fails:

```python
from eval.data_loading import load_file
samples = load_file("data/odd.csv", text_column="user_prompt",
                    label_column="verdict", invert_labels=True)
```

### Loading indirect-injection fixtures

By default samples are presented as `user_input`, which exercises the *direct*
injection path. To evaluate indirect injection, load them with the source tag
they represent so they route through the untrusted path:

```python
load_file("data/rag_attacks.csv", source_tag="retrieved_document")
load_file("data/web_attacks.csv", source_tag="web_content")
```

This is what makes the direct/indirect split in your results real rather than
assumed.

## Configurations

Matches the thesis Experimental Design (Detection Effectiveness) table:

| `--config` | Meaning |
|---|---|
| `baseline` | No defense. Control condition — ASR is 1.0 by construction |
| `stage1` | Syntactic filtering only (Phase 6) |
| `stage12` | Syntactic + semantic (Phase 8) |
| `full` | All three stages (Phase 9) |

The stage flags are accepted now so the CLI is stable; they take effect as each
stage lands.

## Metrics

Implemented directly in `eval/metrics.py` rather than delegated to
scikit-learn, so the code visibly matches the thesis Derived Metrics table.
`ConfusionMatrix.cross_check()` verifies against scikit-learn when installed —
and there is a test asserting they agree.

| Metric | Formula | Thesis reference |
|---|---|---|
| Accuracy | (TP+TN)/Total | Derived Metrics |
| Precision | TP/(TP+FP) | Derived Metrics |
| Recall / Detection Rate | TP/(TP+FN) | Derived + Secondary Metrics |
| F1-Score | 2PR/(P+R) | Derived Metrics |
| False Positive Rate | FP/(FP+TN) | Secondary Metrics, target < 1.0% |
| Attack Success Rate | FN/(FN+TP) | Primary Mitigation Metric (H05) |
| Mitigation Rate | TP/(TP+FN) | Primary Mitigation Metric (H06) |

> **Positive class = malicious.** "Recall" therefore reads as *proportion of
> attacks caught*, which is the security-relevant direction.

> **ASR and Mitigation Rate are complements** (they sum to 1.0) when measured
> from the same confusion matrix. There is a test asserting this. If your
> Chapter IV reports them as independent findings, expect a panelist to notice.

## Statistical significance

`eval/stats.py` implements the paired t-test, Cohen's d, and 95% confidence
intervals, keyed to the six null hypotheses H01–H06.

```python
from eval.stats import run_all_hypotheses, format_hypothesis_report

results = run_all_hypotheses(before_folds, after_folds)
print(format_hypothesis_report(results))
```

Each metric needs several paired observations, which is what 5-fold
cross-validation supplies: one value per fold per condition.

### One caveat worth pre-empting

The thesis justifies the paired t-test partly on the data being *"essentially
normally distributed"* — asserted, not tested. `paired_ttest()` runs a
Shapiro-Wilk check on the paired differences and reports it; if it fails, the
result carries a warning and `wilcoxon_test()` provides the non-parametric
alternative. Reporting the check is more defensible than asserting the
assumption, and it costs nothing.

Note also that with 5 folds, n=5 — a small sample for a t-test. The effect
sizes should be enormous (Cohen's d in the tens) precisely *because* the
baseline is 0 by construction. Be prepared to explain that an implausible-looking
d is an artifact of comparing against a null control, not evidence of anything
subtle.

## Current results

With no detector built yet, the harness measures the control condition:

```
  Confusion Matrix
                        Predicted Benign   Predicted Malicious
  Actual Benign                        8                     0
  Actual Malicious                     8                     0

  Accuracy                    0.5000
  Precision                   0.0000
  Recall                      0.0000
  F1-Score                    0.0000
  Attack Success Rate         1.0000    <- no defense: every attack succeeds
  Mitigation Rate             0.0000
```

That is the correct and expected baseline, and it is a real row in your
results table. Every subsequent phase is measured as a delta against it.

## Files

| File | Purpose |
|---|---|
| `eval/data_loading.py` | Loading, label normalization, stratified splits |
| `eval/metrics.py` | Confusion matrix and derived metrics |
| `eval/predictors.py` | Adapters from samples to the MOCHI pipeline |
| `eval/run_detection.py` | Detection evaluation CLI (thesis Phase 2) |
| `eval/stats.py` | Paired t-test, Cohen's d, H01–H06 |
| `eval/run_mitigation.py` | Attack simulation (thesis Phase 3) — arrives in Phase 13 |
