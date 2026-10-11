# What is actually used

A map of the repository, written because 37 scripts in `eval/` is more than
anyone can hold in their head, and because "is this still used?" is not
answerable by looking at a filename.

Verified against the code on 2026-10-11, not from memory: every "produces"
claim below was checked by running the script or by reading the path it writes
to, and the superseded entries were checked by confirming their output file no
longer exists.

---

## 1. The system itself — `mochi/`

All of it is used. This is the product, not scaffolding.

| Path | What it does |
|---|---|
| `mochi/gateway/` | The OpenAI-compatible proxy. `app.py` is the entry point. |
| `mochi/detect/` | Segmentation, Stage I rules, Stage II classifier, the cascade. |
| `mochi/mitigate/` | The ALLOW / SANITIZE / BLOCK decision and redaction. |
| `mochi/preprocess/` | Normalization: zero-width, base64, homoglyphs, HTML, PDF, images. |
| `mochi/session/` | Cross-turn risk accumulation. |
| `mochi/telemetry/` | The JSONL log schema every result is derived from. |

Run it with `python -m mochi.gateway.app`, or `docker compose up --build`.

## 2. What trained the shipped model — `training/`

| Path | Used? | Notes |
|---|---|---|
| `finetune_e5.py` | **yes** | Produced `models/e5-fine-tuned` and `models/e5-banking`. |
| `model.py` | **yes, at runtime** | Stage II imports it to rebuild the classifier head. Not just a training file. |
| `split_manifest.py` | **yes** | Records a hash of every training row. This is what makes "sealed" checkable. |
| `finetune_rnn.py`, `rnn_models.py` | yes, for Track B | Produced `models/bilstm` and `models/bigru`, the neural baselines. |
| `fetch_fasttext.py` | once | Downloads embeddings for the RNN baselines. Not needed again. |

## 3. The models — `models/`

| Directory | Size | Status |
|---|---|---|
| `e5-fine-tuned/` | 468 MB | **The shipped model.** Everything defaults to it. |
| `e5-banking/` | 468 MB | Retrained with BANKING77 benign data. Lower false positives (0.13% vs 2.66%), 5pp less detection. Decision pending — see 4.2 in `docs/STATE_AND_TODO.md`. |
| `bilstm/`, `bigru/` | small | Track B neural baselines, for the comparison table only. Not used at runtime. |

Both E5 directories carry a `split_manifest.json`. That is how
`eval/build_sealed_test.py` knows which rows the model has seen.

## 4. The datasets — `data/`

| Path | Status | What it is |
|---|---|---|
| `data/clean/` | **in use** | The training corpus, cleaned. Six files. **Note:** `banking77_benign.csv` was added 2026-10-09 and is *not* part of the corpus the September reports describe — see the warning below. |
| `data/sealed/taglish_heldout.csv` | **in use** | 300 code-switched rows. Enforced out of training by `tests/test_sealed_data.py`. |
| `data/external/` | **in use** | BANKING77, CLINC150 and Gandalf, plus intent-name maps. Third-party evaluation sets. |
| `data/attacks/` | **in use** | The 100 generated attacks, the Gandalf corpus, and `system_prompt.txt` — the declared prompt the gateway compares against. |
| `data/features/features.csv` | **in use** | The 23-column engineered feature table for Track A. Rebuilt 2026-10-11. |
| `data/splits/` | derived | Regenerate with `eval/build_sealed_test.py` and `eval/build_ood_set.py`. Not worth keeping. |
| `data/_corpus_sept14/` | derived | The September corpus, staged for regenerating old reports. See the warning. |
| `data/taglish_candidates/`, `taglish_review/`, `taglish_source/`, `taglish_train/` | intermediate | Working files from building the Taglish set. Keep for provenance; nothing reads them now. |
| `data/excluded/`, `data/vectors/` | intermediate | Same. |

> **The corpus changed on 2026-10-09.** `banking77_benign.csv` (9,985 rows) was
> added to `data/clean/` for the banking retrain. Any script pointed at
> `data/clean` today sees a different corpus from the one the September reports
> describe — 95,150 rows against 85,165. That is why the regeneration in 4.12
> uses `data/_corpus_sept14/`, which is `data/clean` minus that one file.
> Confirmed: the 5-file corpus is 85,165 rows, and + 300 sealed Taglish rows
> gives 85,465, exactly the `e5-fine-tuned` manifest total.

## 5. Which script produced which result

Every number in `reports/mochi_results.xlsx` comes from one of these.

| Result file | Produced by |
|---|---|
| `simulation_runs.jsonl` | `eval/attack_simulation.py` (corpus: `attack_corpus.py`, targets: `targets.py`) |
| `gandalf_runs.jsonl` | `eval/attack_simulation.py` with `gandalf_corpus.py` |
| `gandalf_detection*.jsonl` | `eval/attack_simulation.py --detect-only` |
| `image_runs*.jsonl` | `eval/image_simulation.py` |
| `benign_runs*.jsonl` | `eval/benign_run.py` |
| `detection_stage{1,2,12}_sealed.json` | `eval/run_detection.py --config <arm>` on `eval/build_sealed_test.py` output |
| `detection_stage12_per_source.json` | `eval/per_source_detection.py` |
| `ood_stage{1,2,12}.json` | `eval/per_source_detection.py` on `eval/build_ood_set.py` output |
| `long_document_fpr.json` | `eval/long_document_fpr.py` |
| `taglish_contamination.json` | `eval/taglish_contamination.py` |
| `utility_runs.jsonl` | `eval/utility_run.py` |
| `thresholds.json`, `e5_scores.json` | `eval/threshold_sweep.py` |
| `token_association.json` | `eval/token_association.py` |
| `track_a.json` | `eval/baseline_models.py --track-a` |
| `comparison.json` | `eval/compare_tracks.py` |
| `translation_ablation.json` | `eval/translation_ablation.py` |
| `mochi_results.xlsx` | `eval/build_report.py` — **run this last** |

**`taglish_ablation.json` has no producer script.** It is a hand-recorded
summary of the translation ablation (three arms, `[tp, malicious, fp, benign]`
each), and `build_report.py` reads it. Worth regenerating from
`translation_ablation.py` output so it is reproducible, or at least noting its
provenance in the write-up.

## 6. Support libraries — imported, never run directly

`eval/data_loading.py` (imported by 22 others), `predictors.py`, `metrics.py`,
`stats.py`, `targets.py`, `attack_corpus.py`, `gandalf_corpus.py`.

## 7. Dataset preparation — ran once, keep for provenance

`fetch_datasets.py`, `clean_datasets.py`, `audit_datasets.py`,
`build_features.py`, `build_tagalog_lexicon.py`,
`generate_taglish_candidates.py`, `wordnet_synonyms.py`,
`fit_malicious_word_weights.py`.

Rerun only if the corpus changes. `build_features.py` must be rerun whenever
`data/clean` changes, because `feature_stats.py` and Track A read its output.

## 8. Superseded — safe to ignore

| Path | Why |
|---|---|
| `eval/export_simulation.py` | The eleven-sheet workbook. Replaced by `build_report.py` (five sheets). Its output, `reports/attack_simulation.xlsx`, no longer exists on disk. Still contains one unverified string about a 0.7508 score — see section 6 of `STATE_AND_TODO.md`. |
| `reports/*_PRE_FIX.*`, `*_OLDSCENARIO.*`, `*_MISMATCHED.*` | Superseded runs, kept deliberately for before/after comparison. Do not quote them. |
| `reports/audit.json`, `baseline.json`, `baselines.json`, `stage1.json`, `clean_*.json`, `hybrid_ablation.json`, `evasion-matrix.csv` | Pre-September phases. Nothing reads them and no current result depends on them. |

## 9. Utilities

`eval/watch_simulation.py` — live progress for a long run.
`eval/wordcloud_figures.py` — figures for the write-up.
`eval/feature_stats.py` — re-verifies the frozen Track A feature set.

---

## Reproducing the headline numbers

In order. Steps 1–2 need nothing running; steps 3–4 need Ollama.

```bash
# 1. detection only - no LLM, a few minutes
python eval/build_sealed_test.py
python eval/run_detection.py --data data/splits --config stage12 \
    --json reports/detection_stage12_sealed.json
python eval/per_source_detection.py

# 2. out of distribution
python eval/build_ood_set.py
python eval/per_source_detection.py --data data/splits/ood_test.csv \
    --config stage12 --out reports/ood_stage12.json

# 3. against real models - needs `ollama serve` and the three targets
python -m mochi.gateway.app                       # in another terminal
python eval/attack_simulation.py

# 4. false positives
python eval/benign_run.py

# 5. the workbook, always last
python eval/build_report.py
```
