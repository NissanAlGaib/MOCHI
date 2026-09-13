# MOCHI Classification Study — Build Plan

The forward plan for the model comparison, reorganised around the two-track
method agreed with the adviser. Same format as `BUILD_PLAN.md`: each step has a
goal, deliverables, key files, and a definition of done.

**This does not replace `BUILD_PLAN.md`.** That document remains the record of
Phases 0–11, the adviser register, and the design decisions behind the gateway.
This one supersedes only its forward half — the Wave B–E table and Phases 8.5
and 13 — because the method changed shape: what was one evaluation is now two
independent tracks that meet once, at scoring.

---

## Status

| Step | Work | State |
|---|---|---|
| 0 | Re-split 70 / 30 | ✅ Done — 3 Sep 2026 |
| 0b | Double 70/30 split + 10-fold CV | ✅ Done — 13 Sep 2026 |
| 1a | Materialise feature table on new splits | ✅ Done — 3 Sep 2026 |
| — | Obfuscation family removed from `FeatureVector` (22 cols) | ✅ Done — 3 Sep 2026 |
| 1b | Feature statistics (`eval/feature_stats.py`) | ✅ Done — 3 Sep 2026 |
| 1c | Prune derived columns, freeze the set | ✅ Done — 3 Sep 2026 |
| 2 | Track A — SVM, decision tree, random forest | ✅ Done — 13 Sep 2026 |
| 3 | Track B — e5, BiLSTM, BiGRU | ⬜ blocked on GPU |
| 4 | The comparison | ⬜ blocked on 2 + 3 |
| 5 | Taglish evaluation set | ⬜ needs native validation |
| 6 | Ship the winner | ⬜ blocked on 4 |

Inherited from `BUILD_PLAN.md` and unaffected by this plan: Phases 0–8, 10 and
11 committed; Phase 9 omitted by design.

`data/features/features.csv` now carries only the identity/tracking columns
(`text_hash`, `dataset`, `split`, `source_tag`, `label`) plus the **18 Track A
columns**: the 17 frozen `mochi.preprocess.features.TRACK_A_FEATURES` (pure
functions of the text) and the one corpus-fitted
`eval.token_association.TRACK_A_FITTED_FEATURES` column,
`malicious_word_weight_sum` — 23 columns total, not the full extractor output.
The broader table used for the Step 1b association pass was intermediate, not a
retained artifact; its findings are recorded in this document and in
`docs/BUILD_PLAN.md`, and `eval/feature_stats.py` can still be re-run against the
trimmed table to re-verify the frozen set whenever the corpus changes.

**Step 3 (Track B) is next**, and is blocked on torch/GPU.

### Step 0b — the split protocol in force (13 Sep 2026)

A **double 70/30 split**, then 10-fold cross-validation inside what remains:

```
corpus (82,765) ──70/30──> development (57,936)  |  test (24,829, sealed)
development     ──70/30──> train (40,556)        |  validation (17,380)
train           ──10-fold stratified CV──> hyperparameter selection
```

Shares live in `eval/data_loading.py` as `TEST_SHARE`, `VALIDATION_SHARE` and
`CV_FOLDS`, imported by both split builders so the two cannot drift. Class
balance is 43.30% malicious in all three tiers, exactly.

Each tier answers one question and is not reused for another:

| tier | decides |
|---|---|
| train (49%) | model parameters; hyperparameters, via the 10 CV folds |
| validation (21%) | the decision threshold, after the model is fixed |
| test (30%) | nothing — scored once, reported |

The validation tier is **permanently** withheld from training, not temporarily.
Refitting the final model on train + validation would shift its score
distribution and invalidate the threshold chosen on validation, and the only
clean data left to re-choose it on would be the test set. The 11,588 training
rows this costs buy a threshold holdout that no fold of the CV can provide,
because every fold's model is discarded before the final model is fit.

---

## The rule this plan exists to enforce

Two tracks. They share the data split beneath them and the scoring code above
them, and **nothing in between**.

| | Track A — Classical | Track B — NLP |
|---|---|---|
| Input | engineered feature table | raw prompt text |
| Models | SVM, decision tree, random forest | fine-tuned e5, BiLSTM, BiGRU |
| Forbidden | tokenizer, embeddings, TF-IDF, any neural layer | engineered columns, `stage1_*` output, hand-built features |
| Hardware | CPU | RTX 4070, 8 GB |

A model that reads both kinds of input belongs to neither track and cannot
appear in the comparison table. This is a rule about **inputs**, not about
algorithms: TF-IDF + LinearSVC is a classical algorithm, but it reads text, so
it is not Track A.

Feature engineering comes first because Track A cannot start without it and
Track B does not care — which means the whole feature effort is on the critical
path for exactly one of the two tracks, and free for the other.

---

## Step 0 — Re-split 70 / 30 ✅ **done**

**Goal:** one stratified split both tracks agree on.

A prerequisite chore, not a phase, but nothing downstream was valid until it
landed. Every table produced before it has to be recomputed after it.

`stratified_split()` defaulted to `train=0.6, validation=0.1`, and both split
builders honoured PromptShield's own files. Those files are not 70/30 and cannot
be made to be:

| file | rows | share |
|---|---:|---:|
| `promptshield_train` | 18,576 | 43.4% |
| `promptshield_validation` | 970 | 2.3% |
| `promptshield_test` | 23,219 | 54.3% |
| `jayavibhav` | 40,000 | — |
| **pooled** | **82,765** | |

PromptShield's test file is **larger than its train file**. No weighting of
those files reaches 70/30, so the only route is pooling all four and splitting
from scratch.

### What changed

- `eval/data_loading.py` — `stratified_split()` now takes `test=0.3,
  validation=0.0`. **The test set is cut first and does not move when
  `validation` changes**; validation is carved out of the training portion
  instead. That single property is what lets the two tracks request different
  validation tiers and still be scored on identical rows.
- `eval/build_features.py` — `OFFICIAL_SPLITS` removed; every file is pooled,
  and the returned splits are `train` / `test` only.
- `training/finetune_e5.py` — `build_splits` pools identically, then asks for
  `validation=0.1`, meaning 10% of the 70% (≈7% of the corpus).
- `tests/test_eval.py` — the 60/10/30 assertion replaced, plus a new test that
  requesting validation does not move the test boundary.
- `tests/test_build_features.py` — **new.** `build_features.py` had claimed this
  file existed since Phase 6.5; it did not, so the "duplication cannot drift"
  guarantee was never actually enforced. It is now, including a case that fails
  if a file named `*_test.csv` is ever routed by filename again.

### Verified on the real corpus

```
total   82,765
  train  57,936   70.0%   malicious 43.30%
  test   24,829   30.0%   malicious 43.30%
```

Class balance is identical across both halves to two decimal places. Full suite:
**561 passed**.

**Definition of done:** ✅ both split functions return the same assignment for
the same seed; the pinning test exists and passes; no code path reads
`promptshield_test.csv` as a designated test set.

---

## Step 1 — Feature engineering

**The first substantive phase.** Phase 6.5 built the extractor; this step turns
its output into a defensible dataset. Three sub-steps, in order.

### 1a — Materialise against the new splits ✅ **done**

Re-run the extractor so the `split` column is truthful.

- `python eval/build_features.py --data data/clean` → `data/features/features.csv`

This run also carries the obfuscation removal below, so the table changed
shape twice in the same pass: new splits, and 22 fewer columns.

### Amendment — obfuscation removed from the feature set entirely

Not part of the original A11/A12 scope, decided during Step 1b review: **the
obfuscation family does not belong in this dataset at all**, and was removed
from `FeatureVector` itself rather than filtered out at analysis time.

Revealing obfuscation is Phase 3's job, and it has already run by the time
any classifier — Track A or Track B — sees a prompt: base64 is decoded into a
scannable variant, homoglyphs are folded, zero-width characters are stripped,
before extraction starts. A column meaning "was something obfuscated"
describes what preprocessing had to undo, not a property of the prompt — the
wrong kind of feature for a dataset whose job is describing text a classifier
judges. `NormalizationResult.flags` still carries all of it, unchanged, for
telemetry and for Stage I.

The measurement agreed: every one of the 22 columns came back negligible in
the association pass below, because the corpora in hand carry very little
obfuscation. Removed from `mochi/preprocess/features.py`: `has_zero_width`,
`has_bidi`, `homoglyphs_normalized`, `is_mixed_script`, `nfkc_applied`,
`excessive_special_chars`, `base64_decoded`, `hex_decoded`, `rot13_decoded`,
`url_decoded`, `html_stripped`, `hidden_css_detected`,
`html_comment_extracted`, `attribute_text_extracted`,
`file_metadata_extracted`, `decode_depth_exceeded`, `oversized_after_decode`,
`truncated_for_inspection`, `n_flags`, `n_variants_recovered`,
`decoded_char_gain`, `normalization_delta`. Full rationale in
`BUILD_PLAN.md`'s "obfuscation dropped from the dataset" amendment;
`tests/test_features.py::test_no_obfuscation_columns_in_the_feature_set` pins
the absence.

The raw column count drops from 73 to 51.

### Amendment — the materialised row is int/bool only, no float survives

Not part of the original A11/A12 scope either, decided after Step 1c froze
`TRACK_A_FEATURES`: every value `FeatureVector.as_dict()` emits must be an
`int` or a `bool` - a materialised dataset with a decimal column is a dataset
that still needs an encoding decision made against it later, and the whole
point of `.as_dict()` is that no later consumer has to make one.

The categorical fields (`stage1_max_severity`, `dominant_script`,
`detected_language`, `payload_region`) were already converted to ordinal/
one-hot int and boolean columns as part of Step 1c's encoding work. This
amendment covers the twelve genuinely continuous fields -
`avg_word_len`, `uppercase_ratio`, `digit_ratio`, `question_ratio`,
`newline_ratio`, `special_char_ratio`, `instruction_verb_ratio`,
`english_ratio`, `tagalog_ratio`, `max_url_query_entropy`,
`first_hit_offset_ratio`, `payload_share` - none of which has a natural
integer form, since every one of them is a ratio or an average.

**Fixed-point scaling, not truncation.** Each value is multiplied by
`RATIO_SCALE = 10_000` and rounded to the nearest int, and the column is
renamed with an `_x10k` suffix (`instruction_verb_ratio` →
`instruction_verb_ratio_x10k`) so the scale is legible from the column name
rather than left for a reader to discover by comparing magnitudes. Plain
rounding to the nearest whole number was rejected: most of these ratios sit
well under 1.0 (`instruction_verb_ratio`'s mean is 0.015 in Step 1b), and
rounding to a bare int would collapse the entire column to zero.

**This changes nothing about correctness for any Track A model.** A decision
tree or random forest splits on an ordering, and multiplying a column by a
positive constant does not change any pair's relative order. The SVM step
standardises its inputs before training (`eval/baseline_models.py`'s scaling
note), which removes the effect of a prior fixed multiplier exactly. The
Step 1b effect sizes are likewise unaffected - Cliff's delta and Cramér's V
are both rank-based and cannot change under a monotonic transform. The four
decimal digits of precision `RATIO_SCALE` preserves are finer than any effect
size reported anywhere in this document.

`TRACK_A_FEATURES` was updated for its two affected entries:
`instruction_verb_ratio` → `instruction_verb_ratio_x10k`,
`special_char_ratio` → `special_char_ratio_x10k`. Pinned by
`tests/test_features.py::test_float_fields_matches_every_float_on_the_dataclass`
(a new float field added to `FeatureVector` without also being added to
`FLOAT_FIELDS` fails this test before it can reach the CSV unscaled) and
`test_materialised_row_is_int_and_bool_only` (fails on any row containing
anything but `int`/`bool` — checked by exact `type()`, not `isinstance()`,
since `isinstance(True, int)` is `True` in Python and would let a stray float
through a looser check).

`eval/baseline_models.py`'s `engineered_transformer()` one-hot width dropped
from 79 to 57 in the Step 1c categorical work and is unaffected by this
amendment (fixed-point scaling renames a column, it does not expand it) —
`tests/test_features.py::test_engineered_transformer_matches_the_frozen_track_a_set`
pins the current number (17, matching `len(TRACK_A_FEATURES)`, since no
one-hot expansion happens on the frozen numeric/boolean set at all).

### 1b — Feature statistics ✅ **done**

**This is the deliverable the adviser is asking for**, and it did not exist
before this step. `eval/stats.py` is paired t-tests for comparing *models*;
nothing measured a column against the label.

New file, `eval/feature_stats.py`, producing three outputs:

1. **Per-column association with the label.** Cliff's delta for numeric
   columns (chosen over Cohen's d — these are counts and ratios with heavy
   right tails, and delta only reads ordering, so a long tail cannot inflate
   it), signed Cramér's V for booleans. Effect size is reported ahead of the
   p-value, since at ~58,000 training rows almost everything clears
   significance and the p-value alone says nothing about whether an effect is
   large enough to build on.
2. **Feature-to-feature correlation matrix**, with deterministic derivatives
   (`est_token_count = char_count // 4`) reported separately from genuine
   correlations so a reader does not mistake extractor arithmetic for a
   finding.
3. **Benjamini–Hochberg correction across all columns** (`--by-dataset`
   additionally recomputes the top effects within each source corpus, to catch
   an effect that only exists because of how the corpora were merged).
   Reuses `eval/token_association.py`'s BH implementation.

Everything is fitted on the **train split only** — the script refuses to run
without a `split` column and reports the test-row count as "held out and
untouched" rather than reading it.

**Findings, confirmed on the final 57,936-row train split** (identical to the
interim run on the 42,575-row pre-re-split subset to three decimal places — the
re-split changed which rows landed in train, not the conclusions):

- **28 of 46 columns are negligible**, down from 50 of 68 before obfuscation
  was removed — the family that was cut was already carrying almost none of
  the dead weight's total, but cutting it still simplified the set
  meaningfully.
- **The imperative family is the strongest signal in the dataset**:
  `imperative_verb_count` (+0.363), `instruction_verb_ratio` (+0.342, materialised
  as `instruction_verb_ratio_x10k` per the fixed-point amendment below — the
  effect size is identical either way, since scaling by a positive constant
  cannot change a rank-based measure), `starts_with_imperative` (+0.234,
  corpus-specific — see below). A12's imperative hypothesis is vindicated, and
  by a wide margin over every other family.
- **The question-mark hypothesis is directionally right, weak**:
  `ends_with_question` −0.170 (small; benign prompts end in "?" more often),
  but `question_mark_count` (−0.025) and `question_ratio` (−0.051) are
  negligible — the boolean carries the whole signal, the count and ratio are
  noise beside it. A12's second half is confirmed but modest, exactly the
  outcome `BUILD_PLAN.md` said to report honestly either way.
- **Length is real, not a merge artifact.** `char_count`/`word_count`/
  `max_line_len` all land medium-to-small, and the `--by-dataset` check — the
  one `BUILD_PLAN.md` demands before trusting length in this corpus — found the
  effect gets *stronger* inside every individual source corpus (+0.50 to +0.65)
  than it looks pooled (+0.397). It clears the trap rather than falling into it.
- **`colon_count`, `quote_count`, `starts_with_imperative`, and
  `negation_count` are PromptShield house style, not a universal injection
  marker.** Confirmed by `--by-dataset`: strong across all three PromptShield
  splits (+0.39 to +0.63) but near-zero in jayavibhav (−0.003 to +0.092). Keep
  them — three of four still land medium in the pooled ranking — but never
  present an importance score for them without the per-dataset breakdown
  beside it, and expect a panelist to ask why they vanish in one corpus.
- Stage I columns did **not** dominate, contrary to the circularity concern
  raised for this step — only `hit_direct_prompt_injection` (+0.201) and
  `stage1_would_block` (+0.176) showed real signal; `stage1_hit_count` itself
  and six of the nine `hit_*` one-hots were negligible, and three
  (`hit_data_exfiltration`, `hit_indirect_prompt_injection`,
  `hit_obfuscation_encoding`, `hit_url_exfiltration`) were not even
  statistically significant after BH correction. The Step 2 ablation is still
  worth running, but the fairness risk to Track A was smaller than expected.
- `source_tag` is **constant** across every row in this corpus. It is metadata
  from `Sample`, not a `FeatureVector` column, so it never reaches
  `engineered_transformer()` and needs no pruning decision — noted here only so
  a direct CSV read (outside the modelling pipeline) does not mistake it for a
  live column.

### 1c — Prune, then freeze the column set ✅ **done**

**Final decision — 17 columns**, chosen from the Step 1b association pass and
frozen as `mochi.preprocess.features.TRACK_A_FEATURES`:

```
char_count               instruction_verb_ratio_x10k     quote_count
word_count                question_mark_count            bracket_count
is_code_switched          ends_with_question              special_char_ratio_x10k
second_person_count       colon_count                     line_count
obligation_count          negation_count                   max_line_len
imperative_verb_count    starts_with_imperative
```

The two ratio-valued columns carry the `_x10k` suffix per the amendment below —
the materialised value is the ratio × 10,000 as a plain int, not a float.

This is narrower than the interim pruning plan (which had kept the injection
lexicon and the language ratios pending further checks). The final cut removes
three whole families rather than individual columns:

- **The entire injection-lexicon family is out** — `stage1_hit_count`,
  `stage1_would_block`, `stage1_max_severity`, and all nine `hit_*` one-hots.
  This resolves the Step 2 circularity concern **structurally** rather than by
  ablation: a model trained on `TRACK_A_FEATURES` cannot be rediscovering
  Stage I's own regexes, because none of Stage I's output is in its input. The
  "run twice, with and without `stage1_*`" plan in Step 2 is no longer
  necessary and has been dropped from that step.
- **The URL/entity and position families are out** — `url_count` through
  `has_html_tag`, and `first_hit_offset_ratio`/`payload_share`/
  `payload_region`. All measured negligible in Step 1b.
- **The script/language family is reduced to `is_code_switched` alone** —
  `dominant_script`, `detected_language`, `english_ratio`, `tagalog_ratio`, and
  `language_switch_count` are out, despite `language_switch_count` measuring
  +0.169 (small, real signal). `is_code_switched` is kept regardless of its
  own not-significant reading on this corpus, for construct validity on the
  A14 dimension — the Step 5 Taglish evaluation set is where this column is
  expected to start doing real work.
- **`est_token_count` is out** as the deterministic derivative of `char_count`
  it always was.
- **`question_mark_count` and `ends_with_question` both stay**, even though
  Step 1b found the boolean carries nearly all of the signal. A12 names
  question marks specifically, so the raw count stays as the direct answer to
  the adviser's literal hypothesis alongside the column that actually predicts.
- **`colon_count`, `quote_count`, `starts_with_imperative`, `negation_count`
  all stay** despite being PromptShield house style rather than universal
  markers (Step 1b's `--by-dataset` finding). Any importance figure reported
  for these four must carry the per-dataset breakdown alongside it, or the
  finding is silently corpus-specific.

**Every one of the 17 is numeric or boolean.** `eval/baseline_models.py`'s
`engineered_transformer()` was rewritten to build exactly this list — no
one-hot or ordinal encoding is needed anymore, since the columns that required
it (`payload_region`, `stage1_max_severity`, `dominant_script`,
`detected_language`) are all excluded. It also no longer runs a Stage I scan or
imports `mochi.detect` at all, since nothing in the frozen set depends on it —
a real dependency reduction, not just a column-count one.

Frozen and pinned the same way `DETECTOR_IDS` is pinned:
`tests/test_features.py::test_track_a_features_are_the_step_1c_frozen_set`
fails if a name stops resolving on `FeatureVector`, and
`test_engineered_transformer_matches_the_frozen_track_a_set` fails if the
transformer's output width drifts from `len(TRACK_A_FEATURES)`.

**Definition of done:** ✅ `eval/feature_stats.py` runs from one command and
emits the association table and correlation matrix; every deterministic
derivative is removed; the surviving column set is frozen and pinned by a test;
the length and question-mark hypotheses each have a reported effect size.

### Amendment — the materialised table is trimmed to the frozen set, not kept full

A further instruction after the frozen set was chosen: `eval/build_features.py`
now writes **only** the identity/tracking columns plus the 17
`TRACK_A_FEATURES` columns — 22 columns, not the 51 the extractor can compute.
The broader table that Step 1b's association pass was run against was not kept
as a retained artifact once its job (choosing the frozen set) was done; its
findings live in this document instead.

This also dropped the per-row Stage I scan from `build_features.py` entirely
(nothing in `TRACK_A_FEATURES` needs it — the same reasoning as
`engineered_transformer()`'s equivalent change), which is most of why the
rebuild went from several minutes to well under one.

To reproduce the Step 1b analysis after a corpus change, `eval/feature_stats.py`
still runs correctly against the trimmed table — it will simply report on 17
columns instead of 46, which is the point: it becomes a re-verification of the
frozen set rather than a rediscovery process.

---

## Step 2 — Track A: classical models

**Goal:** SVM, decision tree, and random forest on the frozen 17-column
`TRACK_A_FEATURES` table.

`eval/baseline_models.py` already has `engineered_transformer()` (rewritten in
Step 1c to build exactly the frozen set) and an `engineered only` pipeline that
runs logistic regression alone. The work is adding three estimators beside it,
not building new infrastructure.

- Scale before the SVM (mandatory); trees are scale-invariant, so the pipeline
  branches
- Class weighting — the corpus is not balanced
- k-fold inside the 70% for any tuning; the 30% stays sealed

**No `stage1_*`/`hit_*` ablation is needed.** Step 1c resolved the circularity
concern by removing the entire injection-lexicon family from
`TRACK_A_FEATURES` — a model trained on the frozen set structurally cannot be
rediscovering Stage I's own regexes, since none of Stage I's output reaches it.
There is nothing left to run twice.

**Definition of done:** three models trained and scored on the sealed 30%;
feature importances reported for the tree models, with the PromptShield-specific
columns (`colon_count`, `quote_count`, `starts_with_imperative`,
`negation_count`) flagged wherever they rank highly; no text-derived feature
anywhere in the pipeline.

---

## Step 3 — Track B: NLP models

**Goal:** the neural ladder on raw text, sharing one training harness.

Order matters: e5 first, because `training/finetune_e5.py` already exists and
establishes the harness the recurrent baselines then reuse.

- Fine-tune `intfloat/multilingual-e5-small`, export weights, enable Stage II
- BiLSTM, then BiGRU on the same splits and the same tokenisation

**8 GB VRAM:** batch 32 at 512 tokens will not fit in fp32. Use fp16 or gradient
accumulation, and record which — it is a reproducibility detail and a panelist
who trains models will ask.

The validation set is carved out of the 70%, never the 30%.

Nothing from `mochi/preprocess/features.py` enters this track. If a feature
column would help the neural model, that is a finding about the features, not a
licence to add them here — adding them collapses the comparison into a hybrid and
there is no longer a Track B to compare against.

**Definition of done:** three neural models trained on the same 70%, scored on
the same 30%, no engineered column in any input tensor, precision setting
recorded.

---

## Step 4 — The comparison

**Goal:** one table, one denominator.

Both tracks scored by `eval/metrics.py` on identical rows with identical
thresholds. Report per model:

F1 · precision · recall · **FPR** · inference latency · parameter count

Latency and parameter count are not decoration. The winner becomes Stage II of a
gateway sitting inline on every request, so a model that wins by two F1 points
and costs 50 ms has not won. This is the same argument `baseline_models.py`
already makes for preferring a linear model over a transformer, applied to the
final choice.

FPR carries the weight it does everywhere else in this project: over-blocking the
legitimate user is a direct utility cost, and pooled accuracy hides it.

**Definition of done:** a single table containing every model from both tracks,
computed by one code path.

---

## Step 5 — Taglish evaluation set

Unchanged from `BUILD_PLAN.md` register item A14, and still outstanding: ~200
attack + ~200 benign code-switched English–Tagalog samples, native-speaker
validated, held out entirely from both tracks.

Report the false-positive rate on benign Taglish **separately** from the pooled
FPR. Pooling hides exactly the failure this set exists to expose.

No multilingual claim is publishable without it. Scope stays English–Tagalog.

**Definition of done:** the set exists, is native-validated, was never trained
on, and both tracks report a separate FPR on it.

---

## Step 6 — Ship the winner

Phases 12–14, all blocked on Step 4.

- Winner wired in as Stage II behind `MOCHI_ENABLE_STAGE2`
- Provider adapters (Anthropic, Gemini)
- Packaging

If Track A wins, the runtime gains a dependency on the feature extractor in the
request path — which `BUILD_PLAN.md` permits only if the ablation shows it earns
its place. Step 4 is that ablation.

---

## Decisions made

**1. The TF-IDF baselines belong to neither track — resolved.**
`eval/baseline_models.py` contains TF-IDF + logistic regression, TF-IDF +
LinearSVC, and naive Bayes: classical algorithms over text features, so not
Track A (they read raw text) and not Track B (no neural model).
`build_hybrid_models()`'s docstring now says so explicitly — a standing
reference ablation, not a contender in the classification study's comparison
table.

**2. `stage1_*` columns in Track A — resolved, structurally.** Step 1c removed
the entire injection-lexicon family from `TRACK_A_FEATURES`. No ablation is
needed because the columns that raised the concern are not in the frozen set
at all.

**3. Discarding PromptShield's official splits — resolved.** Pooling all four
files and splitting 70/30 from scratch, done in Step 0; PromptShield's own
files run 43/2/55 with the test file larger than the train file, so honouring
them made 70/30 unreachable. The cost — no direct comparability with a number
published against PromptShield's own test file — is accepted and stated here
rather than left for a panelist to find.

---

## What does not change

Phases 0–8, 10 and 11 are committed and unaffected. Phase 9 remains omitted by
design (register Q7). The gateway, telemetry, normalization layer, Stage I, the
session risk accumulator, enforcement and outbound interception all stand as
built — this plan governs the classification study that fills Stage II, and
nothing else.
