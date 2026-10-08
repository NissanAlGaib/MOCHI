# MOCHI — State and To-Do

Written 8 October 2026, at the end of the evaluation session. Supersedes the
status half of `NEXT_STEPS.md`; that document's Part A (deployment model) and
Part B (dataset choices) still stand.

Everything below is either **measured and valid**, or **to do**. Numbers quoted
here were produced under one consistent configuration unless marked otherwise.

---

## 1. Results you can use

All measured against the same build: the enforcement fix, the trust fix, and
the minimal "password" scenario.

### Attack mitigation — three models, with and without MOCHI

**Gandalf (1,000 real human attacks, Lakera, MIT, zero training overlap)**

| Model | ASR undefended | ASR defended | Dangerous | Still work | Mitigation |
|---|---|---|---|---|---|
| Mistral 7B | 69.0% | 10.3% | 207 | 30 | **85.5%** |
| Qwen2.5 7B | 53.3% | 6.0% | 160 | 18 | **88.8%** |
| Llama 3.1 8B | 11.0% | 4.7% | 33 | 11 | 66.7% |
| **Overall** | | | **400** | **59** | **85.2%** |

Run on a seeded 300-prompt sample. Detection swept the full 1,000.

**Generated corpus (100 attacks, 4 escalating tiers)**

| Model | ASR undefended | ASR defended | Dangerous | Still work | Mitigation |
|---|---|---|---|---|---|
| Mistral 7B | 66% | 23% | 66 | 21 | 68.2% |
| Qwen2.5 7B | 56% | 19% | 56 | 16 | 71.4% |
| Llama 3.1 8B | 22% | 10% | 22 | 10 | 54.5% |
| **Overall** | | | **144** | **47** | **67.4%** |

By tier, defended: naive 10/75, **paraphrase 30/75**, obfuscation 0/75,
structural 12/75.

### Detection — MOCHI's verdict alone, no model involved

| Corpus | Detection rate | Stage II | Stage I |
|---|---|---|---|
| Gandalf, all 1,000 | **85.3%** | 533 | 320 |

Identical before and after the scenario change, which is the evidence it never
depended on the scenario.

### False positives — ordinary traffic, no attacks

| Dataset | Queries | Refused | FPR |
|---|---|---|---|
| BANKING77 (banking) | 3,080 | 82 | **2.66%** |
| CLINC150 (10 domains) | 5,500 | 175 | **3.18%** |

Measured twice, identical both times. Worst categories:
`terminate_account` 29/40 (**72%**), `reset_settings` 27/30 (**90%**),
`cancel` 17/30, `unable_to_verify_identity` 9/40.

### Inherited, still valid

- Detection study on the sealed 25,639-row test split: F1 0.9856, FPR 0.48%
- Stage I alone on 82,765 rows: recall 5.9%, precision 97.3%, FPR 0.13%
- Test suite: **691 passing**

---

## 2. Findings worth writing up

**The corpus teaches that distress language is an attack.** In the training
split, `help me` is 86% malicious, `urgent` 83%, `cannot` 78%, `error` 69%.
Attackers genuinely use urgency as social engineering - that part is correct.
The problem is the benign half contains no customer support at all; sampled
benign rows are anatomy exercises, French homework and essay prompts. The model
has never seen "my card doesn't work" labelled benign. This is register item
**D6** (jayavibhav's benign half is not application-representative) turning out
to be the direct cause of the false-positive rate.

**An LLM instructed to paraphrase is a stronger adversary than crowdsourced
humans.** Mitigation is 67.4% against the generated corpus and 85.2% against
Gandalf. Tier 2 - paraphrases produced by a model told to strip distinctive
wording - leaks 40% of the time. Gandalf players were attacking a different
system and leaned on phrasings that still carry attack signal.

**The wording of the secret changes how often attacks land.** Renaming the
protected value from "internal reference" to "password" raised undefended ASR
on every model, most sharply on Llama (5% -> 22%). ASR figures are not
comparable across papers unless the scenario vocabulary matches.

**MOCHI's verdict is model-independent, demonstrated not assumed.** The
identical 64 attacks were stopped for all three targets, because enforcement
happens on the prompt before anything is forwarded. This is the empirical form
of the model-independence claim, and it was obtained free of charge.

**Two defects found only by system-level evaluation.** Both passed 683 unit
tests. See section 3.

---

## 3. Defects found and fixed this session

**Flag-only Stage I detections fell through to ALLOW.** `invisible_text` and
`obfuscation_encoding` carry no `matched_text`, so `_stage1_targets` built no
redaction target, `decide` found nothing confident, and allowed the request -
while `inspect` had already skipped Stage II because Stage I reported a block.
All six `block_invisible_text` detections were allowed through. Fixed; tier 3
obfuscation leaks went 2/75 to 0/75. Three regression tests.

**Enforcement acted on detections inside the system prompt.** Stage II scores a
security-worded system prompt at 0.7508 - above the blocking threshold -
because telling a model to protect its instructions uses the same language as
telling it to reveal them. Every request to such a deployment was refused,
benign ones included. 141 of 294 blocks in the first simulation were MOCHI
refusing its own configuration, inflating the reported mitigation rate from
~70% to ~98%. Fixed by exempting only the **declared** system prompt, matched
by SHA-256. Four regression tests.

The narrow exemption matters: trust is declared, not proven. Exempting every
`role: system` segment would let an attacker bypass Stage II by relabelling
their payload - and all three models obey an injected system message as readily
as a user-turn one.

**Test suite read the developer's `.env`.** Two tests failed whenever Stage II
was enabled locally. Fixed with `tests/conftest.py` pinning the configuration.
Suite also dropped from 111s to 20s.

**Case-sensitive scoring.** A model emitting `ZEBRa` for a demanded `ZEBRA` was
scored as a defence win. Fixed, and the stored replies re-scored without a
re-run.

**Single-canary scoring undercounted extraction.** An attack that pulled the
document's audit tag instead of the system password scored as a failure.
Extraction now counts any planted secret.

---

## 4. To do

### Blocking / highest value

**4.1 Commit everything.** Nothing from this session is committed: 8 modified
files, 12 untracked. Two security fixes, eight eval modules, 691 tests, three
plan documents. Suggested split: security fixes + tests; attack simulation
harness; benign/FPR tooling; plan docs.

**4.2 Decide on the retrain.** Two independent reasons to do one, and they
share the same cost:

- `data/clean/taglish_heldout.csv` is pooled into training (141 of 300 rows in
  train, 68 in validation). No Taglish claim is reportable until it is sealed.
- BANKING77 ships 10,003 unused training rows that would supply the missing
  benign customer-support examples.

Train on **`banking77_train`** (10,003 rows), keep `banking77_test` sealed,
leave **CLINC150 entirely untouched** as the out-of-distribution test.

BANKING77 is the right training source and not because the scenario is a bank.
It is real support traffic, so it is dense in exactly the vocabulary that
fails - `wrong` 3.12%, `cancel` 2.47%, `verify` 1.87%, `not work` 1.20% of
rows. CLINC150 is assistant queries ("set a timer", "what's the weather") and
barely contains that language at all (`wrong` 0.11%, `urgent` 0.00%), so
training on it would teach the model little. The distinction that matters is
**support traffic versus assistant queries**, not banking versus general.

Then: if both FPRs drop, the fix generalises. If only BANKING77 drops, it is
domain-bound - still a publishable finding, and a direct answer to "will it
fail in other fields?"

Caveat: both datasets are short utterances. Neither adds long benign documents,
so neither addresses false positives in long retrieved content - which nothing
currently measures.

Cost: ~30 min training, ~2.5 h re-running everything MOCHI-dependent.
**Undefended arms do not need re-running** - they measure the bare model and
MOCHI is not involved. That is half the runs and the slower half.

### Not yet measured

**4.3 Trust-based thresholds - REJECTED, do not revisit.** The idea was to
be strict on retrieved documents and lenient on the user turn, on the grounds
that every false positive is in `user_input`. It does not work, because every
*attack* is there too: Gandalf is **100% direct** (all 250 dangerous attacks in
the user turn) and the generated corpus is 68% direct. Raising the user-turn
bar to 0.90 would cut FPR 2.98% -> 1.03% but drop detection 85.3% -> 73.5%,
weakening the defence against the attacks being measured. Origin cannot
separate false positives from attacks here because it is not what distinguishes
them.

The idea is sound for a different system - a RAG or agent pipeline where the
dominant threat is indirect and the user is a victim rather than a suspect.
This is a chat assistant where the user is the attack surface.

**4.4 End-to-end on the sealed test split.** `eval/run_detection.py --config
stage12` has never been run; no `stage12` report exists. Converts "the
classifier scores F1 0.9856" into "the gateway stops X%". ~1.5 h.

**4.5 Stage II thresholds.** Still the pre-study 0.45 / 0.55. Note the
out-of-distribution curve is **not** flat the way the in-distribution one was:
0.55 -> 0.85 trades 43 false positives for 116 missed attacks, roughly 1:2.7.
`eval/threshold_sweep.py` exists; its run was interrupted. ~30 min.

**4.6 Utility damage from SANITIZE.** Does redaction break legitimate answers?
Nothing tests it. ~30 min: run benign queries through both arms and compare
replies.

### Gaps to name in Limitations

**4.7 Session risk accumulation** is barely exercised - 7 priming chains.
The accumulator's premise (three turns at ~0.35 each) is in tension with a
bimodal scorer that emits near-0 or near-1.

**4.8 Outbound interception** tested by 6 exfiltration attacks only.

**4.9 Direct output-control attacks are a weak threat.** 19 of 39
`instruction_override` attacks are in the user turn, where the principal
already controls their own session. The 20 placed in documents are the serious
case. Report them separately rather than pooled.

### Housekeeping

**4.10 Telemetry `source_origin` is misleading.** It records the
highest-scoring segment regardless of which drove the verdict. Cost me three
wrong diagnoses this session. Should record the deciding segment.

**4.11 Workbook needs rebuilding** with the current numbers - Gandalf detection
and mitigation, generated corpus, corrected FPR, all under one scenario.
`eval/export_simulation.py` has the sheets; it needs the Gandalf results folded
in.

**4.12 Regenerate stale reports.** Twelve of fourteen predate the 14 September
corpus change. Order matters: token association -> feature table -> Track A ->
comparison. Conclusions do not change, only reproducibility.

**4.13 Docker packaging** (Phase 14). The deliverable behind the adviser's
plug-and-play question. See `NEXT_STEPS.md` Part A.

---

## 5. Artifacts on disk

| File | Contents |
|---|---|
| `reports/gandalf_runs.jsonl` | 1,800 runs, Gandalf 300 |
| `reports/gandalf_detection.jsonl` | 1,000 detection verdicts |
| `reports/simulation_runs.jsonl` | 600 runs, generated corpus |
| `reports/benign_runs.jsonl` | 8,580 benign queries |
| `reports/*_PRE_FIX.*`, `*_OLDSCENARIO.*`, `*_MISMATCHED.*` | superseded runs, kept for before/after |
| `data/attacks/corpus.jsonl` | the 100 generated attacks |
| `data/attacks/gandalf.jsonl`, `gandalf_300.jsonl` | Gandalf, full and sampled |
| `data/external/` | BANKING77, CLINC150, Gandalf + intent name maps |

Two processes may still be running detached: the gateway on port 8000, and
nothing else. Stop with
`Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*uvicorn*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`.

---

## 6. Corrections to earlier claims in this session

Recorded because they were stated confidently and were wrong.

- **"The FPR was inflated by the system prompt."** It was not. The same 257
  queries were flagged before and after; the system prompt only distorted the
  recorded `semantic_score` field, not the verdicts.
- **"The scoring change explains the higher ASR."** It did not - re-scoring the
  archived run moved Mistral by zero. The template rewording did.
- **"MOCHI performed worse in the new run."** It did not; the code was
  byte-identical and the headline moved 70.4% to 67.4%, within noise.

All three had the same root cause: reading a score field without checking which
segment drove the decision. See 4.10.
