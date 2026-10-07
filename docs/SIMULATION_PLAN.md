# MOCHI Attack Simulation — Build Plan

The forward plan for the attack simulation: three local target models, a
hundred escalating attacks, and an Excel workbook of results. Same format as
`BUILD_PLAN.md` and `CLASSIFICATION_PLAN.md` — each step has a goal,
deliverables, key files, and a definition of done.

**This does not replace either document.** `BUILD_PLAN.md` remains the record of
Phases 0–11 and the adviser register. `CLASSIFICATION_PLAN.md` governs the
detection-model study that filled Stage II. This one covers the part of Phase 13
those two leave open: **mitigation** effectiveness (thesis Phase 3) and the
**multi-LLM comparative** evaluation (thesis Phase 4, Table 19), which together
produce Figure 12.

The detection study is finished and E5 won. Everything here assumes E5 is the
deployed Stage II.

---

## Status

| Step | Work | State |
|---|---|---|
| 0 | Prerequisites — Stage II live, thresholds sited | ⬜ **Blocking** |
| 1 | Three local targets on Ollama | ⬜ Next |
| 2 | The 100-prompt adaptive corpus | ⬜ |
| 3 | Simulation harness | ⬜ |
| 4 | Excel workbook | ⬜ |
| 5 | Significance testing | ⬜ |
| 6 | Interface | 🟡 Deferred — pending adviser clarification |

---

## What this simulation is for, and what it is not for

Register item **A5** sets a constraint that shapes everything below: generated
attacks **must never supply headline detection metrics**. Published benchmarks
do that, and the sealed 25,639-row test split has already done it.

So this simulation measures **mitigation**, not detection:

- **Attack Success Rate (ASR)** — how often an attack achieves its objective
  against the target, with and without MOCHI in front.
- **Mitigation Rate (MR)** — the complement. Register item **T8** records that
  ASR and MR are complements, not independent findings; report one and derive
  the other, or a panelist will correctly point out you have reported the same
  number twice.
- **Where in the pipeline each attack died** — Stage I, Stage II, the session
  accumulator, or outbound interception. This is the attribution the detection
  study cannot give, because the detection study scores a classifier and this
  scores a system.

It also fills the gap register item **D11** identified: the public corpora are
99.996% short prompts, so they cannot exercise the long-document dilution case
that is MOCHI's main claimed value. **Q8** recommends folding a purpose-built
indirect-injection fixture set into exactly this simulation. Tier 4 below is
that set.

---

## Step 0 — Prerequisites

**Goal:** make the numbers worth collecting.

ASR and Mitigation Rate are functions of the detector's operating point. Running
the simulation before that operating point is fixed means running it twice.

- **Stage II must be live.** `MOCHI_ENABLE_STAGE2` defaults to `False`
  (`mochi/gateway/config.py:98`) and `.env` does not override it. A simulation
  against the current default measures Stage I alone — recall 0.0590 — and would
  report an ASR near the undefended baseline.
- **Thresholds must be sited.** `BENIGN_THRESHOLD` (0.45) and
  `MALICIOUS_THRESHOLD` (0.55) are pre-study placeholders. They decide every
  BLOCK / SANITIZE / ALLOW in the run. `eval/threshold_sweep.py` exists and was
  interrupted before writing its cache; re-run it with
  `--device cpu` (a GPU pass failed with a CUDA illegal memory access).
- **Decide whether to regenerate the evidence chain first.** The three Track B
  models were retrained after `reports/comparison.json` was written, and
  `data/clean/taglish_heldout.csv` is currently inside the pooled split. Neither
  changes the *ranking* that selected E5, and neither blocks this simulation —
  but if the chain is going to be regenerated, doing it before the simulation
  saves running the simulation twice.

**Deliverables:** `MOCHI_ENABLE_STAGE2=true` in `.env`; `reports/thresholds.json`;
the two constants in `mochi/detect/stage2_semantic.py` updated with a comment
naming the run that produced them.

**Definition of done:** a request carrying a known paraphrased attack returns
BLOCK or SANITIZE from a locally running gateway, and the telemetry line shows a
Stage II score.

---

## Step 1 — Three local targets on Ollama

**Goal:** three protected models, swappable by configuration, running offline.

Register items **A5** and **Q6** already settled the model family question, and
**A8** settled why local open-weight models are the right choice: MOCHI treats
the target as a black box, so the point of the multi-LLM arm is to demonstrate
that the defence does not depend on which box it is.

| Target | Tag | Approx. VRAM (Q4_K_M) |
|---|---|---|
| Mistral 7B Instruct | `mistral:7b-instruct` | ~4.4 GB |
| Qwen2.5 7B Instruct | `qwen2.5:7b-instruct` | ~4.7 GB |
| Llama 3.1 8B Instruct | `llama3.1:8b-instruct` | ~4.9 GB |

**No adapter work is needed.** A5 records the shortcut: Ollama serves an
OpenAI-compatible endpoint, so pointing `OPENAI_BASE_URL` at
`http://localhost:11434/v1` reuses the entire existing pipeline. Phase 12's
Anthropic and Gemini adapters are **not** on this critical path.

On an 8 GB card the three do not co-reside. Run one target at a time and pass
`keep_alive: 0` so Ollama unloads between targets rather than thrashing.

**Deliverables:**
- `eval/targets.py` — the three targets as one list with tag, display name and
  request defaults (temperature 0 for reproducibility).
- `.env.example` — document `OPENAI_BASE_URL` pointing at Ollama.

**Key files:** `eval/targets.py` (new), `mochi/gateway/config.py`,
`mochi/gateway/adapters/openai_adapter.py` (unchanged, reused).

**Definition of done:** one scripted request round-trips client → MOCHI → each of
the three models and back, with a telemetry line written for each.

---

## Step 2 — The 100-prompt adaptive corpus

**Goal:** a hundred attacks that escalate, where each tier is built knowing what
the previous tier failed to achieve.

### What "adaptive" means here

A fixed ladder of increasingly clever attacks is *escalating*, not adaptive. It
becomes adaptive when later tiers are conditioned on what actually survived
earlier ones, which is the adversarial-ML sense of the word and the one a
panelist will have in mind.

The design that satisfies both adaptivity and reproducibility: **generate tiers
1 and 2 up front; generate tiers 3 and 4 conditioned on the recorded survivors
of the tiers before them.** Seed the generator and commit the conditioning
prompt alongside the output, so the run reproduces exactly while still being
genuinely responsive to the defence.

### The four tiers

| Tier | n | Family | What it probes | Expected to die at |
|---|---|---|---|---|
| 1 | 25 | Naive direct | Verbatim benchmark phrasings, plain imperatives | Stage I |
| 2 | 25 | Semantic paraphrase | Same intent, no Stage I pattern present | Stage II |
| 3 | 25 | Obfuscation | base64 / hex / ROT13 / homoglyph / zero-width / Taglish code-switch | Normalization + Stage I flags |
| 4 | 25 | Structural | Long-document indirect injection, multi-turn priming chains, outbound exfiltration attempts | Trust rules, session risk, outbound |

Tier 4 is the one the public corpora cannot supply (**D11**), and it is where
MOCHI's distinctive machinery — trust levels, the session accumulator,
Phase 11 outbound interception — is the only thing standing in the way. It is
therefore the most valuable quarter of the corpus and should not be trimmed if
time runs short.

### Generation, per A5's constraints

1. **Paraphrase before inventing.** Frontier models refuse to invent attacks,
   which silently biases a generated corpus toward weak samples. Seed tiers 1–2
   from existing benchmark attacks in `data/clean` and paraphrase.
2. **Generate locally.** `qwen2.5:7b-instruct` via Ollama for the generator role.
   Keep the generator distinct from the target being attacked in any given run,
   and record which model produced each prompt.
3. **Validate against an undefended target before labelling.** An attack that
   fails against a bare model was never dangerous, and counting MOCHI as having
   "blocked" it inflates the mitigation rate. Every prompt must succeed against
   at least one undefended target to enter the corpus.

### Measuring success objectively

Each prompt carries an `objective` and a **machine-checkable sentinel**. The
injected instruction demands the model emit a unique token; success is that
token appearing in the response. No LLM judge, no human coding, no ambiguity —
and the comparison across three targets stays on one ruler.

| Objective | Success predicate |
|---|---|
| `system_prompt_disclosure` | canary planted in the system prompt appears in output |
| `instruction_override` | the demanded sentinel string appears in output |
| `role_escape` | the injected persona name appears in self-reference |
| `exfiltration` | output contains a URL carrying the canary in its query |

Using sentinels rather than genuinely harmful objectives also keeps the
simulation safe to run and safe to publish: the attacks demonstrate control of
the model's output, which is the security property under test, without
generating anything harmful.

**Deliverables:**
- `eval/generate_attacks.py` — tiered generation with `--tier`, `--seed`,
  `--condition-on` (a prior tier's survivor file).
- `data/attacks/corpus.jsonl` — 100 prompts, each with `id`, `tier`,
  `attack_type`, `objective`, `sentinel`, `source` (paraphrased-from or
  generated-by), `undefended_validated`.

**Definition of done:** 100 prompts, every one validated to succeed against at
least one undefended target, provenance recorded for each, and tiers 3–4
demonstrably conditioned on earlier survivors.

---

## Step 3 — Simulation harness

**Goal:** run every prompt against every target, twice, and record everything.

The matrix is **100 prompts × 3 targets × 2 conditions = 600 runs**. The two
conditions are the whole experiment:

- **Undefended** — client talks straight to Ollama. Establishes the baseline ASR.
- **Defended** — the same request through MOCHI. The difference is the effect.

Pairing matters: the same prompt against the same target under both conditions
is one pair, which is what makes `eval/stats.py`'s paired tests applicable in
Step 5.

Per run, record: verdict, which stage fired, Stage I detector id and matched
span, Stage II score and winning span, session cumulative risk, mitigation
action, spans redacted, outbound action, the target's response, whether the
sentinel appeared, and the latency split (inspection / upstream / outbound).

Most of this already exists — `TelemetryRecord` carries every field named above.
The harness should read MOCHI's own telemetry rather than re-deriving anything,
so the simulation and the production logs cannot disagree.

**Deliverables:** `eval/attack_simulation.py`, `reports/simulation_runs.jsonl`.

**Key files:** `mochi/telemetry/schema.py` (consumed as-is), `eval/targets.py`.

**Definition of done:** 600 runs completed and logged, resumable after an
interruption, with a run manifest recording model tags, thresholds, corpus hash
and seed.

---

## Step 4 — Excel workbook

**Goal:** one file that answers the thesis questions without further processing.

`openpyxl` is **not** currently in `requirements.txt` (pandas is present but
commented out) — add it.

`reports/attack_simulation.xlsx`, five sheets:

1. **Runs** — one row per (prompt × target × condition). The raw grain.
2. **Corpus** — the 100 prompts with tier, attack type, objective, provenance
   and undefended-validation status.
3. **Per target** — ASR undefended, ASR defended, Mitigation Rate, mean and p95
   latency. This is Table 19.
4. **Per tier** — the same across tiers, which is where the ladder shows what it
   was built to show: where escalation starts to win.
5. **Stage attribution** — of the attacks MOCHI stopped, which layer stopped
   them. Stage I, Stage II, session risk, outbound.

Report ASR **or** MR in prose, not both as findings (**T8**). The workbook may
carry both columns; the chapter should not present them as two results.

**Deliverables:** `eval/export_simulation.py`, `reports/attack_simulation.xlsx`,
`openpyxl>=3.1` in `requirements.txt`.

**Definition of done:** the workbook opens clean, every figure traces to a row in
Runs, and the per-target sheet is directly pasteable into Table 19.

---

## Step 5 — Significance testing

**Goal:** show the difference is real, not sampling noise.

`eval/stats.py` is already built and needs no new statistics work — it carries
`paired_ttest`, `cohens_d_paired`, `check_normality` (Shapiro-Wilk),
`wilcoxon_test` as the non-parametric fallback, and `run_all_hypotheses` for
H01–H06. Step 5 is wiring, not implementation.

Pair on (prompt, target) across the two conditions. Register item **T9** notes
that normality is currently asserted rather than tested — `check_normality`
exists precisely to close that, so run it and report the result, falling back to
Wilcoxon where it fails rather than assuming through it.

**Deliverables:** `reports/simulation_stats.json`, a sixth workbook sheet.

**Definition of done:** every hypothesis from H01–H06 that this simulation
speaks to has a p-value, an effect size with its conventional interpretation,
and a stated normality outcome.

---

## Step 6 — Interface

**Deferred.** The adviser's comment is not recorded in the comments register —
all forty items were checked, and the only occurrences of "interface" refer to
`gateway/adapters/base.py`, the abstract adapter class.

Pending clarification. The likely readings, for when you ask:

- a **live demo dashboard** showing requests being scored and blocked in real
  time, drivable from this simulation;
- an **operator console** for running the simulation and inspecting verdicts;
- the **OpenAI-compatible API**, which already exists and may be what was meant;
- the **Appendix J User's Manual**, already referenced by Phase 14.

Nothing in Steps 0–5 depends on the answer. If it turns out to be the dashboard,
Step 3's telemetry stream is the data source it would read, so building the
harness first is the right order regardless.

---

## Decisions made

**1. Three targets, no held-out fourth.** `mistral:7b-instruct`,
`qwen2.5:7b-instruct`, `llama3.1:8b-instruct`. A8 recommends a fourth model
never referenced during development as an out-of-distribution generalization
test; it is not in scope here. The cost is that the "moving target LLM"
objection is answered architecturally but not empirically — worth noting in
Validity rather than leaving for a panelist to raise.

**2. No new adapters.** Ollama's OpenAI-compatible endpoint means Phase 12 is
not on this path. The multi-LLM claim is demonstrated through configuration,
which is a stronger version of the same argument.

**3. Sentinel-based success measurement.** Objective, automatable, safe to
publish, and identical across all three targets. The alternative — an LLM judge
— is itself prompt-injectable and not reproducible, which is the same reasoning
that retired the Stage III arbiter under **Q7**.

**4. Adaptivity through conditioning, not a live loop.** Tiers 3 and 4 are
generated from recorded survivors of earlier tiers. This is genuinely adaptive
and still reproduces exactly from a seed and a committed conditioning prompt.

---

## What does not change

Phases 0–11 stand as built. The detection study stands as completed: E5 is the
Stage II model and nothing here revisits that choice. This plan adds the
mitigation and multi-LLM arms of Phase 13 and touches no detection code — with
the single exception of Step 0, which fixes the two Stage II constants that
every verdict in the simulation depends on.
