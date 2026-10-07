# MOCHI — Next Steps

Written 7 October 2026, after the first attack simulation and the
out-of-distribution evaluation. Plain list of what is left, in the order it
should be done, plus the two decisions that are waiting on someone.

---

## Part A — How MOCHI gets used

The adviser asked whether MOCHI can be plug-and-play for non-technical people
who own a website or chatbot, and said developer-facing is acceptable if not.

**Short answer: integration is already plug-and-play. Hosting is not, and
cannot be within this thesis.** Those are two separate things and it is worth
keeping them apart when answering.

### What is already true

MOCHI speaks the same API as OpenAI. A system already talking to OpenAI points
at MOCHI instead by changing one setting:

```
OPENAI_BASE_URL = https://api.openai.com/v1      ->      http://localhost:8000/v1
```

No code changes. No SDK changes. No retraining. Nothing in the application has
to know MOCHI exists.

This was demonstrated on 29 September, not just claimed: the whole attack
simulation ran against three local Ollama models through MOCHI, and **no
adapter was written** to make that work. Changing that one URL was the entire
integration. That is the strongest available evidence for the model-independence
argument, and it should be reported as a result rather than an assumption.

### The three levels of "plug and play"

| Level | What the user does | Who can do it | Status |
|---|---|---|---|
| 1. Integration | change one URL | anyone who can edit a config file | **done** |
| 2. Deployment | run MOCHI somewhere | anyone who can run Docker | Phase 14, not started |
| 3. Zero setup | sign up, paste a key | anyone | **out of scope** |

Level 3 means somebody else hosts MOCHI — a paid service with uptime,
monitoring and a GPU bill. That is a business, not a thesis. It should not be
claimed.

Level 2 is achievable and is the honest target:

```
docker compose up          # one command
# then change one URL in the app
```

That reaches "someone who owns a website and can follow a README." It does not
reach "someone with no technical knowledge," and no self-hosted security proxy
does.

### One real limitation to state plainly

If the chatbot was built on a no-code platform — Chatbase, Voiceflow, Intercom,
Botpress — the owner often **cannot change the API base URL at all**. The
platform hides it. For those users MOCHI is not installable at any level, and
no amount of packaging fixes it. It would need the platform vendor to
integrate.

So the accurate claim is:

> MOCHI requires no application code changes and integrates by changing a
> single configuration value. It is deployable by anyone able to run a Docker
> container. It cannot be installed into closed no-code chatbot platforms that
> do not expose their model endpoint.

### Recommended answer to the adviser

Target **Level 2**, state Level 3 as out of scope, and present Level 1 as an
already-demonstrated result. If a stronger demonstration is wanted, the cheapest
one is a short screencast: an existing chatbot, one URL changed, an attack
blocked — before and after, no code touched.

---

## Part B — Datasets for the three-LLM evaluation

Three models, every test run twice: once straight to the model, once through
MOCHI. The difference is the result.

**Targets** (unchanged): `mochi-mistral`, `mochi-qwen`, `mochi-llama` — the
stock 7-8B instruct models with the context window pinned to 8192. See
`eval/modelfiles/`.

### Attack sets — measure how many attacks get through

| Dataset | Rows | Source | Overlap with training | Needs the LLM? |
|---|---|---|---|---|
| **Gandalf** | 1,000 | real human attacks, Lakera, MIT | **0** | yes |
| **Generated corpus** | 100 | our 4-tier adaptive set | 0 | yes |
| PromptShield self-scoring | 517 | published benchmark, in sealed test split | 0 (held out) | yes |

Gandalf is the headline attack set: human-written, completely independent of
anything the model trained on, and every attack is trying to extract a secret -
which matches the canary already planted in the Acme Bank system prompt, so the
attacks run unmodified.

The 517 PromptShield rows are optional. They are *in-distribution* - the model
trained on 70% of PromptShield - so a high score there means "handles attacks
like the ones it learned from", which is a weaker claim than Gandalf supports.

### Benign sets — measure how many ordinary users get blocked

| Dataset | Rows | Domain | Overlap | Needs the LLM? |
|---|---|---|---|---|
| **BANKING77** | 3,080 | banking only | **0** | no |
| **CLINC150** | 5,500 | 10 domains, 150 intents | 8 (0.15%) | no |

These answer the question the attack sets cannot: does MOCHI break normal use?
Neither needs the LLM at all - to know whether a legitimate request was refused,
you only need MOCHI's verdict. That makes the whole benign side cheap.

**This is the half that is currently missing from every number in the thesis.**

### Measured so far (Stage II alone, not the full pipeline)

| | FPR |
|---|---|
| Sealed test split, in-distribution | 0.48% |
| BANKING77, unseen | 2.66% |
| CLINC150, unseen | 3.18% |

The failures cluster in queries where a user tells the assistant what to do -
`reset_settings` fails 90% of the time, `cancel` 57%. Those sentences are
genuinely close to injections; the difference is who is authorised to say them,
which the text does not carry.

---

## Part C — The steps, in order

### 1. Re-run the attack simulation  *(blocking; ~1 hour)*

The enforcement fix on 7 October changed what the system does, so the 600-run
workbook describes a system that no longer exists. In that run, six detections
leaked and one attack survived - all through the path that is now fixed.

- Restart the gateway
- `python -m eval.attack_simulation --fresh`
- `python -m eval.export_simulation`
- The workbook's "KNOWN DEFECT" note needs removing once it is no longer true

### 2. Measure the false-positive rate properly  *(~30 min)*

Everything measured so far is Stage II scored in isolation. The real number has
to come from the full pipeline - Stage I, trust rules, enforcement.

- Run BANKING77 (3,080) and CLINC150 (5,500) through the gateway, detection only
- Report FPR per dataset and per intent
- Needs a small runner; the simulation harness already does nearly all of it

### 3. Run Gandalf against the three models  *(~2.5 hours)*

- Sample 300 of 1,000 for a result the same day, or run all 1,000 overnight
- Same two arms, same canary scoring
- Produces the headline out-of-distribution ASR and mitigation rate

### 4. Decide what to do about the false positives

Three options, in increasing cost:

- **Raise the threshold.** Free, but a bad trade: 0.55 -> 0.85 cuts FPR from
  2.66% to 1.27% and costs 12 points of recall.
- **Different thresholds per trust level.** Strict on retrieved documents,
  lenient on what the user types. Uses machinery that already exists. Targets
  the failures directly, because every false positive is in `user_input`. Costs
  some direct-injection recall, since Gandalf attacks are all direct - needs
  measuring, not assuming.
- **Retrain with benign data.** Cannot fix the `reset_settings` class, because
  those sentences are genuinely ambiguous. May help the softer cases.

Recommended: measure option 2 after step 2, then decide.

### 5. Re-seal the Taglish set  *(~10 min, plus a retrain if done)*

`data/clean/taglish_heldout.csv` is pooled with everything else, so 141 of its
300 rows are in the training split and 68 more in validation. It is not a
held-out set in its current location.

- Move it to `data/sealed/` where the glob cannot reach it
- Add a test that fails if any of its rows appear in the pooled corpus
- Note: moving it changes the split, which implies a retrain for the numbers to
  be consistent. Headline metrics will not move - 141 rows out of 41,879 - but
  no Taglish claim is reportable until this is done.

### 6. Fix the two failing tests  *(~15 min)*

`test_unbuilt_stages_report_not_run` and
`test_mochi_only_fields_are_stripped_before_upstream` fail whenever `.env` has
Stage II enabled, because the test suite reads the developer's real `.env`.

- Add a `tests/conftest.py` that pins the settings the tests assume
- The first test is also simply out of date: it asserts Stage II reports
  `not_run`, which was true before Stage II existed

### 7. Site the Stage II thresholds  *(~30 min)*

`BENIGN_THRESHOLD` (0.45) and `MALICIOUS_THRESHOLD` (0.55) are still the values
chosen before any measurement existed. `eval/threshold_sweep.py` was written but
its run was interrupted before writing results.

- Re-run with `--device cpu`
- Note that the out-of-distribution curve is **not** flat the way the
  in-distribution one was: 0.55 -> 0.85 trades 43 false positives for 116 missed
  attacks, roughly 1:2.7. The earlier "1:1 exchange rate" was an artifact of the
  test set.

### 8. Package it  *(Phase 14; ~half a day)*

This is what turns the adviser's question into a deliverable.

- `Dockerfile` and `docker-compose.yml`
- README quick-start: `docker compose up`, then change one URL
- Confirm `.env.example` covers every variable actually required

### 9. Regenerate the stale reports  *(~1-2 hours)*

Twelve of fourteen JSON reports predate the 14 September corpus change. Order
matters because one of them is a fitted artifact, not just a record:

```
token association -> feature table -> Track A -> comparison
```

Lower priority than the above: the conclusions do not change, only
reproducibility.

---

## Decisions waiting on someone

1. **Deployment target.** Level 2 (Docker, developer-installable) as the
   claim - confirm with the adviser.
2. **Gandalf sample size.** 300 for a same-day result, or 1,000 overnight.
3. **Taglish set.** Re-seal and retrain, or drop the Taglish claim from this
   version of the thesis.
