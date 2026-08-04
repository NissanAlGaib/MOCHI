# MOCHI Telemetry

Reference for the structured log MOCHI emits. Implements the thesis "Attack
Logging and Telemetry" section (Figure 9) with the extensions agreed in
`ARCHITECTURE.md`.

- **Sample file:** [`samples/telemetry-sample.jsonl`](samples/telemetry-sample.jsonl) — 7 annotated records covering the full decision space
- **Regenerate:** `python docs/samples/generate_sample_log.py`
- **Live output:** `logs/mochi.jsonl` (configurable via `MOCHI_LOG_PATH`)

## Who is this log for?

**It is a developer-facing product feature, not a test artifact.** The thesis
states that structured logging grants the LLM application *"granular visibility
into the types of attacks that are targeted"*, and FR5 requires logging "for
analysis" as a functional requirement — not a debugging convenience.

It serves three audiences at once:

| Audience | Use |
|---|---|
| **Developer / operator** running MOCHI in front of their app | See what is being attacked, tune thresholds, investigate a blocked request a user complained about, feed a SIEM |
| **You, the researcher** | Every Chapter IV number is computed from this file — ASR, Mitigation Rate, FPR, latency, escalation rate |
| **Test suite** | `tests/test_telemetry.py` asserts against it as a contract |

That the same artifact serves all three is deliberate: it means the evaluation
measures the real production code path, not a separate instrumented build. If
research metrics came from a special "eval mode," you would be measuring
something your users never run.

### Current access model

Today the log is a **file on disk** the operator reads directly. That is
sufficient for a single-node deployment and for your evaluation runs, but it
means an application cannot query its own security events programmatically.

Not yet built (candidates for Phase 12/14):

- `GET /v1/telemetry` — read-only, auth-gated recent-events endpoint so a host
  application can surface "3 injection attempts blocked today" in its own admin UI
- Log rotation — the file grows unbounded; a 97k-sample evaluation run produces
  a ~40 MB file
- SIEM/webhook forwarding for real deployments

None of these block the thesis. Worth stating in Recommendations (Chapter V) as
deployment-hardening work rather than leaving the gap unexplained.

### Privacy posture

`MOCHI_LOG_PAYLOADS` defaults to **false**. Routine operation records a
SHA-256 hash and length, never the prompt text — this is the operational form
of the Ethical Considerations commitment on anonymization and de-identification.
Identical payloads still hash identically, so correlation across runs works
without retaining content.

Set it to `true` only for controlled evaluation over **public benchmark
datasets**, where the "personal data" concern does not apply and you need the
text to inspect false positives. Never enable it against real user traffic.

## Field reference

### Top level

| Field | Type | Set by | Meaning |
|---|---|---|---|
| `timestamp` | ISO-8601 UTC | Phase 2 | When the request was received |
| `request_id` | string | Phase 2 | Unique per request; use to correlate with app-side logs |
| `session_id` | string \| null | Phase 2 | Client-supplied conversation ID. Required for multi-turn detection — without it, Phase 7 cannot link turns |
| `source_origin` | string \| null | Phase 4 | Which tagged segment the detection came from: `user_input`, `retrieved_document`, `web_content`, `api_response`, `system_prompt`. **This is what distinguishes a direct from an indirect injection in your results** |
| `attack_type` | string \| null | Phase 6+ | `direct_injection`, `indirect_injection`, `jailbreak`, `data_exfiltration`, `role_manipulation`, `adversarial_prompt`, `url_exfiltration`. `null` on benign traffic |
| `severity_level` | `low`\|`medium`\|`high` \| null | Phase 6+ | Set only when an attack is identified |
| `normalization_flags` | string[] | Phase 3 | What the preprocessor had to undo, e.g. `base64_decoded`, `zero_width_chars_detected`, `hidden_css_detected`, `unicode_nfkc_applied`. A non-empty list on otherwise-benign traffic is itself suspicious |
| `mitigation_action_applied` | `ALLOW`\|`BLOCK`\|`SANITIZE`\|`N/A` | Phase 10 | Final enforcement decision. `N/A` means no decision was reached (transport error, malformed request) |
| `response_status` | int \| null | Phase 2 | HTTP status returned to the client |
| `target_provider` / `target_model` | string \| null | Phase 2 | Which backend served it. Drives the per-model breakdown in thesis Table 19 |

### `payload_characteristics`

| Field | Meaning |
|---|---|
| `char_length` | Exact character count of inspected text |
| `token_length` | Approximate (`len // 4`) until Phase 8 swaps in the real tokenizer. Feeds the 4,096-token budget check |
| `language` | Populated in Phase 3. Relevant because the Stage II model is multilingual E5 — a non-English attack rate is worth reporting |
| `content_sha256` | Always present. Correlates identical payloads without storing them |
| `content` | Raw text. `null` unless `MOCHI_LOG_PAYLOADS=true` |

### `detection_results`

| Field | Meaning |
|---|---|
| `stage_1_syntactic` | `not_run`, `pass`, or `block_<detector_name>` — naming the detector is what lets you report per-detector precision |
| `stage_2_semantic` | `not_run`, `pass_score_<p>`, `block_score_<p>`, `escalate_score_<p>` |
| `stage_3_arbitration` | `N/A`, `safe`, or `unsafe`. `N/A` is the common case — Stage III only fires in the uncertain band |
| `session_risk_contribution` | This turn's risk score |
| `session_cumulative_risk` | Rolling session risk after this turn. When this crosses threshold while the turn's own score is low, you are seeing a multi-turn chain |

### `latency` (milliseconds)

| Field | Meaning |
|---|---|
| `stage_1_ms` | Syntactic filtering. **NFR1 target: < 2 ms** |
| `stage_2_ms` | Semantic detection. **NFR1 target: < 55 ms** |
| `stage_3_ms` | Arbitration. Makes a network call — hundreds of ms, and **cannot** meet the NFR1 targets. Reported separately on purpose: the defensible claim is that it fires on only a small fraction of traffic, so report escalation rate alongside it |
| `inspection_ms` | Total time inside MOCHI's pipeline. **This is the overhead a developer actually pays** |
| `upstream_ms` | Time waiting on the target LLM. Not MOCHI's cost — separating it prevents overstating overhead |
| `total_ms` | End to end |

> The honest overhead figure for Table 18 is `inspection_ms`, not
> `total_ms`. Reporting `total_ms` would fold the target LLM's own latency into
> MOCHI's measured cost and inflate it several-fold.

## Reading the log

```python
import pandas as pd

df = pd.read_json("logs/mochi.jsonl", lines=True)

# Confusion-matrix inputs (requires ground-truth labels joined by request_id)
blocked = df["mitigation_action_applied"].isin(["BLOCK", "SANITIZE"])

# Stage III escalation rate - the number that defends the latency budget
escalated = (df["detection_results"].apply(lambda d: d["stage_3_arbitration"]) != "N/A")
print(f"Stage III escalation rate: {escalated.mean():.2%}")

# Inspection overhead distribution (NFR1 evidence)
overhead = df["latency"].apply(lambda x: x["inspection_ms"])
print(overhead.describe(percentiles=[0.5, 0.95, 0.99]))

# Attack breakdown by origin - direct vs indirect
print(df[df["attack_type"].notna()].groupby(["source_origin", "attack_type"]).size())
```

## Worked examples

The sample file walks the full decision space. Each is worth understanding
because each corresponds to a different row in your results tables:

| # | Scenario | Illustrates |
|---|---|---|
| 1 | Benign question, allowed | Benign traffic **is** logged — this is what makes FPR computable |
| 2 | `Ignore previous instructions...` | Stage I fail-fast: blocked in 1.12 ms total, Stages II/III never ran |
| 3 | Injection inside a retrieved document | Stage I passes, Stage II catches semantically; `SANITIZE` preserves the legitimate request instead of blocking it outright |
| 4 | Ambiguous security-training question | Uncertain band → Stage III → cleared. Note `stage_3_ms` of 486 ms vs `stage_2_ms` of 35 ms |
| 5 | Base64-wrapped jailbreak | `normalization_flags: ["base64_decoded"]` — without Phase 3 the blob is opaque to every detector |
| 6 | Multi-turn chain | Turn scores 0.31 (below threshold) but `session_cumulative_risk` hits 0.78 → escalation. **The case a stateless design structurally cannot catch** |
| 7 | Markdown-image exfiltration in the response | Request was clean; caught outbound. Also shows a non-OpenAI provider |
