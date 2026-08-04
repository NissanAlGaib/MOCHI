# Testing & Evidence

How to test MOCHI, and what evidence exists to present.

## Two kinds of proof — do not conflate them

Your panel may ask for either. They answer different questions and are
produced by different machinery.

| | **Software testing** | **Empirical evaluation** |
|---|---|---|
| Question | Does the code do what the specification says? | Does MOCHI detect attacks at what accuracy? |
| Method | `pytest` — 59 automated tests | Benchmark run over a labeled dataset |
| Evidence | Pass/fail report, coverage report | Accuracy, Precision, Recall, F1, ASR, Mitigation Rate |
| Belongs in | Chapter III (methodology), Appendix E | **Chapter IV (Results)** |
| Status | ✅ Available now | ⬜ Requires Phases 5–6 (dataset harness + Stage I) |

**The distinction matters.** A passing test suite proves the normalization
layer decodes base64 correctly. It says nothing about whether MOCHI catches
95% of prompt injections — that claim requires running the labeled benchmark,
which needs a detector to exist. Presenting test results as if they were
detection accuracy would be a serious misstatement.

What you *can* say today: "the system is implemented and verified to
specification through 59 automated tests at 87% coverage; detection
effectiveness evaluation follows in Phase 13."

---

## Evidence available now

### 1. Automated test suite

```bash
python -m pytest -v
```

59 tests across three modules. Test names state what is being proven, so the
verbose output reads as a specification:

```
test_zero_width_chars_removed PASSED
test_cyrillic_homoglyphs_folded PASSED
test_base64_payload_decoded PASSED
test_nested_encoding_unwrapped PASSED
test_decode_depth_is_capped PASSED
test_display_none_content_extracted PASSED
test_raw_prompt_absent_from_log_by_default PASSED
...
```

| Module | Tests | Proves |
|---|---|---|
| `test_gateway.py` | 9 | Transport, payload parsing, MOCHI-field stripping, error passthrough |
| `test_telemetry.py` | 15 | Logging contract, privacy posture, latency capture |
| `test_preprocess.py` | 35 | Every obfuscation technique is defeated; benign input untouched |

### 2. Obfuscation resistance demonstration

The most presentable artifact for a security thesis — it shows the system
defeating attacks rather than just passing assertions.

```bash
python demo/evasion_demo.py --csv
```

Current result:

```
Evasion technique        Naive   MOCHI   Result  Flags raised
Plaintext (control)      SEES    SEES    PASS    -
Zero-width split         BLIND   SEES    PASS    zero_width_chars_detected
Cyrillic homoglyphs      BLIND   SEES    PASS    mixed_script_detected;homoglyphs_normalized
Full-width Unicode       BLIND   SEES    PASS    unicode_nfkc_applied
Base64 wrapper           BLIND   SEES    PASS    base64_decoded
Base64 nested x2         BLIND   SEES    PASS    base64_decoded
Hex encoding             BLIND   SEES    PASS    hex_decoded
ROT13                    BLIND   SEES    PASS    rot13_decoded
HTML display:none        SEES    SEES    PASS    html_stripped;hidden_css_detected
HTML comment             BLIND   SEES    PASS    html_stripped;html_comment_extracted
Image alt-text           BLIND   SEES    PASS    html_stripped;attribute_text_extracted
White-on-white text      SEES    SEES    PASS    html_stripped;hidden_css_detected
Base64 in hidden div     BLIND   SEES    PASS    base64_decoded;html_stripped;hidden_css_detected
BENIGN control           BLIND   BLIND   PASS    -
BENIGN HTML question     BLIND   BLIND   PASS    -

  Attack cases            : 13
  Missed WITHOUT MOCHI    : 10/13
  Missed WITH MOCHI       : 0/13
  Benign controls clean   : 2/2
```

"Naive" models the standard RAG pipeline: strip HTML with BeautifulSoup, scan
the text. The two benign controls matter as much as the attacks — they show the
layer does not simply flag everything.

### 3. Coverage report

```bash
python -m pytest --cov=mochi --cov-report=html:reports/coverage --cov-report=term
```

**87% overall.** Untested remainder is concentrated in code paths that need a
live provider (`openai_adapter.py`, 45%) or a real PDF fixture
(`file_extract.py`, 65%). Core logic is well covered:

| Module | Coverage |
|---|---|
| `telemetry/schema.py` | 100% |
| `telemetry/logger.py` | 100% |
| `preprocess/html_extract.py` | 98% |
| `gateway/app.py` | 92% |
| `preprocess/normalize.py` | 91% |

### 4. Machine-readable reports

```bash
python -m pytest --html=reports/test-report.html --self-contained-html --junitxml=reports/junit.xml
```

- `reports/test-report.html` — standalone HTML, opens in any browser, good for a printed appendix
- `reports/junit.xml` — standard JUnit XML, the format CI systems and most academic tooling expect
- `reports/coverage/index.html` — line-by-line coverage browser
- `reports/evasion-matrix.csv` — the demo table, ready to paste into the thesis

`reports/` is gitignored — these are generated artifacts. Regenerate with the
one command below before a defense so the timestamps are current.

### Regenerate everything

```bash
python -m pytest -v \
  --html=reports/test-report.html --self-contained-html \
  --junitxml=reports/junit.xml \
  --cov=mochi --cov-report=html:reports/coverage --cov-report=term
python demo/evasion_demo.py --csv
python docs/samples/generate_sample_log.py
```

---

## Manual / live testing

For a defense, a live demonstration is worth more than a report. This sequence
takes about two minutes.

### 1. Start the gateway

```bash
python -m uvicorn mochi.gateway.app:app --reload
```

### 2. Show the API surface

Open <http://127.0.0.1:8000/docs> — FastAPI's generated Swagger UI. Useful for
showing that MOCHI presents a standard, self-documenting API.

### 3. Health check

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","version":"0.1.0","provider":"openai","default_model":"gpt-4o-mini"}
```

### 4. Prove the integration claim

The "plug and play" claim is best demonstrated by pointing a *stock* OpenAI
client at MOCHI with one line changed:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
print(client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

No SDK, no code restructuring — that *is* NFR2.

### 5. Show the telemetry it produced

```bash
tail -n 1 logs/mochi.jsonl | python -m json.tool
```

Point out `inspection_ms` versus `upstream_ms` — MOCHI's own overhead versus
time spent waiting on the target model.

### 6. Show privacy posture

```bash
grep -c "your prompt text here" logs/mochi.jsonl   # → 0
```

Prompt text is never persisted by default; only a SHA-256 hash. This is the
Ethical Considerations commitment, demonstrable in one command.

---

## What is NOT yet testable

Be direct about this if asked — the gaps are phased work, not oversights.

| Claim | Blocked on | Phase |
|---|---|---|
| Detection accuracy / precision / recall / F1 | No detector exists yet | 6, 8, 13 |
| Attack Success Rate, Mitigation Rate | No enforcement yet | 10, 13 |
| < 2 ms syntactic / < 55 ms semantic (NFR1) | Stages I/II not built; timing hooks are in place | 6, 8 |
| Multi-turn chain detection | Session accumulator not built | 7 |
| Model-agnostic effectiveness (Table 19) | Only the OpenAI adapter exists | 12 |
| Statistical significance (H01–H06) | Requires before/after evaluation data | 13 |

The measurement infrastructure for all of these already exists — `stage_timer`
records per-stage latency, and the telemetry schema has fields reserved for
every detection result. Each phase fills in its slot rather than requiring new
plumbing.

---

## Suggested framing for the panel

> "MOCHI is verified at two levels. At the implementation level, 59 automated
> tests at 87% coverage confirm each component meets its specification,
> including a demonstration that the normalization layer defeats 13 documented
> obfuscation techniques that a conventional pipeline misses in 10 of 13 cases.
> At the effectiveness level, detection and mitigation performance are measured
> against a labeled benchmark of public prompt-injection datasets, reported in
> Chapter IV."

Present the first half now; the second half is what Phases 5–13 produce.
