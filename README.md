# MOCHI

**M**iddleware for **O**bserving, **C**lassifying, and **H**andling prompt **I**njections.

A lightweight, deployable security gateway that sits between an LLM application
and its target LLM, detecting and mitigating prompt injection attacks without
requiring changes to application logic or to the model itself.

Undergraduate thesis — BS Computer Science, Western Mindanao State University.
Hans Adrian A. Lao, Jelaine May C. Macias, Rosepel M. Maglangit.

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Environment setup | ✅ Done |
| 1 | Gateway skeleton (pass-through proxy) | ✅ Done |
| 2 | Structured JSON telemetry | ✅ Done |
| 3 | Normalization / de-obfuscation layer | ✅ Done |
| 4 | Source tagging & payload parsing | ✅ Done |
| 5 | Evaluation harness | ⬜ Next |
| 6 | Stage I — syntactic filtering | ⬜ |
| 7 | Session risk accumulator | ⬜ |
| 8 | Stage II — semantic detection | ⬜ |
| 9 | Stage III — cognitive arbitration | ⬜ |
| 10 | Enforcement & sanitization | ⬜ |
| 11 | Outbound interception | ⬜ |
| 12 | Multi-provider adapters | ⬜ |
| 13 | Full evaluation | ⬜ |
| 14 | Packaging | ⬜ |

See [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) for what each phase entails and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env              # then edit .env and add your OPENAI_API_KEY

python -m uvicorn mochi.gateway.app:app --reload
```

The gateway listens on `http://127.0.0.1:8000`.

## Using it

MOCHI speaks the OpenAI chat-completions API, so integration is a one-line
change in the client application — no code restructuring, no SDK to adopt:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",   # <- the only change
    api_key="unused",                       # MOCHI holds the real key server-side
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this document."}],
)
```

### Optional: source tagging

Supplying `context` lets MOCHI inspect each payload segment separately and
attribute a detection to a precise origin (which is what distinguishes a
direct injection from an indirect one). It is entirely optional — untagged
requests are inspected as a single blob.

```json
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Summarize the page I linked."}],
  "session_id": "sess_abc123",
  "context": {
    "user_input": "Summarize the page I linked.",
    "web_content": "<scraped page text>"
  }
}
```

`session_id` and `context` are MOCHI extensions and are stripped before the
request is forwarded upstream.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + active provider/model |
| `POST` | `/v1/chat/completions` | OpenAI-compatible inspection + forwarding |
| `GET` | `/docs` | Auto-generated OpenAPI docs |

## Telemetry

Every request through `/v1/*` emits one JSON Lines record to
`logs/mochi.jsonl` — **regardless of classification**, so benign traffic is
logged too. That is what makes false-positive rate computable. This file is
the data source the Phase 5/13 evaluation harness reads to produce the
Chapter IV results tables:

```python
import pandas as pd
df = pd.read_json("logs/mochi.jsonl", lines=True)
```

Raw prompt text is **not** stored by default — only a SHA-256 hash and length,
per the thesis commitment on anonymization. Set `MOCHI_LOG_PAYLOADS=true` for
controlled evaluation runs over public benchmark datasets only.

See [docs/TELEMETRY.md](docs/TELEMETRY.md) for the full field reference and
[docs/samples/telemetry-sample.jsonl](docs/samples/telemetry-sample.jsonl) for
seven worked example records.

## Tests

```bash
python -m pytest                 # 59 tests
python demo/evasion_demo.py      # obfuscation-resistance demonstration
```

See [docs/TESTING.md](docs/TESTING.md) for how to generate presentable
evidence (HTML/JUnit reports, coverage, evasion matrix) and for a live-demo
walkthrough.

## Known limitations (current phase)

- **Detection is not wired in yet.** Phase 1 is transport only;
  `inspect_request()` in `mochi/gateway/app.py` is the seam where the Phase
  3–10 pipeline attaches.
- **Streaming (`stream: true`) returns 501.** Phase 11 outbound interception
  needs the complete response body to scan for leaked instructions and
  exfiltration URLs, so a streaming path would have to be buffered anyway.
- **OpenAI is the only provider.** Anthropic and Gemini adapters arrive in
  Phase 12. Pointing `OPENAI_BASE_URL` at any OpenAI-compatible server
  (vLLM, Ollama, LM Studio, OpenRouter) works today.
