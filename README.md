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
| 5 | Evaluation harness | ✅ Done |
| 6 | Stage I — syntactic filtering | ✅ Done |
| 6.5 | Feature extraction layer | ⬜ Next |
| 7 | Session risk accumulator | ✅ Done |
| 8 | Stage II — semantic detection | 🟡 Code complete, model not trained |
| 8.5 | BiLSTM / BiGRU baselines | ⬜ |
| 9 | Stage III — cognitive arbitration | ⬛ Omitted by design (register Q7) |
| 10 | Enforcement & sanitization | ✅ Done |
| 11 | Outbound interception | ✅ Done |
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

### Or with Docker

```bash
cp .env.example .env              # then edit .env and add your OPENAI_API_KEY
docker compose up --build
curl http://127.0.0.1:8000/health
```

**If port 8000 is already taken** — it usually is while a gateway is running on
the host for evaluation — publish somewhere else:

```bash
MOCHI_PUBLISH_PORT=8001 docker compose up --build    # PowerShell: $env:MOCHI_PUBLISH_PORT="8001"
```

Worth knowing because the clash is quiet in the worst way: `docker compose up`
fails to bind while `curl http://127.0.0.1:8000/health` still answers 200 —
from the host process. A health check that passes against the wrong gateway is
more misleading than one that fails.

The image installs the **CPU build of torch**. Stage II runs on CPU by design
— when the protected model is also local, an 8 GB card holding a 7B target at
an 8k context has no room for a second resident model — and the CUDA wheel is
4.3 GB of kernels that would never be used. Expect a first build of several
minutes and roughly 2 GB of image.

Three paths are mounted rather than copied in, so `docker compose up` must run
from the repo root:

| Mount | Why |
|---|---|
| `models/e5-fine-tuned` | 468 MB of weights. Read-only, so a compromised gateway cannot rewrite its own detector, and swapping checkpoints (`e5-banking`) needs no rebuild. |
| `data/attacks/system_prompt.txt` | The declared system prompt. Without it the gateway starts but refuses almost everything — see below. |
| `logs/` | The only path the container writes to. |

`compose` overrides `OPENAI_BASE_URL` to `host.docker.internal:11434`. Inside a
container `localhost` is the container, so the value in `.env` — written for
running on the host — would fail to reach Ollama. Ollama itself stays on the
host; it is not containerised here, because it holds the GPU and the pulled
model blobs.

**If the container restart-loops**, read the logs before changing anything:

```bash
docker compose logs gateway
```

Almost always the model mount is wrong. Stage II is loaded eagerly and scores
a warm-up string before the app accepts traffic, so an unreadable
`models/e5-fine-tuned` raises at startup and the process exits. That is
deliberate: serving Stage I alone — 5.9% recall — while an operator believes
Stage II is running is the failure worth avoiding, so the gateway refuses to
start instead of quietly degrading.

Note that `/health` answers from configuration alone and never touches the
detector, so a passing healthcheck proves the process is up, not that
detection is live. To confirm Stage II actually loaded, look for
`Stage II enabled` in the startup log.

### Verified in the container

Built and run on 2026-10-11 (engine 29.8.2, WSL2). Image 2.55 GB, with
`torch 2.14.1+cpu` — `torch.version.cuda` is `None`, so the CPU-only intent
held.

| Check | Result |
|---|---|
| Stage II loads from the mount | `Stage II enabled - model=/app/models/e5-fine-tuned device=cpu` |
| Declared prompt mount | `Declared system prompt loaded ... (178 chars)` |
| Benign request, container to host Ollama | 200, answered |
| Direct injection in the user turn | 403, `direct_injection detected in user_input` |
| Indirect injection in a document | 200, canary not leaked, the overdraft fact survived |
| Image-borne injection | 403, with `image_text_extracted` in telemetry |
| OCR inside the image | `ocr_ran: True` — the apt layer is right |
| Telemetry to the mounted volume | written as uid 10001 |

Stage II scores are **identical to six decimals** between the host
(`torch 2.6.0+cu124`, `transformers 5.17.0`) and the container
(`2.14.1+cpu`, `5.19.0`), across five probes spanning English, Taglish and
benign text. The version drift does not move the detector, so results measured
on the host reproduce in the image.

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

## Evaluation

```bash
python eval/run_detection.py --demo                  # no downloads needed
python eval/run_detection.py --data data/ --folds 5  # once datasets are in place
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for dataset setup and metric
definitions.

## Tests

```bash
python -m pytest                 # 138 tests
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
