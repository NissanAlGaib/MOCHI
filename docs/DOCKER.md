# Running MOCHI with Docker

The proxy, packaged as a container. Run these from the repository root.

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
| `data/attacks/system_prompt.txt` | The declared system prompt, exempted by exact match. This plain prompt scores 0.0083 and would pass anyway; a defensive one would not (see [SETUP.md](SETUP.md#every-request-is-refused-even-harmless-ones)). |
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

## Verified in the container

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
