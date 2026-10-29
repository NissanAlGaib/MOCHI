# How a prompt reaches the model, and where MOCHI can sit

This answers the question raised in review: *how is the prompt sent to MOCHI,
and how does MOCHI send it to an LLM?* — and the follow-up, *what if we append
a MOCHI function to an LLM's send-request function?*

## The mechanism, plainly

When an application calls `client.chat.completions.create(...)`, that function
does one thing underneath: an HTTP POST to `/v1/chat/completions` at whatever
address the client was configured with. There is no magic in the SDK. It builds
JSON, posts it, parses the reply.

```
application  ──►  SDK builds JSON  ──►  POST /v1/chat/completions  ──►  provider
                       ▲                            ▲
                       │                            │
                  wrapper sits here          proxy sits here
```

That gives two places to intercept, and the review question describes the first.

---

## Surface 1 — the wrapper (in-process)

```python
from openai import OpenAI
from mochi.client import guard

client = guard(OpenAI())        # same client, inspected calls
```

`guard()` replaces `client.chat.completions.create` with a function that
inspects the arguments, then calls the original. No network hop, no second
process, nothing to deploy. The wrapper inherits whatever the application
already configured — base URL, API key, timeouts — so there is no second copy
of the key and no provider address to duplicate.

A refused request raises rather than returning a value:

```python
from mochi.client import MochiBlocked

try:
    reply = client.chat.completions.create(model="gpt-4o", messages=messages)
except MochiBlocked as blocked:
    print(blocked.reason, blocked.request_id)
```

An exception, not a sentinel, because a caller who ignores a return value sends
the attack anyway — and the point of this surface is that forgetting to check
should not be possible.

### Telling MOCHI where content came from

The single highest-value thing an integration can do:

```python
client = guard(OpenAI(), context={
    "system_prompt": SYSTEM,
    "user_input": question,
    "retrieved_document": fetched_page,
})
```

Without `context`, trust is inferred from chat roles, which tags everything in
the user's turn as semi-trusted. A document the application pasted into the
user message is then treated as the user's own words — so an injection inside
it is **refused** rather than **redacted**, and the user's real question is
lost with it. Supplying `context` is what makes sanitisation possible.

An application that concatenates fetched content into the user turn is
downgrading its own protection from sanitize to block.

### Settings

| | Default | |
|---|---|---|
| `system_prompt` | `None` | prepended when the caller supplies none |
| `context` | `None` | tagged origins; see above |
| `enable_stage2` | `True` | off also avoids importing torch |
| `stage2_device` | `"cpu"` | |
| `read_attachments` | `False` | OCR / transcription for images, audio, video |
| `forward_mode` | `"inspect"` | `"convert"` replaces attachments with their text |
| `enable_outbound` | `True` | inspect responses as well as requests |
| `allow_buffered_streaming` | `False` | see below |
| `on_verdict` | `None` | callback after every call, including blocked ones |

Stage II loads lazily on first use, so importing `mochi.client` costs about a
second and does not pull in torch.

---

## Surface 2 — the proxy (a server)

```python
client = OpenAI(base_url="http://mochi:8000/v1", api_key="unused")
```

The application posts to MOCHI instead of the provider. MOCHI inspects, then
makes its own outbound request using the provider address and key held in *its*
configuration, and returns the reply. This is what `mochi/gateway/` is, and it
is what the evaluation measures.

The API key lives in MOCHI's environment, not the application's — which is a
deployment advantage on its own, since the application never holds it.

---

## Surface 3 — the sidecar (same pod, loopback only)

The proxy, deployed beside the application and bound to `127.0.0.1`, with the
provider reachable only from the sidecar. Not different code — the same
gateway, with a network policy that removes the bypass.

---

## Choosing between them

The trade is not about features. Both surfaces run the same pipeline and reach
the same verdict; `showcase/compare_surfaces.py` demonstrates that on identical
input, and `tests/test_client_guard.py::test_wrapper_and_proxy_agree` fails the
build if they ever diverge.

The trade is about **who is being defended against**.

| | Wrapper | Proxy | Sidecar |
|---|---|---|---|
| Can be bypassed | **yes** | only by leaving the network | no |
| Languages | Python only | any | any |
| Extra process | no | yes | yes |
| Network hop | no | yes | loopback |
| Holds the API key | application | MOCHI | MOCHI |
| Enforceable on code you don't control | no | yes | yes |

**The wrapper can be bypassed.** It replaces one method on one object. Code
that builds a second client, or posts with `requests`, never passes through it,
and anyone who can edit the file can delete the line.

**The proxy cannot be avoided** when it is the only route out of the network.

So the recommendation is not one or the other:

- **Integrating a defence into your own application** → the wrapper. Less to
  deploy, less to misconfigure, and the trust context is easiest to supply at
  the call site where the application already knows where its data came from.
- **Imposing a defence on code you do not control** → the proxy or sidecar. A
  control that can be removed by editing one line is not a control.
- **Both properties needed** → the wrapper inside a network that only routes
  outbound through the proxy. Inspection twice is idempotent: `guard()` returns
  early on an already-guarded client, and a sanitised request re-inspected is
  still sanitised.

---

## Streaming

Refused by default on both surfaces, and for the same reason: outbound
inspection needs the complete response body, because an exfiltration URL can
straddle two chunks. The proxy returns **501**; the wrapper raises
`MochiBlocked`.

`allow_buffered_streaming` requests the whole response and hands it back in one
piece. That keeps `stream=True` callers working, but it is **not** incremental
delivery and both surfaces say so rather than letting a caller discover it.

---

## What neither surface does

- **Inspect what the model is asked to *do*.** MOCHI reads text in and text
  out. An agent that calls tools between those points is outside what this
  measures.
- **Catch a short secret quoted back.** Outbound leak detection matches on
  eight-word n-grams, so a four-word credential repeated by the model is below
  the floor. See Limitations 4.8.
- **Defend a second client.** Guarding one object guards that object.
