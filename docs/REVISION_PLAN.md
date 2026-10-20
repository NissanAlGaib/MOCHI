# Revision plan — three changes from adviser review

Raised 2026-10-20. Each item below states what was asked, what I measured or
checked before answering, what I propose to build, and what I need decided.

Item 1 contains a disagreement about terminology that is worth settling before
anything is written up. Items 2 and 3 are additive and I think both are right.

---

## 1. Metrics: what "false positive" should mean

### What was asked

> The false positive rate should be the measurement of prompts that are
> malicious and are marked as malicious by the model but still got through to
> the LLM. Therefore the main metric would be the detection and mitigation rate.

### The part I agree with

**Detection rate and mitigation rate as the headline metrics is correct**, and
it is already what the evaluation reports. No change needed there.

**The quantity being described is real, useful, and currently unmeasured.** An
attack that MOCHI flagged, acted on, and forwarded anyway — which then still
worked — is a distinct failure from one MOCHI never noticed. Those two should
not be pooled, and right now they are.

I computed it on existing run data before writing this, so the numbers below
are measured rather than projected.

| Corpus | Defended runs | MOCHI acted | Forwarded after acting | Still succeeded | Leak rate |
|---|---|---|---|---|---|
| Generated 100 | 300 | 195 | 33 | **4** | **12.1%** |
| Gandalf 300 | 900 | 756 | 0 | 0 | n/a — nothing was forwarded |

Gandalf produces no forwarded cases at all because every Gandalf attack sits in
the user's own message, which is refused outright rather than redacted. Only
the generated corpus contains indirect attacks, which are the ones redacted and
passed on.

Set beside the other failure mode:

| Corpus | Flagged, redacted, still worked | Never flagged at all, worked |
|---|---|---|
| Generated 100 | 4 | 48 |
| Gandalf 300 | 0 | 63 |

**This is a finding worth having.** Redaction almost always holds when it
fires; the failures are overwhelmingly cases where nothing fired. 4 against 48,
and 0 against 63. It says effort belongs in detection rather than in hardening
mitigation — which we could not have argued before, because the two were
reported as one number.

So: build the metric. The adviser is right that it is missing.

### The part I want to push back on

**Calling it the false-positive rate would be a mistake**, for three reasons.
I would rather raise this now than have it raised during the defence.

**It collides with a standard term.** In every detection paper, a false
positive is a *benign* item wrongly flagged as malicious. What is described
here is a *malicious* item correctly flagged whose mitigation then failed —
in classification terms a true positive. Anyone on the panel with a machine
learning background will read "false positive rate" and expect the first thing.

**It removes the only guard against a meaningless result.** A mitigation rate
with no benign measurement beside it can be maximised by refusing everything.
A gateway that blocks 100% of traffic scores a perfect mitigation rate and is
useless. The benign false-positive rate is what makes the mitigation number
mean anything, and it is also where this project's most interesting finding
lives — 2.66% of real banking customers refused, traced to a specific cause in
the training data, reduced to 0.13% by retraining.

**It makes the work incomparable.** Published prompt-injection defences all
report FPR in the standard sense. Redefining it means no reviewer can place
these results next to anyone else's.

### What I propose instead

Keep three metrics, named so each is unambiguous:

| Metric | Question it answers | Denominator |
|---|---|---|
| **Detection rate** | Of attacks, how many did MOCHI flag? | all attacks |
| **Mitigation rate** | Of attacks dangerous to the bare model, how many stopped working? | dangerous attacks |
| **Enforcement leakage** *(new)* | Of attacks MOCHI acted on but forwarded, how many still worked? | acted-and-forwarded |
| **False-positive rate** *(keep as-is)* | Of ordinary benign traffic, how much was refused? | benign queries |

"Enforcement leakage" is the adviser's metric under a name that does not
collide. Other defensible names: *sanitization bypass rate*, *residual attack
rate*. The name matters less than not calling it FPR.

### Why I may be wrong

The adviser may have meant "we should also measure this" rather than "replace
FPR with this", in which case we agree entirely and only the label is at issue.
It is also possible the intent was that benign false positives are less
important for this thesis than enforcement failures — which is arguable, though
I would still report both.

**This needs a decision before Chapter IV is written**, because it changes
which numbers are headline.

### Work

| | Task | Effort |
|---|---|---|
| 1.1 | Add `enforcement_leakage()` to `eval/metrics.py`, taking acted-and-forwarded as denominator | 2 h |
| 1.2 | Make `attack_simulation.py` record the verdict and `reached_model` unambiguously for every run (already present; confirm for the image harness, which writes a different row shape) | 2 h |
| 1.3 | Fix the image harness to emit `mitigation_action` — it currently writes `None`, so leakage cannot be computed for the multimodal arms | 3 h |
| 1.4 | Add a Summary block to the workbook: detection / mitigation / leakage / FPR, with the denominator printed beside each | 3 h |
| 1.5 | Write the definitions page so each metric states its denominator in one line | 1 h |

---

## 2. One pipeline that turns every input into text

### What was asked

> Rather than running multiple different LLM models for different types of
> inputs, we can make a pipeline where we take those inputs and normalize them
> all into text, so that no matter what model we use that only accepts text,
> MOCHI itself can convert different types of inputs into text.

### Assessment

**This is right, and it is already the architecture for images — it just was
not generalised.** `image_extract.py` turns pixels into text and hands that to
the ordinary pipeline; nothing downstream knows the text came from an image.
Audio and video would follow exactly the same shape.

It also solves a practical problem that was blocking the multimodal work. We
could not evaluate audio defence because we had no audio-capable model to test
against. If MOCHI converts audio to text itself, **any text model becomes a
valid target**, and the evaluation stops depending on finding a multimodal
model that runs locally.

### The design decision this forces

There are two different things MOCHI can do with extracted text, and they are
not the same system:

| Mode | What the model receives | Consequence |
|---|---|---|
| **Inspect-only** *(today, for images)* | the original image | Defence protects the real path. Needs a multimodal model. |
| **Convert** *(proposed)* | the extracted text only | Works with any text model. The model loses whatever the raw input carried. |

Convert is strictly safer — nothing unexamined reaches the model — but it
degrades capability. A vision model reading a chart directly understands more
than OCR text of that chart. For audio and video, convert is the practical and
probably the only sensible mode. For images it should be a choice.

**Proposal: build the extraction layer once, make the forwarding mode
configurable per modality.** Default images to inspect-only, audio and video to
convert.

### Architecture

Replace the single-purpose image hook with a registry.

```
mochi/preprocess/extract/
    __init__.py       registry: MIME / block-type  ->  extractor
    base.py           Extraction dataclass, shared by all
    image.py          (move existing image_extract.py here)
    audio.py          new  - speech to text
    video.py          new  - audio track + sampled frames
    document.py       (move existing PDF handling here)
```

Every extractor returns the same shape, which already exists as
`ImageExtraction` and generalises cleanly:

```python
@dataclass
class Extraction:
    visible_text: str          # what a person would perceive
    hidden_text: list[str]     # recovered only by a second pass - evidence in itself
    metadata_text: list[str]   # container metadata, e.g. EXIF, ID3 tags
    flags: list[str]
    ran: bool                  # distinguishes "found nothing" from "could not run"
```

The single integration point is `_text_from_content` in `mochi/detect/segments.py`,
which currently branches on `read_images`. That branch becomes a registry
lookup. **Nothing in Stage I, Stage II, the trust model or enforcement changes**
— which is the same property that made the image work cheap.

### Audio

Speech to text with Whisper (`faster-whisper` for CPU speed). The hidden-text
idea transfers: a second pass at altered speed or with noise reduction may
recover speech the plain pass missed, which is the audio equivalent of
near-white text. Whether that is a real attack vector is an open question worth
one experiment before building it.

Audio also has a metadata channel — ID3 tags on MP3 — exactly like EXIF on
images, and exactly as likely to be lifted into a prompt by a transcription
application.

### Video

Two channels, both reusing what exists:

- **Audio track** — `ffmpeg` extracts it, then the audio extractor.
- **Frames** — sample at a fixed interval, then the existing image extractor.
  A single-frame caption that no human would notice is the video version of
  concealed text.

Frame sampling needs a cap, for the same reason `MAX_WINDOWS` exists: a long
video must not cost unbounded time. Suggest one frame per second to a maximum
of 60.

### Work

| | Task | Effort |
|---|---|---|
| 2.1 | Create `extract/` package, move image and PDF code in, define `Extraction` | 1 d |
| 2.2 | Convert `_text_from_content` to registry dispatch; keep behaviour identical for images (tests must stay green) | 0.5 d |
| 2.3 | Add `MOCHI_FORWARD_MODE` per modality: `inspect` or `convert` | 0.5 d |
| 2.4 | Audio extractor with faster-whisper, plus ID3 metadata | 2 d |
| 2.5 | Video extractor: ffmpeg audio track, frame sampling with a cap | 2 d |
| 2.6 | Build an audio attack corpus — synthesise speech from the existing attack text | 1 d |
| 2.7 | Build a video corpus — attack text as a caption on a small number of frames | 1 d |
| 2.8 | Evaluate both arms against a text model, now reachable because MOCHI converts | 1 d |

**Risk:** the corpora are the long pole, as they were for images. No public set
of audio or video prompt injections exists, so these are synthesised and that
limitation has to be stated the same way it is for the image arms.

---

## 3. How a prompt actually reaches the model, and where MOCHI can sit

### What was asked

> I need clarification on how the prompt is sent to MOCHI then how MOCHI sends
> it to an LLM... my adviser says what if we append a MOCHI function to a send
> request function of an LLM.

### The mechanism, plainly

When an application calls `client.chat.completions.create(...)`, that function
does one thing underneath: an HTTP POST to `/v1/chat/completions` at whatever
address the client was configured with. There is no magic in the SDK — it
builds JSON, posts it, parses the reply.

That gives two places to intercept, and the adviser is describing the second.

**At the network layer — a proxy.** The application posts to MOCHI instead of
the provider. MOCHI inspects, then makes its own outbound request using the
provider address and key held in *its* configuration, and returns the reply.
This is what is built today. It works for any language because the boundary is
HTTP.

**At the function layer — a wrapper.** MOCHI wraps `create()` itself: inspect
the arguments, then call the real function. No network hop, no second process.
This is the adviser's suggestion, and it is the better developer experience:

```python
from openai import OpenAI
from mochi import guard

client = guard(OpenAI())    # same client, inspected calls
```

The wrapper inherits whatever the application already configured, so there is
no second copy of the API key, no provider address to duplicate, and nothing to
deploy. It is Python-only, which is why the proxy keeps its place.

One property the proxy has and the wrapper cannot: **the wrapper can be
bypassed.** An application that constructs a raw HTTP client never passes
through it. A proxy with no other route out cannot be avoided.

### Why this is cheap to build

The detection pipeline is already a plain library with no network dependency.
`showcase/try_prompt.py` runs the full cascade — segmentation, normalization,
both stages, enforcement — with no gateway and no socket. The wrapper is
roughly a hundred lines joining what exists:

```python
def guard(client, *, system_prompt=None, context=None):
    """Wrap an OpenAI-compatible client so every call is inspected first."""
    # intercept client.chat.completions.create
    #   -> build segments from messages (+ context if supplied)
    #   -> inspect()  ->  decide()
    #   -> BLOCK: raise MochiBlocked(reason, request_id)
    #      SANITIZE: forward the redacted messages
    #      ALLOW: forward unchanged
```

### Work

| | Task | Effort |
|---|---|---|
| 3.1 | `mochi/client/guard.py` — wrap an OpenAI-compatible client | 1 d |
| 3.2 | `MochiBlocked` exception carrying reason and request id | 2 h |
| 3.3 | Lazy-load Stage II on first call so importing is cheap | 3 h |
| 3.4 | Tests: block, sanitize, allow, and that a sanitized call forwards redacted text | 0.5 d |
| 3.5 | Worked example in `showcase/` showing the same attack through wrapper and proxy with identical verdicts | 0.5 d |
| 3.6 | Document all three surfaces — wrapper, proxy, sidecar — with the bypass trade-off stated | 0.5 d |

---

## Sequencing

None of this requires retraining, which matters: the machine is currently
thermally throttling and heavy jobs are a risk.

| Order | Item | Why first | Effort |
|---|---|---|---|
| 1 | **1.1–1.5** metrics | Unblocks the adviser conversation and changes what gets written up. Pure analysis, no new runtime code. | 1.5 d |
| 2 | **3.1–3.6** wrapper | Contained, self-proving, and the strongest answer to the adoption question. | 2.5 d |
| 3 | **2.1–2.3** extraction refactor | Behaviour-preserving. Must land before audio or video. | 2 d |
| 4 | **2.4, 2.6** audio + corpus | The modality most likely to be asked about. | 3 d |
| 5 | **2.5, 2.7** video + corpus | Builds on audio; largest remaining unknown. | 3 d |
| 6 | **2.8** evaluate both | Needs the corpora. | 1 d |

Roughly 13 working days. Items 1 and 2 alone — four days — answer the adviser's
three points well enough to present.

## Decisions needed before starting

1. **Metric naming.** Keep false-positive rate in its standard sense and add
   enforcement leakage as a separate number, or redefine FPR as asked? This
   changes Chapter IV and should be settled first.
2. **Forwarding mode for images.** Keep inspect-only, so a vision model still
   receives the picture, or convert to text for uniformity and lose what the
   image carried?
3. **Scope of multimodal.** Is audio enough, or is video required? Video is
   three extra days and its corpus is the weakest evidence we would produce.
4. **Where the wrapper is claimed.** As the primary integration path, with the
   proxy as the enforceable fallback — or the reverse?
