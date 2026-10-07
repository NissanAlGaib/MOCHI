"""The three protected models for the attack simulation.

Register items A5 and Q6 settled the model family question: local open-weight
models served through Ollama. A8 settled why it has to be more than one - MOCHI
treats the target as a black box, so the multi-LLM arm exists to show the
defence does not depend on which box it is.

**No provider adapter is needed.** Ollama serves an OpenAI-compatible endpoint,
and the gateway already speaks that API, so the whole pipeline is reused by
pointing ``OPENAI_BASE_URL`` at ``localhost:11434/v1``. Phase 12's Anthropic and
Gemini adapters are not on this path.

Two endpoints matter to the simulation and they differ in exactly one way:

    defended    client -> MOCHI :8000 -> Ollama :11434 -> model
    undefended  client ------------------> Ollama :11434 -> model

Same target, same prompt, same decoding parameters. The only variable is
whether MOCHI is in the middle, which is what makes the pair comparable and
what ``eval.stats.paired_ttest`` requires.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Ollama's OpenAI-compatible base URL - the undefended arm talks here directly.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

#: The gateway - the defended arm talks here instead.
MOCHI_BASE_URL = os.getenv(
    "MOCHI_BASE_URL",
    f"http://{os.getenv('MOCHI_HOST', '127.0.0.1')}:{os.getenv('MOCHI_PORT', '8000')}/v1",
)

#: Ollama ignores the key; the OpenAI client library requires a non-empty one.
PLACEHOLDER_KEY = "ollama"

#: Greedy decoding. A simulation that samples would report a different attack
#: success rate on every run, and the thesis Reliability section asks for
#: reproducibility rather than an average over unrecorded randomness.
TEMPERATURE = 0.0

#: Enough room for a model to comply with an injection. Too small a budget
#: truncates a successful attack into an apparent failure.
MAX_TOKENS = 512

#: Seconds Ollama keeps a model resident after a request. Zero forces an unload
#: so the next target gets a clean 8 GB card - see MODEL_LOOP_NOTE.
KEEP_ALIVE_BETWEEN_TARGETS = 0

#: An 8 GB card holds one 7-8B model at a time. Loop targets on the OUTSIDE and
#: prompts on the inside: 3 model loads instead of 300.
MODEL_LOOP_NOTE = "outer loop = target, inner loop = prompt"


@dataclass(frozen=True)
class Target:
    """One protected model."""

    tag: str
    """Ollama tag. This, not a checkpoint, is what makes the run reproducible -
    ``ollama pull <tag>`` fetches identical weights for anyone reading the
    thesis, which is why the repository records tags and not 14 GB of blobs."""

    name: str
    """Display name for tables and figures."""

    family: str
    """Vendor lineage, so Table 19 can group by it."""

    approx_vram_gb: float
    """Q4_K_M resident size. Sums to more than 8 GB across the three, which is
    the whole reason for the outer-loop rule above."""


#: Why the targets are ``mochi-*`` and not the stock tags.
#:
#: Ollama defaults ``num_ctx`` to roughly 4096 and truncates silently past it.
#: The system prompt sits at the front of the conversation, so on a long
#: retrieved document it is the first thing evicted - taking the canary with
#: it. Measured on ``mistral:7b-instruct``: the canary is recalled at 1,000
#: words and lost by 3,000, with no error and no warning.
#:
#: That would have quietly destroyed the Tier 4 results. Every long-document
#: attack would have "failed" in both arms, the undefended baseline would show
#: no attacks worth stopping, and the dilution case - the one D11 says the
#: public corpora cannot test and Q8 says to build this simulation for - would
#: have reported nothing.
#:
#: Per-request ``options`` do not fix it: Ollama's OpenAI-compatible endpoint
#: ignores that field, and MOCHI has to speak OpenAI. So the window is pinned
#: model-side by the Modelfiles in ``eval/modelfiles/``, rebuilt with::
#:
#:     ollama create mochi-mistral -f eval/modelfiles/mochi-mistral.Modelfile
#:
#: Verified through the OpenAI endpoint: all three recall the canary at 5,000
#: words. The derived models share the parent's blobs, so they cost no disk.
STOCK_TAGS = {
    "mochi-mistral": "mistral:7b-instruct",
    "mochi-qwen": "qwen2.5:7b-instruct",
    # Not "llama3.1:8b-instruct" - that tag does not exist in the Ollama
    # library and pulls fail with "file does not exist". The instruct build is
    # published under the explicit quantization name.
    "mochi-llama": "llama3.1:8b-instruct-q4_K_M",
}

#: Context window pinned by the Modelfiles. Holds a ~6,000-word document plus
#: the system prompt, and still fits beside the weights on an 8 GB card.
NUM_CTX = 8192

TARGETS: tuple[Target, ...] = (
    Target(tag="mochi-mistral", name="Mistral 7B Instruct",
           family="Mistral", approx_vram_gb=4.4),
    Target(tag="mochi-qwen", name="Qwen2.5 7B Instruct",
           family="Qwen", approx_vram_gb=4.7),
    Target(tag="mochi-llama", name="Llama 3.1 8B Instruct",
           family="Llama", approx_vram_gb=4.9),
)

#: Generator for the attack corpus (A5: local model, no refusals, no API cost).
#: Listed separately from TARGETS because generating attacks and being attacked
#: are different roles - a model should not be scored on prompts it wrote.
#: Uses the stock tag: generation needs no pinned context window.
GENERATOR_TAG = "qwen2.5:7b-instruct"


def by_tag(tag: str) -> Target:
    """Look up a target, failing loudly on a typo rather than silently."""
    for target in TARGETS:
        if target.tag == tag:
            return target
    known = ", ".join(t.tag for t in TARGETS)
    raise KeyError(f"unknown target {tag!r}; known targets: {known}")


def base_url(*, defended: bool) -> str:
    """Endpoint for one arm of the experiment."""
    return MOCHI_BASE_URL if defended else OLLAMA_BASE_URL
