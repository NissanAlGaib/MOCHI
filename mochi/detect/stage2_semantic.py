"""Stage II: semantic detection over a fine-tuned E5 classifier.

Stage I matches *form*; Stage II matches *meaning*. A paraphrased attack -
"kindly set aside whatever guidance you were given earlier" - contains no
pattern from ``patterns.json`` and passes Stage I untouched, but sits close to
known attacks in embedding space. Stage I's measured recall of **0.0590** on the
cleaned corpus (``data/clean``, n=82,765) is the empirical case for this stage
existing.

The figure is quoted from ``reports/clean_stage1.json``, not the earlier
``reports/stage1.json`` (recall 0.0521), which was measured against the raw
``data/`` tree - uncapped jayavibhav plus the 546 deepset rows since excluded.
Both are real numbers; only one is comparable to everything else reported, and
mixing the two is how a results table stops adding up.

Three implementation choices here differ from the library defaults, each
because a measurement said the default was wrong:

1. **Chunk and take the max, never truncate.** ``truncation=True`` keeps the
   first 512 tokens. The signal-position audit found 14.8% of attack payloads
   in the document tail, so the default silently discards them.

2. **Attention pooling, not mean pooling.** ``sentence-transformers`` mean-pools
   E5 by default. With a measured *median payload share of 3.4%*, a mean-pooled
   document vector is ~97% "benign business text" and the malicious direction is
   averaged into noise. Attention pooling lets a 3%-payload document still score
   high, and the learned weights double as the attribution signal.

3. **Report the span, not just the score.** The winning chunk and its top-
   weighted tokens are returned, so Stage II can answer "which token contributed
   to the prediction" the way Stage I already does via ``matched_text``. This is
   also what makes SANITIZE implementable: knowing *where* the payload is means
   redacting it instead of rejecting the whole request.

The scorer is a :class:`SemanticScorer` protocol rather than a hard dependency
on torch. Training happens on a GPU elsewhere (see ``training/finetune_e5.py``);
this module loads the exported weights when they exist, and every code path here
is exercised in tests through a deterministic stub. Torch is imported lazily, so
importing MOCHI never costs a 2 GB dependency it may not need.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from mochi.detect.chunking import Chunk, chunk_text, token_budget_to_chars

logger = logging.getLogger(__name__)

#: E5 context window.
MAX_TOKENS = 512

#: Character window handed to the tokenizer. The tokenizer still truncates as a
#: safety net if a window overflows; overlap makes that harmless.
WINDOW_CHARS = token_budget_to_chars(MAX_TOKENS)

#: Characters shared between adjacent windows (~25%), so a payload split across
#: a boundary appears whole in at least one window.
WINDOW_OVERLAP_CHARS = WINDOW_CHARS // 4

#: Below this score a request passes without further inspection.
BENIGN_THRESHOLD = 0.45

#: At or above this score Stage II blocks outright.
MALICIOUS_THRESHOLD = 0.55

#: Scores between the two thresholds are the uncertain band. Resolved by source
#: trust in Phase 10, or escalated to Stage III when that stage is enabled.
#: See docs/BUILD_PLAN.md - the band deliberately does *not* require an LLM.

#: Cap on windows scored per segment. A pathological document would otherwise
#: cost unbounded GPU time. Exceeding it is reported, never silent.
MAX_WINDOWS = 32

#: How many attributed tokens to report per detection.
TOP_TOKENS = 8

#: Sentence boundary for splitting a flagged window. A terminator counts only
#: when whitespace follows it, so ``evil@example.com`` stays one piece.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

#: Most sentences of one flagged window scored on their own. A 512-token
#: window rarely holds more; past this the rest are not scored, and the
#: sanitiser falls back to token attribution for them.
MAX_SENTENCES = 64

#: Where ``training/finetune_e5.py`` exports weights to. It exists in a clone
#: that has the weights, and is the first place looked. In an installed package
#: it resolves to ``site-packages/models/e5-fine-tuned``, which never exists -
#: so it is a first choice, never the only one.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "e5-fine-tuned"

#: The published weights, fetched on first use when no local copy exists, and
#: cached by huggingface_hub so only the first start needs the network.
#: These are ``models/e5-fine-tuned``, uploaded 2026-11-05; every file was
#: checked byte-for-byte against the local copy after the upload.
#:
#: The revision is a commit hash, never a branch. A branch would let a
#: re-upload change the detector under every installed copy without a
#: release, and Chapter IV's numbers describe one exact set of weights.
DEFAULT_MODEL_REPO: str | None = "Athelstan05/Mochi"
DEFAULT_MODEL_REVISION: str | None = "243ba3219f337fb1fa640f41386a1a5d1980bd69"

#: Marks a model setting as a Hugging Face repository rather than a path.
#: Explicit because ``models/e5-fine-tuned`` and ``owner/name`` are the same
#: shape: a local folder that is missing must fail as missing, not be fetched
#: from whoever happens to own a repository by that name.
HUB_PREFIX = "hf:"

#: The files the loader reads. A download fetches these and nothing else.
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "mochi_head.json", "mochi_head.pt")


class ModelUnavailable(RuntimeError):
    """Raised when Stage II is enabled but no usable model can be loaded."""


_NO_MODEL_HELP = (
    "Set MOCHI_STAGE2_MODEL_DIR (read by both the proxy and the wrapper) to\n"
    "  either a folder holding the weights, or hf:owner/name@revision to\n"
    "  download them. Or run with MOCHI_ENABLE_STAGE2=false to skip Stage II,\n"
    "  which leaves Stage I alone - recall 0.059 - and is not a deployment."
)


def resolve_model_dir(spec: str | Path | None) -> Path:
    """The local folder to load Stage II from, downloading it if needed.

    ``spec`` is, in order of precedence:

    1. ``hf:owner/name@revision`` - fetched from the Hugging Face Hub.
    2. Anything else - a local folder, used as given.
    3. Nothing - then ``MOCHI_STAGE2_MODEL_DIR`` if set, read here so the
       wrapper honours the same setting the proxy does; then the clone's
       ``models/e5-fine-tuned`` if it exists; then the published weights at
       :data:`DEFAULT_MODEL_REPO`.

    Whatever is resolved must hold every file in :data:`MODEL_FILES`; a
    partial folder is reported by name rather than surfacing later as a
    tokenizer error that does not mention MOCHI.
    """
    if not spec:
        spec = os.environ.get("MOCHI_STAGE2_MODEL_DIR", "").strip() or None
    if spec is not None and str(spec).startswith(HUB_PREFIX):
        repo, _, revision = str(spec)[len(HUB_PREFIX):].partition("@")
        path = _download(repo, revision or None)
    elif spec:
        path = Path(spec)
        if not path.exists():
            raise ModelUnavailable(f"No Stage II model at {path}.\n{_NO_MODEL_HELP}")
    elif DEFAULT_MODEL_DIR.exists():
        path = DEFAULT_MODEL_DIR
    elif DEFAULT_MODEL_REPO:
        path = _download(DEFAULT_MODEL_REPO, DEFAULT_MODEL_REVISION)
    else:
        raise ModelUnavailable(
            f"No Stage II model found, and none is published to download.\n"
            f"Looked in {DEFAULT_MODEL_DIR}.\n{_NO_MODEL_HELP}")

    missing = [name for name in MODEL_FILES if not (path / name).exists()]
    if missing:
        raise ModelUnavailable(
            f"Stage II model at {path} is incomplete - missing "
            f"{', '.join(missing)}.\n{_NO_MODEL_HELP}")
    return path


def _download(repo: str, revision: str | None) -> Path:
    """Fetch the weights into the Hugging Face cache and return that folder.

    Cached after the first call, so a restart does not download again. An
    unpinned request is allowed - it is how someone tries a new upload - but
    logged, because whatever ``main`` points at tomorrow is what it will load.
    """
    if not revision:
        logger.warning("Stage II weights %s are not pinned to a revision; "
                       "a re-upload will change the detector. Use "
                       "hf:%s@<commit>.", repo, repo)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # installed with transformers, so rare
        raise ModelUnavailable(
            "Downloading Stage II weights needs huggingface_hub:\n"
            "  pip install huggingface_hub") from exc
    try:
        return Path(snapshot_download(repo_id=repo, revision=revision,
                                      allow_patterns=list(MODEL_FILES)))
    except Exception as exc:  # network, auth, unknown repo or revision
        raise ModelUnavailable(
            f"Could not download Stage II weights {repo}@{revision or 'main'}: "
            f"{exc}\nOn a machine without internet access, download the "
            f"folder elsewhere and point MOCHI_STAGE2_MODEL_DIR at it.") from exc


@runtime_checkable
class SemanticScorer(Protocol):
    """Scores text windows for maliciousness in ``[0.0, 1.0]``.

    Implementations must accept a batch and return one score per input, in
    order. Attribution is optional - a scorer that cannot explain itself returns
    empty token lists and Stage II degrades to score-only.
    """

    def score(self, texts: Sequence[str]) -> list[float]:
        ...

    def attribute(self, text: str, *, top_k: int = TOP_TOKENS) -> list[tuple[str, float]]:
        ...


@dataclass(frozen=True)
class ChunkScore:
    """One window's score, with enough position data to locate it again."""

    chunk: Chunk
    score: float

    @property
    def start(self) -> int:
        return self.chunk.start

    @property
    def end(self) -> int:
        return self.chunk.end


@dataclass
class Stage2Result:
    """Aggregate outcome of a Stage II scan over one segment."""

    score: float = 0.0
    """Maximum score across windows - *not* the mean.

    Max is the correct aggregate for this problem: a document is malicious if
    *any* part of it is, which is the multiple-instance-learning framing. A mean
    would reproduce the dilution failure that pooling was chosen to avoid.
    """
    chunk_scores: list[ChunkScore] = field(default_factory=list)
    tokens: list[tuple[str, float]] = field(default_factory=list)
    """Top attributed tokens from the winning window, highest weight first."""
    sentences: list[tuple[str, float]] = field(default_factory=list)
    """Each sentence of the winning window, scored on its own, in order.

    What the sanitiser cuts. Token attribution says which words drove the
    window's score, but a common word ("your") carries weight in the payload
    and also appears in the page title, which then gets cut with it. Scored
    alone, the title is 0.01 and the payload 0.99. Filled only when the window
    scored at least :data:`BENIGN_THRESHOLD`, like ``tokens``."""
    truncated: bool = False
    """Whether :data:`MAX_WINDOWS` cut the scan short."""
    ran: bool = False
    """False when Stage II was skipped (disabled, or no model available)."""

    @property
    def top(self) -> ChunkScore | None:
        return max(self.chunk_scores, key=lambda c: c.score, default=None)

    @property
    def matched_text(self) -> str:
        """The winning window, truncated - the Stage I ``matched_text`` analogue."""
        top = self.top
        return top.chunk.text[:120] if top else ""

    @property
    def span(self) -> tuple[int, int] | None:
        """Character offsets of the winning window in the original text."""
        top = self.top
        return (top.start, top.end) if top else None

    @property
    def should_block(self) -> bool:
        return self.ran and self.score >= MALICIOUS_THRESHOLD

    @property
    def is_uncertain(self) -> bool:
        return self.ran and BENIGN_THRESHOLD <= self.score < MALICIOUS_THRESHOLD

    @property
    def outcome(self) -> str:
        """Telemetry string for ``detection_results.stage_2_semantic``."""
        if not self.ran:
            return "not_run"
        if self.should_block:
            return f"block_semantic_{self.score:.2f}"
        if self.is_uncertain:
            return f"escalate_semantic_{self.score:.2f}"
        return f"pass_truncated_{self.score:.2f}" if self.truncated else "pass"


class Stage2Detector:
    """Chunk, score, take the max, attribute the winner."""

    def __init__(self, scorer: SemanticScorer) -> None:
        self.scorer = scorer

    def scan_text(self, text: str) -> Stage2Result:
        """Score one text as overlapping windows, keeping the highest."""
        if not text or not text.strip():
            return Stage2Result(ran=True)

        chunks = chunk_text(
            text,
            size=WINDOW_CHARS,
            overlap=WINDOW_OVERLAP_CHARS,
            max_chunks=MAX_WINDOWS,
        )
        if not chunks:
            return Stage2Result(ran=True)

        scores = self.scorer.score([chunk.text for chunk in chunks])
        if len(scores) != len(chunks):
            raise ModelUnavailable(
                f"scorer returned {len(scores)} scores for {len(chunks)} windows; "
                "a SemanticScorer must return one score per input, in order"
            )

        chunk_scores = [ChunkScore(chunk=c, score=float(s))
                        for c, s in zip(chunks, scores)]
        result = Stage2Result(
            score=max(cs.score for cs in chunk_scores),
            chunk_scores=chunk_scores,
            truncated=chunks[-1].end < len(text),
            ran=True,
        )

        # Attribute only the winning window. Attribution is the expensive part,
        # and only the span that drove the decision needs explaining.
        top = result.top
        if top is not None and result.score >= BENIGN_THRESHOLD:
            try:
                result.tokens = self.scorer.attribute(top.chunk.text, top_k=TOP_TOKENS)
            except NotImplementedError:
                result.tokens = []
            result.sentences = self._score_sentences(top.chunk.text)
        return result

    def _score_sentences(self, window: str) -> list[tuple[str, float]]:
        """Score each sentence of ``window`` alone, in one batch.

        A window that is a single sentence already has its score; scoring it
        again would only repeat the work.
        """
        sentences = [part.strip() for part in SENTENCE_SPLIT.split(window)
                     if part and part.strip()][:MAX_SENTENCES]
        if len(sentences) < 2:
            return []
        scores = self.scorer.score(sentences)
        if len(scores) != len(sentences):
            return []
        return [(text, float(score)) for text, score in zip(sentences, scores)]

    def scan(self, texts: Sequence[str]) -> Stage2Result:
        """Score every variant of a segment, keeping the worst outcome.

        ``texts`` is a segment's ``scannable`` list - the normalized original
        plus any payloads decoded in Phase 3 - matching the Stage I interface.
        """
        best: Stage2Result | None = None
        for text in texts:
            candidate = self.scan_text(text)
            if best is None or candidate.score > best.score:
                if best is not None:
                    candidate.truncated = candidate.truncated or best.truncated
                best = candidate
            elif candidate.truncated:
                best.truncated = True
        return best or Stage2Result(ran=True)


# --- scorers ---------------------------------------------------------------


class E5Scorer:
    """Fine-tuned ``multilingual-e5-small`` with an attention-pooling head.

    Torch and transformers are imported inside :meth:`_load`, not at module
    import, so the gateway starts without them when Stage II is disabled.
    """

    def __init__(self, model_dir: str | Path | None = None, *,
                 device: str | None = None) -> None:
        self._spec = model_dir
        self.model_dir: Path | None = None
        """The folder actually loaded from. Set on first use, because resolving
        it may mean a download, and constructing a scorer should not."""
        self._device = device
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # Imports first: they are instant, and a missing torch should not be
        # discovered after a 468 MB download.
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailable(
                "Stage II needs torch and transformers:\n"
                "  pip install torch transformers\n"
                "Or run with MOCHI_ENABLE_STAGE2=false."
            ) from exc

        self.model_dir = resolve_model_dir(self._spec)

        from mochi.detect.classifier import InjectionClassifier  # torch-dependent

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = InjectionClassifier.load(self.model_dir, device=self._device)
        self._model.eval()

    def score(self, texts: Sequence[str]) -> list[float]:
        self._load()
        assert self._torch is not None and self._model is not None
        batch = self._tokenizer(  # type: ignore[misc]
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            logits, _ = self._model(**batch)
            probabilities = self._torch.sigmoid(logits).squeeze(-1)
        return [float(p) for p in probabilities.reshape(-1).tolist()]

    def attribute(self, text: str, *,
                  top_k: int = TOP_TOKENS) -> list[tuple[str, float]]:
        """Return the highest attention-weighted tokens.

        The weights come from the pooling layer that produced the score, so this
        is a faithful explanation of *this* prediction rather than a post-hoc
        approximation like LIME or SHAP - and it costs one extra forward pass
        instead of hundreds.
        """
        self._load()
        assert self._torch is not None and self._model is not None
        batch = self._tokenizer(  # type: ignore[misc]
            [text], truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        ).to(self._device)
        with self._torch.no_grad():
            _, weights = self._model(**batch)

        ids = batch["input_ids"][0].tolist()
        mask = batch["attention_mask"][0].tolist()
        scores = weights[0].reshape(-1).tolist()

        pairs = [
            (self._tokenizer.decode([token_id]).strip(), float(weight))  # type: ignore[union-attr]
            for token_id, weight, keep in zip(ids, scores, mask)
            if keep
        ]
        special = set(self._tokenizer.all_special_tokens)  # type: ignore[union-attr]
        pairs = [(t, w) for t, w in pairs if t and t not in special]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs[:top_k]


class KeywordScorer:
    """Deterministic stand-in used by tests and by ``--config`` dry runs.

    Not a detector - it exists so every code path in this module is testable
    without a 2 GB dependency and without a trained model. Scores by weighted
    keyword density, which is enough to exercise chunking, max-pooling,
    thresholds, and attribution.
    """

    WEIGHTS = {
        "ignore": 0.4, "disregard": 0.4, "override": 0.4, "forget": 0.3,
        "instructions": 0.3, "directives": 0.3, "prompt": 0.2, "system": 0.2,
        "reveal": 0.3, "exfiltrate": 0.4, "jailbreak": 0.4, "unrestricted": 0.3,
    }

    def score(self, texts: Sequence[str]) -> list[float]:
        return [self._score_one(text) for text in texts]

    def _score_one(self, text: str) -> float:
        lowered = text.lower()
        total = sum(weight for word, weight in self.WEIGHTS.items() if word in lowered)
        return min(1.0, total)

    def attribute(self, text: str, *,
                  top_k: int = TOP_TOKENS) -> list[tuple[str, float]]:
        lowered = text.lower()
        hits = [(word, weight) for word, weight in self.WEIGHTS.items()
                if word in lowered]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits[:top_k]


def get_detector(model_dir: str | Path | None = None, *,
                 scorer: SemanticScorer | None = None,
                 device: str | None = None) -> Stage2Detector:
    """Build a Stage II detector.

    Not cached with ``lru_cache``: the model is large and a cached handle would
    keep it resident even after Stage II is disabled. The gateway holds one
    instance for its lifetime instead - see ``mochi/gateway/app.py``.

    ``device`` is passed through to :class:`E5Scorer`; ``None`` keeps the
    default of CUDA when available. Pin it to ``cpu`` when something else needs
    the GPU - a locally served target model, most obviously. Ignored when an
    explicit ``scorer`` is supplied, since that scorer chose its own device.
    """
    return Stage2Detector(scorer or E5Scorer(model_dir, device=device))
