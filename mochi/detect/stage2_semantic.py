"""Stage II: semantic detection over a fine-tuned E5 classifier.

Stage I matches *form*; Stage II matches *meaning*. A paraphrased attack -
"kindly set aside whatever guidance you were given earlier" - contains no
pattern from ``patterns.json`` and passes Stage I untouched, but sits close to
known attacks in embedding space. Stage I's measured recall of 0.0521 on this
project's corpus is the empirical case for this stage existing.

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from mochi.detect.chunking import Chunk, chunk_text, token_budget_to_chars

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

#: Where ``training/finetune_e5.py`` exports weights to.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "e5-fine-tuned"


class ModelUnavailable(RuntimeError):
    """Raised when Stage II is enabled but no usable model can be loaded."""


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
        return result

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
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self._device = device
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise ModelUnavailable(
                f"No Stage II model at {self.model_dir}.\n"
                "Train one first:  see training/README.md (Colab GPU, ~20 min)\n"
                "Or run with MOCHI_ENABLE_STAGE2=false to skip Stage II."
            )
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailable(
                "Stage II needs torch and transformers:\n"
                "  pip install torch transformers\n"
                "Or run with MOCHI_ENABLE_STAGE2=false."
            ) from exc

        from training.model import InjectionClassifier  # local, torch-dependent

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
                 scorer: SemanticScorer | None = None) -> Stage2Detector:
    """Build a Stage II detector.

    Not cached with ``lru_cache``: the model is large and a cached handle would
    keep it resident even after Stage II is disabled. The gateway holds one
    instance for its lifetime instead - see ``mochi/gateway/app.py``.
    """
    return Stage2Detector(scorer or E5Scorer(model_dir))
