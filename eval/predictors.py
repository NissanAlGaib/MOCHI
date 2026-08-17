"""Prediction adapters between evaluation samples and the MOCHI pipeline.

A predictor turns a :class:`~eval.data_loading.Sample` into a binary verdict plus
the telemetry record produced along the way, so detection quality and latency
are measured in the same pass.

:class:`MochiPredictor` is written against the pipeline's *telemetry output*
rather than against any particular stage. It therefore needs no changes as
Stages I-III land - it already reports whatever the pipeline decides, which
today is "nothing" and from Phase 6 onward will be real verdicts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from eval.data_loading import Sample
from mochi.detect import inspect
from mochi.gateway.models import ChatCompletionRequest
from mochi.telemetry import PayloadCharacteristics, TelemetryRecord

#: Substrings in a stage result that indicate the stage rejected the input.
BLOCK_MARKERS = ("block", "unsafe")


@dataclass
class Prediction:
    label: int
    """0 = predicted benign, 1 = predicted malicious."""
    record: TelemetryRecord
    latency_ms: float


class Predictor(Protocol):
    name: str

    def predict(self, sample: Sample) -> Prediction: ...


def sample_to_request(sample: Sample) -> ChatCompletionRequest:
    """Present a sample to the pipeline the way a real client would.

    ``source_tag`` is honoured so that an indirect-injection fixture exercises
    the untrusted path rather than being scored as if the user typed it.
    """
    payload = {
        "model": "eval",
        "messages": [{"role": "user", "content": sample.text}],
    }
    if sample.source_tag != "user_input":
        payload["context"] = {sample.source_tag: sample.text}
    return ChatCompletionRequest.model_validate(payload)


def _stage_blocked(record: TelemetryRecord) -> bool:
    results = record.detection_results
    values = (
        results.stage_1_syntactic,
        results.stage_2_semantic,
        results.stage_3_arbitration,
    )
    return any(
        isinstance(value, str) and any(marker in value.lower() for marker in BLOCK_MARKERS)
        for value in values
    )


class NoDefensePredictor:
    """Control condition: no inspection at all, everything passes.

    This is the thesis Table 10 "Control: No defense" row. Its recall is 0 and
    its ASR is 1.0 by construction - which is the point. Every later
    configuration is measured as a delta against it.
    """

    name = "baseline_no_defense"

    def predict(self, sample: Sample) -> Prediction:
        record = TelemetryRecord()
        record.payload_characteristics = PayloadCharacteristics.from_text(
            sample.text, include_content=False
        )
        return Prediction(label=0, record=record, latency_ms=0.0)


#: Which stages each ablation configuration turns on. This *is* Table 10's
#: ablation - a config that silently ran more stages than its name claims would
#: make the whole ablation meaningless, so the mapping is data, not branching.
STAGE_CONFIGS: dict[str, tuple[bool, bool]] = {
    # name        (stage1, stage2)
    "stage1":     (True, False),
    "stage2":     (False, True),   # Stage II alone, to isolate its contribution
    "stage12":    (True, True),
    "full":       (True, True),    # Stage III joins here in Phase 9
}


class MochiPredictor:
    """Runs the real MOCHI inspection pipeline.

    Args:
        stages: Which stages to enable, per :data:`STAGE_CONFIGS`.
        stage2: Detector to use when the config enables Stage II. Injectable so
            the ablation can be run against a stub, and so one loaded model is
            shared across an entire evaluation run instead of being rebuilt per
            sample.
    """

    def __init__(self, stages: str = "full", *, stage2=None) -> None:
        if stages not in STAGE_CONFIGS:
            raise ValueError(
                f"Unknown configuration {stages!r}. "
                f"Choose from: {', '.join(STAGE_CONFIGS)}"
            )
        self.stages = stages
        self.name = f"mochi_{stages}"
        self.enable_stage1, self.enable_stage2 = STAGE_CONFIGS[stages]

        if self.enable_stage2 and stage2 is None:
            from mochi.detect.stage2_semantic import get_detector as get_stage2

            stage2 = get_stage2()
        self.stage2 = stage2

    def predict(self, sample: Sample) -> Prediction:
        record = TelemetryRecord()
        record.payload_characteristics = PayloadCharacteristics.from_text(
            sample.text, include_content=False
        )
        request = sample_to_request(sample)

        start = time.perf_counter()
        inspect(
            request,
            record,
            enable_stage1=self.enable_stage1,
            enable_stage2=self.enable_stage2,
            stage2=self.stage2,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        record.latency.inspection_ms = round(elapsed_ms, 4)
        return Prediction(
            label=1 if _stage_blocked(record) else 0,
            record=record,
            latency_ms=elapsed_ms,
        )


PREDICTORS: dict[str, type] = {
    "baseline": NoDefensePredictor,
    **{name: MochiPredictor for name in STAGE_CONFIGS},
}


def get_predictor(config: str, *, stage2=None) -> Predictor:
    """Instantiate the predictor for an evaluation configuration."""
    if config == "baseline":
        return NoDefensePredictor()
    if config in STAGE_CONFIGS:
        return MochiPredictor(stages=config, stage2=stage2)
    raise ValueError(
        f"Unknown configuration {config!r}. Choose from: {', '.join(PREDICTORS)}"
    )
