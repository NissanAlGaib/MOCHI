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


class MochiPredictor:
    """Runs the real MOCHI inspection pipeline.

    Args:
        stages: Which stages to enable, for the ablation in Table 10
            ("Stage I only", "Stage I+II", "Full MOCHI"). Accepted now so the
            CLI surface is stable; enforcement arrives with the stages
            themselves in Phases 6/8/9.
    """

    def __init__(self, stages: str = "full") -> None:
        self.stages = stages
        self.name = f"mochi_{stages}"

    def predict(self, sample: Sample) -> Prediction:
        record = TelemetryRecord()
        record.payload_characteristics = PayloadCharacteristics.from_text(
            sample.text, include_content=False
        )
        request = sample_to_request(sample)

        start = time.perf_counter()
        inspect(request, record)
        elapsed_ms = (time.perf_counter() - start) * 1000

        record.latency.inspection_ms = round(elapsed_ms, 4)
        return Prediction(
            label=1 if _stage_blocked(record) else 0,
            record=record,
            latency_ms=elapsed_ms,
        )


PREDICTORS: dict[str, type] = {
    "baseline": NoDefensePredictor,
    "stage1": MochiPredictor,
    "stage12": MochiPredictor,
    "full": MochiPredictor,
}


def get_predictor(config: str) -> Predictor:
    """Instantiate the predictor for an evaluation configuration."""
    if config == "baseline":
        return NoDefensePredictor()
    if config in PREDICTORS:
        return MochiPredictor(stages=config)
    raise ValueError(
        f"Unknown configuration {config!r}. Choose from: {', '.join(PREDICTORS)}"
    )
