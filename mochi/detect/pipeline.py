"""Detection pipeline orchestration.

Phase 4 wired segmentation and preprocessing into the request path. Phase 6
adds Stage I syntactic filtering. Stages II and III attach at the marked
points in :func:`inspect` in Phases 8 and 9.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mochi.detect.segments import InjectionClass, Segment, TrustLevel, build_segments
from mochi.detect.stage1_syntactic import (
    DEFAULT_BLOCK_SEVERITY,
    Detection,
    Stage1Result,
    get_detector,
)
from mochi.telemetry import TelemetryRecord, stage_timer


@dataclass
class InspectionResult:
    """Everything the pipeline learned about a request."""

    segments: list[Segment] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    stage1: list[tuple[Segment, Stage1Result]] = field(default_factory=list)

    # --- segment views ---

    @property
    def untrusted_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_untrusted]

    @property
    def has_untrusted_content(self) -> bool:
        """Whether this request carries any externally-sourced data at all.

        Requests without it cannot be indirect-injection vectors, which is a
        useful denominator when reporting indirect ASR in Chapter IV.
        """
        return any(s.is_untrusted for s in self.segments)

    def by_trust(self, level: TrustLevel) -> list[Segment]:
        return [s for s in self.segments if s.trust is level]

    def scannable_pairs(self) -> list[tuple[Segment, str]]:
        """Flatten to ``(segment, text)`` for every variant needing a scan."""
        return [
            (segment, text)
            for segment in self.segments
            for text in segment.scannable
        ]

    @property
    def primary_script(self) -> str | None:
        for segment in self.segments:
            if segment.normalized.script:
                return segment.normalized.script
        return None

    # --- Stage I views ---

    @property
    def top_detection(self) -> tuple[Segment, Detection] | None:
        """Most severe detection across all segments, with its origin.

        Ties break toward the earliest segment, so a request whose user input
        and retrieved document both trip the same rule attributes to the user
        input rather than arbitrarily.
        """
        best: tuple[Segment, Detection] | None = None
        for segment, result in self.stage1:
            detection = result.highest
            if detection is None:
                continue
            if best is None or detection.rank > best[1].rank:
                best = (segment, detection)
        return best

    @property
    def blocked_segments(self) -> list[Segment]:
        return [segment for segment, result in self.stage1 if result.should_block]

    @property
    def stage1_blocked(self) -> bool:
        return bool(self.blocked_segments)

    @property
    def injection_class(self) -> str | None:
        """Direct vs indirect, derived from where the detection fired."""
        top = self.top_detection
        if top is None:
            return None
        segment, _ = top
        value = segment.injection_class
        return None if value is InjectionClass.NOT_APPLICABLE else value.value


def inspect(request, record: TelemetryRecord, *,
            block_severity: str = DEFAULT_BLOCK_SEVERITY,
            enable_stage1: bool = True) -> InspectionResult:
    """Segment, preprocess, and run the detection cascade.

    Later phases extend this function in place:

    * Phase 7  - session risk accumulation
    * Phase 8  - Stage II scores segments that survive Stage I
    * Phase 9  - Stage III arbitrates the uncertain band
    * Phase 10 - enforcement uses ``segment.trust`` to choose BLOCK vs SANITIZE
    """
    segments = build_segments(request)

    aggregated_flags: list[str] = []
    for segment in segments:
        for flag in segment.normalized.flags:
            if flag not in aggregated_flags:
                aggregated_flags.append(flag)

    result = InspectionResult(segments=segments, flags=aggregated_flags)

    # --- Stage I: syntactic filtering ---
    if enable_stage1:
        detector = get_detector()
        with stage_timer(record.latency, "stage_1"):
            result.stage1 = [
                (
                    segment,
                    detector.scan(
                        segment.scannable,
                        segment.normalized.flags,
                        block_severity=block_severity,
                    ),
                )
                for segment in segments
            ]

    # --- telemetry ---
    record.normalization_flags = list(aggregated_flags)
    record.segments_inspected = [segment.source_tag for segment in segments]
    if record.payload_characteristics is not None:
        record.payload_characteristics.language = result.primary_script

    if enable_stage1:
        top = result.top_detection
        if top is None:
            record.detection_results.stage_1_syntactic = "pass"
        else:
            segment, detection = top
            prefix = "block" if result.stage1_blocked else "signal"
            record.detection_results.stage_1_syntactic = (
                f"{prefix}_{detection.detector_id}"
            )
            record.source_origin = segment.source_tag
            record.attack_type = detection.attack_type
            record.severity_level = detection.severity
            record.injection_class = result.injection_class

    return result
