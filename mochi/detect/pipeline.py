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
    TRUNCATION_FLAG,
    Detection,
    Stage1Result,
    get_detector,
)
from mochi.detect.stage2_semantic import (
    BENIGN_THRESHOLD,
    Stage2Detector,
    Stage2Result,
)
from mochi.telemetry import TelemetryRecord, stage_timer


@dataclass
class InspectionResult:
    """Everything the pipeline learned about a request."""

    segments: list[Segment] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    stage1: list[tuple[Segment, Stage1Result]] = field(default_factory=list)
    stage2: list[tuple[Segment, Stage2Result]] = field(default_factory=list)

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
    def stage1_truncated(self) -> bool:
        """Whether any Stage I scan ran out of budget before finishing."""
        return any(result.truncated for _, result in self.stage1)

    # --- Stage II views ---

    @property
    def top_semantic(self) -> tuple[Segment, Stage2Result] | None:
        """Highest-scoring segment, with its origin. Ties break toward the first."""
        best: tuple[Segment, Stage2Result] | None = None
        for segment, result in self.stage2:
            if not result.ran:
                continue
            if best is None or result.score > best[1].score:
                best = (segment, result)
        return best

    @property
    def semantic_score(self) -> float | None:
        top = self.top_semantic
        return top[1].score if top else None

    @property
    def stage2_blocked(self) -> bool:
        return any(result.should_block for _, result in self.stage2)

    @property
    def stage2_uncertain(self) -> bool:
        """Whether the worst segment landed in the escalation band.

        Only meaningful when nothing blocked - a request with one blocking
        segment and one uncertain segment is blocked, not uncertain.
        """
        return not self.stage2_blocked and any(
            result.is_uncertain for _, result in self.stage2
        )

    # --- combined ---

    @property
    def blocked(self) -> bool:
        """Whether any stage reached a blocking verdict."""
        return self.stage1_blocked or self.stage2_blocked

    @property
    def injection_class(self) -> str | None:
        """Direct vs indirect, derived from where the detection fired.

        Prefers Stage I's attribution when it fired, since a regex match names an
        exact span; falls back to Stage II's winning segment otherwise.

        Returns ``None`` when nothing flagged. A benign request has no injection
        class, and Stage II's top-scoring segment is only an origin if its score
        cleared the benign threshold - otherwise every clean request would be
        labelled "direct" purely because some segment had to come first.
        """
        origin = self.top_detection
        if origin is None:
            semantic = self.top_semantic
            if semantic is None or semantic[1].score < BENIGN_THRESHOLD:
                return None
            origin = semantic
        value = origin[0].injection_class
        return None if value is InjectionClass.NOT_APPLICABLE else value.value


def inspect(request, record: TelemetryRecord, *,
            block_severity: str = DEFAULT_BLOCK_SEVERITY,
            enable_stage1: bool = True,
            enable_stage2: bool = False,
            stage2: Stage2Detector | None = None) -> InspectionResult:
    """Segment, preprocess, and run the detection cascade.

    Stage II is opt-in and requires a detector instance. It is off by default so
    importing MOCHI never pulls in torch and the gateway starts without a trained
    model; ``mochi/gateway/app.py`` builds the detector once at startup when
    ``MOCHI_ENABLE_STAGE2`` is set.

    Later phases extend this function in place:

    * Phase 7  - session risk accumulation
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
        if result.stage1_truncated and TRUNCATION_FLAG not in aggregated_flags:
            aggregated_flags.append(TRUNCATION_FLAG)

    # --- Stage II: semantic detection ---
    # Fail-fast: a Stage I block is already a verdict, and Stage II is ~100x more
    # expensive. Scoring a request that is going to be rejected anyway buys
    # nothing but latency.
    if enable_stage2 and stage2 is not None and not result.stage1_blocked:
        with stage_timer(record.latency, "stage_2"):
            result.stage2 = [
                (segment, stage2.scan(segment.scannable)) for segment in segments
            ]

    # --- telemetry ---
    record.normalization_flags = list(aggregated_flags)
    record.segments_inspected = [segment.source_tag for segment in segments]
    if record.payload_characteristics is not None:
        record.payload_characteristics.language = result.primary_script

    if enable_stage1:
        top = result.top_detection
        if top is None:
            record.detection_results.stage_1_syntactic = (
                "pass_truncated" if result.stage1_truncated else "pass"
            )
        else:
            segment, detection = top
            prefix = "block" if result.stage1_blocked else "signal"
            record.detection_results.stage_1_syntactic = (
                f"{prefix}_{detection.detector_id}"
            )
            record.source_origin = segment.source_tag
            record.attack_type = detection.attack_type
            record.severity_level = detection.severity

    if result.stage2:
        top_semantic = result.top_semantic
        if top_semantic is not None:
            segment, semantic = top_semantic
            record.detection_results.stage_2_semantic = semantic.outcome
            record.detection_results.semantic_score = semantic.score
            record.detection_results.semantic_span = semantic.span
            record.detection_results.attributed_tokens = [
                token for token, _ in semantic.tokens
            ]
            # Stage I owns attribution when it fired - an exact span beats a
            # window. Stage II fills in only what Stage I left blank.
            if record.source_origin is None and semantic.score >= BENIGN_THRESHOLD:
                record.source_origin = segment.source_tag

    if enable_stage1 or result.stage2:
        record.injection_class = result.injection_class

    return result
