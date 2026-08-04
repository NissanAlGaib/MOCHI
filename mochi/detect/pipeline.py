"""Detection pipeline orchestration.

Phase 4 wires segmentation and preprocessing into the request path and records
the outcome in telemetry. Stages I-III attach to :func:`inspect` in Phases 6,
8, and 9; the segment list this produces is what they iterate over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mochi.detect.segments import InjectionClass, Segment, TrustLevel, build_segments
from mochi.telemetry import TelemetryRecord


@dataclass
class InspectionResult:
    """Everything preprocessing learned about a request."""

    segments: list[Segment] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

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
        """Flatten to ``(segment, text)`` for every variant needing a scan.

        Stage I and Stage II iterate this: one segment can yield several texts
        (original plus decoded payloads plus hidden HTML), and each must be
        scanned while staying attributable to its origin.
        """
        return [
            (segment, text)
            for segment in self.segments
            for text in segment.scannable
        ]

    @property
    def primary_script(self) -> str | None:
        """Dominant Unicode script of the first substantive segment."""
        for segment in self.segments:
            if segment.normalized.script:
                return segment.normalized.script
        return None


def inspect(request, record: TelemetryRecord) -> InspectionResult:
    """Segment, preprocess, and record. No detection decisions yet.

    Later phases extend this function in place:

    * Phase 6  - Stage I scans ``result.scannable_pairs()``
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

    # --- telemetry ---
    record.normalization_flags = list(aggregated_flags)
    record.segments_inspected = [segment.source_tag for segment in segments]
    if record.payload_characteristics is not None:
        record.payload_characteristics.language = result.primary_script

    return result
