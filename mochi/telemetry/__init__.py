"""Structured JSON telemetry (Phase 2).

Every inspected request is scored, logged, and time-stamped regardless of the
final classification, per the thesis Attack Logging and Telemetry section.
This log file is the data source the Phase 5/13 evaluation harness reads to
produce the Chapter IV results tables.
"""

from mochi.telemetry.logger import TelemetryWriter, stage_timer
from mochi.telemetry.schema import (
    DetectionResults,
    LatencyBreakdown,
    MitigationAction,
    PayloadCharacteristics,
    SeverityLevel,
    StageOutcome,
    TelemetryRecord,
)

__all__ = [
    "DetectionResults",
    "LatencyBreakdown",
    "MitigationAction",
    "PayloadCharacteristics",
    "SeverityLevel",
    "StageOutcome",
    "TelemetryRecord",
    "TelemetryWriter",
    "stage_timer",
]
