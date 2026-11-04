"""Telemetry sink and timing helpers.

Records are written as JSON Lines (one JSON object per line) so the evaluation
harness in Phase 5 can load an entire run with ``pandas.read_json(..., lines=True)``
without any custom parsing.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from mochi.telemetry.schema import LatencyBreakdown, TelemetryRecord

logger = logging.getLogger("mochi.telemetry")

StageName = Literal["stage_1", "stage_2", "stage_3", "inspection", "upstream"]


class TelemetryWriter:
    """Append-only JSON Lines writer.

    The file handle is kept open for the process lifetime and guarded by a
    lock, since FastAPI serves requests concurrently and interleaved partial
    writes would corrupt lines.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, record: TelemetryRecord) -> None:
        line = record.to_json_line()
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@contextmanager
def stage_timer(latency: LatencyBreakdown, stage: StageName) -> Iterator[None]:
    """Time a pipeline stage and record it on ``latency``.

    Used from Phase 6 onward::

        with stage_timer(record.latency, "stage_1"):
            result = stage1.scan(text)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        setattr(latency, f"{stage}_ms", elapsed_ms)
