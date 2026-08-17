"""Stage I: lightweight syntactic filtering.

Rule-based pattern matching against normalized text. Runs first and is
expected to resolve overt attacks without invoking the expensive stages -
the "fail-fast" property in the thesis architecture.

Cost scales with input length at roughly :data:`MS_PER_KILOCHAR`. Ordinary
prompts are sub-millisecond; long documents are bounded by
:data:`MAX_TOTAL_SCAN_CHARS`. Any claim of a flat per-request budget would be
true only for short input.

Two design choices differ from a literal reading of Table 8, both driven by
false-positive cost:

1. **Severity tiers.** Only ``high`` patterns block at Stage I. ``medium`` and
   ``low`` matches are recorded as signals and escalated, because several
   Table 8 patterns (bare ``\\bOR\\b``, ``\\bAPI_KEY\\b``, ``\\bDAN\\b``,
   ``\\bbash\\b``) match ordinary English and would swamp the FPR < 1% target.

2. **Tightened context.** ``\\bignore\\s+previous\\b`` becomes
   ``ignore ... previous <instruction|rule|...>``. The attack always names what
   it is overriding; the bare form matches "ignore previous emails".

Patterns live in ``mochi/patterns.json`` so they can be revised without code
changes, per the deployment configuration in the thesis.

Long input is scanned as **overlapping windows**, not truncated. An earlier
version cut every text to the first 20,000 characters, which silently hid any
injection past that point - the same tail-truncation failure that the
signal-position audit flagged for Stage II, one stage earlier than anyone was
looking for it. Text past :data:`MAX_TOTAL_SCAN_CHARS` is still dropped, but
that case is now reported rather than passing as a clean scan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from mochi.detect.chunking import chunk_count_for, chunk_text

DEFAULT_PATTERNS_PATH = Path(__file__).resolve().parents[1] / "patterns.json"

#: Ranking used to compare severities.
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}

#: Minimum severity that causes Stage I to block outright.
DEFAULT_BLOCK_SEVERITY = "high"

#: Size of one regex window. Bounds the cost of a single ``re.search`` against a
#: pathological payload; text longer than this is scanned as overlapping windows
#: rather than truncated.
MAX_SCAN_CHARS = 20_000

#: Measured cost of the full pattern set, ~0.5 ms per 1,000 characters (47
#: patterns, ~0.2 ms each per 20,000-char window, flat - no single pattern
#: dominates). A typical 500-char prompt therefore costs well under a
#: millisecond, which is what the 0.430 ms mean / 0.883 ms p95 figures reflect.
#: Long documents cost proportionally more, which is what
#: :data:`MAX_TOTAL_SCAN_CHARS` bounds.
MS_PER_KILOCHAR = 0.5

#: Characters shared between adjacent windows. Must exceed the longest possible
#: single match so a payload straddling a boundary is still seen whole. The
#: longest bounded gap in ``patterns.json`` is 40 characters, and matched text is
#: reported truncated to 120; 512 is comfortably above both.
SCAN_OVERLAP_CHARS = 512

#: Hard ceiling on total characters scanned per text. A 10 MB document would
#: otherwise cost ~500 windows and blow the latency budget.
#:
#: Set to 100,000 (~25,000 tokens) because at :data:`MS_PER_KILOCHAR` that
#: bounds Stage I's worst case near 50 ms - the same order as the Stage II
#: budget, so a pathological document cannot make Stage I the dominant cost.
#: Coverage is generous: 100,000 characters exceeds virtually any realistic
#: retrieved document or web page.
#:
#: Reaching this limit is reported via :attr:`Stage1Result.truncated` and the
#: ``pass_truncated`` outcome rather than passing silently - the distinction
#: between "scanned and clean" and "gave up early" must survive into telemetry,
#: because only one of them is evidence of safety.
MAX_TOTAL_SCAN_CHARS = 100_000

#: Synthetic flag raised when :data:`MAX_TOTAL_SCAN_CHARS` cut a scan short.
TRUNCATION_FLAG = "stage1_scan_truncated"


class PatternError(RuntimeError):
    """Raised when the pattern file is malformed."""


@dataclass(frozen=True)
class ScanCoverage:
    """How much of a text Stage I actually looked at."""

    chars: int
    windows: int
    truncated: bool


@dataclass(frozen=True)
class Detection:
    """One pattern or flag hit."""

    detector_id: str
    attack_type: str
    severity: str
    evidence: str
    """The pattern (or flag name) that fired."""
    matched_text: str = ""
    """The substring that matched, truncated. Empty for flag detectors."""

    @property
    def rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 0)


@dataclass
class Stage1Result:
    """Aggregate outcome of a Stage I scan."""

    detections: list[Detection] = field(default_factory=list)
    block_severity: str = DEFAULT_BLOCK_SEVERITY
    truncated: bool = False
    """Whether any scanned text exceeded :data:`MAX_TOTAL_SCAN_CHARS`.

    A clean result with ``truncated`` set is *not* evidence that the text is
    safe, only that Stage I ran out of budget. Callers that treat ``pass`` as
    safe must check this.
    """
    chars_scanned: int = 0
    windows_scanned: int = 0

    @property
    def matched(self) -> bool:
        return bool(self.detections)

    @property
    def highest(self) -> Detection | None:
        return max(self.detections, key=lambda d: d.rank, default=None)

    @property
    def should_block(self) -> bool:
        top = self.highest
        threshold = SEVERITY_ORDER[self.block_severity]
        return top is not None and top.rank >= threshold

    @property
    def outcome(self) -> str:
        """Telemetry string for ``detection_results.stage_1_syntactic``.

        A truncated clean scan reports ``pass_truncated`` rather than ``pass``,
        so the log never claims coverage it did not have.
        """
        if not self.detections:
            return "pass_truncated" if self.truncated else "pass"
        top = self.highest
        assert top is not None
        return f"{'block' if self.should_block else 'signal'}_{top.detector_id}"

    @property
    def attack_type(self) -> str | None:
        top = self.highest
        return top.attack_type if top else None

    @property
    def severity(self) -> str | None:
        top = self.highest
        return top.severity if top else None


@dataclass(frozen=True)
class _CompiledPattern:
    detector_id: str
    attack_type: str
    severity: str
    source: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class _FlagRule:
    detector_id: str
    attack_type: str
    severity: str
    flag: str


class Stage1Detector:
    """Compiled pattern set with a scan interface.

    Patterns are compiled once at construction; :func:`get_detector` caches the
    instance so per-request cost is matching only.
    """

    def __init__(self, patterns_path: str | Path | None = None) -> None:
        self.path = Path(patterns_path or DEFAULT_PATTERNS_PATH)
        self._patterns: list[_CompiledPattern] = []
        self._flag_rules: dict[str, _FlagRule] = {}
        self._load()

    def _load(self) -> None:
        try:
            spec = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PatternError(f"Pattern file not found: {self.path}") from exc
        except json.JSONDecodeError as exc:
            raise PatternError(f"Malformed pattern file {self.path}: {exc}") from exc

        for detector in spec.get("detectors", []):
            detector_id = detector["id"]
            attack_type = detector.get("attack_type", detector_id)
            kind = detector.get("type", "regex")

            if kind == "regex":
                for entry in detector.get("patterns", []):
                    flags = 0 if entry.get("case_sensitive") else re.IGNORECASE
                    try:
                        compiled = re.compile(entry["regex"], flags)
                    except re.error as exc:
                        raise PatternError(
                            f"Invalid regex in {detector_id}: {entry['regex']!r} ({exc})"
                        ) from exc
                    self._patterns.append(
                        _CompiledPattern(
                            detector_id=detector_id,
                            attack_type=attack_type,
                            severity=entry.get("severity", "medium"),
                            source=entry["regex"],
                            regex=compiled,
                        )
                    )
            elif kind == "normalization_flag":
                for entry in detector.get("flags", []):
                    self._flag_rules[entry["flag"]] = _FlagRule(
                        detector_id=detector_id,
                        attack_type=attack_type,
                        severity=entry.get("severity", "low"),
                        flag=entry["flag"],
                    )
            else:
                raise PatternError(f"Unknown detector type {kind!r} in {detector_id}")

        if not self._patterns and not self._flag_rules:
            raise PatternError(f"{self.path} defines no detectors")

    # --- introspection, used by tests and reporting ---

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def detector_ids(self) -> list[str]:
        ids = [p.detector_id for p in self._patterns]
        ids += [r.detector_id for r in self._flag_rules.values()]
        return sorted(set(ids))

    # --- scanning ---

    def _scan_window(self, text: str) -> dict[str, Detection]:
        """Match every pattern against one window, most severe hit per detector."""
        best: dict[str, Detection] = {}
        for pattern in self._patterns:
            match = pattern.regex.search(text)
            if match is None:
                continue
            detection = Detection(
                detector_id=pattern.detector_id,
                attack_type=pattern.attack_type,
                severity=pattern.severity,
                evidence=pattern.source,
                matched_text=match.group(0)[:120],
            )
            existing = best.get(pattern.detector_id)
            if existing is None or detection.rank > existing.rank:
                best[pattern.detector_id] = detection
        return best

    def scan_text(self, text: str) -> tuple[list[Detection], ScanCoverage]:
        """Match every pattern against one text, in overlapping windows.

        Returns at most one detection per detector, keeping the most severe -
        ten hits from one detector is not ten times the evidence, and collapsing
        them keeps telemetry readable.

        Text longer than :data:`MAX_SCAN_CHARS` is scanned as overlapping
        windows rather than truncated to the first window. Truncation only
        happens past :data:`MAX_TOTAL_SCAN_CHARS`, and is reported in the
        returned :class:`ScanCoverage` so it cannot pass unnoticed.
        """
        if not text:
            return [], ScanCoverage(chars=0, windows=0, truncated=False)

        window_cap = max(
            1, chunk_count_for(min(len(text), MAX_TOTAL_SCAN_CHARS),
                               size=MAX_SCAN_CHARS, overlap=SCAN_OVERLAP_CHARS)
        )
        chunks = chunk_text(
            text,
            size=MAX_SCAN_CHARS,
            overlap=SCAN_OVERLAP_CHARS,
            max_chunks=window_cap,
        )

        best: dict[str, Detection] = {}
        for chunk in chunks:
            for detector_id, detection in self._scan_window(chunk.text).items():
                existing = best.get(detector_id)
                if existing is None or detection.rank > existing.rank:
                    best[detector_id] = detection

        scanned_to = chunks[-1].end if chunks else 0
        coverage = ScanCoverage(
            chars=scanned_to,
            windows=len(chunks),
            truncated=scanned_to < len(text),
        )
        return list(best.values()), coverage

    def scan_flags(self, flags: Iterable[str]) -> list[Detection]:
        """Convert Phase 3 normalization flags into detections."""
        best: dict[str, Detection] = {}
        for flag in flags:
            rule = self._flag_rules.get(flag)
            if rule is None:
                continue
            detection = Detection(
                detector_id=rule.detector_id,
                attack_type=rule.attack_type,
                severity=rule.severity,
                evidence=flag,
            )
            existing = best.get(rule.detector_id)
            if existing is None or detection.rank > existing.rank:
                best[rule.detector_id] = detection
        return list(best.values())

    def scan(self, texts: Sequence[str], flags: Iterable[str] = (),
             *, block_severity: str = DEFAULT_BLOCK_SEVERITY) -> Stage1Result:
        """Scan every text variant plus the normalization flags.

        ``texts`` is a segment's ``scannable`` list: the normalized original
        plus any decoded payloads recovered in Phase 3.
        """
        best: dict[str, Detection] = {}
        chars = windows = 0
        truncated = False

        for text in texts:
            detections, coverage = self.scan_text(text)
            chars += coverage.chars
            windows += coverage.windows
            truncated = truncated or coverage.truncated
            for detection in detections:
                existing = best.get(detection.detector_id)
                if existing is None or detection.rank > existing.rank:
                    best[detection.detector_id] = detection

        # Truncation is a coverage gap, not evidence of an attack, so it is not
        # folded into ``detections`` - doing so would flag every large benign
        # document and cost precision. It surfaces via ``truncated`` and the
        # ``pass_truncated`` outcome, and the pipeline records
        # :data:`TRUNCATION_FLAG` in telemetry.
        for detection in self.scan_flags(flags):
            existing = best.get(detection.detector_id)
            if existing is None or detection.rank > existing.rank:
                best[detection.detector_id] = detection

        return Stage1Result(
            detections=list(best.values()),
            block_severity=block_severity,
            truncated=truncated,
            chars_scanned=chars,
            windows_scanned=windows,
        )


@lru_cache(maxsize=4)
def get_detector(patterns_path: str | None = None) -> Stage1Detector:
    """Cached detector instance - compiles patterns once per process."""
    return Stage1Detector(patterns_path)
