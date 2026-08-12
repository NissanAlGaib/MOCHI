"""Stage I: lightweight syntactic filtering.

Rule-based pattern matching against normalized text. Runs first and is
expected to resolve overt attacks without invoking the expensive stages -
the "fail-fast" property in the thesis architecture.

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
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_PATTERNS_PATH = Path(__file__).resolve().parents[1] / "patterns.json"

#: Ranking used to compare severities.
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}

#: Minimum severity that causes Stage I to block outright.
DEFAULT_BLOCK_SEVERITY = "high"

#: Longest text Stage I will scan. Guards the < 2ms budget against a
#: pathological payload; longer input is chunked by the caller.
MAX_SCAN_CHARS = 20_000


class PatternError(RuntimeError):
    """Raised when the pattern file is malformed."""


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
        """Telemetry string for ``detection_results.stage_1_syntactic``."""
        if not self.detections:
            return "pass"
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

    def scan_text(self, text: str) -> list[Detection]:
        """Match every pattern against one text.

        Returns at most one detection per detector, keeping the most severe -
        ten hits from one detector is not ten times the evidence, and collapsing
        them keeps telemetry readable.
        """
        if not text:
            return []
        if len(text) > MAX_SCAN_CHARS:
            text = text[:MAX_SCAN_CHARS]

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
        return list(best.values())

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
        for detection in [d for text in texts for d in self.scan_text(text)]:
            existing = best.get(detection.detector_id)
            if existing is None or detection.rank > existing.rank:
                best[detection.detector_id] = detection
        for detection in self.scan_flags(flags):
            existing = best.get(detection.detector_id)
            if existing is None or detection.rank > existing.rank:
                best[detection.detector_id] = detection

        return Stage1Result(
            detections=list(best.values()), block_severity=block_severity
        )


@lru_cache(maxsize=4)
def get_detector(patterns_path: str | None = None) -> Stage1Detector:
    """Cached detector instance - compiles patterns once per process."""
    return Stage1Detector(patterns_path)
