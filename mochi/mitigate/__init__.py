"""Mitigation and enforcement (Phase 10).

Detection decides *what* a request is; this package decides *what to do about
it*. Without it MOCHI observes attacks and forwards them anyway.
"""

from mochi.mitigate.sanitizer import (
    BLOCK_STATUS,
    REDACTION_MARKER,
    Decision,
    Verdict,
    apply,
    decide,
    enforce,
)

__all__ = [
    "BLOCK_STATUS",
    "Decision",
    "REDACTION_MARKER",
    "Verdict",
    "apply",
    "decide",
    "enforce",
]
