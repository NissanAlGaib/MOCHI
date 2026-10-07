"""Mitigation and enforcement (Phase 10).

Detection decides *what* a request is; this package decides *what to do about
it*. Without it MOCHI observes attacks and forwards them anyway.
"""

from mochi.mitigate.outbound import (
    LEAK_REDACTION,
    URL_REDACTION,
    OutboundResult,
    find_leaked_sentences,
    inspect_response,
    protected_text,
    scan_completion,
)
from mochi.mitigate.sanitizer import (
    BLOCK_STATUS,
    REDACTION_MARKER,
    SESSION_ESCALATION_FLOOR,
    Decision,
    Verdict,
    apply,
    decide,
    digest_prompt,
    enforce,
)
from mochi.mitigate.url_scanner import (
    FetchMode,
    UrlFinding,
    UrlRisk,
    scan_urls,
)

__all__ = [
    "BLOCK_STATUS",
    "Decision",
    "FetchMode",
    "LEAK_REDACTION",
    "OutboundResult",
    "REDACTION_MARKER",
    "SESSION_ESCALATION_FLOOR",
    "URL_REDACTION",
    "UrlFinding",
    "UrlRisk",
    "Verdict",
    "apply",
    "decide",
    "digest_prompt",
    "enforce",
    "find_leaked_sentences",
    "inspect_response",
    "protected_text",
    "scan_completion",
    "scan_urls",
]
