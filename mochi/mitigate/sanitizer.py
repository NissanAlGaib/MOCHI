"""Phase 10: enforcement. Turning detections into ALLOW / BLOCK / SANITIZE.

Until this module existed, MOCHI detected attacks and forwarded them anyway.
Stage I would return ``block_direct_prompt_injection``, telemetry would record
it, and the request went upstream regardless - an observability layer with a
security-shaped hole in it. This is what closes FR2.

The policy has one organising principle:

    **The action targets the segment that is guilty. Blocking is only correct
    when the guilty segment is the request itself.**

That single rule produces the asymmetry the thesis threat model calls for:

* A user typing "ignore your instructions" *is* the attack. There is no
  legitimate request underneath to serve, so BLOCK.
* A poisoned web page inside an otherwise legitimate "summarise this" request is
  not the user's fault. Rejecting the whole request punishes the principal for
  the attacker's content. Redact the payload and serve the real question.

For the uncertain Stage II band the asymmetry runs the other way round, and for
the same reason: a *weak* signal in untrusted content is worth acting on because
untrusted content should never carry instructions anyway, while a weak signal in
the user's own words is worth tolerating because over-blocking the principal is
a direct utility cost (see :class:`~mochi.detect.segments.TrustLevel`).

**Sanitisation that cannot find its target blocks instead.** A SANITIZE verdict
that silently failed to remove the payload would be the worst outcome available:
the log would claim mitigation while the attack went upstream intact. Redaction
is verified, and an unverifiable redaction is escalated to BLOCK. See
:func:`apply`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mochi.detect.pipeline import InspectionResult
from mochi.detect.segments import Segment, TrustLevel
from mochi.telemetry import MitigationAction

#: Replaces redacted content. Deliberately inert: no imperative verb, no
#: addressee, nothing an LLM could read as an instruction. A marker like
#: "IGNORE THE ABOVE" would itself be an injection.
REDACTION_MARKER = "[content removed by security filter]"

#: HTTP status for a blocked request. 403 rather than 400: the request is
#: well-formed, it is the content that is refused.
BLOCK_STATUS = 403

#: Shortest span worth redacting. Removing one or two characters cannot
#: neutralise a payload and only corrupts the text.
MIN_REDACTION_CHARS = 4

#: Sentence boundary. A terminator only counts when whitespace or end-of-text
#: follows it, so the dot in ``evil@example.com`` is not treated as the end of a
#: sentence - otherwise redaction stops mid-domain and forwards the exfiltration
#: address it was supposed to remove.
SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)|\n+")

#: Splits on the same boundary, for widening a token attribution to a sentence.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class Decision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"


@dataclass
class Verdict:
    """What enforcement decided, and the evidence for it."""

    decision: Decision = Decision.ALLOW
    reason: str = ""
    """Operator-facing explanation. Also the client-facing error message on BLOCK,
    so it must never quote the payload back - that would reflect attacker-
    controlled text into an error response."""
    targets: list[tuple[Segment, str]] = field(default_factory=list)
    """``(segment, text_to_remove)`` pairs for a SANITIZE."""
    redacted_origins: list[str] = field(default_factory=list)
    """Segment origins actually modified, e.g. ``["context.web_content"]``."""
    spans_removed: int = 0
    escalated: bool = False
    """True when a SANITIZE became a BLOCK because redaction could not be
    verified."""

    @property
    def action(self) -> str:
        """Value for ``record.mitigation_action_applied``."""
        return {
            Decision.ALLOW: MitigationAction.ALLOW,
            Decision.BLOCK: MitigationAction.BLOCK,
            Decision.SANITIZE: MitigationAction.SANITIZE,
        }[self.decision]

    @property
    def blocks(self) -> bool:
        return self.decision is Decision.BLOCK


def _stage1_targets(result: InspectionResult) -> list[tuple[Segment, str]]:
    """Exact matched spans from Stage I, per blocking segment.

    Stage I attribution is a precise substring, which makes it the best possible
    redaction target - far better than Stage II's window.
    """
    targets = []
    for segment, stage1 in result.stage1:
        if not stage1.should_block:
            continue
        for detection in stage1.detections:
            text = detection.matched_text
            if len(text) >= MIN_REDACTION_CHARS:
                targets.append((segment, text))
    return targets


def _stage2_targets(result: InspectionResult, *,
                    include_uncertain: bool) -> list[tuple[Segment, str]]:
    """Redaction targets derived from Stage II's span and token attribution.

    Prefers whole sentences containing an attributed token over the raw window:
    a 2,048-character window usually holds legitimate content too, and removing
    all of it destroys the answer the user asked for. Falls back to the window
    when no sentence can be isolated.
    """
    targets = []
    for segment, stage2 in result.stage2:
        if not (stage2.should_block or (include_uncertain and stage2.is_uncertain)):
            continue
        top = stage2.top
        if top is None:
            continue

        window = top.chunk.text
        tokens = [token for token, _ in stage2.tokens if len(token) >= MIN_REDACTION_CHARS]
        sentences = [
            sentence for sentence in SENTENCE_SPLIT.split(window)
            if sentence.strip() and any(t.lower() in sentence.lower() for t in tokens)
        ]
        chosen = sentences or ([window] if len(window) >= MIN_REDACTION_CHARS else [])
        targets.extend((segment, text) for text in chosen)
    return targets


def decide(result: InspectionResult, *,
           sanitize_untrusted: bool = True,
           resolve_band_by_trust: bool = True) -> Verdict:
    """Choose ALLOW, BLOCK, or SANITIZE. Pure - mutates nothing.

    Args:
        sanitize_untrusted: When False, an untrusted segment that trips a
            detector is blocked rather than redacted. Blunter and safer; useful
            as an ablation arm and for deployments that prefer refusal to
            partial answers.
        resolve_band_by_trust: Resolve Stage II's 0.45-0.55 band using source
            trust instead of escalating to a Stage III LLM. This is the
            zero-latency, deterministic alternative to arbitration - and it uses
            provenance, which is ground truth an LLM judge does not have.
    """
    # --- confident detections ---
    stage1 = _stage1_targets(result)
    stage2 = _stage2_targets(result, include_uncertain=False)
    confident = stage1 + stage2

    if confident:
        guilty = {segment.origin: segment for segment, _ in confident}
        # A detection anywhere the principal controls is a direct attack on the
        # request itself. Nothing legitimate survives redaction, so refuse.
        direct = [s for s in guilty.values() if s.trust is not TrustLevel.UNTRUSTED]
        if direct or not sanitize_untrusted:
            return Verdict(
                decision=Decision.BLOCK,
                reason=_block_reason(result, direct or list(guilty.values())),
            )
        return Verdict(
            decision=Decision.SANITIZE,
            reason=(
                "Injected instructions detected in untrusted content "
                f"({', '.join(sorted(s.source_tag for s in guilty.values()))}); "
                "the payload was removed and the remaining request forwarded."
            ),
            targets=confident,
        )

    # --- Stage II uncertain band ---
    if resolve_band_by_trust and result.stage2_uncertain:
        uncertain = _stage2_targets(result, include_uncertain=True)
        untrusted = [(seg, text) for seg, text in uncertain if seg.is_untrusted]
        if untrusted and sanitize_untrusted:
            return Verdict(
                decision=Decision.SANITIZE,
                reason=(
                    "Ambiguous content in an untrusted segment. Untrusted data "
                    "should not carry instructions, so the suspect span was "
                    "removed rather than escalated."
                ),
                targets=untrusted,
            )
        if untrusted:
            return Verdict(decision=Decision.BLOCK,
                           reason="Ambiguous content in an untrusted segment.")
        # Ambiguous, but it came from the principal. Allow and log: over-blocking
        # the user is a direct utility cost, and the turn still feeds session risk.
        return Verdict(
            decision=Decision.ALLOW,
            reason=("Ambiguous content in user input; allowed and logged. "
                    "Recorded against session risk."),
        )

    return Verdict(decision=Decision.ALLOW)


def _block_reason(result: InspectionResult, guilty: list[Segment]) -> str:
    """Client-facing message. Never quotes the payload."""
    top = result.top_detection
    kind = top[1].attack_type if top else "prompt_injection"
    where = ", ".join(sorted({segment.source_tag for segment in guilty}))
    return (
        f"Request blocked by MOCHI: {kind} detected in {where}. "
        "If this is a false positive, the request id in the response identifies "
        "the log entry."
    )


def _expand_to_sentence(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen ``[start, end)`` to the enclosing sentence.

    Redacting only the matched span is not enough, and the failure is not
    cosmetic. A detector's match is a *partial* view of the instruction: the
    pattern for "email X to Y" may match only "email the admin password", so
    removing exactly that span forwards the destination address ``Y`` to the
    model along with whatever fragments sit between two overlapping matches.
    An injected instruction is a sentence-level unit, so that is the unit
    removed.
    """
    preceding = [match.end() for match in SENTENCE_END.finditer(text, 0, start)]
    begin = preceding[-1] if preceding else 0
    while begin < start and text[begin].isspace():
        begin += 1

    following = SENTENCE_END.search(text, end)
    finish = following.end() if following else len(text)
    return begin, min(finish, len(text))


def _merge(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse overlapping or adjacent regions.

    Without this, two detectors matching different parts of one sentence would
    each redact it and produce back-to-back markers.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _redact(text: str, spans: list[str]) -> tuple[str, int]:
    """Replace the sentences containing any span with the marker.

    Returns the rewritten text and how many distinct regions were removed.
    """
    regions: list[tuple[int, int]] = []
    for span in {s for s in spans if s}:
        position = text.find(span)
        while position != -1:
            regions.append(_expand_to_sentence(text, position, position + len(span)))
            position = text.find(span, position + len(span))

    if not regions:
        return text, 0

    merged = _merge(regions)
    out: list[str] = []
    cursor = 0
    for start, end in merged:
        out.append(text[cursor:start])
        out.append(REDACTION_MARKER)
        cursor = end
    out.append(text[cursor:])
    # Expansion can leave doubled separators where a whole sentence went.
    return re.sub(r"\s{2,}", " ", "".join(out)).strip(), len(merged)


def apply(request: Any, verdict: Verdict) -> Verdict:
    """Carry out a SANITIZE verdict against ``request``, in place.

    Redaction works by substring removal across ``messages``, not by segment
    origin. That is deliberate: when a client supplies an explicit ``context``,
    that context is MOCHI-only metadata and is stripped before dispatch - the
    text that actually reaches the LLM lives in ``messages``. Rewriting the
    context would sanitise the copy nobody sees.

    **Verified, or escalated.** If a target span cannot be located in any
    message, redaction has not happened. That occurs legitimately - a payload
    recovered from base64 or folded homoglyphs in Phase 3 has no literal
    counterpart in the raw text - and in that case the only honest options are
    to refuse or to forward the attack. This escalates to BLOCK and records
    ``escalated`` so the ablation can report how often it happens.
    """
    if verdict.decision is not Decision.SANITIZE:
        return verdict

    spans = [text for _, text in verdict.targets]
    total_removed = 0

    for message in getattr(request, "messages", []) or []:
        content = message.content
        if isinstance(content, str):
            new_content, removed = _redact(content, spans)
            if removed:
                message.content = new_content
                total_removed += removed
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    new_text, removed = _redact(block["text"], spans)
                    if removed:
                        block["text"] = new_text
                        total_removed += removed

    verdict.spans_removed = total_removed
    verdict.redacted_origins = sorted({segment.origin for segment, _ in verdict.targets})

    if total_removed == 0:
        verdict.decision = Decision.BLOCK
        verdict.escalated = True
        verdict.reason = (
            "Request blocked by MOCHI: an injected payload was detected but "
            "could not be isolated for removal (it was recovered from encoded "
            "or obfuscated text), so the request was refused rather than "
            "forwarded unmodified."
        )
    return verdict


def enforce(request: Any, result: InspectionResult, *,
            sanitize_untrusted: bool = True,
            resolve_band_by_trust: bool = True) -> Verdict:
    """Decide and apply in one call. The gateway's entry point."""
    verdict = decide(
        result,
        sanitize_untrusted=sanitize_untrusted,
        resolve_band_by_trust=resolve_band_by_trust,
    )
    return apply(request, verdict)
