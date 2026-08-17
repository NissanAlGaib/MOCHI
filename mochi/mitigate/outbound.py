"""Phase 11: outbound interception. Inspecting what the LLM sends back.

Everything before this guarded one direction. A prompt injection that survives
the inbound cascade - or that was never inbound at all, because the payload
arrived through a channel MOCHI does not see - completes its objective in the
*response*. Two ways:

1. **Exfiltration.** The model emits a markdown image whose URL carries the
   secret. The client renders it, the fetch happens automatically, the data
   leaves. See :mod:`mochi.mitigate.url_scanner`.
2. **Disclosure.** The model repeats its system prompt, or the contents of a
   document the user was not entitled to see.

The same principle as inbound applies: **target the guilty part.** A response
containing one exfiltration URL and four paragraphs of correct answer should
have the URL removed and the answer delivered. Suppressing the whole response
punishes the user for the attacker's success and produces a worse outcome than
the attack itself in the common false-positive case.

Disclosure detection is deliberately *verbatim-only*. Detecting paraphrased
disclosure would need a semantic model of what each deployment considers secret,
which MOCHI does not have and cannot guess; claiming to detect it would be a
promise this code cannot keep. What it does reliably is catch the model quoting
protected text back, which is what the standard "repeat your instructions"
attack produces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mochi.detect.segments import Segment, SourceTag
from mochi.mitigate.url_scanner import UrlFinding, UrlRisk, scan_urls

#: Words per n-gram when checking for verbatim disclosure. Eight is long enough
#: that ordinary English does not collide with a system prompt by chance, and
#: short enough to catch a partial quote.
LEAK_NGRAM_WORDS = 8

#: Protected text shorter than this is not checked. A three-word system prompt
#: ("Be concise.") would otherwise match any response that happened to contain
#: the same phrase.
MIN_PROTECTED_WORDS = LEAK_NGRAM_WORDS

#: Replaces a removed URL. Inert, and says what happened.
URL_REDACTION = "[link removed: possible data exfiltration]"

#: Replaces disclosed protected text.
LEAK_REDACTION = "[removed: disclosure of protected instructions]"

_WORDS = re.compile(r"\w+")


class OutboundAction(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass
class OutboundResult:
    """What outbound inspection found and did."""

    action: OutboundAction = OutboundAction.ALLOW
    urls: list[UrlFinding] = field(default_factory=list)
    urls_removed: int = 0
    leaked_spans: int = 0
    findings: list[str] = field(default_factory=list)
    """Operator-facing summary lines, safe to log."""

    @property
    def modified(self) -> bool:
        return self.urls_removed > 0 or self.leaked_spans > 0

    @property
    def exfiltration_risk(self) -> str:
        return max(
            (f.risk for f in self.urls),
            key=lambda r: {UrlRisk.NONE: 0, UrlRisk.LOW: 1,
                           UrlRisk.MEDIUM: 2, UrlRisk.HIGH: 3}[r],
            default=UrlRisk.NONE,
        ).value


def protected_text(segments: list[Segment]) -> list[str]:
    """Text the response must not quote back.

    Only the system prompt. Retrieved documents are deliberately excluded: the
    user asked MOCHI's client to fetch them, so quoting them back is the
    application working. Treating them as secret would break every summarisation
    and RAG use case.
    """
    return [
        segment.raw_text
        for segment in segments
        if segment.source_tag == SourceTag.SYSTEM_PROMPT
        and len(_WORDS.findall(segment.raw_text)) >= MIN_PROTECTED_WORDS
    ]


def _ngrams(text: str, size: int) -> set[str]:
    words = [w.lower() for w in _WORDS.findall(text)]
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def find_leaked_sentences(response: str, protected: list[str]) -> list[str]:
    """Return response sentences that quote ``protected`` text verbatim.

    Matching is on word n-grams over case-folded, punctuation-stripped text, so
    reformatting the quote does not evade it. Whole sentences are returned
    because that is the unit redaction removes - see the Phase 10 finding that
    removing a partial span leaves the operative half behind.
    """
    if not response or not protected:
        return []

    protected_ngrams: set[str] = set()
    for text in protected:
        protected_ngrams |= _ngrams(text, LEAK_NGRAM_WORDS)
    if not protected_ngrams:
        return []

    leaked = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", response):
        if not sentence.strip():
            continue
        if _ngrams(sentence, LEAK_NGRAM_WORDS) & protected_ngrams:
            leaked.append(sentence)
    return leaked


def inspect_response(text: str, *, protected: list[str] | None = None,
                     remove_click_urls: bool = True) -> tuple[str, OutboundResult]:
    """Scan and clean one response body.

    Args:
        protected: Text the response must not quote, from :func:`protected_text`.
        remove_click_urls: Also strip medium-risk URLs that need a click. On by
            default; turning it off keeps only the auto-fetch cases, which is the
            narrower, higher-precision arm for the ablation.

    Returns the cleaned text and what was done to it.
    """
    result = OutboundResult()
    if not text:
        return text, result

    result.urls = scan_urls(text)
    threshold = UrlRisk.MEDIUM if remove_click_urls else UrlRisk.HIGH
    removable = [
        finding for finding in result.urls
        if finding.should_remove
        and (finding.risk is UrlRisk.HIGH or threshold is UrlRisk.MEDIUM)
    ]

    cleaned = text
    for finding in removable:
        if finding.raw in cleaned:
            cleaned = cleaned.replace(finding.raw, URL_REDACTION)
            result.urls_removed += 1
            result.findings.append(
                f"removed {finding.fetch_mode.value} URL ({finding.risk.value}): "
                f"{finding.reason}"
            )

    for sentence in find_leaked_sentences(cleaned, protected or []):
        if sentence in cleaned:
            cleaned = cleaned.replace(sentence, LEAK_REDACTION)
            result.leaked_spans += 1
    if result.leaked_spans:
        result.findings.append(
            f"removed {result.leaked_spans} span(s) quoting protected instructions"
        )

    result.action = OutboundAction.REDACT if result.modified else OutboundAction.ALLOW
    return cleaned, result


# --- response-envelope plumbing -------------------------------------------


def _clean_content(content: Any, **kwargs) -> tuple[Any, list[OutboundResult]]:
    """Apply :func:`inspect_response` to a message ``content`` of any shape."""
    if isinstance(content, str):
        cleaned, result = inspect_response(content, **kwargs)
        return cleaned, [result]

    if isinstance(content, list):
        results = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"], result = inspect_response(block["text"], **kwargs)
                results.append(result)
        return content, results

    return content, []


def scan_completion(payload: Any, *, protected: list[str] | None = None,
                    remove_click_urls: bool = True) -> OutboundResult:
    """Inspect and clean an OpenAI chat-completion response, in place.

    Non-dict payloads and unexpected shapes pass through untouched rather than
    raising: a provider returning something unusual should not turn a working
    request into a 500. The unmodified body is forwarded and the telemetry record
    shows ``allow``, so the gap is visible in the log.
    """
    combined = OutboundResult()
    if not isinstance(payload, dict):
        return combined

    kwargs = {"protected": protected, "remove_click_urls": remove_click_urls}
    results: list[OutboundResult] = []

    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and "content" in message:
            message["content"], sub = _clean_content(message["content"], **kwargs)
            results.extend(sub)

    for result in results:
        combined.urls.extend(result.urls)
        combined.urls_removed += result.urls_removed
        combined.leaked_spans += result.leaked_spans
        combined.findings.extend(result.findings)

    combined.action = (
        OutboundAction.REDACT if combined.modified else OutboundAction.ALLOW
    )
    return combined
