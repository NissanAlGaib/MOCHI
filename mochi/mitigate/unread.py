"""What to do with an attachment MOCHI could not read.

An attachment can go unread for several reasons: a type nothing reads (an
executable, an archive), a corrupt file, an extraction runtime that is not
installed. Until this module, such a request was allowed - detection had read
an empty string, and an empty string is clean - and the attachment was
forwarded with nothing in the log to say so. Under ``forward_mode=inspect``
that hands the model bytes nobody checked, which is the one outcome a filter
exists to prevent.

Three policies, chosen per deployment:

``block``  refuse the request. MOCHI cannot vouch for what it did not read.
``strip``  replace the attachment with a marker and forward the rest, so the
           user's question is still answered.
``allow``  forward it unread - the old behaviour - but flagged.

``auto``, the default, picks by forwarding mode. Converting deployments
*strip*: conversion already replaces every attachment with its text, so an
unread one was never going to reach the model, and refusing would cost the
user their answer for no security gain. Inspecting deployments *block*,
because there the original bytes are what the model receives.

The policy applies only when attachment reading is on. With reading off,
every attachment is unread by the operator's choice, not by failure.

**Remote attachments are flagged, not blocked.** A URL or an uploaded-file id
is never fetched - fetching would let any caller make MOCHI issue requests to
a host of their choosing - so it is unread by design, and refusing every
request that carries an image URL would break ordinary vision applications.
It is recorded as ``attachment_remote`` and named as a limitation.
"""

from __future__ import annotations

from typing import Any

from mochi.mitigate.sanitizer import Decision, Verdict
from mochi.preprocess.extract import Unread, find_unread, strip_unread
from mochi.preprocess.flags import NormalizationFlag as F

POLICIES = ("auto", "block", "strip", "allow")


def resolve_policy(policy: str | None, forward_mode: str) -> str:
    """``block``, ``strip`` or ``allow`` for this deployment."""
    policy = (policy or "auto").strip().lower()
    if policy not in POLICIES:
        raise ValueError(f"Unreadable-attachment policy must be one of "
                         f"{', '.join(POLICIES)}; got {policy!r}")
    if policy == "auto":
        return "strip" if forward_mode == "convert" else "block"
    return policy


def enforce_unread(request: Any, verdict: Verdict, *, policy: str,
                   flags: list[str] | None = None) -> list[Unread]:
    """Apply ``policy`` to the request's unread attachments, in place.

    Turns ``verdict`` into a BLOCK under ``block``; rewrites ``request`` under
    ``strip``. Appends the telemetry flags to ``flags`` when given. A request
    already blocked is left as it is - its reason is the more useful one.
    """
    unread = find_unread(request)
    inline = [u for u in unread if u.inline]
    if flags is not None:
        if inline and F.ATTACHMENT_UNREAD.value not in flags:
            flags.append(F.ATTACHMENT_UNREAD.value)
        if len(inline) < len(unread) and F.ATTACHMENT_REMOTE.value not in flags:
            flags.append(F.ATTACHMENT_REMOTE.value)
    if not inline or verdict.blocks:
        return unread

    if policy == "block":
        named = "; ".join(f"{u.name}: {u.reason}" for u in inline[:3])
        verdict.decision = Decision.BLOCK
        verdict.reason = (
            f"Request blocked by MOCHI: {len(inline)} attachment(s) could not be "
            f"read ({named}), so they were refused rather than forwarded "
            f"unchecked. Send the content in a readable form, or ask the "
            f"operator to set MOCHI_UNREADABLE_ATTACHMENTS=strip.")
    elif policy == "strip":
        strip_unread(request)
    return unread
