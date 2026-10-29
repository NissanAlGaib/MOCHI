"""Wrap an OpenAI-compatible client so every call is inspected first.

The whole mechanism in one paragraph: ``client.chat.completions.create(...)``
is an ordinary Python function that builds JSON and posts it. Replacing that
attribute with a function that inspects the arguments, then calls the original,
puts MOCHI in the path with no network hop and no second process.

What a guarded call does, in the gateway's order
------------------------------------------------
Deliberately the same order ``mochi/gateway/app.py`` uses, because a verdict
that depended on which surface was used would make the evaluation meaningless:

1. Segment and inspect the messages.
2. Enforce - ALLOW, SANITIZE or BLOCK. A BLOCK raises :class:`MochiBlocked`;
   a SANITIZE rewrites the messages in place before they are sent.
3. Convert attachments to text, when configured to, *after* enforcement so a
   redacted request stays redacted.
4. Call the real ``create``.
5. Inspect the response and redact what leaks.

Streaming
---------
Refused by default, as the gateway refuses it, and for the same reason: an
exfiltration URL can straddle two chunks, so outbound inspection needs the
whole body. ``allow_buffered_streaming=True`` requests the complete response
and hands it back in one piece - not incremental delivery, and the error says
so rather than letting a caller discover it.

Async
-----
``AsyncOpenAI`` is wrapped too. Inspection is CPU-bound and synchronous, so an
async call is not made concurrent by this - it is awaited around the same
work. That is honest rather than ideal: making Stage II async would mean a
thread pool and a queue, which is the gateway's job, and an application that
needs it should use the gateway.
"""

from __future__ import annotations

import inspect as _inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from mochi.detect import inspect as inspect_request
from mochi.detect.pipeline import InspectionResult
from mochi.gateway.models import ChatCompletionRequest
from mochi.mitigate import Decision, Verdict, apply, decide
from mochi.mitigate.outbound import OutboundResult, protected_text, scan_completion
from mochi.telemetry import TelemetryRecord


class MochiBlocked(Exception):
    """Raised instead of sending a request MOCHI refused.

    An exception rather than a sentinel return value, because a caller that
    ignores a return value sends the attack anyway, and the whole point of
    this surface is that forgetting to check is not possible. The reason is
    the same string the proxy puts in its 403 body, so a message a developer
    sees here matches what they would have seen there.
    """

    def __init__(self, reason: str, *, request_id: str | None = None,
                 verdict: Verdict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.request_id = request_id
        """Ties this refusal to its telemetry record, as the proxy's
        ``X-Mochi-Request-Id`` does."""

        self.verdict = verdict
        """The full decision, for a caller that wants to know which segment
        was blamed rather than only that something was."""


@dataclass
class GuardedCall:
    """What MOCHI did to one call. Attached to the response as
    ``response._mochi`` when the response object allows attributes, and
    always available on ``client.mochi_last``."""

    request_id: str | None = None
    decision: str = "allow"
    reason: str = ""
    spans_redacted: int = 0
    attachments_converted: int = 0
    outbound_modified: bool = False
    outbound_findings: list[str] = field(default_factory=list)
    inspection: InspectionResult | None = None
    record: TelemetryRecord | None = None


@dataclass
class _Config:
    """Settings for a guarded client. Mirrors the gateway's, minus anything
    about serving: there is no host, port, or upstream address here, because
    the wrapped client already has them."""

    system_prompt: str | None = None
    context: dict[str, Any] | None = None
    enable_stage2: bool = True
    stage2_model_dir: str | None = None
    stage2_device: str = "cpu"
    read_attachments: bool = False
    forward_mode: str = "inspect"
    sanitize_untrusted: bool = True
    resolve_band_by_trust: bool = True
    enforce_on_trusted: bool = False
    enable_outbound: bool = True
    outbound_remove_click_urls: bool = True
    allow_buffered_streaming: bool = False
    on_verdict: Callable[[GuardedCall], None] | None = None


_DETECTOR_CACHE: dict[tuple[str | None, str], Any] = {}


def _stage2(config: _Config):
    """The Stage II detector, built once per (weights, device).

    Lazy, so importing ``mochi.client`` does not pull in torch. A developer
    who wraps a client at import time should not pay a 2 GB import before
    their first call, and a deployment that only ever runs Stage I should not
    pay it at all.
    """
    if not config.enable_stage2:
        return None
    key = (config.stage2_model_dir, config.stage2_device)
    if key not in _DETECTOR_CACHE:
        from mochi.detect.stage2_semantic import get_detector

        _DETECTOR_CACHE[key] = get_detector(config.stage2_model_dir,
                                            device=config.stage2_device)
    return _DETECTOR_CACHE[key]


def _build_request(kwargs: dict[str, Any], config: _Config) -> ChatCompletionRequest:
    """The call's arguments as the same model the gateway validates.

    Reusing ``ChatCompletionRequest`` is what makes the two surfaces agree:
    segmentation reads roles off this object, so a wrapper that invented its
    own shape could tag a segment differently and reach a different verdict on
    identical input.
    """
    payload: dict[str, Any] = {
        "model": kwargs.get("model"),
        "messages": kwargs.get("messages") or [],
    }
    if config.context is not None:
        payload["context"] = config.context
    if kwargs.get("session_id"):
        payload["session_id"] = kwargs["session_id"]
    return ChatCompletionRequest.model_validate(payload)


def inspect_messages(messages: list[dict], *, config: _Config | None = None,
                     **overrides: Any) -> tuple[Verdict, InspectionResult,
                                                TelemetryRecord]:
    """Run the pipeline over a message list and return the verdict.

    Exposed because it is useful on its own - a caller that wants to check a
    prompt without sending it anywhere, or to test a policy change, should not
    have to construct a client to do it.
    """
    settings = config or _Config(**overrides)
    request = _build_request({"messages": messages}, settings)
    record = TelemetryRecord()
    result = inspect_request(
        request, record,
        enable_stage1=True,
        enable_stage2=settings.enable_stage2,
        stage2=_stage2(settings),
        read_images=settings.read_attachments,
    )
    verdict = decide(
        result,
        sanitize_untrusted=settings.sanitize_untrusted,
        resolve_band_by_trust=settings.resolve_band_by_trust,
        enforce_on_trusted=settings.enforce_on_trusted,
    )
    return verdict, result, record


def _guard_kwargs(kwargs: dict[str, Any], config: _Config) -> GuardedCall:
    """Inspect and enforce, mutating ``kwargs['messages']`` on SANITIZE.

    Raises :class:`MochiBlocked` rather than returning, so a caller cannot
    proceed by ignoring the result.
    """
    if config.system_prompt and kwargs.get("messages"):
        messages = list(kwargs["messages"])
        if not messages or messages[0].get("role") not in ("system", "developer"):
            messages.insert(0, {"role": "system",
                                "content": config.system_prompt})
            kwargs["messages"] = messages

    request = _build_request(kwargs, config)
    record = TelemetryRecord()
    result = inspect_request(
        request, record,
        enable_stage1=True,
        enable_stage2=config.enable_stage2,
        stage2=_stage2(config),
        read_images=config.read_attachments,
    )
    verdict = decide(
        result,
        sanitize_untrusted=config.sanitize_untrusted,
        resolve_band_by_trust=config.resolve_band_by_trust,
        enforce_on_trusted=config.enforce_on_trusted,
    )

    call = GuardedCall(
        request_id=record.request_id,
        decision=verdict.decision.value,
        reason=verdict.reason,
        inspection=result,
        record=record,
    )

    if verdict.decision is Decision.SANITIZE:
        # apply() mutates the request in place and may escalate to BLOCK when
        # the payload cannot actually be removed - an unverifiable redaction
        # is worse than a refusal, because the log would claim mitigation
        # while the attack went upstream intact.
        apply(request, verdict)
        call.decision = verdict.decision.value
        call.reason = verdict.reason
        call.spans_redacted = len(verdict.targets)

    if verdict.decision is Decision.BLOCK:
        call.decision = "block"
        call.reason = verdict.reason
        if config.on_verdict:
            config.on_verdict(call)
        raise MochiBlocked(verdict.reason, request_id=record.request_id,
                           verdict=verdict)

    # Attachments become text after enforcement, so a redacted request stays
    # redacted, and after the block check, so nothing is transcribed for a
    # request that is not being sent. Same order as the gateway.
    if config.forward_mode == "convert" and config.read_attachments:
        from mochi.preprocess.extract import convert_attachments

        call.attachments_converted = convert_attachments(request)

    # Hand back the possibly-rewritten messages. model_dump rather than the
    # original list, because apply() and convert_attachments() both edited the
    # validated object rather than the caller's dicts.
    kwargs["messages"] = [
        message.model_dump(exclude_none=True) if hasattr(message, "model_dump")
        else message
        for message in request.messages
    ]
    return call


def _guard_response(response: Any, call: GuardedCall, config: _Config) -> Any:
    """Inspect what came back, and redact what leaks."""
    if not config.enable_outbound or call.inspection is None:
        return response

    # The SDK returns a pydantic model; scan_completion works on the dict
    # shape the proxy sees. Converting, scanning and writing back keeps one
    # implementation rather than two that can drift apart.
    payload = response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
    if not isinstance(payload, dict):
        return response

    outbound: OutboundResult = scan_completion(
        payload,
        protected=protected_text(call.inspection.segments),
        remove_click_urls=config.outbound_remove_click_urls,
    )
    call.outbound_modified = outbound.modified
    call.outbound_findings = list(outbound.findings)

    if not outbound.modified:
        return response

    # Write the cleaned text back onto the original object so the caller keeps
    # the SDK type they expect, rather than being handed a dict.
    try:
        for index, choice in enumerate(payload.get("choices") or []):
            cleaned = (choice.get("message") or {}).get("content")
            target = response.choices[index].message
            if cleaned is not None and getattr(target, "content", None) != cleaned:
                target.content = cleaned
        return response
    except (AttributeError, IndexError, TypeError):
        # An unexpected response shape should not turn a working call into a
        # crash. The cleaned dict is still correct, so return that.
        return payload


def _wrap_create(original: Callable[..., Any], config: _Config,
                 owner: Any) -> Callable[..., Any]:
    """Return a replacement for ``create`` that inspects around it."""

    def _check_stream(kwargs: dict[str, Any]) -> None:
        if kwargs.get("stream") and not config.allow_buffered_streaming:
            raise MochiBlocked(
                "Streaming is not supported by default: outbound inspection "
                "needs the complete response body, and an exfiltration URL "
                "can straddle two chunks. Pass allow_buffered_streaming=True "
                "to receive the whole response as one piece - that is not "
                "incremental streaming.")
        kwargs.pop("stream", None)

    if _inspect.iscoroutinefunction(original):
        async def guarded_async(*args: Any, **kwargs: Any) -> Any:
            _check_stream(kwargs)
            call = _guard_kwargs(kwargs, config)
            response = await original(*args, **kwargs)
            response = _guard_response(response, call, config)
            owner.mochi_last = call
            if config.on_verdict:
                config.on_verdict(call)
            return response

        return guarded_async

    def guarded(*args: Any, **kwargs: Any) -> Any:
        _check_stream(kwargs)
        call = _guard_kwargs(kwargs, config)
        response = original(*args, **kwargs)
        response = _guard_response(response, call, config)
        owner.mochi_last = call
        if config.on_verdict:
            config.on_verdict(call)
        return response

    return guarded


def guard(client: Any, **settings: Any) -> Any:
    """Wrap ``client`` so every chat completion is inspected first.

    Returns the same object, with ``chat.completions.create`` replaced. The
    client is mutated rather than copied: a copy would let a reference taken
    before the call escape inspection, which is exactly the bypass this
    surface is already weakest against.

    Args:
        client: An OpenAI-compatible client, sync or async. Anything exposing
            ``chat.completions.create`` works - the OpenAI SDK, Azure's, or a
            stub in a test.
        system_prompt: Prepended when the caller does not supply one.
        context: Tagged context, for an application that knows a message holds
            retrieved content. Without it, trust is inferred from chat roles,
            which tags everything in the user's turn as semi-trusted - so a
            pasted document is treated as the user's own words and refused
            rather than redacted. Supplying this is what makes sanitisation
            possible, and is the single highest-value thing an integration can
            do.
        read_attachments: Extract text from images, audio and video.
        forward_mode: ``inspect`` or ``convert``.
        enable_stage2: Off disables the semantic stage, and with it the torch
            import.
        enable_outbound: Inspect responses as well as requests.
        allow_buffered_streaming: Accept ``stream=True`` by buffering.
        on_verdict: Called with the :class:`GuardedCall` after every call,
            including a blocked one. For logging without wrapping again.

    Raises:
        TypeError: If the client has no ``chat.completions.create``, which
            means it is not the kind of object this can guard - said plainly
            rather than failing at the first call.
    """
    unknown = set(settings) - {f for f in _Config.__dataclass_fields__}
    if unknown:
        raise TypeError(
            f"guard() got unexpected setting(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(sorted(_Config.__dataclass_fields__))}")
    config = _Config(**settings)

    try:
        completions = client.chat.completions
        original = completions.create
    except AttributeError as exc:
        raise TypeError(
            f"{type(client).__name__} has no chat.completions.create, so it "
            f"cannot be guarded. guard() expects an OpenAI-compatible client."
        ) from exc

    if getattr(original, "__mochi_guarded__", False):
        return client

    guarded = _wrap_create(original, config, client)
    guarded.__mochi_guarded__ = True          # type: ignore[attr-defined]
    guarded.__wrapped__ = original            # type: ignore[attr-defined]
    completions.create = guarded              # type: ignore[assignment]
    client.mochi_config = config
    client.mochi_last = None
    return client


def unguard(client: Any) -> Any:
    """Undo :func:`guard`. Mainly for tests, which should not leak a wrapped
    client into the next one."""
    try:
        completions = client.chat.completions
        original = getattr(completions.create, "__wrapped__", None)
    except AttributeError:
        return client
    if original is not None:
        completions.create = original
    return client
