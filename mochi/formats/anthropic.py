"""Anthropic Messages API - ``client.messages.create`` and ``POST /v1/messages``.

Mapping to the chat shape, and why each piece lands where it does:

- ``system`` (a string or text blocks) is the system message.
- ``tool_result`` blocks travel inside a *user* message in this API, but they
  are tool output, the main indirect-injection channel. Each becomes a chat
  ``tool`` message, so it is tagged ``api_response`` and untrusted - redacted
  rather than refused - exactly as an OpenAI tool message is. Read as the
  user's turn, a poisoned search result would be treated as the user's own
  words and the whole request refused.
- ``image`` and ``document`` blocks become the chat attachment blocks the
  extractors read. A URL source is never fetched.
- ``tool_use`` becomes ``tool_calls``. Not inspected, as in the chat path:
  it is the model's own earlier output, carried forward.
- ``thinking`` blocks are skipped: signed model output, which cannot be
  edited without invalidating the signature, and which a provider of another
  format could not accept anyway.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from mochi.formats.base import (
    DEFAULT_MAX_TOKENS,
    Binding,
    ChatView,
    FormatError,
    WireFormat,
    arguments,
    attachment_block,
    bound_attachment,
    bound_text,
    get,
    item_setter,
    parse_arguments,
    put,
    remote_block,
    setter,
    split_data_url,
    text_of,
)

STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                "tool_use": "tool_calls", "refusal": "content_filter",
                "pause_turn": "stop"}
FINISH_REASONS = {"stop": "end_turn", "length": "max_tokens",
                  "tool_calls": "tool_use", "content_filter": "refusal"}


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


class AnthropicFormat(WireFormat):
    name = "anthropic"

    # -- requests ------------------------------------------------------------

    def to_chat(self, body: dict) -> ChatView:
        view = ChatView()
        system = get(body, "system")
        if isinstance(system, str):
            view.add("system", [bound_text(system, setter(body, "system"))])
        elif isinstance(system, list):
            view.add("system", self._blocks(system))

        for message in get(body, "messages", default=[]) or []:
            role = "assistant" if get(message, "role") == "assistant" else "user"
            content = get(message, "content")
            if isinstance(content, str):
                view.add(role, [bound_text(content, setter(message, "content"))])
                continue
            pending: list[tuple[dict, Binding]] = []
            calls: list[dict] = []
            for index, block in enumerate(content or []):
                kind = get(block, "type")
                if kind == "tool_result":
                    view.add(role, pending)          # keep turn order
                    pending = []
                    view.add("tool", self._tool_result(block),
                             tool_call_id=get(block, "tool_use_id"))
                elif kind == "tool_use":
                    calls.append({"id": get(block, "id"), "type": "function",
                                  "function": {"name": get(block, "name"),
                                               "arguments": arguments(get(block, "input"))}})
                else:
                    pending += self._blocks([block], container=content, start=index)
            view.add(role, pending, tool_calls=calls or None)
        return view

    def _blocks(self, blocks: list, *, container: list | None = None,
                start: int = 0) -> list[tuple[dict, Binding]]:
        """Text, image and document blocks, each bound back to its place."""
        container = blocks if container is None else container
        out: list[tuple[dict, Binding]] = []
        for offset, block in enumerate(blocks):
            index = start + offset
            kind = get(block, "type")
            if kind == "text" and isinstance(get(block, "text"), str):
                out.append(bound_text(get(block, "text"), setter(block, "text")))
            elif kind in ("image", "document"):
                out.append(self._attachment(block, container, index))
        return out

    def _attachment(self, block: Any, container: list, index: int) -> tuple[dict, Binding]:
        source = get(block, "source", default={})
        kind = get(source, "type")
        replace = _replacer(container, index)
        if kind == "base64":
            name = get(block, "title")
            return bound_attachment(
                attachment_block(get(source, "media_type"), get(source, "data"),
                                 filename=name), replace)
        if kind == "text":                       # a plain-text document
            return bound_text(get(source, "data", default=""), setter(source, "data"))
        if kind == "content":                    # a document of text blocks
            parts = [b for b in get(source, "content", default=[]) or []
                     if get(b, "type") == "text"]
            if parts:
                block_text = parts[0]
                return bound_text(get(block_text, "text", default=""),
                                  setter(block_text, "text"))
        uri = get(source, "url") or get(source, "file_id") or "attachment"
        mime = "image/*" if get(block, "type") == "image" else None
        return bound_attachment(remote_block(mime, uri), replace)

    def _tool_result(self, block: Any) -> list[tuple[dict, Binding]]:
        content = get(block, "content")
        if isinstance(content, str):
            return [bound_text(content, setter(block, "content"))]
        if isinstance(content, list):
            return self._blocks(content)
        return []

    def chat_params(self, body: dict) -> dict:
        params: dict[str, Any] = {"max_tokens": get(body, "max_tokens")}
        for name in ("temperature", "top_p"):
            params[name] = get(body, name)
        if get(body, "stop_sequences"):
            params["stop"] = list(get(body, "stop_sequences"))
        tools = get(body, "tools")
        if tools:
            params["tools"] = [{"type": "function", "function": {
                "name": get(t, "name"), "description": get(t, "description", default=""),
                "parameters": get(t, "input_schema", default={"type": "object"})}}
                for t in tools if get(t, "name")]
        choice = get(body, "tool_choice")
        if choice:
            params["tool_choice"] = {
                "auto": "auto", "any": "required", "none": "none",
            }.get(get(choice, "type"), {"type": "function",
                                        "function": {"name": get(choice, "name")}})
        return {k: v for k, v in params.items() if v is not None}

    def from_chat(self, payload: dict) -> dict:
        """A chat request as a Messages request, for an Anthropic upstream."""
        system: list[str] = []
        messages: list[dict] = []

        def push(role: str, blocks: list[dict]) -> None:
            if not blocks:
                return
            if messages and messages[-1]["role"] == role:   # roles must alternate
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": blocks})

        for message in payload.get("messages") or []:
            role, content = message.get("role"), message.get("content")
            if role in ("system", "developer"):
                system.append(text_of(content))
            elif role in ("tool", "function"):
                push("user", [{"type": "tool_result",
                               "tool_use_id": message.get("tool_call_id") or "",
                               "content": text_of(content)}])
            elif role == "assistant":
                blocks = [_text_block(text_of(content))] if text_of(content) else []
                for call in message.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    blocks.append({"type": "tool_use", "id": call.get("id"),
                                   "name": fn.get("name"),
                                   "input": parse_arguments(fn.get("arguments"))})
                push("assistant", blocks)
            else:
                push("user", _user_blocks(content))

        body: dict[str, Any] = {
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens") or payload.get("max_completion_tokens")
            or DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if system:
            body["system"] = "\n\n".join(s for s in system if s)
        for name in ("temperature", "top_p"):
            if payload.get(name) is not None:
                body[name] = payload[name]
        stop = payload.get("stop")
        if stop:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        if payload.get("tools"):
            body["tools"] = [{"name": t["function"]["name"],
                              "description": t["function"].get("description", ""),
                              "input_schema": t["function"].get("parameters")
                              or {"type": "object"}}
                             for t in payload["tools"] if t.get("type") == "function"]
        choice = payload.get("tool_choice")
        if isinstance(choice, str) and choice in ("auto", "required", "none"):
            body["tool_choice"] = {"type": {"auto": "auto", "required": "any",
                                            "none": "none"}[choice]}
        elif isinstance(choice, dict):
            body["tool_choice"] = {"type": "tool",
                                   "name": (choice.get("function") or {}).get("name")}
        return body

    def is_streaming(self, body: dict) -> bool:
        return bool(get(body, "stream"))

    def ensure_system(self, body: dict, text: str) -> None:
        if not get(body, "system"):
            put(body, "system", text)

    # -- responses -----------------------------------------------------------

    def response_to_chat(self, response: dict) -> dict:
        text, calls = [], []
        for block in response.get("content") or []:
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append({"id": block.get("id"), "type": "function",
                              "function": {"name": block.get("name"),
                                           "arguments": arguments(block.get("input"))}})
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = calls
        usage = response.get("usage") or {}
        return {
            "id": response.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.get("model", ""),
            "choices": [{"index": 0, "message": message,
                         "finish_reason": STOP_REASONS.get(response.get("stop_reason"),
                                                           "stop")}],
            "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                      "completion_tokens": usage.get("output_tokens", 0),
                      "total_tokens": usage.get("input_tokens", 0)
                      + usage.get("output_tokens", 0)},
        }

    def response_from_chat(self, completion: dict, *, model: str | None) -> dict:
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content: list[dict] = []
        if text_of(message.get("content")):
            content.append(_text_block(text_of(message.get("content"))))
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            content.append({"type": "tool_use", "id": call.get("id") or _id("toolu"),
                            "name": fn.get("name"),
                            "input": parse_arguments(fn.get("arguments"))})
        usage = completion.get("usage") or {}
        return {
            "id": _id("msg"),
            "type": "message",
            "role": "assistant",
            "model": model or completion.get("model", ""),
            "content": content,
            "stop_reason": FINISH_REASONS.get(choice.get("finish_reason"), "end_turn"),
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0)},
        }

    def response_texts(self, response: Any) -> list[tuple[str, Callable[[str], None]]]:
        return [(get(block, "text"), setter(block, "text"))
                for block in get(response, "content", default=[]) or []
                if get(block, "type") == "text" and isinstance(get(block, "text"), str)]

    def error(self, status: int, message: str, *, extra: dict | None = None) -> dict:
        kind = {400: "invalid_request_error", 401: "authentication_error",
                403: "permission_error", 404: "not_found_error",
                429: "rate_limit_error", 501: "invalid_request_error"}.get(status, "api_error")
        error: dict[str, Any] = {"type": kind, "message": message}
        if extra:
            error.update(extra)
        return {"type": "error", "error": error}


def _replacer(container: list, index: int) -> Callable[[str], None]:
    write = item_setter(container, index)
    return lambda text: write(_text_block(text))


def _user_blocks(content: Any) -> list[dict]:
    """Chat user content as Messages blocks."""
    if isinstance(content, str):
        return [_text_block(content)] if content else []
    blocks: list[dict] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if isinstance(block.get("text"), str):
            blocks.append(_text_block(block["text"]))
        elif kind == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            mime, data = split_data_url(url)
            blocks.append({"type": "image", "source":
                           {"type": "base64", "media_type": mime, "data": data}
                           if mime else {"type": "url", "url": url}})
        elif kind == "file":
            inner = block.get("file") or {}
            mime, data = split_data_url(inner.get("file_data", ""))
            if mime == "application/pdf":
                blocks.append({"type": "document", "source": {
                    "type": "base64", "media_type": mime, "data": data}})
            else:
                raise FormatError(
                    f"A {mime or 'file'} attachment cannot be sent to Anthropic. "
                    "Set MOCHI_FORWARD_MODE=convert to forward its text instead.")
        else:
            raise FormatError(
                f"A {kind} block cannot be sent to Anthropic. Set "
                "MOCHI_FORWARD_MODE=convert to forward its transcript instead.")
    return blocks


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"
