"""OpenAI Responses API - ``client.responses.create`` and ``POST /v1/responses``.

The newer of the OpenAI SDK's two surfaces, and a different shape from chat
completions: ``instructions`` instead of a system message, ``input`` as a
string or a list of typed items, and tool output as ``function_call_output``
items rather than ``tool`` messages.

Mapping to the chat shape:

- ``instructions`` and ``system``/``developer`` items are system messages.
- ``function_call_output`` items are tool output - chat ``tool`` messages,
  untrusted, redacted rather than refused.
- ``input_image``, ``input_file`` and ``input_audio`` become the attachment
  blocks the extractors read. A URL or uploaded-file id is never fetched.
- ``function_call`` items become ``tool_calls``, merged into one assistant
  turn when consecutive, as the chat format requires.

**Stored conversations are out of reach.** With ``previous_response_id`` the
earlier turns live at OpenAI, not in the request, so MOCHI inspects only what
is sent now - the same limit session accumulation has, and the reason a
request carrying one cannot be forwarded to any provider but OpenAI.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from mochi.formats.base import (
    Binding,
    ChatView,
    WireFormat,
    arguments,
    bound_attachment,
    bound_text,
    get,
    item_setter,
    put,
    remote_block,
    setter,
    text_of,
)

ROLES = {"system": "system", "developer": "system", "assistant": "assistant",
         "user": "user"}
TEXT_PARTS = {"input_text", "output_text", "text"}


class ResponsesFormat(WireFormat):
    name = "openai_responses"

    def to_chat(self, body: dict) -> ChatView:
        view = ChatView()
        instructions = get(body, "instructions")
        if isinstance(instructions, str) and instructions:
            view.add("system", [bound_text(instructions, setter(body, "instructions"))])

        items = get(body, "input")
        if isinstance(items, str):
            view.add("user", [bound_text(items, setter(body, "input"))])
            return view

        for item in items or []:
            kind = get(item, "type")
            if kind == "function_call_output":
                output = get(item, "output")
                blocks = ([bound_text(output, setter(item, "output"))]
                          if isinstance(output, str) else _parts(output or [], "input_text"))
                view.add("tool", blocks, tool_call_id=get(item, "call_id"))
            elif kind == "function_call":
                call = {"id": get(item, "call_id"), "type": "function",
                        "function": {"name": get(item, "name"),
                                     "arguments": arguments(get(item, "arguments"))}}
                last = view.messages[-1] if view.messages else None
                if last and last["role"] == "assistant" and last.get("tool_calls") \
                        and not last["content"]:
                    last["tool_calls"].append(call)
                else:
                    view.add("assistant", [], tool_calls=[call])
            elif kind in (None, "message") and get(item, "role") in ROLES:
                role = ROLES[get(item, "role")]
                content = get(item, "content")
                text_type = "output_text" if role == "assistant" else "input_text"
                if isinstance(content, str):
                    view.add(role, [bound_text(content, setter(item, "content"))])
                else:
                    view.add(role, _parts(content or [], text_type))
            # reasoning, item_reference and built-in tool calls carry nothing
            # MOCHI can inspect or edit, and pass through untouched.
        return view

    def chat_params(self, body: dict) -> dict:
        params: dict[str, Any] = {
            "max_tokens": get(body, "max_output_tokens"),
            "temperature": get(body, "temperature"),
            "top_p": get(body, "top_p"),
        }
        tools = [t for t in get(body, "tools", default=[]) or []
                 if get(t, "type") == "function"]
        if tools:
            params["tools"] = [{"type": "function", "function": {
                "name": get(t, "name"), "description": get(t, "description", default=""),
                "parameters": get(t, "parameters", default={"type": "object"})}}
                for t in tools]
        choice = get(body, "tool_choice")
        if isinstance(choice, str):
            params["tool_choice"] = choice
        elif choice is not None and get(choice, "name"):
            params["tool_choice"] = {"type": "function",
                                     "function": {"name": get(choice, "name")}}
        return {k: v for k, v in params.items() if v is not None}

    def ensure_system(self, body: dict, text: str) -> None:
        if get(body, "instructions"):
            return
        items = get(body, "input")
        if isinstance(items, list) and any(get(i, "role") in ("system", "developer")
                                           for i in items):
            return
        put(body, "instructions", text)

    # -- responses -----------------------------------------------------------

    def response_from_chat(self, completion: dict, *, model: str | None) -> dict:
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        output: list[dict] = []
        text = text_of(message.get("content"))
        if text:
            output.append({"type": "message", "id": _id("msg"), "status": "completed",
                           "role": "assistant",
                           "content": [{"type": "output_text", "text": text,
                                        "annotations": []}]})
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            output.append({"type": "function_call", "id": _id("fc"),
                           "call_id": call.get("id") or _id("call"),
                           "name": fn.get("name"), "arguments": fn.get("arguments") or "{}",
                           "status": "completed"})
        usage = completion.get("usage") or {}
        return {
            "id": _id("resp"),
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model or completion.get("model", ""),
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "error": None,
            "incomplete_details": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0),
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        }

    def response_texts(self, response: Any) -> list[tuple[str, Callable[[str], None]]]:
        out = []
        for item in get(response, "output", default=[]) or []:
            if get(item, "type") != "message":
                continue
            for part in get(item, "content", default=[]) or []:
                text = get(part, "text")
                if get(part, "type") == "output_text" and isinstance(text, str):
                    out.append((text, setter(part, "text")))
        return out


def _parts(parts: list, text_type: str) -> list[tuple[dict, Binding]]:
    """Content parts, each bound back to its place."""
    out: list[tuple[dict, Binding]] = []
    for index, part in enumerate(parts):
        kind = get(part, "type")
        replace = _replacer(parts, index, text_type)
        if kind in TEXT_PARTS and isinstance(get(part, "text"), str):
            out.append(bound_text(get(part, "text"), setter(part, "text")))
        elif kind == "input_image":
            url = get(part, "image_url")
            if isinstance(url, str) and url.startswith("data:"):
                out.append(bound_attachment({"type": "image_url", "image_url": {"url": url}},
                                            replace))
            else:
                out.append(bound_attachment(
                    remote_block("image/*", url or get(part, "file_id") or "image"), replace))
        elif kind == "input_file":
            data = get(part, "file_data")
            if isinstance(data, str) and data:
                out.append(bound_attachment(
                    {"type": "file", "file": {"file_data": data,
                                              "filename": get(part, "filename")
                                              or "document"}}, replace))
            else:
                out.append(bound_attachment(
                    remote_block(None, get(part, "file_url") or get(part, "file_id")
                                 or get(part, "filename") or "file"), replace))
        elif kind == "input_audio":
            audio = get(part, "input_audio", default={})
            out.append(bound_attachment(
                {"type": "input_audio", "input_audio": {"data": get(audio, "data"),
                                                        "format": get(audio, "format")}},
                replace))
    return out


def _replacer(parts: list, index: int, text_type: str) -> Callable[[str], None]:
    write = item_setter(parts, index)
    return lambda text: write({"type": text_type, "text": text})


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"
