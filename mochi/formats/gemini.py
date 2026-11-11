"""Google Gemini - ``client.models.generate_content`` and ``:generateContent``.

The same shape arrives two ways. Over REST it is camelCase JSON
(``inlineData``, ``systemInstruction``); through the ``google-genai`` SDK it
is snake_case, the system instruction lives in ``config``, and ``contents``
may be a bare string, a list of parts, or SDK objects. The wrapper turns the
SDK's spellings into dicts first (:func:`normalize_contents`); every reader
here accepts both spellings.

Mapping to the chat shape:

- The system instruction is the system message; a ``model`` turn is the
  assistant.
- ``functionResponse`` parts are tool output and become chat ``tool``
  messages - untrusted, so redacted rather than refused. Each string inside
  the response object is its own text block, so redaction rewrites the
  string it found the payload in and the object keeps its shape.
- ``inlineData`` becomes the attachment block its MIME type calls for; a
  ``fileData`` URI is never fetched.
- Thought parts are skipped, as Anthropic's thinking blocks are.
"""

from __future__ import annotations

import io
import time
from typing import Any, Callable

from mochi.formats.base import (
    Binding,
    ChatView,
    WireFormat,
    arguments,
    attachment_block,
    bound_attachment,
    bound_text,
    get,
    item_setter,
    parse_arguments,
    plain,
    put,
    remote_block,
    setter,
    split_data_url,
    text_of,
)

FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
          "RECITATION": "content_filter", "PROHIBITED_CONTENT": "content_filter",
          "BLOCKLIST": "content_filter", "SPII": "content_filter"}
FINISH_BACK = {"stop": "STOP", "length": "MAX_TOKENS", "content_filter": "SAFETY",
               "tool_calls": "STOP"}
STATUS = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED",
          404: "NOT_FOUND", 429: "RESOURCE_EXHAUSTED", 501: "UNIMPLEMENTED"}


class GeminiFormat(WireFormat):
    name = "gemini"

    # -- requests ------------------------------------------------------------

    def to_chat(self, body: dict) -> ChatView:
        view = ChatView()
        holder, key = _system_location(body)
        if holder is not None:
            view.add("system", _system_blocks(holder, key))

        for content in get(body, "contents", default=[]) or []:
            role = {"model": "assistant"}.get(get(content, "role"), "user")
            parts = get(content, "parts", default=[]) or []
            pending: list[tuple[dict, Binding]] = []
            calls: list[dict] = []
            for index, part in enumerate(parts):
                if get(part, "thought"):
                    continue
                text = get(part, "text")
                blob = get(part, "inlineData", "inline_data")
                remote = get(part, "fileData", "file_data")
                call = get(part, "functionCall", "function_call")
                result = get(part, "functionResponse", "function_response")
                if isinstance(text, str):
                    pending.append(bound_text(text, setter(part, "text")))
                elif blob is not None:
                    pending.append(bound_attachment(
                        attachment_block(get(blob, "mimeType", "mime_type"),
                                         get(blob, "data"),
                                         filename=get(blob, "displayName", "display_name")),
                        _replacer(parts, index)))
                elif remote is not None:
                    pending.append(bound_attachment(
                        remote_block(get(remote, "mimeType", "mime_type"),
                                     get(remote, "fileUri", "file_uri", default="file")),
                        _replacer(parts, index)))
                elif call is not None:
                    name = get(call, "name")
                    calls.append({"id": get(call, "id") or name, "type": "function",
                                  "function": {"name": name,
                                               "arguments": arguments(get(call, "args"))}})
                elif result is not None:
                    view.add(role, pending)
                    pending = []
                    view.add("tool", _leaves(get(result, "response")),
                             tool_call_id=get(result, "id") or get(result, "name"))
            view.add(role, pending, tool_calls=calls or None)
        return view

    def chat_params(self, body: dict) -> dict:
        config = get(body, "generationConfig", "generation_config", "config", default={})
        params: dict[str, Any] = {
            "temperature": get(config, "temperature"),
            "top_p": get(config, "topP", "top_p"),
            "max_tokens": get(config, "maxOutputTokens", "max_output_tokens"),
        }
        stop = get(config, "stopSequences", "stop_sequences")
        if stop:
            params["stop"] = list(stop)
        tools = get(body, "tools") or get(config, "tools") or []
        declared = [d for t in tools
                    for d in get(t, "functionDeclarations", "function_declarations",
                                 default=[]) or []]
        if declared:
            params["tools"] = [{"type": "function", "function": {
                "name": get(d, "name"), "description": get(d, "description", default=""),
                "parameters": plain(get(d, "parameters", "parametersJsonSchema",
                                        "parameters_json_schema",
                                        default={"type": "object"}))}}
                for d in declared]
        return {k: v for k, v in params.items() if v is not None}

    def from_chat(self, payload: dict) -> dict:
        """A chat request as a generateContent body, for a Gemini upstream.

        The model is not part of the body - it is in the URL - so the adapter
        takes it separately.
        """
        system: list[str] = []
        contents: list[dict] = []
        names: dict[str, str] = {}

        def push(role: str, parts: list[dict]) -> None:
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        for message in payload.get("messages") or []:
            role, content = message.get("role"), message.get("content")
            if role in ("system", "developer"):
                system.append(text_of(content))
            elif role == "assistant":
                parts = [{"text": text_of(content)}] if text_of(content) else []
                for call in message.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    names[call.get("id") or fn.get("name")] = fn.get("name")
                    parts.append({"functionCall": {
                        "name": fn.get("name"), "args": parse_arguments(fn.get("arguments"))}})
                push("model", parts)
            elif role in ("tool", "function"):
                call_id = message.get("tool_call_id") or message.get("name") or ""
                push("user", [{"functionResponse": {
                    "name": names.get(call_id, call_id),
                    "response": {"output": text_of(content)}}}])
            else:
                push("user", _user_parts(content))

        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(s for s in system if s)}]}
        generation = {
            "temperature": payload.get("temperature"),
            "topP": payload.get("top_p"),
            "maxOutputTokens": payload.get("max_tokens") or payload.get("max_completion_tokens"),
        }
        stop = payload.get("stop")
        if stop:
            generation["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
        generation = {k: v for k, v in generation.items() if v is not None}
        if generation:
            body["generationConfig"] = generation
        if payload.get("tools"):
            body["tools"] = [{"functionDeclarations": [
                {"name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "parameters": t["function"].get("parameters") or {"type": "object"}}
                for t in payload["tools"] if t.get("type") == "function"]}]
        choice = payload.get("tool_choice")
        if isinstance(choice, str) and choice in ("auto", "required", "none"):
            body["toolConfig"] = {"functionCallingConfig": {
                "mode": {"auto": "AUTO", "required": "ANY", "none": "NONE"}[choice]}}
        return body

    def is_streaming(self, body: dict) -> bool:
        return False          # streaming is a different method, not a flag

    def ensure_system(self, body: dict, text: str) -> None:
        holder, _ = _system_location(body)
        if holder is not None:
            return
        if "config" in body:                     # the SDK's call, see guard()
            config = body.get("config")
            if config is None:
                body["config"] = {"system_instruction": text}
                return
            put(config, "system_instruction", text)
            return
        body["systemInstruction"] = {"parts": [{"text": text}]}

    # -- responses -----------------------------------------------------------

    def response_to_chat(self, response: dict) -> dict:
        candidate = (response.get("candidates") or [{}])[0]
        text, calls = [], []
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("thought"):
                continue
            if isinstance(part.get("text"), str):
                text.append(part["text"])
            elif part.get("functionCall"):
                call = part["functionCall"]
                calls.append({"id": call.get("id") or call.get("name"), "type": "function",
                              "function": {"name": call.get("name"),
                                           "arguments": arguments(call.get("args"))}})
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = calls
        usage = response.get("usageMetadata") or {}
        finish = "tool_calls" if calls else FINISH.get(candidate.get("finishReason"), "stop")
        return {
            "id": response.get("responseId", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.get("modelVersion", ""),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage.get("promptTokenCount", 0),
                      "completion_tokens": usage.get("candidatesTokenCount", 0),
                      "total_tokens": usage.get("totalTokenCount", 0)},
        }

    def response_from_chat(self, completion: dict, *, model: str | None) -> dict:
        candidates = []
        for choice in completion.get("choices") or []:
            message = choice.get("message") or {}
            parts: list[dict] = []
            if text_of(message.get("content")):
                parts.append({"text": text_of(message.get("content"))})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                parts.append({"functionCall": {"name": fn.get("name"),
                                               "args": parse_arguments(fn.get("arguments"))}})
            candidates.append({
                "content": {"role": "model", "parts": parts},
                "finishReason": FINISH_BACK.get(choice.get("finish_reason"), "STOP"),
                "index": choice.get("index", len(candidates)),
            })
        usage = completion.get("usage") or {}
        return {
            "candidates": candidates,
            "usageMetadata": {"promptTokenCount": usage.get("prompt_tokens", 0),
                              "candidatesTokenCount": usage.get("completion_tokens", 0),
                              "totalTokenCount": usage.get("total_tokens", 0)},
            "modelVersion": model or completion.get("model", ""),
        }

    def response_texts(self, response: Any) -> list[tuple[str, Callable[[str], None]]]:
        out = []
        for candidate in get(response, "candidates", default=[]) or []:
            content = get(candidate, "content")
            parts = (get(content, "parts", default=[]) or []) if content is not None else []
            for part in parts:
                text = get(part, "text")
                if isinstance(text, str):
                    out.append((text, setter(part, "text")))
        return out

    def error(self, status: int, message: str, *, extra: dict | None = None) -> dict:
        error: dict[str, Any] = {"code": status, "message": message,
                                 "status": STATUS.get(status, "INTERNAL")}
        if extra:
            error.update(extra)
        return {"error": error}


# --- helpers ----------------------------------------------------------------


def _system_location(body: dict) -> tuple[Any, str | None]:
    """Where the system instruction is, as ``(holder, key)``; ``(None, None)``."""
    for key in ("systemInstruction", "system_instruction"):
        if get(body, key) is not None:
            return body, key
    config = get(body, "config")
    if config is not None and get(config, "system_instruction") is not None:
        return config, "system_instruction"
    return None, None


def _system_blocks(holder: Any, key: str) -> list[tuple[dict, Binding]]:
    value = get(holder, key)
    if isinstance(value, str):
        return [bound_text(value, setter(holder, key))]
    items = value if isinstance(value, list) else [value]
    out: list[tuple[dict, Binding]] = []
    for item_index, item in enumerate(items):
        if isinstance(item, str):
            out.append(bound_text(item, item_setter(items, item_index)))
            continue
        parts = get(item, "parts")
        for part in (parts if parts is not None else [item]):
            if isinstance(get(part, "text"), str):
                out.append(bound_text(get(part, "text"), setter(part, "text")))
    return out


def _leaves(value: Any) -> list[tuple[dict, Binding]]:
    """Every string inside a function response, each bound to its own place."""
    out: list[tuple[dict, Binding]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(child, str):
                    out.append(bound_text(child, setter(node, key)))
                else:
                    walk(child)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                if isinstance(child, str):
                    out.append(bound_text(child, item_setter(node, index)))
                else:
                    walk(child)

    walk(value)
    return out


def _replacer(parts: list, index: int) -> Callable[[str], None]:
    write = item_setter(parts, index)
    return lambda text: write({"text": text})


def _user_parts(content: Any) -> list[dict]:
    """Chat user content as Gemini parts."""
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts: list[dict] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
            continue
        if kind == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
        elif kind == "input_audio":
            audio = block.get("input_audio") or {}
            url = f"data:audio/{audio.get('format', 'wav')};base64,{audio.get('data', '')}"
        elif kind in ("video", "video_url"):
            url = (block.get(kind) or {}).get("url", "")
        elif kind == "file":
            url = (block.get("file") or {}).get("file_data", "")
        else:
            continue
        mime, data = split_data_url(url)
        if mime:
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        elif url:
            parts.append({"fileData": {"fileUri": url}})
    return parts


# --- the SDK's input forms --------------------------------------------------


def normalize_contents(contents: Any) -> list[dict]:
    """The SDK's ``contents`` argument as a list of content dicts.

    The SDK accepts a string, a part, a list of parts (one user turn), a list
    of contents (a conversation), SDK objects, dicts and PIL images. Mirrors
    its rule: content-shaped items are turns, and runs of anything else are
    gathered into a user turn. The result is a valid ``contents`` argument
    for the SDK, so the wrapper can hand back the edited list as it is.
    """
    if contents is None:
        return []
    items = contents if isinstance(contents, list) else [contents]
    out: list[dict] = []
    pending: list[Any] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "parts": list(pending)})
            pending.clear()

    for item in items:
        data = plain(item) if not isinstance(item, (str, bytes)) else item
        if isinstance(data, dict) and ("parts" in data or "role" in data):
            flush()
            data.setdefault("parts", [])
            data["parts"] = [_part(p) for p in data.get("parts") or []]
            out.append(data)
        else:
            pending.append(_part(item))
    flush()
    return out


def _part(item: Any) -> Any:
    if isinstance(item, str):
        return {"text": item}
    if _is_pil(item):
        buffer = io.BytesIO()
        item.save(buffer, format="PNG")
        return {"inline_data": {"mime_type": "image/png", "data": buffer.getvalue()}}
    data = plain(item)
    return data if isinstance(data, dict) else item


def _is_pil(item: Any) -> bool:
    return hasattr(item, "save") and hasattr(item, "mode") and hasattr(item, "size")


__all__ = ["GeminiFormat", "normalize_contents"]
