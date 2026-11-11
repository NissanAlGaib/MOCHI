"""OpenAI chat completions - MOCHI's own shape.

Inspection reads this format natively, so its translations are identities.
It is registered beside the others so the gateway and the wrapper can treat
every format alike - in particular so a chat request can be forwarded to an
Anthropic or Gemini upstream through the same path as anything else.
"""

from __future__ import annotations

from typing import Any, Callable

from mochi.formats.base import Binding, ChatView, WireFormat, bound_text, get, setter


class ChatFormat(WireFormat):
    name = "openai_chat"

    def to_chat(self, body: dict) -> ChatView:
        view = ChatView()
        for message in get(body, "messages", default=[]) or []:
            content = get(message, "content")
            extra = {k: get(message, k) for k in ("tool_calls", "tool_call_id")}
            blocks = ([bound_text(content, setter(message, "content"))]
                      if isinstance(content, str)
                      else [(b, Binding()) for b in content or []])
            view.add(get(message, "role", default="user"), blocks, **extra)
        return view

    def from_chat(self, payload: dict) -> dict:
        return payload

    def response_to_chat(self, response: dict) -> dict:
        return response

    def response_from_chat(self, completion: dict, *, model: str | None) -> dict:
        return completion

    def response_texts(self, response: Any) -> list[tuple[str, Callable[[str], None]]]:
        out = []
        for choice in get(response, "choices", default=[]) or []:
            message = get(choice, "message")
            if message is not None and isinstance(get(message, "content"), str):
                out.append((get(message, "content"), setter(message, "content")))
        return out

    def ensure_system(self, body: dict, text: str) -> None:
        messages = list(get(body, "messages", default=[]) or [])
        if messages and get(messages[0], "role") in ("system", "developer"):
            return
        body["messages"] = [{"role": "system", "content": text}, *messages]
