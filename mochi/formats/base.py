"""What every wire format provides, and the view inspection reads.

MOCHI inspects one shape: an OpenAI chat-completions message list. Every
other SDK's request is translated into that shape for inspection, and the
translation remembers where each piece came from, so that whatever
enforcement changes - a redacted sentence, an attachment replaced by its
transcript - is written back into the request in its own format. The
application's SDK never sees the chat shape; it gets its own request back,
cleaned.

That write-back is positional, and it can be, because the two things that
edit an inspected request both keep positions: redaction rewrites a text
block in place, and attachment conversion swaps one block for one block.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Callable

#: Anthropic requires ``max_tokens``; a chat request may omit it. Used only
#: when a request crosses formats and the caller named no limit.
DEFAULT_MAX_TOKENS = 4096


class FormatError(ValueError):
    """A request MOCHI cannot read or translate. Becomes a 400, not a 500."""


# --- reading and writing SDK objects and plain dicts alike ------------------
#
# A request built from the SDK's own types - a tool-use block copied out of
# the previous response, a Gemini ``Part`` - is an object; one built by hand
# or received as JSON is a dict. Every accessor below takes both, so no format
# module has to care which it was handed.


def get(obj: Any, *names: str, default: Any = None) -> Any:
    """The first of ``names`` that ``obj`` has, as a key or an attribute.

    Several names because Gemini's REST API spells fields in camelCase
    (``inlineData``) and its Python SDK in snake_case (``inline_data``).
    """
    for name in names:
        if isinstance(obj, dict):
            if obj.get(name) is not None:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def put(obj: Any, name: str, value: Any) -> None:
    """Set ``name`` on ``obj``, as a key or an attribute."""
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def setter(obj: Any, name: str) -> Callable[[Any], None]:
    """A function that writes ``name`` on ``obj``."""
    return lambda value: put(obj, name, value)


def item_setter(container: list, index: int) -> Callable[[Any], None]:
    """A function that replaces ``container[index]``."""
    def write(value: Any) -> None:
        container[index] = value
    return write


def plain(obj: Any) -> Any:
    """``obj`` as plain data: SDK objects dumped, containers recursed."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:
            return dump()
    if isinstance(obj, dict):
        return {key: plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(value) for value in obj]
    return obj


def b64(data: Any) -> str | None:
    """Base64 text for ``data``, which the SDKs hold as bytes and JSON as text."""
    if isinstance(data, (bytes, bytearray)):
        return base64.b64encode(bytes(data)).decode("ascii")
    if isinstance(data, str) and data:
        return data
    return None


def data_url(mime: str | None, data: Any) -> str | None:
    """A ``data:`` URL - how the chat format carries an inline attachment."""
    encoded = b64(data)
    if encoded is None:
        return None
    if encoded.startswith("data:"):
        return encoded
    return f"data:{mime or 'application/octet-stream'};base64,{encoded}"


def split_data_url(url: str) -> tuple[str | None, str]:
    """``(mime, base64)`` from a ``data:`` URL; ``(None, url)`` otherwise."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, url
    header, _, payload = url.partition(",")
    return header[5:].split(";")[0] or None, payload


def attachment_block(mime: str | None, data: Any, *, filename: str | None = None) -> dict:
    """The chat-format block for an inline attachment of type ``mime``.

    The block types are the ones ``mochi.preprocess.extract`` reads, so an
    image, recording, video or document attached through any SDK reaches the
    same extractor an OpenAI request's would.
    """
    kind = (mime or "").lower()
    url = data_url(mime, data)
    if url is None:
        return unreadable_block(filename or kind or "attachment")
    if kind.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": url}}
    if kind.startswith("audio/"):
        return {"type": "input_audio",
                "input_audio": {"data": split_data_url(url)[1],
                                "format": kind.split("/", 1)[1].split(";")[0]}}
    if kind.startswith("video/"):
        return {"type": "video", "video": {"url": url}}
    return {"type": "file", "file": {"file_data": url,
                                     "filename": filename or _default_name(kind)}}


def remote_block(mime: str | None, uri: str) -> dict:
    """A block for an attachment held elsewhere (a URL, an uploaded file).

    Never fetched - fetching would let any caller make MOCHI issue requests to
    a host of their choosing, the rule every extractor already follows. An
    image keeps its URL so a vision model still receives it; anything else
    becomes a document block with no bytes, which extraction reports as
    unread and conversion replaces with a marker, rather than calling it clean.
    """
    if (mime or "").lower().startswith("image/") or mime is None and _looks_like_image(uri):
        return {"type": "image_url", "image_url": {"url": uri}}
    return unreadable_block(uri)


def unreadable_block(name: str) -> dict:
    return {"type": "file", "file": {"filename": name}}


def _looks_like_image(uri: str) -> bool:
    return isinstance(uri, str) and uri.lower().split("?")[0].endswith(
        (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"))


def _default_name(mime: str) -> str:
    if "pdf" in mime:
        return "document.pdf"
    if "html" in mime:
        return "page.html"
    return "document.txt"


def text_of(content: Any) -> str:
    """The text of chat-format content: a string, or the text of its blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content
                         if isinstance(block, dict) and isinstance(block.get("text"), str))
    return ""


def arguments(value: Any) -> str:
    """Tool-call arguments as the JSON string the chat format carries."""
    if isinstance(value, str):
        return value
    return json.dumps(plain(value) if value is not None else {}, ensure_ascii=False)


def parse_arguments(value: Any) -> Any:
    """Tool-call arguments as the object Anthropic and Gemini carry."""
    if not isinstance(value, str):
        return value or {}
    try:
        return json.loads(value) if value.strip() else {}
    except ValueError:
        return {"arguments": value}


# --- the view ---------------------------------------------------------------


@dataclass
class Binding:
    """Where one chat-format content block came from in the native request.

    ``set_text`` writes edited text back; ``replace`` swaps the native block
    for a text block holding the given text, which is what conversion does to
    an attachment. A block with neither is read-only - inspected, never edited.
    """

    original: str | None = None
    set_text: Callable[[str], None] | None = None
    replace: Callable[[str], None] | None = None


@dataclass
class ChatView:
    """A native request seen as chat messages, with the way back.

    ``messages`` is what inspection reads and, when the request is forwarded
    to a provider of another format, what is sent. Each message's content is
    a list of blocks, and ``bindings[i][j]`` says where block ``j`` of message
    ``i`` lives in the native request.
    """

    messages: list[dict] = field(default_factory=list)
    bindings: list[list[Binding]] = field(default_factory=list)

    def add(self, role: str, blocks: list[tuple[dict, Binding]], **extra: Any) -> None:
        """Append a message. ``extra`` carries ``tool_calls``/``tool_call_id``."""
        extra = {k: v for k, v in extra.items() if v is not None}
        if not blocks and not extra:
            return
        message: dict[str, Any] = {"role": role, "content": [b for b, _ in blocks]}
        message.update(extra)
        self.messages.append(message)
        self.bindings.append([binding for _, binding in blocks])

    def write_back(self, edited: list[Any]) -> int:
        """Copy enforcement's edits into the native request. Returns the count.

        ``edited`` is the inspected request's messages after enforcement -
        validated objects whose content may have been rewritten in place
        (redaction) or swapped block for block (conversion).
        """
        changed = 0
        for index, message in enumerate(edited):
            if index >= len(self.bindings):
                break
            content = get(message, "content")
            if not isinstance(content, list):
                continue
            for position, block in enumerate(content):
                if position >= len(self.bindings[index]):
                    break
                binding = self.bindings[index][position]
                if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                    continue
                new = block["text"]
                if binding.original is not None:
                    if new != binding.original and binding.set_text is not None:
                        binding.set_text(new)
                        changed += 1
                elif binding.replace is not None:
                    binding.replace(new)
                    changed += 1
        return changed

    def forwardable(self) -> list[dict]:
        """``messages`` as a chat request any OpenAI-compatible server accepts.

        All-text content is collapsed to a string: some compatible servers
        (Ollama among them) accept block lists only in user turns. A message
        carrying only tool calls gets ``content: null``, as the chat format
        expects.
        """
        out: list[dict] = []
        for message in self.messages:
            message = dict(message)
            blocks = message.get("content") or []
            if all(isinstance(b, dict) and b.get("type") == "text" for b in blocks):
                message["content"] = text_of(blocks) if blocks else None
            out.append(message)
        return out


def text_block(text: str, binding: Binding) -> tuple[dict, Binding]:
    return {"type": "text", "text": text}, binding


def bound_text(text: str, write: Callable[[str], None]) -> tuple[dict, Binding]:
    """A text block whose edits are written back with ``write``."""
    return text_block(text, Binding(original=text, set_text=write))


def bound_attachment(block: dict, replace: Callable[[str], None]) -> tuple[dict, Binding]:
    """An attachment block that conversion may replace using ``replace``."""
    return block, Binding(original=None, replace=replace)


# --- the format interface ---------------------------------------------------


class WireFormat:
    """One SDK's request and response shapes, and the way to and from chat.

    The four translations a gateway needs, plus the two the wrapper and the
    gateway need for outbound scanning. Subclasses implement what their
    format supports; the base raises :class:`FormatError` for the rest.
    """

    #: Registry key, also the name in telemetry.
    name = "base"

    # -- requests --

    def to_chat(self, body: dict) -> ChatView:
        """The request as chat messages, bound back to ``body``."""
        raise NotImplementedError

    def chat_params(self, body: dict) -> dict:
        """Everything but the messages, as chat-completions parameters."""
        return {}

    def from_chat(self, payload: dict) -> dict:
        """A chat request in this format - for a provider that speaks it."""
        raise FormatError(f"{self.name} is not a provider format")

    def model(self, body: dict) -> str | None:
        return get(body, "model")

    def is_streaming(self, body: dict) -> bool:
        return bool(get(body, "stream"))

    def ensure_system(self, body: dict, text: str) -> None:
        """Declare ``text`` as the system prompt when the request has none."""

    # -- responses --

    def response_to_chat(self, response: dict) -> dict:
        """This format's response as a chat completion."""
        raise FormatError(f"{self.name} is not a provider format")

    def response_from_chat(self, completion: dict, *, model: str | None) -> dict:
        """A chat completion as this format's response."""
        raise NotImplementedError

    def response_texts(self, response: Any) -> list[tuple[str, Callable[[str], None]]]:
        """Every text in a response, with how to write it back.

        Works on the SDK's response object and on the JSON alike, so the
        wrapper and the gateway scan responses with one implementation.
        """
        raise NotImplementedError

    def error(self, status: int, message: str, *,
              extra: dict | None = None) -> dict:
        """An error body this format's SDK parses into its own exception."""
        error: dict[str, Any] = {"message": message, "type": "mochi_error"}
        if extra:
            error.update(extra)
        return {"error": error}
