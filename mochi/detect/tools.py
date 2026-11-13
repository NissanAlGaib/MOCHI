"""The names of the tools a request offers the model.

Untrusted content has no business naming them. A retrieved article, a web page
or a tool's own output that says "call transfer_funds" is addressing the agent,
not the reader - data does not tell anyone which function to invoke. Stage I
turns these names into an untrusted-only rule (see
:meth:`mochi.detect.stage1_syntactic.Stage1Detector.scan`).

Every SDK declares tools in its own shape, so this reads all four:

* OpenAI chat:      ``[{"type": "function", "function": {"name": ...}}]``
* OpenAI Responses: ``[{"type": "function", "name": ...}]``
* Anthropic:        ``[{"name": ..., "input_schema": ...}]``
* Gemini:           ``[{"functionDeclarations": [{"name": ...}]}]`` (or snake_case)

Objects from an SDK (pydantic models, dataclasses) are read by attribute as
well as by key, since the wrapper sees the caller's arguments before they are
serialised.
"""

from __future__ import annotations

import re
from typing import Any

#: A name must look like an identifier to be matched. Anything else (a server
#: tool such as Anthropic's ``web_search_20250305`` still qualifies) is ignored
#: rather than compiled into a pattern.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{2,63}$")


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def tool_names(tools: Any) -> list[str]:
    """Distinct tool names declared in ``tools``, in order of appearance."""
    if not tools or isinstance(tools, (str, bytes)):
        return []
    names: list[str] = []

    def add(name: Any) -> None:
        if isinstance(name, str) and _IDENTIFIER.match(name) and name not in names:
            names.append(name)

    for tool in tools if isinstance(tools, (list, tuple)) else [tools]:
        function = _get(tool, "function")
        if function is not None and not callable(function):
            add(_get(function, "name"))
        add(_get(tool, "name"))
        for key in ("functionDeclarations", "function_declarations"):
            for declaration in _get(tool, key) or []:
                add(_get(declaration, "name"))
    return names


def request_tools(body: Any) -> Any:
    """Where a request body keeps its tools: top level, or Gemini's ``config``."""
    tools = _get(body, "tools")
    if tools:
        return tools
    config = _get(body, "config")
    return _get(config, "tools") if config is not None else None
