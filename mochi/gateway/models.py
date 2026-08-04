"""Request/response schemas for the OpenAI-compatible gateway surface.

Models are deliberately permissive (``extra="allow"``) so that OpenAI
parameters MOCHI does not care about (temperature, top_p, tools, ...) pass
through untouched. MOCHI only needs to *understand* the fields it inspects;
everything else is forwarded verbatim.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Source tags from the thesis (Chapter III, "Source Identification and Payload
# Tracing"). Used from Phase 4 onward; defined here so the schema is stable.
SOURCE_TAGS = (
    "user_input",
    "retrieved_document",
    "web_content",
    "api_response",
    "system_prompt",
)

# MOCHI-specific fields that must be stripped before forwarding upstream -
# the target provider's API would reject them as unknown parameters.
MOCHI_ONLY_FIELDS = frozenset({"session_id", "context"})


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None


class TaggedContext(BaseModel):
    """Optional per-source payload segments.

    When a client supplies this, MOCHI inspects each segment separately and
    can attribute a detection to a precise origin. When absent, the pipeline
    falls back to treating the message content as a single untagged blob -
    this fallback is what keeps integration a one-line ``base_url`` change.
    """

    model_config = ConfigDict(extra="forbid")

    user_input: str | None = None
    retrieved_document: str | None = None
    web_content: str | None = None
    api_response: str | None = None
    system_prompt: str | None = None

    def segments(self) -> list[tuple[str, str]]:
        """Return ``(source_tag, text)`` pairs for every populated segment."""
        return [
            (tag, value)
            for tag in SOURCE_TAGS
            if (value := getattr(self, tag)) is not None
        ]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    stream: bool = False

    # --- MOCHI extensions (stripped before upstream dispatch) ---
    session_id: str | None = None
    context: TaggedContext | None = None

    def upstream_payload(self, default_model: str) -> dict[str, Any]:
        """Serialize for the target LLM, dropping MOCHI-only fields."""
        payload = self.model_dump(exclude_none=True, exclude=MOCHI_ONLY_FIELDS)
        if not payload.get("model"):
            payload["model"] = default_model
        return payload

    def inspectable_text(self) -> str:
        """All text the detection pipeline should examine, as one string.

        Prefers the tagged ``context`` when supplied; otherwise falls back to
        message content. Handles multimodal content blocks by extracting their
        text parts and ignoring non-text (images, audio).
        """
        if self.context is not None:
            return "\n".join(text for _, text in self.context.segments())

        parts: list[str] = []
        for message in self.messages:
            content = message.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
        return "\n".join(parts)
