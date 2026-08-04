"""Phase 1 gateway tests.

These exercise the transport path with a stubbed adapter, so no API key or
network access is required.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from mochi.gateway.adapters.base import LLMAdapter, UpstreamError
from mochi.gateway.app import app
from mochi.gateway.models import ChatCompletionRequest, TaggedContext


class StubAdapter(LLMAdapter):
    """Records what the gateway forwarded, returns a canned completion."""

    name = "stub"

    def __init__(self, error: UpstreamError | None = None) -> None:
        self.received: dict[str, Any] | None = None
        self.closed = False
        self._error = error

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.received = payload
        if self._error is not None:
            raise self._error
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "stub reply"},
                    "finish_reason": "stop",
                }
            ],
        }

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def stub() -> StubAdapter:
    return StubAdapter()


@pytest.fixture
def client(stub: StubAdapter):
    with TestClient(app) as c:
        # Replace the real adapter created during lifespan startup.
        c.app.state.adapter = stub
        yield c


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_completion_forwards_and_returns(client: TestClient, stub: StubAdapter) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "stub reply"
    assert stub.received is not None
    assert stub.received["messages"] == [{"role": "user", "content": "hi"}]


def test_mochi_only_fields_are_stripped_before_upstream(
    client: TestClient, stub: StubAdapter
) -> None:
    """session_id/context are MOCHI extensions - forwarding them would 400."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "summarize this"}],
            "session_id": "sess_abc123",
            "context": {"web_content": "<p>scraped page</p>"},
        },
    )

    assert response.status_code == 200
    assert stub.received is not None
    assert "session_id" not in stub.received
    assert "context" not in stub.received


def test_default_model_applied_when_omitted(client: TestClient, stub: StubAdapter) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert stub.received is not None
    assert stub.received["model"]  # falls back to TARGET_LLM_MODEL


def test_streaming_rejected_with_501(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 501


def test_malformed_json_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_upstream_error_status_is_preserved() -> None:
    """A 401 from the provider should reach the client as 401, not 500."""
    failing = StubAdapter(
        error=UpstreamError(
            "Target LLM returned 401",
            status_code=401,
            payload={"error": {"message": "Invalid API key"}},
        )
    )
    with TestClient(app) as c:
        c.app.state.adapter = failing
        response = c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid API key"


# --- schema-level tests (no HTTP) ---


def test_tagged_context_segments_only_returns_populated() -> None:
    ctx = TaggedContext(user_input="hello", web_content="<p>page</p>")
    assert ctx.segments() == [
        ("user_input", "hello"),
        ("web_content", "<p>page</p>"),
    ]


def test_unknown_openai_params_pass_through() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
        }
    )
    payload = request.upstream_payload(default_model="fallback")
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
