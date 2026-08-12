"""Phase 2 telemetry tests.

The log file produced here is the same artifact the Phase 5/13 evaluation
harness parses, so these tests double as a contract check on the Chapter IV
data source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mochi.gateway.app import app
from mochi.telemetry import (
    LatencyBreakdown,
    MitigationAction,
    PayloadCharacteristics,
    TelemetryRecord,
    TelemetryWriter,
    stage_timer,
)
from tests.test_gateway import StubAdapter


def read_records(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line]


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "mochi.jsonl"


@pytest.fixture
def client(log_path: Path):
    stub = StubAdapter()
    with TestClient(app) as c:
        c.app.state.adapter = stub
        # Redirect telemetry to a temp file so tests never touch logs/.
        c.app.state.telemetry_writer.close()
        c.app.state.telemetry_writer = TelemetryWriter(log_path)
        yield c
        c.app.state.telemetry_writer.close()


# --- writer / schema unit tests ---


def test_writer_emits_one_json_line_per_record(log_path: Path) -> None:
    with TelemetryWriter(log_path) as writer:
        writer.write(TelemetryRecord())
        writer.write(TelemetryRecord())

    records = read_records(log_path)
    assert len(records) == 2
    assert records[0]["request_id"] != records[1]["request_id"]


def test_writer_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "mochi.jsonl"
    with TelemetryWriter(nested) as writer:
        writer.write(TelemetryRecord())
    assert nested.exists()


def test_writer_appends_across_sessions(log_path: Path) -> None:
    """Re-opening must not truncate - evaluation runs accumulate."""
    with TelemetryWriter(log_path) as writer:
        writer.write(TelemetryRecord())
    with TelemetryWriter(log_path) as writer:
        writer.write(TelemetryRecord())
    assert len(read_records(log_path)) == 2


def test_payload_hashed_not_stored_by_default() -> None:
    chars = PayloadCharacteristics.from_text("secret prompt", include_content=False)
    assert chars.content is None
    assert chars.content_sha256 is not None
    assert chars.char_length == len("secret prompt")


def test_payload_stored_when_explicitly_enabled() -> None:
    chars = PayloadCharacteristics.from_text("secret prompt", include_content=True)
    assert chars.content == "secret prompt"


def test_identical_payloads_hash_identically() -> None:
    a = PayloadCharacteristics.from_text("same", include_content=False)
    b = PayloadCharacteristics.from_text("same", include_content=False)
    assert a.content_sha256 == b.content_sha256


def test_stage_timer_records_elapsed_time() -> None:
    latency = LatencyBreakdown()
    with stage_timer(latency, "stage_1"):
        pass
    assert latency.stage_1_ms is not None
    assert latency.stage_1_ms >= 0


def test_stage_timer_records_even_on_exception() -> None:
    latency = LatencyBreakdown()
    with pytest.raises(ValueError):
        with stage_timer(latency, "stage_2"):
            raise ValueError("stage blew up")
    assert latency.stage_2_ms is not None


# --- end-to-end via the gateway ---


def test_request_emits_telemetry_record(client: TestClient, log_path: Path) -> None:
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    records = read_records(log_path)
    assert len(records) == 1
    record = records[0]
    assert record["response_status"] == 200
    assert record["mitigation_action_applied"] == MitigationAction.ALLOW
    assert record["latency"]["total_ms"] is not None
    assert record["latency"]["upstream_ms"] is not None
    assert record["target_provider"] == "openai"


def test_health_is_not_logged(client: TestClient, log_path: Path) -> None:
    """Infrastructure endpoints must not pollute evaluation data."""
    client.get("/health")
    assert not log_path.exists() or read_records(log_path) == []


def test_session_id_is_recorded(client: TestClient, log_path: Path) -> None:
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "session_id": "sess_abc123",
        },
    )
    assert read_records(log_path)[0]["session_id"] == "sess_abc123"


def test_raw_prompt_absent_from_log_by_default(
    client: TestClient, log_path: Path
) -> None:
    """Ethical commitment: routine operation stores a hash, never the text."""
    secret = "my private medical question"
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": secret}]},
    )

    assert secret not in log_path.read_text(encoding="utf-8")
    chars = read_records(log_path)[0]["payload_characteristics"]
    assert chars["content"] is None
    assert chars["content_sha256"] is not None
    assert chars["char_length"] == len(secret)


def test_rejected_request_still_logged(client: TestClient, log_path: Path) -> None:
    """A 501/4xx must still produce a record - failures are data too."""
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    records = read_records(log_path)
    assert len(records) == 1
    assert records[0]["response_status"] == 501


def test_tagged_context_is_the_inspected_text(
    client: TestClient, log_path: Path
) -> None:
    """When context is supplied it defines what gets inspected, not messages."""
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "short"}],
            "context": {"web_content": "a much longer scraped page body"},
        },
    )

    chars = read_records(log_path)[0]["payload_characteristics"]
    assert chars["char_length"] == len("a much longer scraped page body")


def test_unbuilt_stages_report_not_run(client: TestClient, log_path: Path) -> None:
    """Stage I runs from Phase 6; later stages must not claim a verdict yet."""
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    results = read_records(log_path)[0]["detection_results"]
    assert results["stage_1_syntactic"] == "pass"
    assert results["stage_2_semantic"] == "not_run"
    assert results["stage_3_arbitration"] == "N/A"
