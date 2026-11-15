from __future__ import annotations

import base64
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.main import create_app
from app.models import Encounter, EncounterStatus, Patient
from app.models.domain import utcnow
from tests.fakes import SULFA_ALERT, FakeOpenAI, verdict

TOKEN = "stream-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


async def _seed(app: FastAPI, status: EncounterStatus) -> uuid.UUID:
    async with app.state.sessionmaker() as session:
        patient = Patient(
            allergies=[{"allergen": "Sulfa drugs", "reaction": "hives"}],
            active_medications=[{"name": "Warfarin", "dosage": "5 mg", "frequency": "daily"}],
        )
        encounter = Encounter(
            patient=patient, provider_id="dr-test", scheduled_time=utcnow(), status=status
        )
        session.add(encounter)
        await session.commit()
        return encounter.id


async def _status(app: FastAPI, encounter_id: uuid.UUID) -> EncounterStatus:
    async with app.state.sessionmaker() as session:
        result = await session.scalar(select(Encounter.status).where(Encounter.id == encounter_id))
        assert result is not None
        return result


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[tuple[TestClient, FastAPI, FakeOpenAI]]:
    settings = Settings(
        _env_file=None,
        demo_mode=True,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'stream.db'}",
        twilio_validate_signature=False,
        stream_api_token=TOKEN,
        mock_transcription_latency_ms=0,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        fake = FakeOpenAI()
        app.state.openai_client = fake
        yield client, app, fake


def seed(
    client: TestClient, app: FastAPI, status: EncounterStatus = EncounterStatus.SCHEDULED
) -> uuid.UUID:
    return client.portal.call(_seed, app, status)  # type: ignore[union-attr]


def transcript(seq: int, text: str) -> dict[str, Any]:
    return {"type": "transcript_chunk", "seq": seq, "text": text, "speaker": "clinician"}


def test_conflict_pushes_clinical_guardrail_alert(harness) -> None:
    client, app, fake = harness
    encounter_id = seed(client, app)
    fake.completions.responses.append(verdict(SULFA_ALERT))

    with client.websocket_connect(f"/encounter/{encounter_id}/stream", headers=AUTH) as ws:
        ready = ws.receive_json()
        assert ready["type"] == "session_ready"
        assert ready["guardrail_enabled"] is True

        ws.send_json(transcript(1, "Let's start you on Bactrim DS twice daily."))
        alert = ws.receive_json()
        assert alert["type"] == "clinical_guardrail_alert"
        assert alert["severity"] == "high"
        assert alert["trigger_entity"] == "Bactrim"
        assert alert["seq"] == 1
        assert "Sulfa" in alert["context_conflict"]

        ws.send_json({"type": "end_stream"})
        closed = ws.receive_json()
        assert closed == {"type": "session_closed", "reason": "client_ended", "alerts_raised": 1}

    assert client.portal.call(_status, app, encounter_id) is EncounterStatus.IN_PROGRESS  # type: ignore[union-attr]


def test_duplicate_alert_is_suppressed_and_safe_chunks_are_quiet(harness) -> None:
    client, app, fake = harness
    encounter_id = seed(client, app)
    fake.completions.responses.extend([verdict(SULFA_ALERT), verdict(None), verdict(SULFA_ALERT)])

    with client.websocket_connect(f"/encounter/{encounter_id}/stream", headers=AUTH) as ws:
        ws.receive_json()
        ws.send_json(transcript(1, "Start Bactrim now."))
        assert ws.receive_json()["type"] == "clinical_guardrail_alert"
        ws.send_json(transcript(2, "Drink plenty of fluids."))
        ws.send_json(transcript(3, "So Bactrim twice daily."))
        ws.send_json({"type": "end_stream"})
        assert ws.receive_json() == {
            "type": "session_closed",
            "reason": "client_ended",
            "alerts_raised": 1,
        }

    # Rolling context window is passed to later evaluations.
    last_prompt = fake.completions.calls[-1]["messages"][1]["content"]
    assert "clinician: Start Bactrim now." in last_prompt


def test_interim_transcripts_are_not_evaluated(harness) -> None:
    client, app, fake = harness
    encounter_id = seed(client, app)

    with client.websocket_connect(f"/encounter/{encounter_id}/stream", headers=AUTH) as ws:
        ws.receive_json()
        ws.send_json({**transcript(1, "Let's start Bact"), "is_final": False})
        ws.send_json({"type": "end_stream"})
        assert ws.receive_json()["type"] == "session_closed"
    assert fake.completions.calls == []


def test_audio_chunk_is_mock_transcribed_then_evaluated(harness) -> None:
    client, app, fake = harness
    encounter_id = seed(client, app)
    audio = base64.b64encode(b"\x00\x01" * 160).decode()

    with client.websocket_connect(f"/encounter/{encounter_id}/stream", headers=AUTH) as ws:
        ws.receive_json()
        ws.send_json({"type": "audio_chunk", "seq": 7, "data": audio})
        event = ws.receive_json()
        assert event["type"] == "transcript"
        assert event["source"] == "mock_asr"
        assert event["seq"] == 7
        assert event["text"] == "What brings you in today?"
        ws.send_json({"type": "end_stream"})
        assert ws.receive_json()["type"] == "session_closed"
    assert len(fake.completions.calls) == 1


def test_guardrail_failure_is_reported_not_silent(harness) -> None:
    client, app, fake = harness
    encounter_id = seed(client, app)
    fake.completions.responses.append(TimeoutError())

    with client.websocket_connect(f"/encounter/{encounter_id}/stream", headers=AUTH) as ws:
        ws.receive_json()
        ws.send_json(transcript(4, "Start Bactrim."))
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "guardrail_unavailable"
        assert error["seq"] == 4
        assert error["retryable"] is True


def test_invalid_message_does_not_end_session(harness) -> None:
    client, app, _ = harness
    encounter_id = seed(client, app)

    with client.websocket_connect(f"/encounter/{encounter_id}/stream?token={TOKEN}") as ws:
        ws.receive_json()
        ws.send_json({"type": "bogus"})
        assert ws.receive_json()["code"] == "invalid_message"
        ws.send_bytes(b"\x00")
        assert ws.receive_json()["code"] == "unsupported_frame"
        ws.send_json({"type": "end_stream"})
        assert ws.receive_json()["type"] == "session_closed"


def test_unknown_encounter_closes_4404(harness) -> None:
    client, _, _ = harness
    with client.websocket_connect(f"/encounter/{uuid.uuid4()}/stream", headers=AUTH) as ws:
        assert ws.receive_json()["code"] == "encounter_not_found"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4404


def test_closed_encounter_closes_4409(harness) -> None:
    client, app, _ = harness
    encounter_id = seed(client, app, EncounterStatus.COMPLETED)
    with client.websocket_connect(f"/encounter/{encounter_id}/stream", headers=AUTH) as ws:
        assert ws.receive_json()["code"] == "encounter_closed"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4409


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_unauthorized_connection_rejected(harness, headers: dict[str, str]) -> None:
    client, app, _ = harness
    encounter_id = seed(client, app)
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(f"/encounter/{encounter_id}/stream", headers=headers),
    ):
        pass
    assert exc.value.code == 1008


def test_guardrail_disabled_without_openai_key(harness) -> None:
    client, app, _ = harness
    app.state.openai_client = None
    encounter_id = seed(client, app)
    with client.websocket_connect(f"/encounter/{encounter_id}/stream", headers=AUTH) as ws:
        assert ws.receive_json()["guardrail_enabled"] is False
