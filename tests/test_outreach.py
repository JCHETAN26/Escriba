"""Tests for OutreachService and outbound communication endpoints."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select
from twilio.base.exceptions import TwilioRestException

from app.core.config import Settings
from app.main import create_app
from app.models.domain import (
    WORKSPACE_PROVIDER_ID,
    ClinicalDocument,
    DocumentStatus,
    Encounter,
    EncounterStatus,
    IntakeChat,
    MessageDirection,
    Patient,
    utcnow,
)
from app.services.outreach import OutreachService

SAMPLE_SUMMARY = (
    "You have a bacterial sinus infection. Take your antibiotic medicine twice every day for "
    "7 days. Rinse your nose with saline spray. Call us if your fever gets worse."
)


class MockTwilioClient:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []

    async def send_sms(self, to_number: str, body: str) -> str:
        sid = f"SM_mock_{uuid.uuid4().hex[:8]}"
        self.sent_messages.append({"to": to_number, "body": body, "sid": sid})
        return sid


async def _seed_outreach_encounter(
    app: FastAPI,
    phone: str | None = "+15555550123",
    first_name: str = "Jane",
    has_summary: bool = True,
    status: DocumentStatus = DocumentStatus.APPROVED,
    provider_id: str = "dr-test",
) -> uuid.UUID:
    async with app.state.sessionmaker() as session:
        patient = Patient(
            phone_number=phone,
            demographics={"first_name": first_name, "last_name": "Doe"},
            allergies=[],
            active_medications=[],
        )
        encounter = Encounter(
            patient=patient,
            provider_id=provider_id,
            scheduled_time=utcnow(),
            status=EncounterStatus.COMPLETED,
        )
        if has_summary:
            doc = ClinicalDocument(
                encounter=encounter,
                generated_soap={"assessment": "Sinusitis"},
                staged_orders=[],
                visit_summary_for_patient=SAMPLE_SUMMARY,
                status=status,
            )
            session.add(doc)
        session.add(encounter)
        await session.commit()
        return encounter.id


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[tuple[TestClient, FastAPI]]:
    settings = Settings(
        _env_file=None,
        demo_mode=True,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'outreach.db'}",
        twilio_validate_signature=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


async def test_dispatch_care_plan_service(harness: tuple[TestClient, FastAPI]) -> None:
    client, app = harness
    encounter_id = client.portal.call(_seed_outreach_encounter, app)  # type: ignore[union-attr]
    mock_twilio = MockTwilioClient()

    async with app.state.sessionmaker() as session:
        service = OutreachService(session, app.state.settings, twilio_client=mock_twilio)
        result = await service.dispatch_care_plan(encounter_id)

    assert result["encounter_id"] == str(encounter_id)
    assert result["patient_phone"] == "+15555550123"
    assert result["summary_text"] == SAMPLE_SUMMARY
    assert len(mock_twilio.sent_messages) == 1
    assert SAMPLE_SUMMARY in mock_twilio.sent_messages[0]["body"]

    # Verify IntakeChat record
    async with app.state.sessionmaker() as session:
        chats = list(
            await session.scalars(
                select(IntakeChat)
                .where(IntakeChat.encounter_id == encounter_id)
                .order_by(IntakeChat.timestamp)
            )
        )
        assert len(chats) == 1
        assert chats[0].message_direction is MessageDirection.OUTBOUND
        assert SAMPLE_SUMMARY in chats[0].message_body


async def test_trigger_demo_triage_service(harness: tuple[TestClient, FastAPI]) -> None:
    client, app = harness
    encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_outreach_encounter, app, "+15555550123", "Marcus"
    )
    mock_twilio = MockTwilioClient()

    async with app.state.sessionmaker() as session:
        service = OutreachService(session, app.state.settings, twilio_client=mock_twilio)
        result = await service.trigger_demo_triage(encounter_id)

    assert result["encounter_id"] == str(encounter_id)
    assert "Marcus" in result["triage_message"]
    assert "2 days after your visit" in result["triage_message"]
    assert len(mock_twilio.sent_messages) == 1


def test_api_dispatch_summary_endpoint(harness: tuple[TestClient, FastAPI]) -> None:
    client, app = harness
    encounter_id = client.portal.call(_seed_outreach_encounter, app)  # type: ignore[union-attr]

    response = client.post(f"/encounter/{encounter_id}/dispatch-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["encounter_id"] == str(encounter_id)
    assert data["patient_phone"] == "+15555550123"
    assert data["status"] == "dispatched"
    assert SAMPLE_SUMMARY in data["summary_text"]


def test_api_trigger_demo_triage_endpoint(harness: tuple[TestClient, FastAPI]) -> None:
    client, app = harness
    encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_outreach_encounter, app, "+15555550123", "Carlos"
    )

    response = client.post(f"/encounter/{encounter_id}/demo-trigger-triage")
    assert response.status_code == 200
    data = response.json()
    assert data["encounter_id"] == str(encounter_id)
    assert data["status"] == "sent"
    assert "Carlos" in data["triage_message"]


def test_api_dispatch_summary_missing_document_returns_400(
    harness: tuple[TestClient, FastAPI],
) -> None:
    client, app = harness
    encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_outreach_encounter, app, "+15555550123", "Jane", False
    )

    response = client.post(f"/encounter/{encounter_id}/dispatch-summary")
    assert response.status_code == 400
    assert "no approved clinical summary" in response.json()["detail"].lower()


def test_api_outreach_encounter_not_found(harness: tuple[TestClient, FastAPI]) -> None:
    client, _app = harness
    random_id = uuid.uuid4()
    response = client.post(f"/encounter/{random_id}/dispatch-summary")
    assert response.status_code == 404


def test_legacy_dispatch_does_not_send_unapproved_summary(
    harness: tuple[TestClient, FastAPI],
) -> None:
    client, app = harness
    encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_outreach_encounter, app, "+15555550123", "Jane", True, DocumentStatus.REVIEW_PENDING
    )
    assert client.post(f"/encounter/{encounter_id}/dispatch-summary").status_code == 400


def test_workspace_texts_only_approved_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        demo_mode=True,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'workspace-sms.db'}",
        twilio_validate_signature=False,
    )
    app = create_app(settings)
    headers = {"X-Escriba-Workspace": "1"}
    phone = "+15555550123"
    calls: list[dict[str, str]] = []
    fail = False

    def send(**kwargs: str) -> SimpleNamespace:
        calls.append(kwargs)
        if fail:
            raise TwilioRestException(400, "/Messages", msg="private details", code=21606)
        return SimpleNamespace(sid="SM_test", status="queued")

    monkeypatch.setattr(
        "twilio.rest.Client", lambda *_: SimpleNamespace(messages=SimpleNamespace(create=send))
    )
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 54321)) as client:
        assert client.portal is not None
        approved = client.portal.call(
            _seed_outreach_encounter,
            app, None, "Approved", True, DocumentStatus.APPROVED, WORKSPACE_PROVIDER_ID,
        )
        draft = client.portal.call(
            _seed_outreach_encounter,
            app, None, "Draft", True, DocumentStatus.REVIEW_PENDING, WORKSPACE_PROVIDER_ID,
        )
        payload = {"phone_number": phone}
        route = f"/api/visits/{approved}/text-summary"
        assert client.post(route, json=payload).status_code == 403
        assert client.post(route, headers=headers, json=payload).status_code == 503
        invalid_phone = client.post(route, headers=headers, json={"phone_number": "5550123"})
        assert invalid_phone.status_code == 422
        settings.twilio_account_sid = "AC_test"
        settings.twilio_auth_token = SecretStr("test-token")
        settings.twilio_from_number = "+15555550100"
        assert client.post(
            f"/api/visits/{draft}/text-summary", headers=headers, json=payload
        ).status_code == 409
        assert calls == []

        fail = True
        failed = client.post(route, headers=headers, json=payload)
        assert failed.status_code == 502
        assert "21606" in failed.json()["detail"]
        assert "private details" not in failed.text
        fail = False
        response = client.post(route, headers=headers, json=payload)
        assert response.status_code == 200
        assert response.json() == {"message_sid": "SM_test"}
        assert calls[-1] == {
            "to": phone,
            "from_": "+15555550100",
            "body": "Escriba Care Plan Summary:\n\n" + SAMPLE_SUMMARY,
        }
