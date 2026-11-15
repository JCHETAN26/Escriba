"""Tests for Twilio inbound webhook routing: Pre-Visit Intake vs Post-Visit Triage."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.main import create_app
from app.models.domain import (
    Encounter,
    EncounterStatus,
    IntakeChat,
    MessageDirection,
    Patient,
    utcnow,
)
from app.services.intake import (
    TRIAGE_NEGATIVE_REPLY,
    TRIAGE_POSITIVE_REPLY,
)
from tests.fakes import FakeOpenAI

PHONE = "+15555550999"


async def _seed_patient_and_encounter(
    app: FastAPI,
    status: EncounterStatus,
    phone: str = PHONE,
) -> tuple[uuid.UUID, uuid.UUID]:
    async with app.state.sessionmaker() as session:
        patient = Patient(
            phone_number=phone,
            demographics={"first_name": "Elena"},
            allergies=[],
            active_medications=[],
        )
        encounter = Encounter(
            patient=patient,
            provider_id="dr-test",
            scheduled_time=utcnow(),
            status=status,
        )
        session.add(encounter)
        session.add(patient)
        await session.commit()
        return patient.id, encounter.id


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[tuple[TestClient, FastAPI]]:
    settings = Settings(
        _env_file=None,
        demo_mode=True,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'triage_routing.db'}",
        twilio_validate_signature=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def post_sms(client: TestClient, body: str, from_number: str = PHONE) -> str:
    response = client.post(
        "/webhook/twilio",
        data={
            "MessageSid": f"SM{uuid.uuid4().hex}",
            "AccountSid": f"AC{uuid.uuid4().hex}",
            "From": from_number,
            "To": "+15555550100",
            "Body": body,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    return response.text


def test_inbound_sms_routes_to_pre_visit_when_scheduled(
    harness: tuple[TestClient, FastAPI]
) -> None:
    client, app = harness
    _patient_id, encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_patient_and_encounter, app, EncounterStatus.SCHEDULED
    )

    xml_reply = post_sms(client, "I have had a sore throat for 3 days.")
    assert "prepare for your visit" in xml_reply

    # Verify encounter moved to intake_in_progress, NOT flagged for triage
    async def verify_db() -> None:
        async with app.state.sessionmaker() as session:
            enc = await session.scalar(select(Encounter).where(Encounter.id == encounter_id))
            assert enc is not None
            assert enc.status is EncounterStatus.INTAKE_IN_PROGRESS
            assert enc.triage_alert is False

    client.portal.call(verify_db)  # type: ignore[union-attr]


def test_inbound_sms_routes_to_post_visit_triage_when_completed_negative(
    harness: tuple[TestClient, FastAPI]
) -> None:
    client, app = harness
    _patient_id, encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_patient_and_encounter, app, EncounterStatus.COMPLETED
    )

    xml_reply = post_sms(client, "I feel worse and I am so dizzy today.")
    assert "alerted your care team" in xml_reply
    assert "priority review" in xml_reply

    # Verify encounter is flagged on clinician dashboard
    async def verify_db() -> None:
        async with app.state.sessionmaker() as session:
            enc = await session.scalar(select(Encounter).where(Encounter.id == encounter_id))
            assert enc is not None
            assert enc.status is EncounterStatus.COMPLETED
            assert enc.triage_alert is True
            assert any(
                kw in (enc.triage_notes or "").lower() for kw in ("worse", "dizzy", "symptom")
            )

            chats = list(
                await session.scalars(
                    select(IntakeChat)
                    .where(IntakeChat.encounter_id == encounter_id)
                    .order_by(IntakeChat.timestamp)
                )
            )
            assert len(chats) == 2
            assert chats[0].message_direction is MessageDirection.INBOUND
            assert "dizzy" in chats[0].message_body
            assert chats[1].message_direction is MessageDirection.OUTBOUND
            assert TRIAGE_NEGATIVE_REPLY in chats[1].message_body

    client.portal.call(verify_db)  # type: ignore[union-attr]


def test_inbound_sms_routes_to_post_visit_triage_when_completed_positive(
    harness: tuple[TestClient, FastAPI]
) -> None:
    client, app = harness
    _patient_id, encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_patient_and_encounter, app, EncounterStatus.COMPLETED
    )

    xml_reply = post_sms(client, "Feeling much better, thank you! No issues at all.")
    assert "glad to hear you are improving" in xml_reply

    async def verify_db() -> None:
        async with app.state.sessionmaker() as session:
            enc = await session.scalar(select(Encounter).where(Encounter.id == encounter_id))
            assert enc is not None
            assert enc.triage_alert is False
            assert TRIAGE_POSITIVE_REPLY in xml_reply

    client.portal.call(verify_db)  # type: ignore[union-attr]


def test_inbound_triage_uses_llm_classification_when_available(
    harness: tuple[TestClient, FastAPI]
) -> None:
    client, app = harness
    _patient_id, encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_patient_and_encounter, app, EncounterStatus.COMPLETED
    )

    # Attach fake OpenAI client
    fake = FakeOpenAI()
    llm_payload: dict[str, Any] = {
        "is_negative_outcome": True,
        "triage_reason": "Patient reports severe blurred vision after starting medication.",
    }
    fake.completions.responses.append(json.dumps(llm_payload))
    app.state.openai_client = fake

    xml_reply = post_sms(client, "Something is wrong with my eyes.")
    assert "alerted your care team" in xml_reply

    async def verify_db() -> None:
        async with app.state.sessionmaker() as session:
            enc = await session.scalar(select(Encounter).where(Encounter.id == encounter_id))
            assert enc is not None
            assert enc.triage_alert is True
            assert "blurred vision" in (enc.triage_notes or "").lower()

    client.portal.call(verify_db)  # type: ignore[union-attr]
