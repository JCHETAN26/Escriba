"""Integration tests for the POST /encounter/{encounter_id}/synthesize REST endpoint."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import Settings
from app.main import create_app
from app.models.domain import (
    ClinicalDocument,
    DocumentStatus,
    Encounter,
    EncounterStatus,
    Patient,
    utcnow,
)
from tests.fakes import SAMPLE_SYNTHESIS_DICT, FakeOpenAI, make_rate_limit_error

SAMPLE_TRANSCRIPT = (
    "Doctor: Good morning! How can I help you today? "
    "Patient: Doctor, my face hurts and I have intense sinus pressure for the past 4 days. "
    "Doctor: Let's check... Purulent drainage and maxillary sinus tenderness. "
    "I will write a prescription for Augmentin twice a day for a week. "
    "If this does not resolve after the antibiotic course, we will refer you to ENT."
)


async def _seed_encounter(
    app: FastAPI,
    status: EncounterStatus = EncounterStatus.IN_PROGRESS,
    transcript: str | None = None,
) -> uuid.UUID:
    async with app.state.sessionmaker() as session:
        patient = Patient(
            allergies=[{"allergen": "Sulfa drugs", "reaction": "hives"}],
            active_medications=[{"name": "Warfarin", "dosage": "5 mg", "frequency": "daily"}],
        )
        encounter = Encounter(
            patient=patient,
            provider_id="dr-test",
            scheduled_time=utcnow(),
            status=status,
            transcript=transcript,
        )
        session.add(encounter)
        await session.commit()
        return encounter.id


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[tuple[TestClient, FastAPI, FakeOpenAI]]:
    settings = Settings(
        _env_file=None,
        demo_mode=True,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'synthesis.db'}",
        twilio_validate_signature=False,
        synthesis_max_transcript_chars=50_000,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        fake = FakeOpenAI()
        app.state.openai_client = fake
        yield client, app, fake


def test_synthesize_endpoint_success_from_body_transcript(harness) -> None:
    client, app, fake = harness
    encounter_id = client.portal.call(_seed_encounter, app)  # type: ignore[union-attr]
    fake.completions.responses.append(SAMPLE_SYNTHESIS_DICT)

    response = client.post(
        f"/encounter/{encounter_id}/synthesize",
        json={"transcript": SAMPLE_TRANSCRIPT},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["encounter_id"] == str(encounter_id)
    assert data["status"] == "review_pending"
    assert "soap_note" in data
    assert data["soap_note"]["subjective"] != ""
    assert data["soap_note"]["plan"] != ""
    assert len(data["staged_orders"]) == 2
    assert data["staged_orders"][0]["order_type"] == "prescription"
    assert "visit_summary_for_patient" in data

    # Verify DB persistence
    async def verify_db() -> None:
        async with app.state.sessionmaker() as session:
            enc = await session.scalar(
                select(Encounter)
                .options(selectinload(Encounter.clinical_documents))
                .where(Encounter.id == encounter_id)
            )
            assert enc is not None
            assert enc.status is EncounterStatus.DOCUMENTATION_PENDING
            assert enc.transcript == SAMPLE_TRANSCRIPT
            assert len(enc.clinical_documents) == 1
            doc: ClinicalDocument = enc.clinical_documents[0]
            assert doc.status is DocumentStatus.REVIEW_PENDING
            assert doc.generated_soap["assessment"] != ""
            assert len(doc.staged_orders) == 2
            assert doc.visit_summary_for_patient is not None

    client.portal.call(verify_db)  # type: ignore[union-attr]


def test_synthesize_endpoint_success_from_stored_transcript(harness) -> None:
    client, app, fake = harness
    encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_encounter, app, EncounterStatus.IN_PROGRESS, SAMPLE_TRANSCRIPT
    )
    fake.completions.responses.append(SAMPLE_SYNTHESIS_DICT)

    # Calling without body should use stored transcript
    response = client.post(f"/encounter/{encounter_id}/synthesize")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "review_pending"
    assert len(data["staged_orders"]) == 2


def test_synthesize_encounter_not_found(harness) -> None:
    client, _app, _fake = harness
    random_id = uuid.uuid4()
    response = client.post(
        f"/encounter/{random_id}/synthesize",
        json={"transcript": SAMPLE_TRANSCRIPT},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_synthesize_missing_transcript_returns_400(harness) -> None:
    client, app, _fake = harness
    encounter_id = client.portal.call(_seed_encounter, app)  # type: ignore[union-attr]

    response = client.post(
        f"/encounter/{encounter_id}/synthesize",
        json={"transcript": "   "},
    )
    assert response.status_code == 400
    assert "no transcript available" in response.json()["detail"].lower()


def test_synthesize_exceeding_max_chars_returns_413(harness) -> None:
    client, app, _fake = harness
    app.state.settings.synthesis_max_transcript_chars = 30
    encounter_id = client.portal.call(_seed_encounter, app)  # type: ignore[union-attr]

    response = client.post(
        f"/encounter/{encounter_id}/synthesize",
        json={"transcript": SAMPLE_TRANSCRIPT},
    )
    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json()["detail"]


def test_synthesize_rate_limit_returns_429(harness) -> None:
    client, app, fake = harness
    encounter_id = client.portal.call(_seed_encounter, app)  # type: ignore[union-attr]
    fake.completions.responses.append(make_rate_limit_error())

    response = client.post(
        f"/encounter/{encounter_id}/synthesize",
        json={"transcript": SAMPLE_TRANSCRIPT},
    )
    assert response.status_code == 429
    assert "rate limit exceeded" in response.json()["detail"].lower()


def test_synthesize_unconfigured_openai_returns_503(harness) -> None:
    client, app, _fake = harness
    app.state.openai_client = None
    encounter_id = client.portal.call(_seed_encounter, app)  # type: ignore[union-attr]

    response = client.post(
        f"/encounter/{encounter_id}/synthesize",
        json={"transcript": SAMPLE_TRANSCRIPT},
    )
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
