from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from twilio.request_validator import RequestValidator

from app.core.config import Settings
from app.main import create_app
from app.models import Encounter, EncounterStatus, IntakeChat, MessageDirection, Patient
from app.schemas.intake import IntakeExtractionSchema

AUTH_TOKEN = "test-auth-token"
BASE_URL = "https://escriba.test"
FROM = "+15555550100"


def sms_form(sid_suffix: str = "0", body: str = "I have had a cough") -> dict[str, str]:
    return {
        "MessageSid": "SM" + sid_suffix.rjust(32, "0"),
        "AccountSid": "AC" + "a" * 32,
        "From": FROM,
        "To": "+15555550199",
        "Body": body,
        "NumMedia": "0",
    }


def sign(params: dict[str, str]) -> str:
    return str(RequestValidator(AUTH_TOKEN).compute_signature(f"{BASE_URL}/webhook/twilio", params))


@pytest.fixture
async def app_client(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, object]]:
    settings = Settings(
        _env_file=None,
        demo_mode=True,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        twilio_auth_token=AUTH_TOKEN,
        public_base_url=BASE_URL,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
            yield client, app


async def post_sms(client: AsyncClient, params: dict[str, str], signature: str | None = None):
    headers = {"X-Twilio-Signature": signature if signature is not None else sign(params)}
    return await client.post("/webhook/twilio", data=params, headers=headers)


async def test_first_message_creates_encounter_and_returns_twiml(app_client) -> None:
    client, app = app_client
    response = await post_sms(client, sms_form())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "<Response><Message>" in response.text
    assert "main reason" in response.text

    async with app.state.sessionmaker() as session:
        patient = await session.scalar(select(Patient).where(Patient.phone_number == FROM))
        assert patient is not None
        encounter = await session.scalar(
            select(Encounter).where(Encounter.patient_id == patient.id)
        )
        assert encounter is not None
        assert encounter.status is EncounterStatus.INTAKE_IN_PROGRESS
        directions = (
            await session.scalars(
                select(IntakeChat.message_direction).where(IntakeChat.encounter_id == encounter.id)
            )
        ).all()
        assert sorted(directions) == [MessageDirection.INBOUND, MessageDirection.OUTBOUND]


async def test_follow_up_reuses_open_encounter(app_client) -> None:
    client, app = app_client
    await post_sms(client, sms_form("1"))
    response = await post_sms(client, sms_form("2", body="About 3 days"))

    assert response.status_code == 200
    assert "added that to your visit notes" in response.text
    async with app.state.sessionmaker() as session:
        assert await session.scalar(select(func.count()).select_from(Encounter)) == 1
        assert await session.scalar(select(func.count()).select_from(IntakeChat)) == 4


async def test_twilio_retry_is_idempotent(app_client) -> None:
    client, app = app_client
    params = sms_form("3")
    await post_sms(client, params)
    response = await post_sms(client, params)

    assert response.status_code == 200
    assert "received your message" in response.text
    async with app.state.sessionmaker() as session:
        assert await session.scalar(select(func.count()).select_from(IntakeChat)) == 2


async def test_invalid_signature_rejected(app_client) -> None:
    client, _ = app_client
    response = await post_sms(client, sms_form(), signature="forged")
    assert response.status_code == 403


async def test_missing_signature_rejected(app_client) -> None:
    client, _ = app_client
    response = await client.post("/webhook/twilio", data=sms_form())
    assert response.status_code == 403


async def test_malformed_payload_rejected(app_client) -> None:
    client, _ = app_client
    params = sms_form()
    params["From"] = "not-a-phone"
    response = await post_sms(client, params)
    assert response.status_code == 400


async def test_patient_delete_cascades(app_client) -> None:
    client, app = app_client
    await post_sms(client, sms_form("4"))
    async with app.state.sessionmaker() as session:
        patient = await session.scalar(select(Patient))
        await session.delete(patient)
        await session.commit()
        assert await session.scalar(select(func.count()).select_from(IntakeChat)) == 0
        assert await session.scalar(select(func.count()).select_from(Encounter)) == 0


def test_extraction_schema_normalizes_and_forbids_extra() -> None:
    parsed = IntakeExtractionSchema.model_validate(
        {
            "chief_complaint": " cough ",
            "symptom_duration": "3 days",
            "reported_medications": [
                {"name": "Lisinopril", "dosage": "10 mg"},
                {"name": "lisinopril", "dosage": "10 MG"},
            ],
            "reported_allergies": [{"allergen": "Penicillin", "reaction": "rash"}],
        }
    )
    assert parsed.chief_complaint == "cough"
    assert [m.display() for m in parsed.reported_medications] == ["Lisinopril 10 mg"]

    with pytest.raises(ValueError):
        IntakeExtractionSchema.model_validate(
            {"chief_complaint": "x", "symptom_duration": "y", "diagnosis": "invented"}
        )


def test_production_rejects_unsafe_config() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, environment="production", twilio_auth_token="t")
