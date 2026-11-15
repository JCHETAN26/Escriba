"""Integration tests for the Order Orchestrator endpoints: formulary check & 1-click approval."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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

SAMPLE_STAGED_ORDERS: list[dict[str, Any]] = [
    {
        "order_type": "prescription",
        "description": "Ozempic 0.5 mg/dose subcutaneous pen",
        "icd10_rationale": "Type 2 diabetes mellitus (E11.9)",
        "transcript_timestamp_reference": "Let's try you on Ozempic.",
    },
    {
        "order_type": "prescription",
        "description": "Amoxicillin 500 mg oral capsule",
        "icd10_rationale": "Acute otitis media (H66.90)",
        "transcript_timestamp_reference": "Take amoxicillin for 10 days.",
    },
    {
        "order_type": "lab",
        "description": "Hemoglobin A1c (HbA1c)",
        "icd10_rationale": "Diabetes monitoring (E11.9)",
        "transcript_timestamp_reference": "We will draw blood for an A1c today.",
    },
]


async def _seed_document(
    app: FastAPI,
    status: DocumentStatus = DocumentStatus.REVIEW_PENDING,
    orders: list[dict[str, Any]] | None = None,
) -> tuple[uuid.UUID, uuid.UUID]:
    async with app.state.sessionmaker() as session:
        patient = Patient(allergies=[], active_medications=[])
        encounter = Encounter(
            patient=patient,
            provider_id="dr-test",
            scheduled_time=utcnow(),
            status=EncounterStatus.DOCUMENTATION_PENDING,
        )
        doc = ClinicalDocument(
            encounter=encounter,
            generated_soap={"assessment": "Type 2 diabetes mellitus"},
            staged_orders=orders if orders is not None else SAMPLE_STAGED_ORDERS,
            status=status,
        )
        session.add(encounter)
        session.add(doc)
        await session.commit()
        return doc.id, encounter.id


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[tuple[TestClient, FastAPI]]:
    settings = Settings(
        _env_file=None,
        demo_mode=True,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'orchestrator.db'}",
        twilio_validate_signature=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def test_verify_orders_enriches_prescriptions(harness: tuple[TestClient, FastAPI]) -> None:
    client, app = harness
    doc_id, _encounter_id = client.portal.call(_seed_document, app)  # type: ignore[union-attr]

    response = client.post(f"/document/{doc_id}/verify-orders")
    assert response.status_code == 200
    orders = response.json()
    assert len(orders) == 3

    # Ozempic: flagged as requiring prior auth and having generic alternative
    ozempic = next(o for o in orders if "ozempic" in o["description"].lower())
    assert ozempic["formulary_verification"] is not None
    assert ozempic["formulary_verification"]["is_covered"] is False
    assert ozempic["formulary_verification"]["requires_prior_auth"] is True
    assert ozempic["formulary_verification"]["generic_alternative"] == "Metformin 500mg oral tablet"
    assert ozempic["formulary_verification"]["estimated_copay_usd"] == 950.00
    assert ozempic["order_id"] != ""

    # Amoxicillin: covered
    amox = next(o for o in orders if "amoxicillin" in o["description"].lower())
    assert amox["formulary_verification"] is not None
    assert amox["formulary_verification"]["is_covered"] is True
    assert amox["formulary_verification"]["requires_prior_auth"] is False
    assert amox["formulary_verification"]["generic_alternative"] is None

    # Lab order: formulary verification is None
    lab = next(o for o in orders if o["order_type"] == "lab")
    assert lab["formulary_verification"] is None


def test_verify_orders_document_not_found(harness: tuple[TestClient, FastAPI]) -> None:
    client, _app = harness
    random_id = uuid.uuid4()
    response = client.post(f"/document/{random_id}/verify-orders")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_1_click_approval_flow(harness: tuple[TestClient, FastAPI]) -> None:
    client, app = harness
    doc_id, _encounter_id = client.portal.call(_seed_document, app)  # type: ignore[union-attr]

    # Verify orders first to ensure IDs are assigned
    verify_resp = client.post(f"/document/{doc_id}/verify-orders")
    assert verify_resp.status_code == 200
    orders = verify_resp.json()
    approved_ids = [orders[0]["order_id"], orders[1]["order_id"]]

    # Execute 1-click approval
    approve_resp = client.post(
        f"/document/{doc_id}/approve",
        json={
            "clinician_id": "dr-smith-md",
            "approved_order_ids": approved_ids,
            "approval_notes": "All orders reviewed and confirmed with patient.",
        },
    )

    assert approve_resp.status_code == 200
    data = approve_resp.json()
    assert data["document_id"] == str(doc_id)
    assert data["status"] == "approved"
    assert data["approved_by"] == "dr-smith-md"
    assert data["approved_at"] is not None
    assert data["approved_orders_count"] == 2
    assert data["encounter_status"] == "completed"

    # Verify DB state directly
    async def verify_db() -> None:
        async with app.state.sessionmaker() as session:
            doc = await session.scalar(
                select(ClinicalDocument)
                .options(selectinload(ClinicalDocument.encounter))
                .where(ClinicalDocument.id == doc_id)
            )
            assert doc is not None
            assert doc.status is DocumentStatus.APPROVED
            assert doc.approved_by == "dr-smith-md"
            assert doc.approved_at is not None
            assert doc.encounter.status is EncounterStatus.COMPLETED

    client.portal.call(verify_db)  # type: ignore[union-attr]


def test_approve_already_approved_document_returns_409(
    harness: tuple[TestClient, FastAPI]
) -> None:
    client, app = harness
    doc_id, _encounter_id = client.portal.call(  # type: ignore[union-attr]
        _seed_document, app, DocumentStatus.APPROVED
    )

    response = client.post(
        f"/document/{doc_id}/approve",
        json={"clinician_id": "dr-smith-md"},
    )
    assert response.status_code == 409
    assert "already been approved" in response.json()["detail"].lower()


def test_approve_document_not_found(harness: tuple[TestClient, FastAPI]) -> None:
    client, _app = harness
    random_id = uuid.uuid4()
    response = client.post(
        f"/document/{random_id}/approve",
        json={"clinician_id": "dr-smith-md"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
