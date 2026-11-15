"""Orchestrator endpoints: real-time formulary verification and 1-click clinical order approval."""

from __future__ import annotations

import logging
import uuid
from typing import cast

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.database import DbSession
from app.core.dependencies import SettingsDep
from app.models.domain import (
    WORKSPACE_PROVIDER_ID,
    ClinicalDocument,
    DocumentStatus,
    EncounterStatus,
    utcnow,
)
from app.schemas.clinical import OrderType
from app.schemas.orchestrator import (
    EnrichedOrder,
    OrderApprovalRequest,
    OrderApprovalResponse,
)
from app.services.formulary_mock import FormularyService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["orchestrator"])


@router.post(
    "/{document_id}/verify-orders",
    response_model=list[EnrichedOrder],
    summary="Verify formulary coverage and copay tiers for staged prescriptions",
    responses={
        200: {"description": "Staged orders enriched with formulary coverage results."},
        404: {"description": "Clinical document not found."},
    },
)
async def verify_orders(
    document_id: uuid.UUID,
    session: DbSession,
    settings: SettingsDep,
) -> list[EnrichedOrder]:
    """Inspect staged orders and evaluate real-time pharmacy benefit coverage for prescriptions."""
    if not settings.demo_mode:
        raise HTTPException(status_code=503, detail="A formulary provider is not configured.")
    doc = await session.scalar(
        select(ClinicalDocument)
        .options(selectinload(ClinicalDocument.encounter))
        .where(ClinicalDocument.id == document_id)
    )
    if doc is None or doc.encounter.provider_id == WORKSPACE_PROVIDER_ID:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical document {document_id} not found.",
        )

    formulary_service = FormularyService()
    patient_id = str(doc.encounter.patient_id) if doc.encounter else "patient-unknown"

    enriched_orders: list[EnrichedOrder] = []
    updated_staged_dicts: list[dict[str, object]] = []

    for raw_order in doc.staged_orders:
        order_dict = dict(raw_order)
        order_id = str(order_dict.get("order_id") or order_dict.get("id") or uuid.uuid4())
        order_dict["order_id"] = order_id

        order_type = order_dict.get("order_type")
        if order_type == "prescription":
            desc = str(order_dict.get("description", ""))
            verification = await formulary_service.check_coverage(desc, patient_id)
        else:
            verification = None

        enriched = EnrichedOrder(
            order_id=order_id,
            order_type=cast(OrderType, order_dict["order_type"]),
            description=str(order_dict.get("description", "")),
            icd10_rationale=str(order_dict.get("icd10_rationale", "")),
            transcript_timestamp_reference=str(
                order_dict.get("transcript_timestamp_reference", "")
            ),
            formulary_verification=verification,
        )
        enriched_orders.append(enriched)
        updated_staged_dicts.append(enriched.model_dump(mode="json"))

    doc.staged_orders = updated_staged_dicts
    await session.commit()

    logger.info(
        "Formulary verification complete for doc=%s orders=%d", document_id, len(enriched_orders)
    )
    return enriched_orders


@router.post(
    "/{document_id}/approve",
    response_model=OrderApprovalResponse,
    summary="1-Click approval for clinical document and staged orders",
    responses={
        200: {"description": "Document approved locally."},
        404: {"description": "Clinical document not found."},
        409: {"description": "Document is already approved."},
    },
)
async def approve_document(
    document_id: uuid.UUID,
    body: OrderApprovalRequest,
    session: DbSession,
) -> OrderApprovalResponse:
    """Execute 1-click clinical sign-off, transition document to APPROVED, and complete encounter.
    """
    doc = await session.scalar(
        select(ClinicalDocument)
        .options(selectinload(ClinicalDocument.encounter))
        .where(ClinicalDocument.id == document_id)
    )
    if doc is None or doc.encounter.provider_id == WORKSPACE_PROVIDER_ID:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical document {document_id} not found.",
        )

    if doc.status == DocumentStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Clinical document {document_id} has already been approved.",
        )

    now = utcnow()
    doc.status = DocumentStatus.APPROVED
    doc.approved_by = body.clinician_id
    doc.approved_at = now

    approved_count = (
        len(body.approved_order_ids)
        if body.approved_order_ids
        else len(doc.staged_orders)
    )

    if doc.encounter is not None:
        doc.encounter.status = EncounterStatus.COMPLETED

    await session.commit()
    await session.refresh(doc)

    logger.info(
        "Document approved doc=%s by clinician=%s orders=%d",
        document_id,
        body.clinician_id,
        approved_count,
    )

    encounter_status = (
        doc.encounter.status if doc.encounter is not None else EncounterStatus.COMPLETED
    )

    return OrderApprovalResponse(
        document_id=doc.id,
        encounter_id=doc.encounter_id,
        status=doc.status,
        approved_by=body.clinician_id,
        approved_at=now,
        approved_orders_count=approved_count,
        encounter_status=encounter_status,
    )
