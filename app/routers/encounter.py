"""Encounter documentation endpoints: SOAP note and staged order synthesis."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, status
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.database import DbSession
from app.core.dependencies import SettingsDep
from app.core.prompts import SummaryLanguage
from app.models.domain import (
    WORKSPACE_PROVIDER_ID,
    ClinicalDocument,
    DocumentStatus,
    Encounter,
    EncounterStatus,
)
from app.schemas.clinical import EncounterSynthesisResult, SOAPNote, StagedOrder
from app.services.clinical_nlp import (
    ClinicalNLPService,
    ContextWindowExceededError,
    RateLimitExceededError,
    SynthesisFailedError,
)
from app.services.guardrail import PatientContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/encounter", tags=["encounter"])


class SynthesizeRequest(BaseModel):
    """Optional request payload for on-demand synthesis with a direct transcript."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    transcript: str | None = Field(
        default=None,
        description="Full visit transcript. If omitted, uses stored encounter transcript.",
    )


class SynthesizeResponse(BaseModel):
    """JSON response payload for the frontend's 1-Click Review Drawer."""

    model_config = ConfigDict(extra="ignore")

    document_id: uuid.UUID
    encounter_id: uuid.UUID
    status: DocumentStatus
    soap_note: SOAPNote
    staged_orders: list[StagedOrder]
    visit_summary_for_patient: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document_and_synthesis(
        cls, doc: ClinicalDocument, synthesis: EncounterSynthesisResult
    ) -> SynthesizeResponse:
        return cls(
            document_id=doc.id,
            encounter_id=doc.encounter_id,
            status=doc.status,
            soap_note=synthesis.soap_note,
            staged_orders=synthesis.staged_orders,
            visit_summary_for_patient=synthesis.visit_summary_for_patient,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )


@router.post(
    "/{encounter_id}/synthesize",
    response_model=SynthesizeResponse,
    summary="Synthesize clinical documentation and staged orders for an encounter",
    responses={
        200: {"description": "Structured clinical documentation generated successfully."},
        400: {"description": "Transcript missing or empty."},
        404: {"description": "Encounter not found."},
        413: {"description": "Transcript exceeds context window limit."},
        429: {"description": "OpenAI rate limit exceeded."},
        502: {"description": "Synthesis failed upstream."},
        503: {"description": "OpenAI client not configured."},
    },
)
async def synthesize_encounter(
    encounter_id: uuid.UUID,
    session: DbSession,
    settings: SettingsDep,
    request: Request,
    body: SynthesizeRequest | None = None,
) -> SynthesizeResponse:
    """Trigger clinical NLP synthesis to generate SOAP notes, staged orders, and patient summary."""
    openai_client = request.app.state.openai_client
    if openai_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clinical NLP synthesis service is not configured (missing OpenAI API key).",
        )

    encounter = await session.scalar(
        select(Encounter)
        .options(
            selectinload(Encounter.patient),
            selectinload(Encounter.clinical_documents),
        )
        .where(Encounter.id == encounter_id)
    )
    if encounter is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Encounter {encounter_id} not found.",
        )
    if encounter.provider_id == WORKSPACE_PROVIDER_ID:
        raise HTTPException(status_code=404, detail="Encounter not found.")

    body_transcript = body.transcript if body and body.transcript else None
    transcript_text = body_transcript or encounter.transcript
    if not transcript_text or not transcript_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No transcript available to synthesize for this encounter.",
        )

    if encounter.status in (EncounterStatus.COMPLETED, EncounterStatus.CANCELLED) or any(
        doc.status == DocumentStatus.APPROVED for doc in encounter.clinical_documents
    ):
        raise HTTPException(status_code=409, detail="This visit is closed.")

    clean_transcript = transcript_text.strip()
    synthesis = await generate_draft(
        openai_client, settings, clean_transcript, PatientContext.from_patient(encounter.patient)
    )

    encounter.transcript = clean_transcript

    clinical_doc: ClinicalDocument
    if encounter.clinical_documents:
        clinical_doc = encounter.clinical_documents[0]
    else:
        clinical_doc = ClinicalDocument(encounter_id=encounter.id)
        session.add(clinical_doc)

    populate_draft(clinical_doc, synthesis)

    if encounter.status in (
        EncounterStatus.SCHEDULED,
        EncounterStatus.IN_PROGRESS,
        EncounterStatus.INTAKE_COMPLETE,
    ):
        encounter.status = EncounterStatus.DOCUMENTATION_PENDING

    await session.commit()
    await session.refresh(clinical_doc)

    logger.info(
        "Synthesized documentation for encounter=%s doc_id=%s orders=%d",
        encounter.id,
        clinical_doc.id,
        len(synthesis.staged_orders),
    )

    return SynthesizeResponse.from_document_and_synthesis(clinical_doc, synthesis)


def populate_draft(doc: ClinicalDocument, synthesis: EncounterSynthesisResult) -> None:
    doc.generated_soap = synthesis.soap_note.model_dump(mode="json")
    doc.staged_orders = [order.model_dump(mode="json") for order in synthesis.staged_orders]
    doc.visit_summary_for_patient = synthesis.visit_summary_for_patient
    doc.status = DocumentStatus.REVIEW_PENDING


async def generate_draft(
    client: AsyncOpenAI,
    settings: SettingsDep,
    transcript: str,
    patient_context: PatientContext,
    *,
    summary_language: SummaryLanguage = "en",
    note_language: SummaryLanguage = "en",
) -> EncounterSynthesisResult:
    service = ClinicalNLPService(
        client,
        model=settings.synthesis_model,
        timeout_seconds=settings.synthesis_timeout_seconds,
        temperature=settings.synthesis_temperature,
        max_transcript_chars=settings.synthesis_max_transcript_chars,
    )
    try:
        return await service.synthesize_encounter(
            transcript,
            patient_context,
            summary_language=summary_language,
            note_language=note_language,
        )
    except ContextWindowExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429, detail="Note service rate limit exceeded. Try again."
        ) from exc
    except (SynthesisFailedError, ValueError) as exc:
        logger.warning("Clinical synthesis failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="The note service could not generate a draft. Try again."
        ) from exc
