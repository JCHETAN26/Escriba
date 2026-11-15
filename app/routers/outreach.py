"""Outreach endpoints for patient care plan dispatch and 48-hour follow-up triage."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict

from app.core.database import DbSession
from app.core.dependencies import SettingsDep
from app.services.outreach import OutreachService, SmsProviderError, SmsUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/encounter", tags=["outreach"])


class DispatchSummaryResponse(BaseModel):
    """Response returned upon care plan SMS dispatch."""

    model_config = ConfigDict(extra="ignore")

    encounter_id: uuid.UUID
    patient_phone: str
    message_sid: str
    summary_text: str
    status: str = "dispatched"


class DemoTriageResponse(BaseModel):
    """Response returned upon 48-hour check-in dispatch."""

    model_config = ConfigDict(extra="ignore")

    encounter_id: uuid.UUID
    patient_phone: str
    message_sid: str
    triage_message: str
    status: str = "sent"


@router.post(
    "/{encounter_id}/dispatch-summary",
    response_model=DispatchSummaryResponse,
    summary="Dispatch approved 6th-grade care plan summary to patient via SMS",
    responses={
        200: {"description": "Care plan summary successfully dispatched."},
        400: {"description": "Patient phone or clinical summary missing."},
        404: {"description": "Encounter not found."},
    },
)
async def dispatch_care_plan(
    encounter_id: uuid.UUID,
    session: DbSession,
    settings: SettingsDep,
) -> DispatchSummaryResponse:
    """Send the clinician-approved care plan summary to the patient's mobile phone."""
    service = OutreachService(session, settings)
    try:
        result = await service.dispatch_care_plan(encounter_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except SmsUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SmsProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return DispatchSummaryResponse(
        encounter_id=uuid.UUID(result["encounter_id"]),
        patient_phone=result["patient_phone"],
        message_sid=result["message_sid"],
        summary_text=result["summary_text"],
    )


@router.post(
    "/{encounter_id}/demo-trigger-triage",
    response_model=DemoTriageResponse,
    summary="Trigger the demo 48-hour follow-up check-in SMS to patient",
    responses={
        200: {"description": "Follow-up triage SMS sent."},
        400: {"description": "Patient phone missing."},
        404: {"description": "Encounter not found."},
    },
)
async def demo_trigger_triage(
    encounter_id: uuid.UUID,
    session: DbSession,
    settings: SettingsDep,
) -> DemoTriageResponse:
    """Send the 48-hour post-visit triage check-in message to patient."""
    if not settings.demo_mode:
        raise HTTPException(status_code=404, detail="Endpoint not available.")
    service = OutreachService(session, settings)
    try:
        result = await service.trigger_demo_triage(encounter_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except SmsUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SmsProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return DemoTriageResponse(
        encounter_id=uuid.UUID(result["encounter_id"]),
        patient_phone=result["patient_phone"],
        message_sid=result["message_sid"],
        triage_message=result["triage_message"],
    )
