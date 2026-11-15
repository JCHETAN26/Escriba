"""Proactive patient outreach service for care plan delivery and 48-hour follow-up triage."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from twilio.base.exceptions import TwilioException

from app.core.config import Settings
from app.models.domain import (
    ClinicalDocument,
    DocumentStatus,
    Encounter,
    IntakeChat,
    MessageDirection,
    utcnow,
)

logger = logging.getLogger(__name__)


class SmsUnavailableError(RuntimeError):
    """The SMS sender is not configured."""


class SmsProviderError(RuntimeError):
    """The SMS provider did not confirm that it accepted the message."""


class OutreachService:
    """Orchestrates outbound patient communications over SMS."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        twilio_client: Any | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._twilio_client = twilio_client

    async def _send_sms(self, to_number: str, body: str, *, require_live: bool = False) -> str:
        """Send an outbound SMS without blocking the asyncio event loop."""
        from_number = self._settings.twilio_from_number or "+15555550100"

        # If a custom/mock client was passed directly
        if self._twilio_client is not None:
            if hasattr(self._twilio_client, "messages"):
                msg = await asyncio.to_thread(
                    self._twilio_client.messages.create,
                    to=to_number,
                    from_=from_number,
                    body=body,
                )
                return str(getattr(msg, "sid", "SM_mock"))
            if callable(getattr(self._twilio_client, "send_sms", None)):
                result = await self._twilio_client.send_sms(to_number, body)
                return str(result)

        # Official Twilio client if credentials are configured
        if (
            self._settings.twilio_account_sid
            and self._settings.twilio_auth_token
            and self._settings.twilio_from_number
        ):
            from twilio.rest import Client

            client = Client(
                self._settings.twilio_account_sid,
                self._settings.twilio_auth_token.get_secret_value(),
            )
            try:
                msg = await asyncio.to_thread(
                    client.messages.create,
                    to=to_number,
                    from_=from_number,
                    body=body,
                )
            except TwilioException as exc:
                code = getattr(exc, "code", None)
                suffix = f" (Twilio error {code})" if isinstance(code, int) else ""
                raise SmsProviderError(
                    f"Twilio did not accept the message{suffix}. Check the Twilio console."
                ) from exc
            except OSError as exc:
                raise SmsProviderError(
                    "Twilio acceptance could not be confirmed. Check the logs before retrying."
                ) from exc
            if not msg.sid or msg.status in {"failed", "undelivered", "canceled"}:
                raise SmsProviderError("Twilio reported a failed send. Check the Twilio console.")
            return str(msg.sid)

        if not require_live and self._settings.demo_mode:
            logger.info("Simulated outbound SMS")
            return f"SM_{uuid.uuid4().hex}"
        raise SmsUnavailableError(
            "SMS is not configured. Add the Twilio sender details and restart."
        )

    async def dispatch_care_plan(
        self,
        encounter_id: str | uuid.UUID,
        *,
        recipient: str | None = None,
        require_live: bool = False,
    ) -> dict[str, Any]:
        """Fetch approved clinical document and send the 6th-grade summary to patient."""
        enc_uuid = uuid.UUID(str(encounter_id))
        encounter = await self._session.scalar(
            select(Encounter)
            .options(
                selectinload(Encounter.patient),
                selectinload(Encounter.clinical_documents),
            )
            .where(Encounter.id == enc_uuid)
        )
        if encounter is None:
            raise LookupError(f"Encounter {encounter_id} not found.")

        phone_number = recipient or (encounter.patient.phone_number if encounter.patient else None)
        if not phone_number:
            raise ValueError(
                f"Patient associated with encounter {encounter_id} has no phone number."
            )

        doc: ClinicalDocument | None = next(
            (
                d
                for d in encounter.clinical_documents
                if d.status == DocumentStatus.APPROVED and d.visit_summary_for_patient
            ),
            None,
        )
        if doc is None or not doc.visit_summary_for_patient:
            raise ValueError(
                f"No approved clinical summary available for encounter {encounter_id}."
            )

        language = (
            encounter.patient.demographics.get("patient_language")
            or encounter.patient.demographics.get("preferred_language")
            if encounter.patient
            else None
        )
        heading = (
            "Escriba · Instrucciones de su visita"
            if language == "es"
            else "Escriba Care Plan Summary:"
        )
        message_body = f"{heading}\n\n{doc.visit_summary_for_patient}"
        if len(message_body) > 1600:
            raise ValueError("The approved summary is too long for one SMS message.")
        sid = await self._send_sms(phone_number, message_body, require_live=require_live)

        chat = IntakeChat(
            encounter_id=encounter.id,
            message_direction=MessageDirection.OUTBOUND,
            message_body=message_body,
            timestamp=utcnow(),
            twilio_message_sid=sid,
        )
        self._session.add(chat)
        await self._session.commit()

        logger.info(
            "Dispatched care plan SMS encounter=%s sid=%s",
            encounter.id,
            sid,
        )
        return {
            "encounter_id": str(encounter.id),
            "patient_phone": phone_number,
            "message_sid": sid,
            "summary_text": doc.visit_summary_for_patient,
        }

    async def trigger_demo_triage(self, encounter_id: str | uuid.UUID) -> dict[str, Any]:
        """Simulate the 48-hour follow-up check-in via outbound SMS."""
        enc_uuid = uuid.UUID(str(encounter_id))
        encounter = await self._session.scalar(
            select(Encounter)
            .options(selectinload(Encounter.patient))
            .where(Encounter.id == enc_uuid)
        )
        if encounter is None:
            raise LookupError(f"Encounter {encounter_id} not found.")

        if not encounter.patient or not encounter.patient.phone_number:
            raise ValueError(
                f"Patient associated with encounter {encounter_id} has no phone number."
            )

        patient_name = "there"
        if isinstance(encounter.patient.demographics, dict):
            first_name = encounter.patient.demographics.get("first_name")
            if not first_name and encounter.patient.demographics.get("name"):
                first_name = str(encounter.patient.demographics["name"]).split()[0]
            if first_name:
                patient_name = str(first_name).strip()

        message_body = (
            f"Hi {patient_name}, Escriba checking in 2 days after your visit. "
            "Are you experiencing any side effects from your new plan, or are your symptoms "
            "improving?"
        )
        sid = await self._send_sms(encounter.patient.phone_number, message_body)

        chat = IntakeChat(
            encounter_id=encounter.id,
            message_direction=MessageDirection.OUTBOUND,
            message_body=message_body,
            timestamp=utcnow(),
            twilio_message_sid=sid,
        )
        self._session.add(chat)
        await self._session.commit()

        logger.info(
            "Sent 48-hour triage check-in encounter=%s phone=%s sid=%s",
            encounter.id,
            encounter.patient.phone_number,
            sid,
        )
        return {
            "encounter_id": str(encounter.id),
            "patient_phone": encounter.patient.phone_number,
            "message_sid": sid,
            "triage_message": message_body,
        }
