"""Inbound SMS intake and 48-hour post-visit triage orchestration."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import Settings
from app.models.domain import (
    DocumentStatus,
    Encounter,
    EncounterStatus,
    IntakeChat,
    MessageDirection,
    Patient,
    utcnow,
)
from app.schemas.intake import TwilioInboundSMS

logger = logging.getLogger(__name__)

OPEN_INTAKE_STATUSES: tuple[EncounterStatus, ...] = (
    EncounterStatus.SCHEDULED,
    EncounterStatus.INTAKE_IN_PROGRESS,
    EncounterStatus.INTAKE_COMPLETE,
    EncounterStatus.IN_PROGRESS,
)

EMERGENCY_NOTICE = "If this is an emergency, call 911."
GREETING_REPLY = (
    "Thanks for contacting your care team. To prepare for your visit, please tell us "
    f"the main reason you're coming in and how long it has been going on. {EMERGENCY_NOTICE}"
)
FOLLOW_UP_REPLY = f"Thank you, we've added that to your visit notes. {EMERGENCY_NOTICE}"
DUPLICATE_REPLY = "Thank you, we've received your message."

TRIAGE_NEGATIVE_REPLY = (
    "We have alerted your care team about your symptoms and flagged this for priority review. "
    "A clinician will follow up with you. If this is a medical emergency, call 911 or go to "
    "the nearest emergency room immediately."
)
TRIAGE_POSITIVE_REPLY = (
    "Thank you for updating us! We are glad to hear you are improving. Please continue taking "
    "your medications as prescribed, and text us if any new symptoms arise."
)

RED_FLAG_KEYWORDS = (
    "worse",
    "dizzy",
    "dizziness",
    "pain",
    "hurt",
    "hurts",
    "fever",
    "nausea",
    "vomit",
    "vomiting",
    "bleeding",
    "rash",
    "swelling",
    "breath",
    "shortness of breath",
    "chest pain",
    "side effect",
    "adverse",
    "sick",
    "terrible",
    "reaction",
)


@dataclass(frozen=True, slots=True)
class IntakeTurnResult:
    encounter_id: uuid.UUID
    reply_text: str
    is_duplicate: bool


class IntakeService:
    """Routes inbound SMS turns to Pre-Visit Intake or Post-Visit Triage."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._openai_client = openai_client

    async def handle_inbound_sms(self, sms: TwilioInboundSMS) -> IntakeTurnResult:
        duplicate = await self._find_duplicate(sms.message_sid)
        if duplicate is not None:
            return duplicate

        try:
            result = await self._record_turn(sms)
            await self._session.commit()
        except IntegrityError:
            # Concurrent delivery of the same MessageSid (Twilio retry) won the race.
            await self._session.rollback()
            duplicate = await self._find_duplicate(sms.message_sid)
            if duplicate is None:
                raise
            return duplicate

        logger.info(
            "Recorded message sid=%s encounter=%s", sms.message_sid, result.encounter_id
        )
        return result

    async def _find_duplicate(self, message_sid: str) -> IntakeTurnResult | None:
        existing = await self._session.scalar(
            select(IntakeChat).where(IntakeChat.twilio_message_sid == message_sid)
        )
        if existing is None:
            return None
        logger.info("Duplicate Twilio delivery ignored sid=%s", message_sid)
        return IntakeTurnResult(
            encounter_id=existing.encounter_id, reply_text=DUPLICATE_REPLY, is_duplicate=True
        )

    async def _record_turn(self, sms: TwilioInboundSMS) -> IntakeTurnResult:
        patient = await self._get_or_create_patient(sms.from_number)

        latest_encounter = await self._session.scalar(
            select(Encounter)
            .options(selectinload(Encounter.clinical_documents))
            .where(Encounter.patient_id == patient.id)
            .order_by(Encounter.created_at.desc(), Encounter.scheduled_time.desc())
            .limit(1)
        )

        is_post_visit = False
        if latest_encounter is not None:
            is_post_visit = latest_encounter.status is EncounterStatus.COMPLETED or any(
                doc.status == DocumentStatus.APPROVED
                for doc in latest_encounter.clinical_documents
            )

        if is_post_visit and latest_encounter is not None:
            return await self._record_triage_turn(sms, latest_encounter)

        return await self._record_intake_turn(sms, patient)

    async def _record_intake_turn(
        self, sms: TwilioInboundSMS, patient: Patient
    ) -> IntakeTurnResult:
        encounter, is_new_conversation = await self._get_or_create_open_encounter(patient)

        reply_text = GREETING_REPLY if is_new_conversation else FOLLOW_UP_REPLY
        now = utcnow()
        self._session.add_all(
            [
                IntakeChat(
                    encounter_id=encounter.id,
                    message_direction=MessageDirection.INBOUND,
                    message_body=sms.body,
                    timestamp=now,
                    twilio_message_sid=sms.message_sid,
                ),
                IntakeChat(
                    encounter_id=encounter.id,
                    message_direction=MessageDirection.OUTBOUND,
                    message_body=reply_text,
                    timestamp=now,
                ),
            ]
        )
        if encounter.status is EncounterStatus.SCHEDULED:
            encounter.status = EncounterStatus.INTAKE_IN_PROGRESS

        await self._session.flush()
        return IntakeTurnResult(
            encounter_id=encounter.id, reply_text=reply_text, is_duplicate=False
        )

    async def _record_triage_turn(
        self, sms: TwilioInboundSMS, encounter: Encounter
    ) -> IntakeTurnResult:
        now = utcnow()
        is_negative, reason = await self._evaluate_triage_outcome(sms.body)

        reply_text = TRIAGE_NEGATIVE_REPLY if is_negative else TRIAGE_POSITIVE_REPLY
        if is_negative:
            encounter.triage_alert = True
            encounter.triage_notes = reason
            logger.warning(
                "Post-visit triage alert flagged encounter=%s reason='%s'",
                encounter.id,
                reason,
            )

        self._session.add_all(
            [
                IntakeChat(
                    encounter_id=encounter.id,
                    message_direction=MessageDirection.INBOUND,
                    message_body=sms.body,
                    timestamp=now,
                    twilio_message_sid=sms.message_sid,
                ),
                IntakeChat(
                    encounter_id=encounter.id,
                    message_direction=MessageDirection.OUTBOUND,
                    message_body=reply_text,
                    timestamp=now,
                ),
            ]
        )
        await self._session.flush()
        return IntakeTurnResult(
            encounter_id=encounter.id, reply_text=reply_text, is_duplicate=False
        )

    async def _evaluate_triage_outcome(self, message_body: str) -> tuple[bool, str]:
        """Classify message for negative clinical outcomes, side effects, or complications."""
        normalized = message_body.lower().strip()

        # 1. Try LLM classification if OpenAI client is available
        if self._openai_client is not None:
            try:
                system_prompt = (
                    "You are a clinical post-visit triage engine. Determine if the patient's "
                    "message indicates a negative outcome, worsening symptoms, severe side "
                    "effects, or a complication requiring provider attention. "
                    "Respond with strict JSON: "
                    '{"is_negative_outcome": bool, "triage_reason": "concise explanation"}'
                )
                completion = await self._openai_client.chat.completions.create(
                    model=self._settings.guardrail_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message_body},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = completion.choices[0].message.content
                if content:
                    data = json.loads(content)
                    is_neg = bool(data.get("is_negative_outcome"))
                    reason = str(data.get("triage_reason", "Reported symptoms require review."))
                    return is_neg, reason
            except Exception:
                logger.exception("LLM triage classification failed; falling back to heuristic")

        # 2. Heuristic fallback
        for kw in RED_FLAG_KEYWORDS:
            if kw in normalized:
                return True, f"Patient reported red-flag symptom: '{kw}'"

        return False, "Patient reports stable or improving condition."

    async def _get_or_create_patient(self, phone_number: str) -> Patient:
        patient = await self._session.scalar(
            select(Patient).where(Patient.phone_number == phone_number)
        )
        if patient is not None:
            return patient
        patient = Patient(
            phone_number=phone_number,
            demographics={"source": "sms_intake", "identity_verified": False},
        )
        self._session.add(patient)
        await self._session.flush()
        return patient

    async def _get_or_create_open_encounter(self, patient: Patient) -> tuple[Encounter, bool]:
        encounter = await self._session.scalar(
            select(Encounter)
            .where(
                Encounter.patient_id == patient.id,
                Encounter.status.in_(OPEN_INTAKE_STATUSES),
            )
            .order_by(Encounter.scheduled_time.desc())
            .limit(1)
        )
        if encounter is not None:
            has_messages = await self._session.scalar(
                select(IntakeChat.id).where(IntakeChat.encounter_id == encounter.id).limit(1)
            )
            return encounter, has_messages is None

        encounter = Encounter(
            patient_id=patient.id,
            provider_id=self._settings.default_provider_id,
            scheduled_time=utcnow(),
            status=EncounterStatus.SCHEDULED,
        )
        self._session.add(encounter)
        await self._session.flush()
        return encounter, True
