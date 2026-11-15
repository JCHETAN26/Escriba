"""Twilio SMS intake webhook."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import ValidationError
from twilio.twiml.messaging_response import MessagingResponse

from app.core.database import DbSession
from app.core.dependencies import SettingsDep
from app.core.security import verify_twilio_signature
from app.schemas.intake import TwilioInboundSMS
from app.services.intake import IntakeService

logger = logging.getLogger(__name__)

TWIML_MEDIA_TYPE = "application/xml"

router = APIRouter(prefix="/webhook", tags=["intake"])


def build_twiml_reply(message: str) -> str:
    twiml = MessagingResponse()
    twiml.message(message)
    return str(twiml)


@router.post(
    "/twilio",
    summary="Receive an inbound patient SMS from Twilio",
    response_class=Response,
    dependencies=[Depends(verify_twilio_signature)],
    responses={
        200: {"content": {TWIML_MEDIA_TYPE: {}}, "description": "TwiML reply"},
        400: {"description": "Malformed Twilio payload"},
        403: {"description": "Invalid or missing X-Twilio-Signature"},
    },
)
async def twilio_inbound_sms(
    request: Request, session: DbSession, settings: SettingsDep
) -> Response:
    form = await request.form()
    params = {key: value for key, value in form.multi_items() if isinstance(value, str)}

    try:
        sms = TwilioInboundSMS.model_validate(params)
    except ValidationError as exc:
        # Log field locations only; values may contain PHI.
        logger.warning(
            "Rejected malformed Twilio payload: %s", [err["loc"] for err in exc.errors()]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed Twilio payload"
        ) from exc

    openai_client = getattr(request.app.state, "openai_client", None)
    result = await IntakeService(
        session, settings, openai_client=openai_client
    ).handle_inbound_sms(sms)
    return Response(content=build_twiml_reply(result.reply_text), media_type=TWIML_MEDIA_TYPE)
