"""Live encounter stream: audio/transcript in, guardrail alerts out."""

from __future__ import annotations

import logging
import secrets
import uuid

from fastapi import APIRouter, WebSocket
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.dependencies import SettingsDep
from app.schemas.audio import ErrorEvent
from app.services.encounter_stream import (
    WS_CLOSE_ENCOUNTER_CLOSED,
    WS_CLOSE_ENCOUNTER_NOT_FOUND,
    WS_CLOSE_INTERNAL_ERROR,
    WS_CLOSE_POLICY_VIOLATION,
    EncounterClosedError,
    EncounterNotFoundError,
    EncounterStreamSession,
    open_encounter_for_streaming,
)
from app.services.guardrail import GuardrailService
from app.services.transcription import MockTranscriber

logger = logging.getLogger(__name__)

router = APIRouter(tags=["encounter-stream"])


def _presented_token(websocket: WebSocket) -> str | None:
    """Bearer header for server clients; ``?token=`` for browsers (no custom WS headers)."""
    authorization = websocket.headers.get("authorization", "")
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() == "bearer" and credential:
        return credential.strip()
    return websocket.query_params.get("token")


def _is_authorized(websocket: WebSocket, settings: Settings) -> bool:
    if settings.stream_api_token is None:
        return True  # Development only; production startup requires a token.
    presented = _presented_token(websocket)
    if presented is None:
        return False
    expected = settings.stream_api_token.get_secret_value()
    return secrets.compare_digest(presented.encode(), expected.encode())


def _build_guardrail(websocket: WebSocket, settings: Settings) -> GuardrailService | None:
    client: AsyncOpenAI | None = websocket.app.state.openai_client
    if client is None:
        return None
    return GuardrailService(
        client,
        model=settings.guardrail_model,
        timeout_seconds=settings.guardrail_timeout_seconds,
        temperature=settings.guardrail_temperature,
    )


async def _reject(websocket: WebSocket, error: ErrorEvent, close_code: int) -> None:
    await websocket.send_text(error.model_dump_json())
    await websocket.close(code=close_code)


@router.websocket("/encounter/{encounter_id}/stream")
async def encounter_stream(
    websocket: WebSocket, encounter_id: uuid.UUID, settings: SettingsDep
) -> None:
    """Bidirectional encounter stream. See ``app.schemas.audio`` for the wire protocol."""
    if not settings.demo_mode:
        await websocket.close(code=WS_CLOSE_POLICY_VIOLATION)
        return
    if not _is_authorized(websocket, settings):
        # Closing before accept yields an HTTP 403 on the upgrade request.
        await websocket.close(code=WS_CLOSE_POLICY_VIOLATION)
        return

    await websocket.accept()

    try:
        patient_context = await open_encounter_for_streaming(
            websocket.app.state.sessionmaker, encounter_id
        )
    except EncounterNotFoundError:
        await _reject(
            websocket,
            ErrorEvent(code="encounter_not_found", message="Encounter not found."),
            WS_CLOSE_ENCOUNTER_NOT_FOUND,
        )
        return
    except EncounterClosedError:
        await _reject(
            websocket,
            ErrorEvent(code="encounter_closed", message="Encounter is completed or cancelled."),
            WS_CLOSE_ENCOUNTER_CLOSED,
        )
        return

    session = EncounterStreamSession(
        websocket=websocket,
        encounter_id=encounter_id,
        patient_context=patient_context,
        guardrail=_build_guardrail(websocket, settings),
        transcriber=MockTranscriber(latency_seconds=settings.mock_transcription_latency_ms / 1000),
        settings=settings,
    )
    try:
        await session.run()
    except Exception:
        logger.exception("Encounter stream crashed encounter=%s", encounter_id)
        try:
            await _reject(
                websocket,
                ErrorEvent(code="internal_error", message="Internal server error."),
                WS_CLOSE_INTERNAL_ERROR,
            )
        except Exception:
            logger.debug("Could not notify client of crash (socket already closed)")
