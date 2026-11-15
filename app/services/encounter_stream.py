"""Per-connection orchestration for the live encounter stream.

Concurrency model (one instance per WebSocket):
* The receive loop never awaits the LLM. Each finalized utterance spawns an evaluation
  task, so a slow guardrail call cannot stall audio ingestion.
* ``_llm_slots`` caps concurrent LLM calls; ``guardrail_max_pending`` caps queued work and
  emits a visible ``backpressure`` error instead of silently skipping a safety check.
* Mock transcription is serialized (FIFO lock) so transcript order matches audio order.
* All socket writes go through ``_send_lock``; Starlette WebSockets are not safe for
  concurrent sends.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Coroutine
from typing import Any

from fastapi import WebSocket
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.core.config import Settings
from app.models.domain import Encounter, EncounterStatus
from app.schemas.audio import (
    CLIENT_MESSAGE_ADAPTER,
    AudioChunkMessage,
    ClinicalGuardrailAlert,
    EndStreamMessage,
    ErrorCode,
    ErrorEvent,
    ServerEvent,
    SessionClosedEvent,
    SessionReadyEvent,
    Speaker,
    TranscriptChunkMessage,
    TranscriptEvent,
)
from app.services.guardrail import GuardrailService, GuardrailUnavailableError, PatientContext
from app.services.transcription import Transcriber

logger = logging.getLogger(__name__)

WS_CLOSE_NORMAL = 1000
WS_CLOSE_POLICY_VIOLATION = 1008
WS_CLOSE_INTERNAL_ERROR = 1011
WS_CLOSE_ENCOUNTER_NOT_FOUND = 4404
WS_CLOSE_ENCOUNTER_CLOSED = 4409

_STREAMABLE_TO_IN_PROGRESS = frozenset(
    {
        EncounterStatus.SCHEDULED,
        EncounterStatus.INTAKE_IN_PROGRESS,
        EncounterStatus.INTAKE_COMPLETE,
    }
)
_CLOSED_STATUSES = frozenset({EncounterStatus.COMPLETED, EncounterStatus.CANCELLED})


class EncounterNotFoundError(LookupError):
    pass


class EncounterClosedError(RuntimeError):
    pass


async def open_encounter_for_streaming(
    sessionmaker: async_sessionmaker[AsyncSession], encounter_id: uuid.UUID
) -> PatientContext:
    """Load the patient snapshot and mark the encounter in progress.

    Uses a short-lived DB session: a WebSocket may stay open for an entire visit and must
    not pin a pooled connection.
    """
    async with sessionmaker() as session:
        encounter = await session.scalar(
            select(Encounter)
            .options(selectinload(Encounter.patient))
            .where(Encounter.id == encounter_id)
        )
        if encounter is None:
            raise EncounterNotFoundError(str(encounter_id))
        if encounter.status in _CLOSED_STATUSES:
            raise EncounterClosedError(str(encounter_id))
        if encounter.status in _STREAMABLE_TO_IN_PROGRESS:
            encounter.status = EncounterStatus.IN_PROGRESS
            await session.commit()
        return PatientContext.from_patient(encounter.patient)


def _summarize_validation_error(exc: ValidationError) -> str:
    # Locations and messages only; never echo input values (may contain PHI).
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or 'message'}: {err['msg']}"
        for err in exc.errors()[:5]
    )


class EncounterStreamSession:
    def __init__(
        self,
        *,
        websocket: WebSocket,
        encounter_id: uuid.UUID,
        patient_context: PatientContext,
        guardrail: GuardrailService | None,
        transcriber: Transcriber,
        settings: Settings,
    ) -> None:
        self._ws = websocket
        self._encounter_id = encounter_id
        self._patient_context = patient_context
        self._guardrail = guardrail
        self._transcriber = transcriber
        self._settings = settings

        self.session_id = uuid.uuid4()
        self._send_lock = asyncio.Lock()
        self._transcribe_lock = asyncio.Lock()
        self._llm_slots = asyncio.Semaphore(settings.guardrail_max_concurrency)
        self._window: deque[str] = deque(maxlen=settings.guardrail_context_window)
        self._tasks: set[asyncio.Task[None]] = set()
        self._alert_keys: set[tuple[str, str]] = set()
        self._alerts_raised = 0
        self._closed = False

    async def run(self) -> None:
        await self._send(
            SessionReadyEvent(
                session_id=self.session_id,
                encounter_id=self._encounter_id,
                guardrail_enabled=self._guardrail is not None,
            )
        )
        try:
            if await self._receive_loop():
                await self._drain()
                await self._send(SessionClosedEvent(alerts_raised=self._alerts_raised))
                await self._ws.close(code=WS_CLOSE_NORMAL)
        except WebSocketDisconnect:
            pass
        finally:
            self._closed = True
            await self._cancel_pending()
            logger.info(
                "Stream ended session=%s encounter=%s alerts=%d",
                self.session_id,
                self._encounter_id,
                self._alerts_raised,
            )

    # ------------------------------------------------------------------ receive path

    async def _receive_loop(self) -> bool:
        """Return True when the client sent ``end_stream``, False on disconnect."""
        while True:
            frame = await self._ws.receive()
            if frame["type"] == "websocket.disconnect":
                return False

            text: str | None = frame.get("text")
            if text is None:
                await self._send_error("unsupported_frame", "Send JSON text frames only.")
                continue
            if len(text) > self._settings.stream_max_message_chars:
                await self._send_error("message_too_large", "Frame exceeds the size limit.")
                continue

            try:
                message = CLIENT_MESSAGE_ADAPTER.validate_json(text)
            except ValidationError as exc:
                await self._send_error("invalid_message", _summarize_validation_error(exc))
                continue

            match message:
                case EndStreamMessage():
                    return True
                case TranscriptChunkMessage():
                    if message.is_final:
                        self._submit_utterance(message.seq, message.text, message.speaker)
                case AudioChunkMessage():
                    self._spawn(self._transcribe_and_submit(message))

    async def _transcribe_and_submit(self, chunk: AudioChunkMessage) -> None:
        async with self._transcribe_lock:
            try:
                segment = await self._transcriber.transcribe(chunk)
            except Exception:
                logger.exception(
                    "Transcription failed session=%s seq=%d", self.session_id, chunk.seq
                )
                await self._send_error(
                    "transcription_failed", "Audio chunk could not be transcribed.", seq=chunk.seq,
                    retryable=True,
                )
                return
            if segment is None:
                return
            await self._send(
                TranscriptEvent(
                    seq=chunk.seq, text=segment.text, speaker=segment.speaker, source="mock_asr"
                )
            )
            self._submit_utterance(chunk.seq, segment.text, segment.speaker)

    def _submit_utterance(self, seq: int, text: str, speaker: Speaker) -> None:
        recent = tuple(self._window)
        self._window.append(f"{speaker}: {text}")
        if self._guardrail is None:
            return
        if len(self._tasks) >= self._settings.guardrail_max_pending:
            self._spawn(
                self._send_error(
                    "backpressure",
                    "Safety check skipped for this utterance (server overloaded); verify manually.",
                    seq=seq,
                    retryable=True,
                )
            )
            return
        self._spawn(self._evaluate(self._guardrail, seq, text, speaker, recent))

    # ------------------------------------------------------------------ guardrail path

    async def _evaluate(
        self,
        guardrail: GuardrailService,
        seq: int,
        text: str,
        speaker: Speaker,
        recent: tuple[str, ...],
    ) -> None:
        async with self._llm_slots:
            if self._closed:
                return
            try:
                finding = await guardrail.evaluate_transcript_chunk(
                    text, self._patient_context, recent_transcript=recent, speaker=speaker
                )
            except GuardrailUnavailableError as exc:
                logger.warning(
                    "Guardrail unavailable session=%s seq=%d: %s", self.session_id, seq, exc
                )
                await self._send_error(
                    "guardrail_unavailable",
                    "Safety check could not be completed for this utterance; verify manually.",
                    seq=seq,
                    retryable=True,
                )
                return

        if finding is None:
            return
        key = (finding.trigger_entity.casefold(), finding.severity)
        if key in self._alert_keys:
            return  # Same drug + severity already surfaced this session.
        self._alert_keys.add(key)
        self._alerts_raised += 1
        await self._send(ClinicalGuardrailAlert.from_finding(finding, seq=seq, utterance=text))

    # ------------------------------------------------------------------ plumbing

    async def _send(self, event: ServerEvent) -> None:
        if self._closed or self._ws.application_state != WebSocketState.CONNECTED:
            return
        async with self._send_lock:
            await self._ws.send_text(event.model_dump_json())

    async def _send_error(
        self, code: ErrorCode, message: str, *, seq: int | None = None, retryable: bool = False
    ) -> None:
        await self._send(ErrorEvent(code=code, message=message, seq=seq, retryable=retryable))

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(self._guarded(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _guarded(self, coro: Coroutine[Any, Any, None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._closed:
                logger.debug("Background task failed after close session=%s", self.session_id)
            else:
                logger.exception("Background task failed session=%s", self.session_id)

    async def _drain(self) -> None:
        """Let in-flight transcription/guardrail work finish before closing (bounded)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settings.stream_drain_timeout_seconds
        while self._tasks:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "Drain timed out session=%s pending=%d", self.session_id, len(self._tasks)
                )
                return
            await asyncio.wait(set(self._tasks), timeout=remaining)

    async def _cancel_pending(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
