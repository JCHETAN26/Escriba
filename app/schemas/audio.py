"""WebSocket wire protocol for ``WS /encounter/{encounter_id}/stream``.

Every frame is a JSON text frame with a ``type`` discriminator.

Client → server: ``audio_chunk`` | ``transcript_chunk`` | ``end_stream``
Server → client: ``session_ready`` | ``transcript`` | ``clinical_guardrail_alert`` |
                 ``error`` | ``session_closed``
"""

from __future__ import annotations

import base64
import binascii
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

Speaker = Literal["clinician", "patient", "unknown"]
SequenceNumber = Annotated[
    int, Field(ge=0, description="Client-assigned, monotonically increasing.")
]

# ~256 KiB of decoded audio per frame (base64 inflates by 4/3).
MAX_AUDIO_CHUNK_B64_CHARS = 350_000
MAX_TRANSCRIPT_CHUNK_CHARS = 2_000


# --------------------------------------------------------------------------- client → server


class _ClientMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class AudioChunkMessage(_ClientMessage):
    type: Literal["audio_chunk"]
    seq: SequenceNumber
    encoding: Literal["pcm_s16le", "opus", "webm_opus"] = "pcm_s16le"
    sample_rate_hz: int = Field(default=16_000, ge=8_000, le=48_000)
    data: str = Field(
        min_length=4,
        max_length=MAX_AUDIO_CHUNK_B64_CHARS,
        description="Standard base64-encoded audio bytes.",
    )

    @field_validator("data")
    @classmethod
    def _must_be_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("data must be valid standard base64") from exc
        return value

    def audio_bytes(self) -> bytes:
        return base64.b64decode(self.data, validate=True)


class TranscriptChunkMessage(_ClientMessage):
    """Text produced by a client-side ASR (or typed, for demos)."""

    type: Literal["transcript_chunk"]
    seq: SequenceNumber
    text: str = Field(min_length=1, max_length=MAX_TRANSCRIPT_CHUNK_CHARS)
    speaker: Speaker = "unknown"
    is_final: bool = Field(
        default=True,
        description="Interim ASR hypotheses (false) are not evaluated by the guardrail.",
    )


class EndStreamMessage(_ClientMessage):
    type: Literal["end_stream"]


ClientMessage = Annotated[
    AudioChunkMessage | TranscriptChunkMessage | EndStreamMessage,
    Field(discriminator="type"),
]
CLIENT_MESSAGE_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


# --------------------------------------------------------------------------- server → client


class _ServerMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionReadyEvent(_ServerMessage):
    type: Literal["session_ready"] = "session_ready"
    session_id: uuid.UUID
    encounter_id: uuid.UUID
    guardrail_enabled: bool


class TranscriptEvent(_ServerMessage):
    type: Literal["transcript"] = "transcript"
    seq: int
    text: str
    speaker: Speaker
    is_final: bool = True
    source: Literal["mock_asr", "asr"]


class GuardrailFinding(_ServerMessage):
    """The LLM's verdict payload. Also the core of ``ClinicalGuardrailAlert``."""

    severity: Literal["high", "medium"]
    trigger_entity: str = Field(
        min_length=1, max_length=200, description="Drug/treatment as spoken, e.g. 'Bactrim'."
    )
    context_conflict: str = Field(
        min_length=1, max_length=500, description="e.g. 'Patient has Sulfa allergy (hives).'"
    )
    suggested_action: str = Field(min_length=1, max_length=500)


class ClinicalGuardrailAlert(GuardrailFinding):
    type: Literal["clinical_guardrail_alert"] = "clinical_guardrail_alert"
    alert_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    seq: int = Field(description="seq of the chunk that triggered the alert.")
    utterance: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_finding(
        cls, finding: GuardrailFinding, *, seq: int, utterance: str
    ) -> ClinicalGuardrailAlert:
        return cls(**finding.model_dump(), seq=seq, utterance=utterance)


ErrorCode = Literal[
    "invalid_message",
    "unsupported_frame",
    "message_too_large",
    "encounter_not_found",
    "encounter_closed",
    "guardrail_unavailable",
    "backpressure",
    "transcription_failed",
    "internal_error",
]


class ErrorEvent(_ServerMessage):
    type: Literal["error"] = "error"
    code: ErrorCode
    message: str
    retryable: bool = False
    seq: int | None = None


class SessionClosedEvent(_ServerMessage):
    type: Literal["session_closed"] = "session_closed"
    reason: Literal["client_ended"] = "client_ended"
    alerts_raised: int = Field(ge=0)


ServerEvent = (
    SessionReadyEvent | TranscriptEvent | ClinicalGuardrailAlert | ErrorEvent | SessionClosedEvent
)
ServerMessage = Annotated[ServerEvent, Field(discriminator="type")]
SERVER_MESSAGE_ADAPTER: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
