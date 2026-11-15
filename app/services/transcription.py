"""Speech-to-text abstraction.

The hackathon MVP ships ``MockTranscriber``, which simulates streaming ASR latency and
emits a scripted visit. A real provider implements the same ``Transcriber`` protocol
and is swapped in at the router without touching the stream or guardrail logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from app.schemas.audio import AudioChunkMessage, Speaker


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    text: str
    speaker: Speaker


class Transcriber(Protocol):
    async def transcribe(self, chunk: AudioChunkMessage) -> TranscriptSegment | None:
        """Return the finalized segment for this chunk, or ``None`` if it held no speech."""
        ...


# A UTI visit for a patient with a documented sulfa allergy and warfarin on the med list.
DEMO_SCRIPT: tuple[TranscriptSegment, ...] = (
    TranscriptSegment("What brings you in today?", "clinician"),
    TranscriptSegment("I've had burning when I pee for about three days.", "patient"),
    TranscriptSegment("Any fever, back pain, or blood in the urine?", "clinician"),
    TranscriptSegment("No, just the burning and going a lot.", "patient"),
    TranscriptSegment("That sounds like a simple urinary tract infection.", "clinician"),
    TranscriptSegment("Let's start you on Bactrim DS twice daily for three days.", "clinician"),
    TranscriptSegment("You can also take ibuprofen for the discomfort.", "clinician"),
)


class MockTranscriber:
    """Deterministic stand-in for streaming ASR. One instance per stream (holds a cursor)."""

    def __init__(
        self, *, latency_seconds: float, script: tuple[TranscriptSegment, ...] = DEMO_SCRIPT
    ) -> None:
        if not script:
            raise ValueError("script must not be empty")
        self._latency_seconds = latency_seconds
        self._script = script
        self._cursor = 0

    async def transcribe(self, chunk: AudioChunkMessage) -> TranscriptSegment | None:
        if not chunk.audio_bytes():  # Decode to exercise the real payload path.
            return None
        await asyncio.sleep(self._latency_seconds)
        segment = self._script[self._cursor % len(self._script)]
        self._cursor += 1
        return segment
