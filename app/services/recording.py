"""Transcribe a completed visit recording with OpenAI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from openai import AsyncOpenAI, OpenAIError
from openai.types.audio import TranscriptionDiarized
from pydantic import BaseModel

from app.core.config import Settings

AUDIO_FORMATS = {
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "mp4": "audio/mp4",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "webm": "audio/webm",
}
MAX_AUDIO_BYTES = 25_000_000


class TranscriptSegment(BaseModel):
    start: float | None
    text: str
    language: Literal["en", "es"] | None = None


@dataclass
class RecordingTranscript:
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)


async def transcribe_recording(
    client: AsyncOpenAI, settings: Settings, audio: bytes, extension: str
) -> RecordingTranscript:
    file = (f"recording.{extension}", audio, AUDIO_FORMATS[extension])
    model = settings.transcription_model
    async with asyncio.timeout(settings.transcription_timeout_seconds):
        transcriptions = client.with_options(
            timeout=settings.transcription_timeout_seconds
        ).audio.transcriptions
        if model == "gpt-4o-transcribe-diarize":
            result = await transcriptions.create(
                model=model,
                file=file,
                response_format="diarized_json",
                chunking_strategy="auto",
                stream=False,
            )
            if not isinstance(result, TranscriptionDiarized):
                raise OpenAIError("Transcription response did not include audio segments.")
            segments = [
                TranscriptSegment(start=segment.start, text=segment.text.strip())
                for segment in result.segments
                if segment.text.strip()
            ]
            return RecordingTranscript("\n".join(segment.text for segment in segments), segments)
        result_text = await transcriptions.create(model=model, file=file)
        text = result_text.text.strip()
        return RecordingTranscript(text, [TranscriptSegment(start=None, text=text)] if text else [])
