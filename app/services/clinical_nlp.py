"""Clinical NLP documentation synthesis service using OpenAI Structured Outputs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError, RateLimitError

from app.core.config import DEFAULT_SYNTHESIS_MODEL
from app.core.prompts import SummaryLanguage, build_clinical_synthesis_messages
from app.models.domain import Patient
from app.schemas.clinical import EncounterSynthesisResult
from app.services.guardrail import PatientContext

logger = logging.getLogger(__name__)

DEFAULT_MAX_TRANSCRIPT_CHARS = 100_000


class ClinicalNLPError(Exception):
    """Base exception for clinical NLP synthesis errors."""


class ContextWindowExceededError(ClinicalNLPError):
    """Raised when the transcript exceeds the allowed length/context window."""


class RateLimitExceededError(ClinicalNLPError):
    """Raised when the LLM provider rate limit is encountered."""


class SynthesisFailedError(ClinicalNLPError):
    """Raised when synthesis fails due to parsing, API error, or refusal."""


class ClinicalNLPService:
    """Orchestrates structured clinical documentation synthesis via OpenAI."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = DEFAULT_SYNTHESIS_MODEL,
        timeout_seconds: float = 30.0,
        temperature: float = 0.1,
        max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_transcript_chars = max_transcript_chars

    async def synthesize_encounter(
        self,
        transcript: str,
        patient_context: Patient | PatientContext,
        *,
        summary_language: SummaryLanguage = "en",
        note_language: SummaryLanguage = "en",
    ) -> EncounterSynthesisResult:
        """Synthesize clinical documentation, staged orders, and patient summary from transcript.

        Raises:
            ValueError: If transcript is empty.
            ContextWindowExceededError: If transcript exceeds max allowed character limit.
            RateLimitExceededError: If OpenAI rate limits are hit.
            SynthesisFailedError: If the API call times out, fails, or model refuses.
        """
        raw_transcript = transcript.strip()
        if not raw_transcript:
            raise ValueError("Transcript cannot be empty.")

        if len(raw_transcript) > self._max_transcript_chars:
            raise ContextWindowExceededError(
                f"Transcript length ({len(raw_transcript)} chars) exceeds maximum limit "
                f"of {self._max_transcript_chars} chars."
            )

        ctx = (
            PatientContext.from_patient(patient_context)
            if isinstance(patient_context, Patient)
            else patient_context
        )

        messages = build_clinical_synthesis_messages(
            raw_transcript, ctx, summary_language=summary_language, note_language=note_language
        )

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "response_format": EncounterSynthesisResult,
            "temperature": self._temperature,
            "timeout": self._timeout_seconds,
        }

        try:
            completion = await asyncio.wait_for(
                self._client.beta.chat.completions.parse(**request_kwargs),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise SynthesisFailedError("Clinical synthesis timed out.") from exc
        except RateLimitError as exc:
            raise RateLimitExceededError(f"OpenAI rate limit exceeded: {exc}") from exc
        except OpenAIError as exc:
            raise SynthesisFailedError(
                f"OpenAI API error during clinical synthesis: {type(exc).__name__}: {exc}"
            ) from exc

        if not completion.choices:
            raise SynthesisFailedError("OpenAI returned no choices for clinical synthesis.")

        choice = completion.choices[0]
        message = choice.message

        if getattr(message, "refusal", None):
            raise SynthesisFailedError(f"Model refused clinical synthesis: {message.refusal}")

        parsed_result = getattr(message, "parsed", None)
        if parsed_result is None:
            raise SynthesisFailedError("OpenAI returned empty parsed synthesis result.")

        return cast(EncounterSynthesisResult, parsed_result)
