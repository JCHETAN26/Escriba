from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
from openai import RateLimitError

SULFA_ALERT = {
    "severity": "high",
    "trigger_entity": "Bactrim",
    "context_conflict": "Patient has a documented Sulfa allergy (hives).",
    "suggested_action": "Use a non-sulfonamide agent such as nitrofurantoin.",
}

SAMPLE_SYNTHESIS_DICT: dict[str, Any] = {
    "soap_note": {
        "subjective": (
            "Patient is a 45-year-old presenting with 4 days of sinus pressure and facial pain. "
            "Reports significant stress at work and caregiver fatigue."
        ),
        "objective": (
            "T 98.6 F, BP 120/80 mmHg. Tenderness over bilateral maxillary sinuses. "
            "Purulent drainage in middle meatus."
        ),
        "assessment": "Acute maxillary sinusitis. Chronic hypertension, stable.",
        "plan": (
            "Start Amoxicillin-Clavulanate 875-125mg BID x 7 days. Continue Lisinopril. "
            "Return precautions for vision changes or high fevers."
        ),
    },
    "staged_orders": [
        {
            "order_type": "prescription",
            "description": "Amoxicillin-Clavulanate 875-125mg PO BID x 7 days",
            "icd10_rationale": "Acute maxillary sinusitis (J01.00)",
            "transcript_timestamp_reference": (
                "I will write a prescription for Augmentin twice a day for a week."
            ),
        },
        {
            "order_type": "referral",
            "description": "ENT referral for persistent chronic sinusitis evaluation",
            "icd10_rationale": "Recurrent sinusitis (J01.00)",
            "transcript_timestamp_reference": (
                "If this does not resolve after the antibiotic course, we will refer you to ENT."
            ),
        },
    ],
    "visit_summary_for_patient": (
        "You have a bacterial sinus infection. Take your antibiotic medicine twice every day for "
        "7 days until the bottle is empty. You can rinse your nose with saline spray. Call our "
        "office or go to urgent care if your fever gets worse or your vision changes."
    ),
}


def verdict(alert: dict[str, str] | None) -> str:
    return json.dumps({"alert": alert})


def make_rate_limit_error(message: str = "Rate limit reached") -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(message=message, response=response, body=None)


class FakeCompletions:
    """Stands in for ``AsyncOpenAI().chat.completions``. Queue responses or exceptions."""

    def __init__(self) -> None:
        self.responses: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.delay_seconds = 0.0

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        response = self.responses.pop(0) if self.responses else verdict(None)
        if isinstance(response, BaseException):
            raise response
        message = SimpleNamespace(content=response, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        response = self.responses.pop(0) if self.responses else SAMPLE_SYNTHESIS_DICT
        if isinstance(response, BaseException):
            raise response
        response_format = kwargs.get("response_format")
        if isinstance(response, str):
            parsed = (
                response_format.model_validate_json(response)
                if response_format
                else json.loads(response)
            )
        elif isinstance(response, dict):
            parsed = response_format.model_validate(response) if response_format else response
        else:
            parsed = response
        message = SimpleNamespace(parsed=parsed, refusal=None, content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))
