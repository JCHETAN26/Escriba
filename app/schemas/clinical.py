"""Canonical clinical value objects shared by API schemas, LLM targets, and ORM JSON columns.

This module has no ORM dependencies so both ``app.schemas`` and ``app.models`` can import it.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

T = TypeVar("T")


class _ClinicalValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _blank_to_none(value: str | None) -> str | None:
    return value or None


class Medication(_ClinicalValue):
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Medication name as reported (brand or generic), e.g. 'Lisinopril'.",
    )
    dosage: str | None = Field(
        default=None,
        max_length=100,
        description="Strength with unit, e.g. '10 mg'. null if not stated.",
    )
    frequency: str | None = Field(
        default=None,
        max_length=100,
        description="How often it is taken, e.g. 'once daily'. null if not stated.",
    )

    _normalize_optional = field_validator("dosage", "frequency")(_blank_to_none)

    def display(self) -> str:
        return " ".join(part for part in (self.name, self.dosage, self.frequency) if part)


class Allergy(_ClinicalValue):
    allergen: str = Field(
        min_length=1,
        max_length=200,
        description="Substance the patient is allergic to, e.g. 'Sulfa drugs'.",
    )
    reaction: str | None = Field(
        default=None,
        max_length=200,
        description="Reaction experienced, e.g. 'hives'. null if not stated.",
    )

    _normalize_optional = field_validator("reaction")(_blank_to_none)

    def display(self) -> str:
        return f"{self.allergen} (reaction: {self.reaction})" if self.reaction else self.allergen


def _dedupe(items: Iterable[T], key: Callable[[T], Hashable]) -> list[T]:
    seen: set[Hashable] = set()
    result: list[T] = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def dedupe_medications(items: Iterable[Medication]) -> list[Medication]:
    """Drop exact duplicates (case-insensitive) while keeping distinct doses of one drug."""
    return _dedupe(
        items,
        lambda m: (m.name.casefold(), (m.dosage or "").casefold(), (m.frequency or "").casefold()),
    )


def dedupe_allergies(items: Iterable[Allergy]) -> list[Allergy]:
    """One entry per allergen. An entry with a documented reaction wins over one without."""
    by_allergen: dict[str, Allergy] = {}
    for allergy in items:
        key = allergy.allergen.casefold()
        existing = by_allergen.get(key)
        if existing is None:
            by_allergen[key] = allergy
        elif existing.reaction is None and allergy.reaction is not None:
            by_allergen[key] = Allergy(allergen=existing.allergen, reaction=allergy.reaction)
    return list(by_allergen.values())


_MEDICATION_LIST: TypeAdapter[list[Medication]] = TypeAdapter(list[Medication])
_ALLERGY_LIST: TypeAdapter[list[Allergy]] = TypeAdapter(list[Allergy])


def normalize_medication_records(value: object) -> list[dict[str, Any]]:
    """Validate and normalize a value destined for a JSON(B) medications column."""
    items = dedupe_medications(_MEDICATION_LIST.validate_python(value))
    return [m.model_dump(mode="json") for m in items]


def normalize_allergy_records(value: object) -> list[dict[str, Any]]:
    """Validate and normalize a value destined for a JSON(B) allergies column."""
    items = dedupe_allergies(_ALLERGY_LIST.validate_python(value))
    return [a.model_dump(mode="json") for a in items]


OrderType = Literal["prescription", "lab", "referral"]


class StagedOrder(BaseModel):
    """An actionable clinical order proposed or decided during the visit."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    order_type: OrderType = Field(
        description="Category of the order: 'prescription', 'lab', or 'referral'."
    )
    description: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Order details stated by the clinician. For a medication, copy the dose and route "
            "without inferring dosage form, pill count, or start timing."
        ),
    )
    icd10_rationale: str = Field(
        min_length=1,
        max_length=200,
        description="Documented reason for the order. Include only an explicitly supplied code.",
    )
    transcript_timestamp_reference: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Verbatim quote or audio anchor from the transcript justifying this order "
            "to ground clinical intent and prevent hallucinations."
        ),
    )


class SOAPNote(BaseModel):
    """Standard clinical SOAP (Subjective, Objective, Assessment, Plan) note structure."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    subjective: str = Field(
        min_length=1,
        description=(
            "Chief complaint, HPI, relevant review of systems, patient-reported symptoms, "
            "and psychosocial stressors (excluding non-clinical small talk)."
        ),
    )
    objective: str = Field(
        min_length=1,
        description=(
            "Vital signs, physical exam findings, and diagnostic results explicitly reported. "
            "Keep decisions to order or defer tests in Plan."
        ),
    )
    assessment: str = Field(
        min_length=1,
        description=(
            "The clinician's stated assessment and reasoning, with the same certainty. "
            "Use the assessed symptom if no diagnosis was stated. Attribute a cause only "
            "when the clinician explicitly did so."
        ),
    )
    plan: str = Field(
        min_length=1,
        description=(
            "The stated management plan, including decisions to order or defer tests and their "
            "timing. Preserve the exact prerequisite for each conditional treatment and the "
            "separate conditions for seeking follow-up."
        ),
    )


class EncounterSynthesisResult(BaseModel):
    """Structured documentation target for post-encounter clinical NLP synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    soap_note: SOAPNote = Field(
        description="Complete structured clinical SOAP note for provider documentation."
    )
    staged_orders: list[StagedOrder] = Field(
        default_factory=list,
        description=(
            "Actionable prescriptions, lab orders, or referrals staged for 1-click "
            "clinician approval."
        ),
    )
    visit_summary_for_patient: str = Field(
        min_length=1,
        description=(
            "Plain-language summary with the clinician's degree of diagnostic certainty. "
            "Include all stated medication details (name, dose, route, frequency, and duration), "
            "test timing, and follow-up instructions. Omit advice and warnings the clinician "
            "did not give. Do not infer pill counts, medication start dates, testing status, "
            "or what to record in a diary."
        ),
    )
