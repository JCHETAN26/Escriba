"""Regression coverage for the verifier's evidence and correction loop."""

import json
import uuid
from typing import cast

import pytest
from openai import AsyncOpenAI

from app.core.config import Settings
from app.schemas.clinical import (
    Allergy,
    EncounterSynthesisResult,
    Medication,
    SOAPNote,
    StagedOrder,
)
from app.services.draft_verification import VerificationUnavailableError, verify_draft
from app.services.guardrail import PatientContext
from tests.fakes import FakeOpenAI

TRANSCRIPT = (
    "My right knee still hurts on stairs. Right knee bends to 110 degrees. "
    "Ongoing right knee pain. The clinic will call tomorrow. I am prescribing Bactrim."
)
DRAFT = EncounterSynthesisResult(
    soap_note=SOAPNote(
        subjective="Right knee hurts especially on stairs.",
        objective="Right knee bends to 110 degrees.",
        assessment="Ongoing right knee pain.",
        plan="The clinic will call tomorrow.",
    ),
    visit_summary_for_patient="The clinic will call tomorrow.",
    staged_orders=[
        StagedOrder(
            order_type="prescription",
            description="Bactrim",
            icd10_rationale="Reason not stated",
            transcript_timestamp_reference="I am prescribing Bactrim.",
        )
    ],
)


def pass_result(*, repaired: bool = False) -> dict:
    quotes = [
        "My right knee still hurts on stairs.",
        "Right knee bends to 110 degrees.",
        "Ongoing right knee pain.",
        "The clinic will call tomorrow.",
        "The clinic will call tomorrow.",
        "I am prescribing Bactrim.",
    ]
    return {
        "items": [
            {
                "id": index,
                "status": "flagged" if index == 0 and not repaired else "supported",
                "evidence": [quote],
                "question": None,
                "replacement": "Right knee still hurts on stairs."
                if index == 0 and not repaired
                else None,
                "medication_details_missing": (
                    ["dose", "route", "frequency", "duration"] if index == 5 else []
                ),
                "ongoing_or_chronic": False,
                "wrong_language": False,
                "directs_prescription_indices": [],
            }
            for index, quote in enumerate(quotes)
        ]
    }


async def test_repairs_prose_and_asks_about_prescription() -> None:
    fake = FakeOpenAI()
    fake.completions.responses.extend(
        [
            pass_result(),
            pass_result(repaired=True),
            json.dumps(
                {
                    "alert": {
                        "severity": "high",
                        "trigger_entity": "Bactrim",
                        "context_conflict": "Bactrim conflicts with the documented sulfa allergy.",
                        "suggested_action": "Confirm with the clinician.",
                    }
                }
            ),
        ]
    )
    chart = PatientContext(
        patient_id=uuid.uuid4(),
        allergies=(Allergy(allergen="Sulfa drugs"),),
        medications=(Medication(name="Amlodipine"),),
    )
    draft, report = await verify_draft(
        cast(AsyncOpenAI, fake),
        Settings(_env_file=None, twilio_validate_signature=False),
        DRAFT,
        TRANSCRIPT,
        chart,
        repair=True,
    )
    assert draft.soap_note.subjective == "Right knee still hurts on stairs."
    assert draft.staged_orders == DRAFT.staged_orders
    assert report.corrections[0].before == DRAFT.soap_note.subjective
    assert any("Re-checking" in step for step in report.trace)
    assert any("frequency" in (finding.question or "") for finding in report.findings)
    assert any(
        finding.severity == "blocking" and "documented sulfa allergy" in (finding.question or "")
        for finding in report.findings
    )


async def test_fabricated_quote_and_missing_item_cannot_pass() -> None:
    fake = FakeOpenAI()
    result = pass_result(repaired=True)
    result["items"][0]["evidence"] = ["She fell down two flights of stairs."]
    fake.completions.responses.append(result)
    chart = PatientContext(patient_id=uuid.uuid4())
    settings = Settings(_env_file=None, twilio_validate_signature=False)
    _, report = await verify_draft(
        cast(AsyncOpenAI, fake),
        settings,
        DRAFT,
        TRANSCRIPT,
        chart,
        repair=False,
    )
    assert report.findings[0].status == "flagged"
    assert report.findings[0].evidence == []
    result["items"].pop()
    fake.completions.responses.append(result)
    with pytest.raises(VerificationUnavailableError, match="omitted"):
        await verify_draft(
            cast(AsyncOpenAI, fake),
            settings,
            DRAFT,
            TRANSCRIPT,
            chart,
            repair=False,
        )


@pytest.mark.parametrize(
    ("plan", "summary", "linked_summary", "expected_areas"),
    [
        ("Start Bactrim.", "Take Bactrim daily.", False, {"plan", "patient_instructions"}),
        ("Start Bactrim.", "Take your antibiotic daily.", True, {"plan", "patient_instructions"}),
        ("Do not start Bactrim.", "Do not take Bactrim.", False, set()),
    ],
)
async def test_links_only_active_prescription_directions(
    plan: str, summary: str, linked_summary: bool, expected_areas: set[str]
) -> None:
    fake = FakeOpenAI()
    result = pass_result(repaired=True)
    result["items"][4]["directs_prescription_indices"] = [0] if linked_summary else []
    fake.completions.responses.extend([result, '{"alert": null}'])
    draft = DRAFT.model_copy(
        update={
            "soap_note": DRAFT.soap_note.model_copy(update={"plan": plan}),
            "visit_summary_for_patient": summary,
        }
    )
    _, report = await verify_draft(
        cast(AsyncOpenAI, fake),
        Settings(_env_file=None, twilio_validate_signature=False),
        draft,
        TRANSCRIPT,
        PatientContext(patient_id=uuid.uuid4()),
        repair=False,
    )
    assert report.prescription_directions is not None
    assert {item.area for item in report.prescription_directions} == expected_areas


async def test_wrong_language_is_distinct_from_unsupported_meaning() -> None:
    quote = "La rodilla derecha se flexiona a ciento diez grados."
    fake = FakeOpenAI()
    result = pass_result(repaired=True)
    result["items"][1].update(wrong_language=True, evidence=[quote])
    fake.completions.responses.extend([result, '{"alert": null}'])
    draft = DRAFT.model_copy(
        update={
            "soap_note": DRAFT.soap_note.model_copy(update={"objective": quote}),
        }
    )
    _, report = await verify_draft(
        cast(AsyncOpenAI, fake),
        Settings(_env_file=None, twilio_validate_signature=False),
        draft,
        TRANSCRIPT + " " + quote,
        PatientContext(patient_id=uuid.uuid4()),
        repair=False,
    )
    finding = next(item for item in report.findings if item.area == "objective")
    assert finding.kind == "language"
    assert finding.status == "flagged"
    assert finding.evidence == [quote]
    assert "English" in (finding.question or "")


@pytest.mark.parametrize(
    ("source", "assessment", "flagged"),
    [
        ("", "Persistent right knee pain with limited right knee flexion.", True),
        ("", "Right knee flexion was limited to 110°.", True),
        ("", "Limited activity due to pain, with right knee flexion at 110°.", False),
        (" Right knee flexion is limited.", "Limited right knee flexion.", False),
    ],
)
async def test_numeric_flexion_alone_does_not_support_a_limitation(
    source: str, assessment: str, flagged: bool
) -> None:
    quote = "La rodilla derecha se flexiona a ciento diez grados."
    fake = FakeOpenAI()
    draft = DRAFT.model_copy(
        update={
            "staged_orders": [],
            "soap_note": DRAFT.soap_note.model_copy(
                update={"objective": "Right knee flexion was 110°.", "assessment": assessment}
            ),
        }
    )
    result = pass_result(repaired=True)
    result["items"].pop()
    result["items"][1]["evidence"] = [quote]
    result["items"][2]["evidence"] = [source.strip() if source else quote]
    fake.completions.responses.append(result)
    _, report = await verify_draft(
        cast(AsyncOpenAI, fake),
        Settings(_env_file=None, twilio_validate_signature=False),
        draft,
        TRANSCRIPT.replace("Right knee bends to 110 degrees.", quote) + source,
        PatientContext(patient_id=uuid.uuid4()),
        repair=False,
    )
    objective = next(item for item in report.findings if item.area == "objective")
    assessments = [item for item in report.findings if item.area == "assessment"]
    assert objective.text == "Right knee flexion was 110°."
    assert objective.status == "supported"
    assert any(item.status == "flagged" for item in assessments) == flagged
    if flagged:
        assert quote in assessments[0].evidence
        assert "inferred limitation" in (assessments[0].question or "")
