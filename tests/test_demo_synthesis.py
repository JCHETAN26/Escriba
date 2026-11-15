"""Regression tests for the synthesis demo's transcript checks."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from app.models.domain import utcnow
from app.routers.encounter import SynthesizeResponse
from demo.synthesis import CASES, check_output


def demo_case(case_id: str) -> tuple[dict[str, Any], SynthesizeResponse]:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    case = next(case for case in cases if case["id"] == case_id)
    return case, SynthesizeResponse.model_validate(
        {
            **case["fixture"],
            "document_id": uuid.uuid4(),
            "encounter_id": uuid.uuid4(),
            "status": "review_pending",
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
    )


@pytest.mark.parametrize(
    ("quote", "grounded"),
    [
        ("I am ordering a complete blood count today.", True),
        ("“Clinician: I am ordering a complete blood count today.”", True),
        ("ordering a complete blood count today", True),
        ("Clinician: I will order a complete blood count today.", False),
        ("Clinician: I am placing a referral to gynecology for the heavy bleeding.", False),
    ],
)
def test_order_quote_must_match_the_clinician_and_the_action(quote: str, grounded: bool) -> None:
    case, response = demo_case("lab-and-referral")
    lab, referral = response.staged_orders
    revised_lab = lab.model_copy(update={"transcript_timestamp_reference": quote})
    response = response.model_copy(update={"staged_orders": [revised_lab, referral]})

    findings = check_output(case, response)
    if grounded:
        assert findings == []
    else:
        assert any("verbatim clinician quote" in finding for finding in findings)


def test_summary_accepts_paraphrases_and_flags_changed_test_purpose() -> None:
    case, response = demo_case("lab-and-referral")
    summary = "Get a blood test today. You will see a gynecologist. Call if bleeding gets worse."
    soap_note = response.soap_note.model_copy(
        update={"assessment": "Heavy periods and fatigue, potentially due to low iron."}
    )
    paraphrased = response.model_copy(
        update={"soap_note": soap_note, "visit_summary_for_patient": summary}
    )
    assert check_output(case, paraphrased) == []

    incorrect = response.model_copy(
        update={
            "visit_summary_for_patient": summary.replace(
                "blood test", "blood test to check your iron level"
            )
        }
    )
    assert any(
        "Unexpected wording in summary" in finding for finding in check_output(case, incorrect)
    )


def test_grounded_order_still_requires_timing_in_its_description() -> None:
    case, response = demo_case("lab-and-referral")
    lab, referral = response.staged_orders
    untimed_lab = lab.model_copy(update={"description": "Complete blood count"})
    response = response.model_copy(update={"staged_orders": [untimed_lab, referral]})

    findings = check_output(case, response)
    assert len(findings) == 1
    assert "Missing required detail in lab description" in findings[0]


def test_checker_accepts_uncertainty_and_transportation_paraphrases() -> None:
    case, response = demo_case("conditional-plan")
    note = response.soap_note.model_copy(update={"assessment": "Sore throat, possibly viral."})
    assert check_output(case, response.model_copy(update={"soap_note": note})) == []

    case, response = demo_case("small-talk-and-access")
    summary = (
        "The clinic will call tomorrow to discuss how to help you get to your physical therapy "
        "visits. Come back in four weeks."
    )
    assert (
        check_output(case, response.model_copy(update={"visit_summary_for_patient": summary})) == []
    )


def test_summary_requires_dose_and_route_but_accepts_spelled_out_units() -> None:
    case, response = demo_case("explicit-prescription")
    summary = response.visit_summary_for_patient.replace("5 mg", "five milligrams").replace(
        "once a day", "once each day"
    )
    paraphrased = response.model_copy(update={"visit_summary_for_patient": summary})
    assert check_output(case, paraphrased) == []

    missing_details = response.model_copy(
        update={
            "visit_summary_for_patient": summary.replace("five milligrams of ", "").replace(
                " by mouth", ""
            )
        }
    )
    assert len(check_output(case, missing_details)) == 2


@pytest.mark.parametrize("tablet_count", ["1 tablet", "one 5 mg tablet", "one pill"])
def test_summary_flags_unspecified_tablet_count(tablet_count: str) -> None:
    case, response = demo_case("explicit-prescription")
    summary = response.visit_summary_for_patient + f" Take {tablet_count} per dose."
    result = response.model_copy(update={"visit_summary_for_patient": summary})

    findings = check_output(case, result)
    assert len(findings) == 1
    assert "tablets?" in findings[0]


def test_summary_flags_a_start_date_before_the_medication_instruction() -> None:
    case, response = demo_case("explicit-prescription")
    summary = "Today, start amlodipine. " + response.visit_summary_for_patient
    result = response.model_copy(update={"visit_summary_for_patient": summary})

    findings = check_output(case, result)
    assert len(findings) == 1
    assert "Unexpected wording in summary" in findings[0]


@pytest.mark.parametrize("timing", ["beginning today", "starting tomorrow", "start on Monday"])
def test_order_description_flags_unspecified_start_timing(timing: str) -> None:
    case, response = demo_case("explicit-prescription")
    [order] = response.staged_orders
    updated_order = order.model_copy(update={"description": f"{order.description}, {timing}."})
    result = response.model_copy(update={"staged_orders": [updated_order]})

    findings = check_output(case, result)
    assert len(findings) == 1
    assert "Unexpected wording in order description" in findings[0]


def test_today_for_blood_pressure_does_not_imply_a_medication_start_date() -> None:
    case, response = demo_case("explicit-prescription")
    summary = "Your blood pressure today was 152/92. " + response.visit_summary_for_patient
    assert (
        check_output(case, response.model_copy(update={"visit_summary_for_patient": summary})) == []
    )


@pytest.mark.parametrize("claim", ["No tests were done today.", "No labs were ordered."])
def test_phone_visit_flags_unspecified_testing(claim: str) -> None:
    case, response = demo_case("missing-objective")
    summary = f"{response.visit_summary_for_patient} {claim}"
    result = response.model_copy(update={"visit_summary_for_patient": summary})

    findings = check_output(case, result)
    assert len(findings) == 1
    assert "testing" in findings[0]


@pytest.mark.parametrize(
    "instructions", ["Write down when you go to bed.", "Record your bedtime and wake-up time."]
)
def test_phone_visit_flags_unspecified_diary_entries(instructions: str) -> None:
    case, response = demo_case("missing-objective")
    assert check_output(case, response) == []

    summary = f"{response.visit_summary_for_patient} {instructions}"
    findings = check_output(
        case, response.model_copy(update={"visit_summary_for_patient": summary})
    )
    assert len(findings) == 1
    assert "bedtimes?" in findings[0]


def test_ankle_testing_deferral_belongs_only_in_the_plan() -> None:
    case, response = demo_case("correction-and-negation")
    paraphrased_plan = response.soap_note.plan.replace("No tests today.", "Testing deferred today.")
    revised_note = response.soap_note.model_copy(update={"plan": paraphrased_plan})
    assert check_output(case, response.model_copy(update={"soap_note": revised_note})) == []

    misplaced_note = response.soap_note.model_copy(
        update={
            "objective": f"{response.soap_note.objective} No tests performed today.",
            "plan": response.soap_note.plan.removesuffix(" No tests today."),
        }
    )
    findings = check_output(case, response.model_copy(update={"soap_note": misplaced_note}))
    assert len(findings) == 2
    assert any("Missing expected wording in plan" in finding for finding in findings)
    assert any("Unexpected wording in objective" in finding for finding in findings)

    duplicated_note = response.soap_note.model_copy(
        update={"objective": f"{response.soap_note.objective} No tests today."}
    )
    findings = check_output(case, response.model_copy(update={"soap_note": duplicated_note}))
    assert len(findings) == 1
    assert "Unexpected wording in objective" in findings[0]


@pytest.mark.parametrize(
    ("case_id", "section", "text", "expected_finding"),
    [
        (
            "explicit-prescription",
            "summary",
            "Your blood pressure is high. Take amlodipine 5 mg once daily. Come back "
            "in two weeks. Call if you get dizzy.",
            "by mouth|orally",
        ),
        (
            "conditional-plan",
            "plan",
            "Defer testing and antibiotics today. Reconsider amoxicillin if symptoms "
            "persist. Drink fluids and call if not improving in three days.",
            "confirm|positive",
        ),
        (
            "small-talk-and-access",
            "assessment",
            "Ongoing right knee pain. Missed physical therapy contributed to the pain.",
            "contribut",
        ),
        (
            "correction-and-negation",
            "summary",
            "You have a right ankle sprain. Rest and keep it raised. Come back in one "
            "week. Call sooner if you cannot put weight on it.",
            "think|may|might",
        ),
    ],
)
def test_checker_detects_the_four_live_regressions(
    case_id: str, section: str, text: str, expected_finding: str
) -> None:
    case, response = demo_case(case_id)
    if section == "summary":
        response = response.model_copy(update={"visit_summary_for_patient": text})
    else:
        note = response.soap_note.model_copy(update={section: text})
        response = response.model_copy(update={"soap_note": note})

    assert any(expected_finding in finding for finding in check_output(case, response))
