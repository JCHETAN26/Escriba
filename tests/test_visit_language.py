import uuid
from datetime import UTC, datetime
from typing import cast

import pytest
from fastapi import HTTPException
from openai import AsyncOpenAI

from app.core.config import Settings
from app.routers.workspace import _audit_resolutions
from app.schemas.clinical import Allergy
from app.services.draft_verification import (
    Finding,
    FindingResolution,
    VerificationReport,
    verify_draft,
)
from app.services.guardrail import PatientContext
from app.services.visit_language import classify_visit
from tests.fakes import FakeOpenAI
from tests.test_draft_verification import DRAFT, TRANSCRIPT, pass_result


@pytest.mark.parametrize(
    ("english", "spanish", "language", "confidence"),
    [
        (100, 0, "en", 1),
        (0, 100, "es", 1),
        (60, 40, "mixed", 0.6),
        (85, 15, "en", 0.85),
        (15, 85, "es", 0.85),
        (84, 16, "mixed", 0.84),
    ],
)
def test_language_thresholds(english, spanish, language, confidence):
    assert classify_visit(english, spanish) == (language, confidence)


async def test_allergy_blocking_and_cross_language_evidence():
    fake = FakeOpenAI()
    result = pass_result(repaired=True)
    result["items"][0]["evidence"] = ["Me duele la rodilla derecha al subir las escaleras."]
    result["items"][5]["ongoing_or_chronic"] = True
    result["items"][5]["medication_details_missing"] = ["duration"]
    fake.completions.responses.extend([result, '{"alert": null}'])
    chart = PatientContext(patient_id=uuid.uuid4(), allergies=(Allergy(allergen="Bactrim"),))
    _, report = await verify_draft(
        cast(AsyncOpenAI, fake),
        Settings(_env_file=None, twilio_validate_signature=False),
        DRAFT,
        TRANSCRIPT + " Me duele la rodilla derecha al subir las escaleras.",
        chart,
        repair=False,
    )
    first = report.findings[0]
    assert first.status == "supported"
    assert first.evidence == result["items"][0]["evidence"]
    assert report.findings[-1].severity == "blocking"
    assert "Bactrim" in (report.findings[-1].chart_evidence or "")
    assert all("duration" not in (finding.question or "") for finding in report.findings)


def test_approve_requires_resolutions_and_minimum_override_reason():
    blocking = Finding(
        area="order",
        text="Conflict",
        status="flagged",
        evidence=[],
        order_index=0,
        severity="blocking",
    )
    question = Finding(
        area="plan",
        text="When?",
        status="flagged",
        evidence=[],
        question="When?",
        severity="question",
    )
    report = VerificationReport(findings=[blocking, question], trace=[])
    today = datetime.now(UTC)
    with pytest.raises(HTTPException) as unresolved:
        _audit_resolutions(report, [], {0}, "Dr. Demo", today)
    assert unresolved.value.status_code == 409
    short = FindingResolution(finding_id=blocking.id, action="overridden", note="   seven!! ")
    assert len(short.note or "") == 7
    with pytest.raises(HTTPException) as too_short:
        _audit_resolutions(report, [short], {0}, "Dr. Demo", today)
    assert too_short.value.status_code == 409
    short.note = "Test reason"
    with pytest.raises(HTTPException, match="Answer or dismiss") as unanswered:
        _audit_resolutions(report, [short], {0}, "Dr. Demo", today)
    assert unanswered.value.status_code == 409
    answer = FindingResolution(finding_id=question.id, action="answered", note="Reviewed.")
    audit = _audit_resolutions(report, [short, answer], {0}, "Dr. Demo", today)
    assert audit[0].note == "Test reason"
    assert audit[0].reviewer == "Dr. Demo"
    assert audit[0].resolved_at == today
