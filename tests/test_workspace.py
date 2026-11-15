"""Regression coverage for visit capture, editing, and approval."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core.config import Settings
from app.main import create_app
from app.routers import workspace
from app.schemas.clinical import EncounterSynthesisResult
from app.services.draft_verification import (
    Correction,
    Finding,
    PrescriptionDirection,
    VerificationReport,
)
from app.services.outreach import OutreachService, SmsUnavailableError
from app.services.recording import RecordingTranscript, TranscriptSegment
from tests.fakes import FakeOpenAI

WORKSPACE = {"X-Escriba-Workspace": "1"}


def test_dropped_prescription_cannot_be_approved_or_texted(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'reconciliation.db'}",
        twilio_validate_signature=False,
    )
    original_plan = "Start amlodipine 5 mg by mouth once daily."
    original_summary = "Start taking amlodipine every day."
    transcript = original_plan + " Return in two weeks."
    fixture = {
        "soap_note": {
            "subjective": "Hypertension.",
            "objective": "No exam documented.",
            "assessment": "Hypertension.",
            "plan": original_plan,
        },
        "visit_summary_for_patient": original_summary,
        "staged_orders": [
            {
                "order_type": "prescription",
                "description": "Amlodipine 5 mg",
                "icd10_rationale": "Hypertension",
                "transcript_timestamp_reference": original_plan,
            }
        ],
    }

    async def transcribe(*_args):
        return RecordingTranscript(transcript, [TranscriptSegment(start=0, text=transcript)])

    async def tag(_client, _settings, recording):
        recording.segments[0].language = "en"
        return "en", 1.0

    async def verify(_client, _settings, draft, *_args, **_kwargs):
        directions = []
        if draft.soap_note.plan == original_plan:
            directions.append(PrescriptionDirection(order_index=0, area="plan", text=original_plan))
        if draft.visit_summary_for_patient == original_summary:
            directions.append(
                PrescriptionDirection(
                    order_index=0,
                    area="patient_instructions",
                    text=original_summary,
                )
            )
        return draft, VerificationReport(
            findings=[
                Finding(
                    area="order",
                    text="Allergy: amlodipine",
                    status="flagged",
                    evidence=[],
                    order_index=0,
                    severity="blocking",
                    kind="medication_safety",
                )
            ],
            trace=[],
            prescription_directions=directions,
        )

    submitted: list[str] = []

    async def send(_self, _phone, message, *, require_live):
        submitted.append(message)
        return "SM123"

    monkeypatch.setattr(workspace, "transcribe_recording", transcribe)
    monkeypatch.setattr(workspace, "tag_recording", tag)
    monkeypatch.setattr(workspace, "verify_draft", verify)
    monkeypatch.setattr(OutreachService, "_send_sms", send)
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 54321)) as client:
        fake = FakeOpenAI()
        fake.completions.responses.append(fixture)
        app.state.openai_client = fake
        created = client.post("/api/visits", headers=WORKSPACE, json={"label": "Synthetic allergy"})
        route = f"/api/visits/{created.json()['id']}"
        assert (
            client.put(
                route + "/recording",
                content=b"audio",
                headers={**WORKSPACE, "X-Audio-Format": "wav"},
            ).status_code
            == 200
        )
        draft = client.post(route + "/draft", headers=WORKSPACE).json()["draft"]
        for plan, summary, blocked_sections in [
            (original_plan, original_summary, "Plan and patient instructions"),
            ("Return in two weeks.", original_summary, "patient instructions"),
            (original_plan, "Return in two weeks.", "Plan"),
            ("Return in two weeks.", "Return in two weeks.", None),
        ]:
            body = {
                "revision": draft["updated_at"],
                "soap_note": dict(draft["soap_note"], plan=plan),
                "visit_summary_for_patient": summary,
                "kept_orders": [],
            }
            saved = client.put(route + "/review", headers=WORKSPACE, json=body)
            assert saved.status_code == 200, saved.text
            checked = client.post(
                route + "/verify",
                headers=WORKSPACE,
                json={
                    "revision": saved.json()["draft"]["updated_at"],
                },
            )
            draft = checked.json()["draft"]
            body.update(
                revision=draft["updated_at"],
                clinician="Synthetic Reviewer",
                resolutions=[
                    {
                        "finding_id": checked.json()["verification"]["findings"][0]["id"],
                        "action": "dropped",
                    }
                ],
                text_on_approval_to="+15550100000",
            )
            approved = client.put(route + "/review", headers=WORKSPACE, json=body)
            if blocked_sections:
                assert approved.status_code == 409, approved.text
                assert f"the {blocked_sections} before" in approved.json()["detail"]
                assert submitted == []
            else:
                assert approved.status_code == 200, approved.text
                assert approved.json()["draft"]["staged_orders"] == []
                assert submitted == ["Escriba Care Plan Summary:\n\nReturn in two weeks."]


@pytest.mark.parametrize(
    ("language", "sms_available"),
    [
        ("en", False),
        ("es", True),
        ("es", False),
    ],
)
def test_recording_to_approved_note(
    tmp_path: Path, monkeypatch: MonkeyPatch, language: str, sms_available: bool
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'workspace.db'}",
        twilio_validate_signature=False,
    )
    fixture = EncounterSynthesisResult.model_validate(
        {
            "soap_note": {
                "subjective": "Right ankle pain",
                "objective": (
                    "Mild swelling. No tests were ordered today as stated by the clinician."
                    if language == "en"
                    else "Mild swelling. No tests today."
                ),
                "assessment": "Possible sprain",
                "plan": (
                    "Rest and elevate. The clinician stated that no tests are planned today."
                    if language == "en"
                    else "Rest and elevate. No tests are planned today."
                ),
            },
            "visit_summary_for_patient": (
                "Descanse y eleve el tobillo. No hay pruebas previstas para hoy."
                if language == "es"
                else "Rest and elevate the ankle. No tests are planned today."
            ),
            "staged_orders": [
                {
                    "order_type": "lab",
                    "description": "CBC today",
                    "icd10_rationale": "Fatigue",
                    "transcript_timestamp_reference": "Let's order a CBC.",
                }
            ],
        }
    )

    transcript = (
        "My right ankle hurts. There is mild swelling. It may be a sprain. "
        "Rest and elevate. No tests today."
    )

    async def transcribe(
        _client: object, _settings: Settings, audio: bytes, extension: str
    ) -> RecordingTranscript:
        assert audio == b"test recording"
        assert extension == "wav"
        return RecordingTranscript(transcript, [TranscriptSegment(start=0.8, text=transcript)])

    monkeypatch.setattr(workspace, "transcribe_recording", transcribe)

    async def tag(_client, _settings, recording):
        recording.segments[0].language = "en"
        return "en", 1.0

    monkeypatch.setattr(workspace, "tag_recording", tag)

    async def verify(
        _client,
        _settings,
        draft,
        _transcript,
        _chart,
        *,
        repair,
        summary_language="en",
        note_language="en",
        progress=None,
    ):
        return draft, VerificationReport(
            findings=[
                Finding(
                    area="order",
                    text=draft.staged_orders[0].description,
                    status="flagged",
                    evidence=[],
                    question="Which quote supports this order?",
                    order_index=0,
                    severity="question",
                    kind="unsupported_order",
                ),
                Finding(
                    area="order",
                    text="Medication conflict",
                    status="flagged",
                    evidence=[],
                    severity="blocking",
                    order_index=0,
                    kind="medication_safety",
                ),
            ],
            trace=["Checked draft against the transcript."],
            corrections=[
                Correction(
                    area="plan",
                    before="Do not elevate the ankle.",
                    after="Rest and elevate the ankle.",
                )
            ],
        )

    monkeypatch.setattr(workspace, "verify_draft", verify)
    messages: list[str] = []

    async def send_sms(
        _self: OutreachService, phone: str, message: str, *, require_live: bool
    ) -> str:
        assert require_live
        assert phone == "+15550100000"
        if not sms_available:
            raise SmsUnavailableError("SMS is not configured.")
        messages.append(message)
        return "SM123"

    monkeypatch.setattr(OutreachService, "_send_sms", send_sms)
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 54321)) as client:
        assert client.get("/visits/new").status_code == 200
        documented_paths = client.get("/openapi.json").json()["paths"]
        assert all(
            path.startswith("/api/visits") or path == "/healthz" for path in documented_paths
        )
        create = client.post(
            "/api/visits",
            headers=WORKSPACE,
            json={
                "label": "Synthetic ankle",
                "preferred_language": language,
            },
        )
        assert create.status_code == 201
        assert create.json()["preferred_language"] == language
        visit_id = create.json()["id"]
        route = f"/api/visits/{visit_id}"
        assert client.get(f"/visits/{visit_id}/capture").status_code == 200
        assert create.json()["transcript_segments"] == []
        headers = {**WORKSPACE, "X-Audio-Format": "wav"}
        assert (
            client.put(route + "/recording", content=b"sound", headers=headers).status_code == 503
        )
        fake = FakeOpenAI()
        fake.completions.responses.append(fixture)
        app.state.openai_client = fake
        assert client.put(route + "/recording", content=b"sound").status_code == 403
        audio = client.put(route + "/recording", content=b"test recording", headers=headers)
        assert audio.status_code == 200, audio.text
        assert audio.json()["transcript"] == transcript
        assert client.get(route).json()["transcript_segments"] == [
            {"start": 0.8, "text": transcript, "language": "en"}
        ]
        assert (
            client.put(route + "/recording", content=b"sound", headers=headers).status_code == 409
        )

        generated = client.post(route + "/draft", headers=WORKSPACE)
        assert generated.status_code == 200, generated.text
        draft = generated.json()["draft"]
        assert draft["status"] == "review_pending"
        assert draft["soap_note"]["objective"] == "Mild swelling."
        assert "no tests are planned today." in draft["soap_note"]["plan"].lower()
        assert draft["visit_summary_for_patient"] == fixture.visit_summary_for_patient
        if language == "es":
            assert (
                "entirely in clear, warm Spanish"
                in fake.completions.calls[0]["messages"][0]["content"]
            )
        assert client.get(route).json()["draft"]["soap_note"]["objective"] == "Mild swelling."
        assert len(draft["staged_orders"]) == 1
        assert (
            client.post(
                f"/document/{draft['document_id']}/approve", json={"clinician_id": "Other"}
            ).status_code
            == 404
        )
        updated_note = dict(draft["soap_note"], plan="Rest and elevate the ankle.")
        final_summary = (
            "Descanse con el tobillo en alto."
            if language == "es"
            else ("Keep the ankle raised and rest.")
        )
        edit = {
            "revision": draft["updated_at"],
            "soap_note": updated_note,
            "visit_summary_for_patient": final_summary,
        }
        if language == "es":
            no_approval = client.put(
                route + "/review",
                headers=WORKSPACE,
                json={**edit, "text_on_approval_to": "+15550100000"},
            )
            assert no_approval.status_code == 422
            assert messages == []
        saved = client.put(route + "/review", headers=WORKSPACE, json=edit)
        assert saved.status_code == 200, saved.text
        assert saved.json()["draft"]["soap_note"] == updated_note
        assert client.put(route + "/review", headers=WORKSPACE, json=edit).status_code == 409
        edit.update(revision=saved.json()["draft"]["updated_at"], clinician="Synthetic Reviewer")
        unverified = client.put(route + "/review", headers=WORKSPACE, json=edit)
        assert unverified.status_code == 409
        checked = client.post(
            route + "/verify", headers=WORKSPACE, json={"revision": edit["revision"]}
        )
        assert checked.status_code == 200, checked.text
        report = checked.json()["verification"]
        assert report["findings"][0]["kind"] == "unsupported_order"
        undone = client.post(
            route + "/verification/undo",
            headers=WORKSPACE,
            json={
                "revision": checked.json()["draft"]["updated_at"],
                "correction_id": report["corrections"][0]["id"],
            },
        )
        assert undone.status_code == 200, undone.text
        assert undone.json()["draft"]["soap_note"]["plan"] == "Do not elevate the ankle."
        report = undone.json()["verification"]
        assert report["findings"][-1]["severity"] == "question"
        edit.update(
            revision=undone.json()["draft"]["updated_at"],
            soap_note=undone.json()["draft"]["soap_note"],
        )
        kept_unsupported = client.put(
            route + "/review",
            headers=WORKSPACE,
            json={**edit, "kept_orders": [0]},
        )
        assert kept_unsupported.status_code == 409
        assert kept_unsupported.json()["detail"] == "Drop proposed orders without source support."
        resolutions = [
            {
                "finding_id": item["id"],
                "action": "dismissed" if item["severity"] == "question" else "dropped",
            }
            for item in report["findings"]
        ]
        assert (
            client.put(
                route + "/review", headers=WORKSPACE, json={**edit, "resolutions": resolutions[:1]}
            ).status_code
            == 409
        )
        edit["resolutions"] = resolutions
        edit["clinician"] = " \t  "
        blank = client.put(route + "/review", headers=WORKSPACE, json=edit)
        assert blank.status_code == 422
        assert "Enter the reviewer's name" in blank.json()["detail"][0]["msg"]
        edit["clinician"] = "  Synthetic Reviewer  "
        if language == "es":
            edit["text_on_approval_to"] = "+15550100000"
        approved = client.put(route + "/review", headers=WORKSPACE, json=edit)
        assert approved.status_code == 200, approved.text
        assert approved.json()["approved_by"] == "Synthetic Reviewer"
        audit = client.get(route).json()["verification"]["resolutions"]
        assert [item["action"] for item in audit] == ["dismissed", "dropped", "dismissed"]
        assert all(
            item["reviewer"] == "Synthetic Reviewer" and item["resolved_at"] for item in audit
        )
        assert approved.json()["summary_text_sent"] is sms_available
        if language == "es" and not sms_available:
            assert approved.json()["summary_text_error"] == "SMS is not configured."
        else:
            assert approved.json()["summary_text_error"] is None
        assert client.get(f"/visits/{visit_id}/approved").status_code == 200
        assert client.get(route).json()["draft"]["status"] == "approved"
        assert (
            client.post(route + "/draft", headers=WORKSPACE).json()["draft"]
            == approved.json()["draft"]
        )
        locked = client.put(route + "/review", headers=WORKSPACE, json=edit)
        assert locked.status_code == 409
        assert locked.json()["detail"] == "This visit is approved and locked."
        assert messages == (
            ["Escriba · Instrucciones de su visita\n\n" + final_summary] if sms_available else []
        )
