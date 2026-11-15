"""Regression coverage for visit capture, editing, and approval."""

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core.config import Settings
from app.main import create_app
from app.routers import workspace
from app.schemas.clinical import Allergy, EncounterSynthesisResult, SOAPNote
from app.services.draft_verification import (
    Correction,
    Finding,
    PrescriptionDirection,
    VerificationReport,
)
from app.services.guardrail import PatientContext
from app.services.outreach import OutreachService, SmsUnavailableError
from app.services.recording import RecordingTranscript, TranscriptSegment
from tests.fakes import FakeOpenAI

WORKSPACE = {"X-Escriba-Workspace": "1"}


def test_hosted_recording_chunks_are_validated_reassembled_and_removed(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    origin = "https://escriba.example"
    database = tmp_path / "uploads.db"
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{database}",
        public_base_url=origin,
        twilio_validate_signature=False,
    )
    app = create_app(settings)
    first = b"a" * 3_000_000
    last = b"b" * 2_614_474

    async def transcribe(_client, _settings, audio, extension):
        assert extension == "wav"
        assert audio == first + last
        return RecordingTranscript("Synthetic visit", [])

    async def tag(_client, _settings, _recording):
        return "en", 1.0

    monkeypatch.setattr(workspace, "transcribe_recording", transcribe)
    monkeypatch.setattr(workspace, "tag_recording", tag)
    with TestClient(app, base_url=origin) as client:
        app.state.settings = settings.model_copy(update={"hosted_demo": True})
        app.state.openai_client = FakeOpenAI()
        config = client.get("/api/visits/config").json()
        assert config["max_audio_bytes"] == 25_000_000
        assert config["audio_chunk_bytes"] == len(first)
        headers = {**WORKSPACE, "Origin": origin, "X-Audio-Format": "wav"}
        visit = client.post("/api/visits", headers=headers, json={"label": "Chunk upload"})
        assert visit.is_success, visit.text
        route = f"/api/visits/{visit.json()['id']}/recording/chunks"
        upload = {"upload_id": str(uuid4())}
        params = {**upload, "total_bytes": len(first) + len(last)}
        assert (
            client.put(
                f"{route}/0",
                headers=headers,
                content=first,
                params={
                    **upload,
                    "total_bytes": 25_000_001,
                },
            ).status_code
            == 422
        )
        assert (
            client.put(
                f"{route}/0", headers=headers, content=first + b"x", params=params
            ).status_code
            == 413
        )
        for _ in range(2):
            part = client.put(f"{route}/0", headers=headers, content=first, params=params)
            assert part.is_success, part.text
        assert client.post(f"{route}/complete", headers=headers, params=upload).status_code == 409
        guest_cookies = dict(client.cookies)
        client.cookies.clear()
        assert (
            client.put(f"{route}/1", headers=headers, content=last, params=params).status_code
            == 404
        )
        client.cookies.clear()
        client.cookies.update(guest_cookies)
        part = client.put(f"{route}/1", headers=headers, content=last, params=params)
        assert part.is_success, part.text
        done = client.post(f"{route}/complete", headers=headers, params=upload)
        assert done.is_success, done.text
        assert done.json()["transcript"] == "Synthetic visit"
        assert client.post(f"{route}/complete", headers=headers, params=upload).status_code == 409
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("SELECT count(*) FROM recording_upload_chunks").fetchone() == (0,)


def test_hosted_demo_keeps_guest_charts_separate(tmp_path: Path) -> None:
    origin = "https://escriba.example"
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'guest.db'}",
        public_base_url=origin,
        twilio_validate_signature=False,
    )
    app = create_app(settings)
    with TestClient(app, base_url=origin) as client:
        # Use a local test database with the hosted request handling enabled.
        app.state.settings = settings.model_copy(
            update={"hosted_demo": True, "environment": "staging"}
        )
        config = client.get("/api/visits/config")
        assert config.json()["hosted_demo"] is True
        cookie = config.headers["set-cookie"]
        assert all(flag in cookie for flag in ("__Host-escriba-demo=", "Secure", "HttpOnly"))
        headers = {**WORKSPACE, "Origin": origin}
        visit = client.post(
            "/api/visits",
            json={"label": "Prior visit", "patient_name": "Morgan"},
            headers=headers,
        )
        assert visit.is_success, visit.text
        visit_id = visit.json()["id"]
        patient_id = visit.json()["patient_id"]
        assert client.get(f"/api/visits/{visit_id}").is_success
        assert client.get("/api/patients").json()[0]["id"] == patient_id
        assert (
            client.post(
                "/api/visits",
                json={"label": "Cross-origin attempt"},
                headers={**WORKSPACE, "Origin": "https://elsewhere.example"},
            ).status_code
            == 403
        )

        client.cookies.clear()
        assert client.get("/api/patients").json() == []
        assert client.get(f"/api/visits/{visit_id}").status_code == 404
        assert (
            client.post(
                "/api/visits",
                json={"label": "Returning visit", "patient_id": patient_id},
                headers=headers,
            ).status_code
            == 404
        )


def test_returning_patient_reuses_allergy_and_keeps_visit_details(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'returning.db'}",
        twilio_validate_signature=False,
    )
    chart_contexts: list[PatientContext] = []

    async def transcribe(_client, _settings, audio, _extension):
        return RecordingTranscript(audio.decode(), [])

    async def tag(_client, _settings, recording):
        return ("en" if "hives" in recording.text else "es"), 1.0

    async def draft(_client, _settings, _transcript, _chart, **_kwargs):
        return EncounterSynthesisResult(
            soap_note=SOAPNote(
                subjective="Patient reported a prior medication reaction.",
                objective="No exam documented.",
                assessment="Medication reaction.",
                plan="Review the reaction.",
            ),
            visit_summary_for_patient="We reviewed your medication history.",
            staged_orders=[],
        )

    async def verify(_client, _settings, note, _transcript, chart, **_kwargs):
        chart_contexts.append(chart)
        return note, VerificationReport(findings=[], trace=[])

    monkeypatch.setattr(workspace, "transcribe_recording", transcribe)
    monkeypatch.setattr(workspace, "tag_recording", tag)
    monkeypatch.setattr(workspace, "generate_draft", draft)
    monkeypatch.setattr(workspace, "verify_draft", verify)
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 54321)) as client:
        app.state.openai_client = FakeOpenAI()

        def call(method, path, **kwargs):
            response = client.request(method, path, headers=WORKSPACE, **kwargs)
            assert response.is_success, response.text
            return response.json()

        def prepare(route, words):
            recorded = client.put(
                route + "/recording",
                content=words,
                headers={**WORKSPACE, "X-Audio-Format": "wav"},
            )
            assert recorded.is_success, recorded.text
            generated = call("POST", route + "/draft")
            return call(
                "POST",
                route + "/verify",
                json={
                    "revision": generated["draft"]["updated_at"],
                },
            )

        first = call("POST", "/api/visits", json={"label": "First visit", "patient_name": "Morgan"})
        prior = f"/api/visits/{first['id']}"
        checked = prepare(prior, b"Amlodipine gave me hives.")
        assert chart_contexts[-1].allergies == ()
        documented = call(
            "POST",
            prior + "/allergies",
            json={
                "allergen": "amlodipine",
                "reaction": "hives",
            },
        )
        assert documented["verification"] is None
        assert documented["draft"]["updated_at"] != checked["draft"]["updated_at"]
        checked = call(
            "POST",
            prior + "/verify",
            json={
                "revision": documented["draft"]["updated_at"],
            },
        )["draft"]
        call(
            "PUT",
            prior + "/review",
            json={
                "revision": checked["updated_at"],
                "soap_note": checked["soap_note"],
                "visit_summary_for_patient": checked["visit_summary_for_patient"],
                "clinician": "Demo clinician",
            },
        )
        charts = call("GET", "/api/patients")
        assert [(chart["id"], chart["label"]) for chart in charts] == [
            (first["patient_id"], "Morgan"),
        ]
        followup = call(
            "POST",
            "/api/visits",
            json={
                "label": "Return visit",
                "patient_id": first["patient_id"],
            },
        )
        assert followup["previous_visit"]["id"] == first["id"]
        assert followup["allergies"] == [{"allergen": "amlodipine", "reaction": "hives"}]
        checked = prepare(f"/api/visits/{followup['id']}", b"Vamos a iniciar amlodipino.")
        assert checked["note_language"] == "es"
        assert chart_contexts[-1].allergies == (Allergy(allergen="amlodipine", reaction="hives"),)
        assert call("GET", prior)["note_language"] == "en"
        assert call("GET", prior)["label"] == "First visit"

        same_name = call(
            "POST", "/api/visits", json={"label": "First visit", "patient_name": "Morgan"}
        )
        assert same_name["patient_id"] != first["patient_id"]
        assert same_name["allergies"] == []


@pytest.mark.parametrize("dismiss_marks", [False, True])
def test_dropped_prescription_cannot_be_approved_or_texted(
    tmp_path: Path, monkeypatch: MonkeyPatch, dismiss_marks: bool
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
            (
                "Amlodipine not started because of a documented allergy (hives). "
                "Return in two weeks.",
                "Return in two weeks.",
                None,
            ),
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
                if dismiss_marks:
                    marks = checked.json()["verification"]["prescription_directions"]
                    for mark in marks:
                        body["resolutions"].append(
                            {"finding_id": mark["id"], "action": "not_a_direction"}
                        )
                        approved = client.put(route + "/review", headers=WORKSPACE, json=body)
                        assert approved.status_code == (200 if mark == marks[-1] else 409)
                    audit = approved.json()["verification"]["resolutions"]
                    assert {
                        item["finding_id"] for item in audit if item["action"] == "not_a_direction"
                    } == {mark["id"] for mark in marks}
                    assert approved.json()["draft"]["staged_orders"] == []
                    break
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
        dropped_prescriptions=frozenset(),
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
            path.startswith("/api/visits") or path in {"/api/patients", "/healthz"}
            for path in documented_paths
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
