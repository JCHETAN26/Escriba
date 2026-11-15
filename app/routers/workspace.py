"""Single-user, loopback visit capture and chart review workspace."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from openai import APIStatusError, AsyncOpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import selectinload

from app.core.database import DbSession
from app.core.dependencies import SettingsDep
from app.core.prompts import SummaryLanguage
from app.models.domain import (
    WORKSPACE_PROVIDER_ID,
    ClinicalDocument,
    DocumentStatus,
    DraftVerification,
    Encounter,
    EncounterStatus,
    Patient,
    RecordingTimeline,
    utcnow,
)
from app.routers.encounter import (
    SynthesizeResponse,
    generate_draft,
    populate_draft,
)
from app.schemas.clinical import (
    Allergy,
    EncounterSynthesisResult,
    Medication,
    SOAPNote,
    StagedOrder,
)
from app.services.draft_verification import (
    Finding,
    FindingResolution,
    ResolutionAudit,
    VerificationReport,
    VerificationUnavailableError,
    verify_draft,
)
from app.services.guardrail import PatientContext
from app.services.outreach import OutreachService, SmsProviderError, SmsUnavailableError
from app.services.recording import (
    AUDIO_FORMATS,
    MAX_AUDIO_BYTES,
    TranscriptSegment,
    transcribe_recording,
)
from app.services.visit_language import (
    DetectedLanguage,
    LanguageUnavailableError,
    localize_draft,
    tag_recording,
)

ASSETS = Path(__file__).resolve().parents[1] / "web"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
LOCKED_VISIT = "This visit is approved and locked."
STALE_DRAFT = "This draft changed. Reopen the visit."
PhoneNumber = Annotated[str, Field(pattern=r"^\+[1-9][0-9]{7,14}$")]
TESTING_DEFERRAL = re.compile(
    r"(?:^|(?<=[.!?\n]))[ \t]*"
    r"(?:no tests? (?:are planned |are being ordered |will be ordered )?today"
    r"|testing (?:is |will be )?deferred today)[.!?]?"
    r"(?=[ \t]*(?:$|\n)|(?<=[.!?])\s)",
    re.IGNORECASE,
)
TESTING_PLAN = re.compile(
    r"\b(?:not (?:going to )?order tests? today|no tests? (?:will be|are being) ordered today|"
    r"no se (?:van a pedir|pedirán|solicitarán) pruebas hoy)\b",
    re.IGNORECASE,
)
OBJECTIVE_TESTING_ABSENCE = re.compile(
    r"(?:^|(?<=[.!?\n]))[ \t]*(?:"
    r"no tests? (?:were|was) (?:obtained|performed|ordered) today"
    r"(?: per (?:the )?clinician(?:'s)? statement)?"
    r"|no se (?:realizaron|solicitaron|hicieron) pruebas hoy"
    r")[.!?]?(?=[ \t]*(?:$|\n)|(?<=[.!?])\s)",
    re.IGNORECASE,
)


def require_local_workspace(request: Request, settings: SettingsDep) -> None:
    if (
        settings.environment != "development"
        or request.client is None
        or request.client.host not in LOCAL_HOSTS
        or request.url.hostname not in LOCAL_HOSTS
    ):
        raise HTTPException(status_code=404, detail="Workspace is only available locally.")
    if request.method != "GET":
        origin = request.headers.get("origin")
        if request.headers.get("x-escriba-workspace") != "1" or (
            origin is not None and origin != str(request.base_url).rstrip("/")
        ):
            raise HTTPException(
                status_code=403, detail="Open the Escriba workspace to continue."
            )


router = APIRouter(
    dependencies=[Depends(require_local_workspace)],
    tags=["workspace"],
)


class VisitLabel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    label: str = Field(min_length=1, max_length=120)
    preferred_language: SummaryLanguage | None = None
    allergies: list[Allergy] = Field(default_factory=list, max_length=30)
    current_medications: list[Medication] = Field(default_factory=list, max_length=50)


class VisitListItem(BaseModel):
    id: uuid.UUID
    label: str
    created_at: datetime
    status: EncounterStatus


class VisitView(VisitListItem):
    transcript: str | None
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    draft: SynthesizeResponse | None
    approved_by: str | None = None
    approved_at: datetime | None = None
    preferred_language: SummaryLanguage = "en"
    detected_language: DetectedLanguage | None = None
    detection_confidence: float | None = None
    note_language: SummaryLanguage = "en"
    patient_language: SummaryLanguage = "en"
    summary_text_sent: bool = False
    summary_text_error: str | None = None
    verification: VerificationReport | None = None
    allergies: tuple[Allergy, ...] = ()
    current_medications: tuple[Medication, ...] = ()


class Review(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    revision: datetime
    soap_note: SOAPNote
    visit_summary_for_patient: str = Field(min_length=1, max_length=20_000)
    clinician: str | None = Field(default=None, min_length=1, max_length=64)
    kept_orders: list[int] = Field(default_factory=list, max_length=100)
    text_on_approval_to: PhoneNumber | None = None
    resolutions: list[FindingResolution] = Field(default_factory=list, max_length=2000)

    @field_validator("clinician", mode="before")
    @classmethod
    def trim_clinician(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("Enter the reviewer's name to approve.")
        return value


class SummaryTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    phone_number: PhoneNumber


class SummaryTextReceipt(BaseModel):
    message_sid: str


class VerifyDraftRequest(BaseModel):
    revision: datetime
    repair: bool = False


class ChangeLanguageRequest(BaseModel):
    revision: datetime
    note_language: SummaryLanguage
    patient_language: SummaryLanguage


class UndoCorrectionRequest(BaseModel):
    revision: datetime
    correction_id: str


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


@router.get("/", include_in_schema=False, response_model=None)
async def workspace() -> FileResponse:
    return FileResponse(
        ASSETS / "index.html",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; frame-ancestors 'none'; base-uri 'none'"
            ),
        },
    )


@router.get("/visits/new", include_in_schema=False, response_model=None)
async def new_visit_page() -> FileResponse:
    return await workspace()


@router.get("/visits/{visit_id}/{page}", include_in_schema=False, response_model=None)
async def visit_page(
    visit_id: uuid.UUID, page: Literal["capture", "review", "approved"], session: DbSession
) -> FileResponse:
    await _load_visit(session, visit_id)
    return await workspace()


@router.get("/workspace/{asset}", include_in_schema=False, response_model=None)
async def workspace_asset(asset: str) -> FileResponse:
    if asset not in {"workspace.css", "workspace.js"}:
        raise HTTPException(status_code=404)
    return FileResponse(ASSETS / asset)


async def _load_visit(session: DbSession, visit_id: uuid.UUID) -> Encounter:
    visit = await session.scalar(
        select(Encounter)
        .options(selectinload(Encounter.patient), selectinload(Encounter.clinical_documents))
        .where(Encounter.id == visit_id, Encounter.provider_id == WORKSPACE_PROVIDER_ID)
    )
    if visit is None:
        raise HTTPException(status_code=404, detail="Visit not found.")
    return visit


def _item(visit: Encounter) -> VisitListItem:
    return VisitListItem(
        id=visit.id,
        label=str(visit.patient.demographics["label"]),
        created_at=visit.created_at,
        status=visit.status,
    )


async def _view(visit: Encounter, session: DbSession) -> VisitView:
    doc = next(iter(visit.clinical_documents), None)
    timeline = await session.get(RecordingTimeline, visit.id) if visit.transcript else None
    draft = None
    verification = None
    if doc is not None:
        draft = SynthesizeResponse.from_document_and_synthesis(doc, _synthesis(doc))
        verification = await _verification(doc, session)
    chart = PatientContext.from_patient(visit.patient)
    return VisitView(
        **_item(visit).model_dump(),
        transcript=visit.transcript,
        transcript_segments=[
            TranscriptSegment.model_validate(segment) for segment in timeline.segments
        ]
        if timeline
        else [],
        draft=draft,
        approved_by=doc.approved_by if doc else None,
        approved_at=doc.approved_at if doc else None,
        preferred_language=_preferred_language(visit),
        detected_language=visit.patient.demographics.get("detected_language"),
        detection_confidence=visit.patient.demographics.get("detection_confidence"),
        note_language=_note_language(visit),
        patient_language=_preferred_language(visit),
        verification=verification,
        allergies=chart.allergies,
        current_medications=chart.medications,
    )


def _synthesis(doc: ClinicalDocument) -> EncounterSynthesisResult:
    return EncounterSynthesisResult(
        soap_note=SOAPNote.model_validate(doc.generated_soap),
        staged_orders=[StagedOrder.model_validate(order) for order in doc.staged_orders],
        visit_summary_for_patient=doc.visit_summary_for_patient or "",
    )


def _revision(value: datetime) -> str:
    return value.replace(tzinfo=value.tzinfo or UTC).astimezone(UTC).isoformat()


async def _verification(doc: ClinicalDocument, session: DbSession) -> VerificationReport | None:
    saved = await session.get(DraftVerification, doc.id)
    if saved and saved.report.get("version") != 2 and doc.status != DocumentStatus.APPROVED:
        return None
    if saved and (
        saved.revision == _revision(doc.updated_at) or doc.status == DocumentStatus.APPROVED
    ):
        return VerificationReport.model_validate(saved.report)
    return None


def _preferred_language(visit: Encounter) -> SummaryLanguage:
    demographics = visit.patient.demographics
    return (
        "es"
        if demographics.get("patient_language", demographics.get("preferred_language")) == ("es")
        else "en"
    )


def _note_language(visit: Encounter) -> SummaryLanguage:
    return "es" if visit.patient.demographics.get("note_language") == "es" else "en"


def _client(request: Request) -> AsyncOpenAI:
    client: AsyncOpenAI | None = request.app.state.openai_client
    if client is None:
        raise HTTPException(
            status_code=503, detail="Configure the OpenAI API key to process audio."
        )
    return client


def _require_change(result: object) -> None:
    if cast("CursorResult[object]", result).rowcount != 1:
        raise HTTPException(status_code=409, detail="This visit changed. Reopen it and try again.")


@router.get("/api/visits", response_model=list[VisitListItem])
async def list_visits(session: DbSession, response: Response) -> list[VisitListItem]:
    _no_store(response)
    visits = await session.scalars(
        select(Encounter)
        .options(selectinload(Encounter.patient))
        .where(Encounter.provider_id == WORKSPACE_PROVIDER_ID)
        .order_by(Encounter.created_at.desc())
        .limit(100)
    )
    return [_item(visit) for visit in visits]


@router.post("/api/visits", response_model=VisitView, status_code=201)
async def create_visit(body: VisitLabel, session: DbSession, response: Response) -> VisitView:
    _no_store(response)
    visit = Encounter(
        patient=Patient(
            demographics={"label": body.label, "preferred_language": body.preferred_language},
            allergies=[allergy.model_dump() for allergy in body.allergies],
            active_medications=[medication.model_dump() for medication in body.current_medications],
        ),
        provider_id=WORKSPACE_PROVIDER_ID,
        scheduled_time=utcnow(),
        clinical_documents=[],
    )
    session.add(visit)
    await session.commit()
    return await _view(visit, session)


@router.get("/api/visits/{visit_id}", response_model=VisitView)
async def get_visit(visit_id: uuid.UUID, session: DbSession, response: Response) -> VisitView:
    _no_store(response)
    return await _view(await _load_visit(session, visit_id), session)


@router.put("/api/visits/{visit_id}/recording", response_model=VisitView)
async def upload_recording(
    visit_id: uuid.UUID,
    request: Request,
    settings: SettingsDep,
    session: DbSession,
    response: Response,
) -> VisitView:
    _no_store(response)
    client = _client(request)
    visit = await _load_visit(session, visit_id)
    if visit.transcript or visit.status != EncounterStatus.SCHEDULED:
        raise HTTPException(status_code=409, detail="This visit already has a recording.")
    extension = request.headers.get("x-audio-format", "").lower()
    if extension not in AUDIO_FORMATS:
        raise HTTPException(status_code=415, detail="Choose a supported audio file.")
    audio = bytearray()
    async for chunk in request.stream():
        if len(audio) + len(chunk) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio files must be 25 MB or less.")
        audio.extend(chunk)
    if not audio:
        raise HTTPException(status_code=400, detail="The recording is empty.")
    try:
        recording = await transcribe_recording(client, settings, bytes(audio), extension)
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="Transcription is busy. Try again.") from exc
    except APIStatusError as exc:
        if exc.status_code in (400, 413, 415, 422):
            raise HTTPException(
                status_code=422, detail="The audio could not be read. Try another recording."
            ) from exc
        raise HTTPException(
            status_code=502, detail="Transcription service failed. Try again."
        ) from exc
    except (OpenAIError, TimeoutError) as exc:
        raise HTTPException(
            status_code=502, detail="Transcription service failed. Try again."
        ) from exc
    if not recording.text:
        raise HTTPException(status_code=422, detail="No speech was found in the recording.")
    try:
        detected, confidence = await tag_recording(client, settings, recording)
    except LanguageUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    result = await session.execute(
        update(Encounter)
        .where(
            Encounter.id == visit_id,
            Encounter.transcript.is_(None),
            Encounter.status == EncounterStatus.SCHEDULED,
        )
        .values(transcript=recording.text, status=EncounterStatus.IN_PROGRESS)
    )
    _require_change(result)
    demographics = visit.patient.demographics
    patient_language = demographics.get("preferred_language") or (
        "en" if detected == "en" else "es"
    )
    visit.patient.demographics = {
        **demographics,
        "detected_language": detected,
        "detection_confidence": confidence,
        "note_language": settings.spanish_visits_note_language if detected == "es" else "en",
        "patient_language": patient_language,
    }
    if recording.segments:
        session.add(
            RecordingTimeline(
                encounter_id=visit_id,
                segments=[segment.model_dump() for segment in recording.segments],
            )
        )
    await session.commit()
    return await _view(visit, session)


@router.post("/api/visits/{visit_id}/draft", response_model=VisitView)
async def create_draft(
    visit_id: uuid.UUID,
    request: Request,
    settings: SettingsDep,
    session: DbSession,
    response: Response,
) -> VisitView:
    _no_store(response)
    visit = await _load_visit(session, visit_id)
    if visit.clinical_documents:
        return await _view(visit, session)
    if not visit.transcript or visit.status != EncounterStatus.IN_PROGRESS:
        raise HTTPException(status_code=409, detail="Transcribe a recording first.")
    synthesis = await generate_draft(
        _client(request),
        settings,
        visit.transcript,
        PatientContext.from_patient(visit.patient),
        summary_language=_preferred_language(visit),
        note_language=_note_language(visit),
    )
    if visit.patient.demographics.get("detected_language") == "mixed":
        try:
            synthesis = await localize_draft(
                _client(request),
                settings,
                synthesis,
                note_language=_note_language(visit),
                patient_language=None,
            )
        except LanguageUnavailableError as exc:
            raise HTTPException(
                status_code=502, detail="Could not normalize the note language. Try again."
            ) from exc

    # Remove a duplicated decision, preserving findings and statements about completed tests.
    note = synthesis.soap_note
    if TESTING_DEFERRAL.search(note.plan) or TESTING_PLAN.search(note.plan):
        objective = OBJECTIVE_TESTING_ABSENCE.sub(
            "", TESTING_DEFERRAL.sub("", note.objective)
        ).strip()
        if objective != note.objective:
            note = note.model_copy(
                update={
                    "objective": objective
                    or (
                        "No se documentan hallazgos objetivos."
                        if _note_language(visit) == "es"
                        else "No objective findings documented."
                    )
                }
            )

    synthesis = synthesis.model_copy(update={"soap_note": note})

    result = await session.execute(
        update(Encounter)
        .where(Encounter.id == visit_id, Encounter.status == EncounterStatus.IN_PROGRESS)
        .values(status=EncounterStatus.DOCUMENTATION_PENDING)
    )
    _require_change(result)
    doc = ClinicalDocument(encounter=visit)
    populate_draft(doc, synthesis)
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return await _view(visit, session)


@router.post("/api/visits/{visit_id}/verify", response_model=VisitView)
async def verify_visit(
    visit_id: uuid.UUID,
    body: VerifyDraftRequest,
    request: Request,
    settings: SettingsDep,
    session: DbSession,
    response: Response,
) -> VisitView | StreamingResponse:
    _no_store(response)
    if request.headers.get("accept") == "application/x-ndjson":

        async def stream() -> AsyncIterator[str]:
            events: asyncio.Queue[str | None] = asyncio.Queue()

            async def run() -> VisitView:
                try:
                    async with request.app.state.sessionmaker() as stream_session:
                        return await _verify_visit(
                            visit_id, body, request, settings, stream_session, events.put_nowait
                        )
                finally:
                    events.put_nowait(None)

            task = asyncio.create_task(run())
            try:
                while (step := await events.get()) is not None:
                    yield json.dumps({"step": step}) + "\n"
                try:
                    view = await task
                except HTTPException as exc:
                    yield json.dumps({"error": str(exc.detail)}) + "\n"
                else:
                    yield json.dumps({"visit": view.model_dump(mode="json")}) + "\n"
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        return StreamingResponse(
            stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"}
        )
    return await _verify_visit(visit_id, body, request, settings, session)


async def _verify_visit(
    visit_id: uuid.UUID,
    body: VerifyDraftRequest,
    request: Request,
    settings: SettingsDep,
    session: DbSession,
    progress: Callable[[str], None] | None = None,
) -> VisitView:
    visit = await _load_visit(session, visit_id)
    doc = next(iter(visit.clinical_documents), None)
    if doc is None or not visit.transcript:
        raise HTTPException(status_code=409, detail="Generate a draft first.")
    if doc.status == DocumentStatus.APPROVED:
        raise HTTPException(status_code=409, detail=LOCKED_VISIT)
    if _revision(body.revision) != _revision(doc.updated_at):
        raise HTTPException(status_code=409, detail=STALE_DRAFT)
    try:
        draft, report = await verify_draft(
            _client(request),
            settings,
            _synthesis(doc),
            visit.transcript,
            PatientContext.from_patient(visit.patient),
            repair=body.repair,
            summary_language=_preferred_language(visit),
            note_language=_note_language(visit),
            progress=progress,
        )
    except VerificationUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return await _save_verified_revision(session, visit, doc, draft, report)


async def _save_verified_revision(
    session: DbSession,
    visit: Encounter,
    doc: ClinicalDocument,
    draft: EncounterSynthesisResult,
    report: VerificationReport,
) -> VisitView:
    revision = utcnow()
    result = await session.execute(
        update(ClinicalDocument)
        .where(
            ClinicalDocument.id == doc.id,
            ClinicalDocument.status == DocumentStatus.REVIEW_PENDING,
            ClinicalDocument.updated_at == doc.updated_at,
        )
        .values(
            generated_soap=draft.soap_note.model_dump(mode="json"),
            visit_summary_for_patient=draft.visit_summary_for_patient,
            staged_orders=[order.model_dump(mode="json") for order in draft.staged_orders],
            updated_at=revision,
        )
    )
    _require_change(result)
    saved = await session.get(DraftVerification, doc.id)
    if saved is None:
        saved = DraftVerification(document_id=doc.id)
        session.add(saved)
    saved.revision = _revision(revision)
    saved.report = report.model_dump(mode="json")
    await session.commit()
    await session.refresh(doc)
    return await _view(visit, session)


def _editable_doc(visit: Encounter, revision: datetime) -> ClinicalDocument:
    doc = next(iter(visit.clinical_documents), None)
    if doc is not None and doc.status == DocumentStatus.APPROVED:
        raise HTTPException(status_code=409, detail=LOCKED_VISIT)
    if doc is None or _revision(revision) != _revision(doc.updated_at):
        raise HTTPException(status_code=409, detail=STALE_DRAFT)
    return doc


@router.put("/api/visits/{visit_id}/language", response_model=VisitView)
async def change_visit_language(
    visit_id: uuid.UUID,
    body: ChangeLanguageRequest,
    request: Request,
    settings: SettingsDep,
    session: DbSession,
    response: Response,
) -> VisitView:
    _no_store(response)
    visit = await _load_visit(session, visit_id)
    doc = _editable_doc(visit, body.revision)
    draft = _synthesis(doc)
    note_changed = body.note_language != _note_language(visit)
    patient_changed = body.patient_language != _preferred_language(visit)
    if not (note_changed or patient_changed):
        return await _view(visit, session)
    try:
        translated = await localize_draft(
            _client(request),
            settings,
            draft,
            note_language=body.note_language if note_changed else None,
            patient_language=body.patient_language if patient_changed else None,
        )
        translated, report = await verify_draft(
            _client(request),
            settings,
            translated,
            visit.transcript or "",
            PatientContext.from_patient(visit.patient),
            repair=False,
            summary_language=body.patient_language,
            note_language=body.note_language,
        )
    except (LanguageUnavailableError, VerificationUnavailableError, ValueError) as exc:
        raise HTTPException(
            status_code=502, detail="Could not translate and verify the draft."
        ) from exc
    visit.patient.demographics = {
        **visit.patient.demographics,
        "note_language": body.note_language,
        "patient_language": body.patient_language,
    }
    return await _save_verified_revision(session, visit, doc, translated, report)


@router.post("/api/visits/{visit_id}/verification/undo", response_model=VisitView)
async def undo_correction(
    visit_id: uuid.UUID,
    body: UndoCorrectionRequest,
    session: DbSession,
    response: Response,
) -> VisitView:
    _no_store(response)
    visit = await _load_visit(session, visit_id)
    doc = _editable_doc(visit, body.revision)
    report = await _verification(doc, session)
    correction = (
        next(
            (item for item in report.corrections if item.id == body.correction_id),
            None,
        )
        if report
        else None
    )
    if report is None or correction is None or correction.area == "order":
        raise HTTPException(status_code=409, detail="This correction is no longer available.")
    draft = _synthesis(doc)
    area = correction.area
    note = draft.soap_note.model_dump()
    value = draft.visit_summary_for_patient if area == "patient_instructions" else note[area]
    if value.count(correction.after) != 1:
        raise HTTPException(status_code=409, detail="The corrected wording is no longer unique.")
    value = value.replace(correction.after, correction.before, 1)
    if area == "patient_instructions":
        draft = draft.model_copy(update={"visit_summary_for_patient": value})
    else:
        note[area] = value
        draft = draft.model_copy(update={"soap_note": SOAPNote.model_validate(note)})
    report.corrections.remove(correction)
    if area in ("plan", "patient_instructions"):
        report.prescription_directions = None
    report.findings = [
        item
        for item in report.findings
        if item.area != area or item.text.strip() not in correction.after
    ]
    report.findings.append(
        Finding(
            area=area,
            text=correction.before,
            status="flagged",
            severity="question",
            evidence=[],
            kind="undone_fix",
            question="Original wording restored. Can you confirm this wording?",
        )
    )
    report.trace.append("Automatic edit undone. The restored wording needs a decision.")
    return await _save_verified_revision(session, visit, doc, draft, report)


@router.put("/api/visits/{visit_id}/review", response_model=VisitView)
async def save_review(
    visit_id: uuid.UUID,
    body: Review,
    session: DbSession,
    settings: SettingsDep,
    response: Response,
) -> VisitView:
    _no_store(response)
    visit = await _load_visit(session, visit_id)
    doc = next(iter(visit.clinical_documents), None)
    if body.text_on_approval_to and body.clinician is None:
        raise HTTPException(status_code=422, detail="Approve the visit before texting its summary.")
    if doc is not None and doc.status == DocumentStatus.APPROVED:
        raise HTTPException(status_code=409, detail=LOCKED_VISIT)
    if (
        doc is None
        or doc.status != DocumentStatus.REVIEW_PENDING
        or body.revision.replace(tzinfo=body.revision.tzinfo or UTC)
        != doc.updated_at.replace(tzinfo=doc.updated_at.tzinfo or UTC)
    ):
        raise HTTPException(status_code=409, detail=STALE_DRAFT)
    stmt = (
        update(ClinicalDocument)
        .where(
            ClinicalDocument.id == doc.id,
            ClinicalDocument.status == DocumentStatus.REVIEW_PENDING,
            ClinicalDocument.updated_at == doc.updated_at,
        )
        .values(
            generated_soap=body.soap_note.model_dump(mode="json"),
            visit_summary_for_patient=body.visit_summary_for_patient,
            updated_at=utcnow(),
        )
    )
    if body.clinician is not None:
        kept = set(body.kept_orders)
        if any(index < 0 or index >= len(doc.staged_orders) for index in kept):
            raise HTTPException(status_code=422, detail="Invalid proposed order selection.")
        verification = await _verification(doc, session)
        if (
            verification is None
            or body.soap_note.model_dump(mode="json") != doc.generated_soap
            or body.visit_summary_for_patient != doc.visit_summary_for_patient
        ):
            raise HTTPException(status_code=409, detail="Save and verify the current draft first.")
        approved_at = utcnow()
        dropped_prescriptions = {
            index
            for index, order in enumerate(doc.staged_orders)
            if order["order_type"] == "prescription" and index not in kept
        }
        _require_consistent_draft(verification, kept, dropped_prescriptions)
        verification.resolutions = _audit_resolutions(
            verification, body.resolutions, kept, body.clinician, approved_at
        )
        saved = await session.get(DraftVerification, doc.id)
        if saved is None:
            raise HTTPException(status_code=409, detail="Verify the current draft first.")
        saved.report = verification.model_dump(mode="json")
        stmt = stmt.values(
            status=DocumentStatus.APPROVED,
            approved_by=body.clinician,
            approved_at=approved_at,
            staged_orders=[order for index, order in enumerate(doc.staged_orders) if index in kept],
        )
    result = await session.execute(stmt)
    if cast("CursorResult[object]", result).rowcount != 1:
        status = await session.scalar(
            select(ClinicalDocument.status).where(ClinicalDocument.id == doc.id)
        )
        detail = LOCKED_VISIT if status == DocumentStatus.APPROVED else STALE_DRAFT
        raise HTTPException(status_code=409, detail=detail)
    if body.clinician is not None:
        visit.status = EncounterStatus.COMPLETED
    await session.commit()
    await session.refresh(doc)
    view = await _view(visit, session)
    if body.text_on_approval_to:
        try:
            await _dispatch_summary(visit_id, body.text_on_approval_to, session, settings)
        except HTTPException as exc:
            view.summary_text_error = str(exc.detail)
        else:
            view.summary_text_sent = True
    return view


def _require_consistent_draft(
    report: VerificationReport, kept: set[int], dropped_prescriptions: set[int]
) -> None:
    if dropped_prescriptions and report.prescription_directions is None:
        raise HTTPException(
            status_code=409,
            detail="Verify the Plan and patient instructions before excluding a prescription.",
        )
    locations = {
        item.area
        for item in report.prescription_directions or []
        if item.order_index in dropped_prescriptions
    }
    if locations:
        labels = {"plan": "Plan", "patient_instructions": "patient instructions"}
        sections = " and ".join(label for area, label in labels.items() if area in locations)
        raise HTTPException(
            status_code=409,
            detail=f"Edit and verify the {sections} before approving a dropped prescription.",
        )
    if any(
        item.kind == "language" and (item.order_index is None or item.order_index in kept)
        for item in report.findings
    ):
        raise HTTPException(
            status_code=409,
            detail="Correct the flagged language mismatch and verify the draft again.",
        )


def _audit_resolutions(
    report: VerificationReport,
    resolutions: list[FindingResolution],
    kept: set[int],
    clinician: str,
    approved_at: datetime,
) -> list[ResolutionAudit]:
    required = {item.id: item for item in report.findings if item.severity != "info"}
    decisions = {item.finding_id: item for item in resolutions}
    if len(decisions) != len(resolutions) or decisions.keys() - required.keys():
        raise HTTPException(
            status_code=409, detail="Resolve findings from the current verification."
        )
    audit = []
    for finding_id, finding in required.items():
        resolution = decisions.get(finding_id)
        if finding.kind == "unsupported_order" and finding.order_index in kept:
            raise HTTPException(
                status_code=409, detail="Drop proposed orders without source support."
            )
        if finding.severity == "blocking":
            dropped = (
                resolution
                and resolution.action == "dropped"
                and (finding.order_index is not None and finding.order_index not in kept)
            )
            overridden = (
                resolution
                and resolution.action == "overridden"
                and (
                    len(resolution.note or "") >= 8
                    and (finding.order_index is None or finding.order_index in kept)
                )
            )
            if not (dropped or overridden):
                raise HTTPException(
                    status_code=409,
                    detail="Resolve each blocking finding. Overrides need at least 8 characters.",
                )
        elif resolution is None or resolution.action not in {"answered", "dismissed"}:
            raise HTTPException(status_code=409, detail="Answer or dismiss every open question.")
        assert resolution is not None
        audit.append(
            ResolutionAudit(
                **resolution.model_dump(),
                reviewer=clinician,
                resolved_at=approved_at,
            )
        )
    return audit


async def _dispatch_summary(
    visit_id: uuid.UUID, phone: str, session: DbSession, settings: SettingsDep
) -> SummaryTextReceipt:
    service = OutreachService(session, settings)
    try:
        result = await service.dispatch_care_plan(visit_id, recipient=phone, require_live=True)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SmsUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SmsProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return SummaryTextReceipt(message_sid=result["message_sid"])


@router.post("/api/visits/{visit_id}/text-summary", response_model=SummaryTextReceipt)
async def text_summary(
    visit_id: uuid.UUID,
    body: SummaryTextRequest,
    session: DbSession,
    settings: SettingsDep,
    response: Response,
) -> SummaryTextReceipt:
    _no_store(response)
    visit = await _load_visit(session, visit_id)
    if visit.status != EncounterStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Approve the visit before texting its summary.")
    return await _dispatch_summary(visit_id, body.phone_number, session, settings)
