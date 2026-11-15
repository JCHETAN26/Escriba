# Escriba

**Autonomous Planning, Human Approval.**

Escriba turns a visit recording into a SOAP draft, proposed orders, and a take-home
checklist. The verification panel cites the transcript, checks prescriptions against
documented allergies and medications, and flags wording the clinician needs to review.
The clinician approves and locks the note before copying it into the chart or texting
the approved instructions through Twilio. Escriba does not place orders.

---

## Run the visit workspace

From the repository root, install `requirements.txt` and set
`ESCRIBA_OPENAI_API_KEY` in `.env`. Start the local workspace:

```bash
ESCRIBA_TWILIO_VALIDATE_SIGNATURE=false uvicorn app.main:create_app --factory \
  --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Open **http://127.0.0.1:8000**. Create a visit, record speech with the microphone or
choose an audio file up to 25 MB, and select **Transcribe and draft**. The app sends
the completed recording to OpenAI. It saves the resulting transcript and uses it
to generate the editable SOAP note, patient instructions, and proposed orders.
The audio file itself is not retained by the workspace.

Review the transcript alongside the draft, use **View source** to jump to the
original evidence, and resolve verification findings. If you drop a prescription,
edit any directions to use it in the Plan and patient instructions, then select
**Save & verify**. Enter the reviewing clinician's name and select **Approve & lock**.
Copy the approved SOAP note as a whole or section by section. Approvals and drafts persist in the
local database. Place orders separately in your clinical system.

This workspace is for single-user local evaluation with synthetic visits.
The local approval records the entered name. Shared use with patient data
requires authenticated clinician accounts, access controls, and a production
deployment configured for that data. The workspace processes completed recordings
and displays the transcript without verified speaker identities.

For a synthetic demo, use `demo/benchmark/audio/mixed-language-demo.wav` to review
an English note and Spanish instructions with original Spanish evidence. Use
`demo/benchmark/audio/amlodipine-allergy-demo.wav` with a documented amlodipine
allergy (hives) to demonstrate the blocking finding and dropped-order review.

The workspace API and health check are exposed by default. Set
`ESCRIBA_DEMO_MODE=true` in development to mount the legacy encounter,
document, webhook, and outreach routes described below.

## Host the hackathon demo on Vercel

Link the Escriba project in Vercel and connect a dedicated Neon PostgreSQL database
to its Production environment. Vercel exposes the connection URL as
`DATABASE_URL` or `DATABASE_URL_UNPOOLED`. You can also set
`ESCRIBA_DATABASE_URL` to a separate `postgresql+asyncpg://` connection URL.
Use a blank database for the demo.

Set these Production environment variables before running `vercel deploy --prod`:

| Variable | Value |
| --- | --- |
| `ESCRIBA_ENVIRONMENT` | `staging` |
| `ESCRIBA_HOSTED_DEMO` | `true` |
| `ESCRIBA_PUBLIC_BASE_URL` | The production demo's HTTPS origin |
| `ESCRIBA_OPENAI_API_KEY` | A sensitive server-side environment variable |
| `ESCRIBA_TWILIO_VALIDATE_SIGNATURE` | `false` |

Open the production URL in a private window to confirm guest access. The hosted
demo uses a browser cookie to keep visits and returning-patient charts together
for that browser. It accepts fictional recordings up to 25 MB and disables SMS.
For larger uploads, the browser sends 3 MB chunks that Neon holds until Escriba
assembles the recording. Unfinished uploads expire after one hour and are cleared
when another chunked upload starts.

## The 5-Phase End-to-End Architecture

```
                                  ESCRIBA LIFECYCLE

 [Pre-Visit]              [During Visit]             [Post-Visit Synthesis]           [Loop-Closing]
 Patient SMS               Live Audio Stream             1-Click Review Drawer         Outreach & Triage
     │                            │                               │                           │
     ▼                            ▼                               ▼                           ▼
Twilio Inbound           WS /stream (WebSockets)          POST /synthesize             POST /dispatch-summary
Webhook (Intake)         • Base64 Audio Ingestion         • OpenAI Structured Outputs  • Outbound 6th-Grade SMS
     │                   • Mock ASR Stream                • Small Talk Filter          • 48-Hr Check-In Ping
     │                   • Real-Time Safety Guardrail     • SOAP Note + Staged Orders  • Inbound Triage Webhook
     │                     (Bactrim vs. Sulfa Allergy)    • Formulary Verification     • Red-Flag Dashboard Alert
     │                            │                         (Ozempic Prior-Auth Flag)         │
     ▼                            ▼                               │                           ▼
Patient Record           Encounter In-Progress                    ▼                    Clinician Notified
Created in DB            Alerts pushed to UI             1-Click Doctor Approval        of Complications
```

| Phase | Milestone | Capabilities | Status |
|:-----:|:----------|:-------------|:------:|
| **1** | **Intake & Foundation** | Async SQLAlchemy 2.0 (SQLite/Postgres), domain models (`Patient`, `Encounter`, `IntakeChat`, `ClinicalDocument`), Twilio HMAC signature security, idempotent pre-visit SMS ingestion. | **Complete** |
| **2** | **Audio & Safety Guardrails** | Full-duplex WebSocket (`/encounter/{id}/stream`), base64 audio streaming, mock ASR, real-time OpenAI Structured Outputs LLM guardrail catching drug-allergy contraindications with fail-visible posture. | **Complete** |
| **3** | **Clinical NLP Synthesis** | `POST /encounter/{id}/synthesize` using `client.beta.chat.completions.parse`. Central prompts with Small Talk Filter (filters chit-chat, preserves SDOH/psychosocial stressors), structured SOAP note, grounded staged orders, and 6th-grade patient summary. | **Complete** |
| **4** | **Order Orchestrator** | Real-time formulary verification (`POST /document/{id}/verify-orders`) catching high-cost brand rejections (Ozempic, Vyvanse, Synthroid) and recommending generics; 1-click clinician approval (`POST /document/{id}/approve`) with full audit timestamps. | **Complete** |
| **5** | **Loop-Closing Triage** | Care plan SMS dispatch (`POST /dispatch-summary`), simulated 48-hr check-in (`POST /demo-trigger-triage`), dual-mode Twilio webhook routing pre-visit intake vs. post-visit triage with automatic red-flag complication escalation. | **Complete** |

---

## Codebase Structure

```
app/
├── core/
│   ├── config.py           # Pydantic Settings, env variables, production invariants
│   ├── database.py         # Async engine, sessionmaker, Base, DB session dependency
│   ├── dependencies.py     # FastAPI shared request dependencies
│   ├── prompts.py          # Central LLM prompts, Small Talk Filter, prompt-injection fences
│   └── security.py         # Twilio HMAC-SHA1 signature verification
├── models/
│   └── domain.py           # Patient, Encounter, IntakeChat, ClinicalDocument, ORM validators
├── schemas/
│   ├── audio.py            # WebSocket protocol: AudioChunk, Transcript, GuardrailAlert
│   ├── clinical.py         # Medication, Allergy, StagedOrder, SOAPNote, EncounterSynthesis
│   ├── intake.py           # TwilioInboundSMS, IntakeExtractionSchema
│   └── orchestrator.py     # FormularyVerificationResult, EnrichedOrder, OrderApproval
├── services/
│   ├── clinical_nlp.py     # Structured OpenAI synthesis with context window limits
│   ├── encounter_stream.py # Async WebSocket connection manager, backpressure, FIFO transcriber
│   ├── formulary_mock.py   # Surescripts-style e-prescribing benefit & prior-auth simulator
│   ├── guardrail.py        # Real-time contraindication checker with grounding checks
│   ├── intake.py           # Pre-visit intake & post-visit triage routing engine
│   ├── outreach.py         # Outbound SMS care plan dispatcher & 48-hour follow-up
│   └── transcription.py    # Non-blocking mock streaming ASR
└── routers/
    ├── audio.py            # WS /encounter/{encounter_id}/stream
    ├── encounter.py        # POST /encounter/{encounter_id}/synthesize
    ├── intake.py           # POST /webhook/twilio
    ├── orchestrator.py     # POST /document/{id}/verify-orders & POST /approve
    └── outreach.py         # POST /dispatch-summary & POST /demo-trigger-triage
tests/
├── fakes.py                # Mock OpenAI client with Structured Outputs & RateLimit fixtures
├── test_clinical_nlp.py    # Unit tests for clinical synthesis & prompt templates
├── test_encounter_stream.py# End-to-end WebSocket tests for live guardrail alerts
├── test_encounter_synthesis_api.py # REST tests for SOAP & order synthesis
├── test_formulary_mock.py  # Unit tests for Ozempic/Vyvanse/generic coverage logic
├── test_guardrail.py       # Unit tests for allergy cross-reactivity & injection safety
├── test_intake_webhook.py  # Tests for pre-visit SMS intake & idempotency
├── test_orchestrator_api.py# Tests for formulary verification & 1-click approval
├── test_outreach.py        # Tests for care plan dispatch & 48-hour check-in
└── test_triage_routing.py  # Tests for post-visit triage routing & complication flagging
```

---

## Getting Started

### Prerequisites
- **Python 3.11+**
- (Optional for live demo) **Twilio Phone Number** + **ngrok** account

### 1. Installation

```bash
# Clone and enter repo
git clone https://github.com/JCHETAN26/Escriba.git
cd Escriba

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment variables
cp .env.example .env
```

### 2. Configuration (`.env`)

For local hackathon testing and demos:

```env
ESCRIBA_ENVIRONMENT=development
ESCRIBA_OPENAI_API_KEY=your-openai-api-key
ESCRIBA_TWILIO_VALIDATE_SIGNATURE=false

# Optional: Add your live Twilio credentials to receive real SMS on your phone:
# ESCRIBA_TWILIO_ACCOUNT_SID=ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ESCRIBA_TWILIO_AUTH_TOKEN=your-auth-token
# ESCRIBA_TWILIO_FROM_NUMBER=+15551234567
```

### 3. Launching the Server

```bash
uvicorn app.main:create_app --factory --reload --port 8000
```

- **Swagger UI:** <http://localhost:8000/docs>
- **Health Check:** <http://localhost:8000/healthz>

---

## Live Hackathon Pitch Guide: Exposing Webhooks via ngrok

To demo live two-way SMS with the judges during the pitch:

Enable the legacy endpoints with `ESCRIBA_DEMO_MODE=true` before running this demo.

### Step 1: Start your local Escriba server
```bash
uvicorn app.main:create_app --factory --port 8000
```

### Step 2: Open an ngrok tunnel
In a separate terminal window:
```bash
ngrok http 8000
```
Note your public HTTPS URL (e.g., `https://abc1234.ngrok-free.app`).

### Step 3: Configure Twilio Console
1. Log into your [Twilio Console](https://console.twilio.com/).
2. Navigate to **Phone Numbers** → **Manage** → **Active Numbers** → Click your number.
3. Under **Messaging Configuration** → **A MESSAGE COMES IN**:
   - Set to: `Webhook`
   - URL: `https://abc1234.ngrok-free.app/webhook/twilio`
   - HTTP Method: `HTTP POST`
4. Click **Save Configuration**.

---

## 3-Minute Live Hackathon Pitch Script

1. **Pre-Visit Intake (Judges' Phone):**
   - The judge texts `"Hi, I've had a bad sinus headache and pain for 4 days"` to the Twilio number.
   - Escriba creates a patient and encounter, replies with intake guidance, and records the chat.
2. **Live Visit & Safety Guardrail:**
   - Connect to `WS /encounter/{id}/stream`.
   - Send live transcript: `"Let's start you on Bactrim DS twice daily."`
   - Instant guardrail alert fires over WebSocket:
     `[HIGH SEVERITY ALERT]: Bactrim conflicts with patient's documented Sulfa allergy (hives). Suggested Action: Use non-sulfonamide agent such as Augmentin.`
3. **1-Click Clinical NLP Synthesis:**
   - Call `POST /encounter/{id}/synthesize`.
   - Shows SOAP note with banter filtered out, actionable staged orders grounded with audio quotes, and a 6th-grade patient summary.
4. **Agentic Order Orchestrator:**
   - Call `POST /document/{id}/verify-orders`.
   - AI detects brand-name *Ozempic* is not covered, requires prior authorization, and proactively suggests *Metformin 500mg* ($15 copay vs. $950).
   - Doctor reviews and clicks Approve (`POST /document/{id}/approve`).
5. **Closing the Loop (Post-Visit Triage):**
   - Call `POST /encounter/{id}/dispatch-summary`: patient instantly receives their 6th-grade care plan on their phone via SMS.
   - Trigger follow-up: `POST /encounter/{id}/demo-trigger-triage` sends the 48-hour check-in.
   - The judge replies: `"I feel dizzy and worse."`
   - Escriba's triage agent detects the red flag, alerts the patient, and marks `triage_alert = True` on the clinician's dashboard!

---

## SOAP note and patient summary demo

From the repo root with development dependencies installed:

```bash
python -m demo.synthesis
python -m demo.synthesis --live
```

The offline run feeds seven synthetic visits and scripted reference outputs through the synthesis
endpoint and checks that each SOAP note, patient summary, and set of proposed orders is saved.
It writes `demo/demo-report.md`. The live run requires `ESCRIBA_OPENAI_API_KEY` from your
environment or `.env` and writes a separate `demo/live-report.md` with the model's outputs.
Use `--output` to choose another report path. Both runs use a temporary database.

The report shows each transcript beside the draft and lists any failed checks. The cases cover
an explicit prescription, a lab and referral, conditional treatment, missing measurements,
barriers to care, corrected laterality and negation, and instructions embedded in a transcript.
Review each draft against its source for clinical accuracy, completeness, and readability.

---

## Quality Gates & Verification

```bash
# Run the complete test suite (74 tests)
pytest

# Linting and formatting
ruff check .

# Strict type checking
mypy app
```
