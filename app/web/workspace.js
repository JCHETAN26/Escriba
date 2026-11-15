"use strict";

const byId = (id) => document.getElementById(id);
const fields = ["subjective", "objective", "assessment", "plan"];
let maxAudioBytes = 25_000_000;
let audioChunkBytes = 0;
const audioLimit = () => maxAudioBytes / 1_000_000 + " MB";
const audioLabel = () => "MP3, M4A, WAV or WebM · up to " + audioLimit();
const audioFormats = new Set(["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]);
const visitStates = {
  scheduled: "Ready to capture",
  in_progress: "Transcript saved",
  documentation_pending: "Draft",
  completed: "Approved",
};
const editors = new Map();
let visits = [];
let patientCharts = [];
let visit = null;
let screen = "capture";
let decisions = [];
let busy = false;
let verifying = false;
const resolutions = new Map();
let dirty = false;
let selection = null;
let recorder = null;
let recordingTimer;
let processingStep = null;

function create(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}

function makeButton(className, label, action) {
  const button = create("button", className, label);
  button.type = "button";
  button.addEventListener("click", action);
  return button;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(/(Z|[+-]\d\d:\d\d)$/.test(value) ? value : value + "Z");
  return date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function say(message, error = false) {
  const status = byId("status");
  status.textContent = message;
  status.dataset.error = String(error);
}

async function api(path, options = {}) {
  const headers = { "X-Escriba-Workspace": "1", ...options.headers };
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : "Check the visit details and try again.");
  }
  return data;
}

const visitPath = (suffix = "") => "/api/visits/" + visit.id + suffix;
const pagePath = (id, page) => "/visits/" + id + "/" + page;
const pageForVisit = (item) => item.status === "completed" ? "approved"
  : item.status === "documentation_pending" ? "review" : "capture";
const json = (method, body) => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});
const normalizedPhone = (id = "sms-phone") => byId(id).value.trim().replace(/[\s().-]/g, "");
const validPhone = (id) => /^\+[1-9][0-9]{7,14}$/.test(normalizedPhone(id));
const textStatusKey = (id) => "escriba-summary-text-" + id;

function matchesSavedDraft() {
  const draft = visit?.draft;
  return draft && fields.every(field => editors.get(field).value.trim() === draft.soap_note[field])
    && byId("summary").value.trim() === draft.visit_summary_for_patient;
}

function currentVerification() {
  return matchesSavedDraft() ? visit.verification : null;
}

function droppedPrescriptions() {
  return (visit?.draft?.staged_orders || []).flatMap((order, index) =>
    order.order_type === "prescription" && decisions[index] === "drop" ? [index] : []);
}

function prescriptionReconciliation(report) {
  const dropped = droppedPrescriptions();
  const marks = (report?.prescription_directions || []).filter(item => dropped.includes(item.order_index));
  return {
    recheck: dropped.length > 0 && !report?.prescription_directions,
    marks,
    directions: marks.filter(item => !findingDecision(item)),
    keptWithheld: report?.findings.some(item => item.kind === "chart_withholding"
      && decisions[item.order_index] === "keep"),
  };
}

function updateApproval() {
  const undecided = decisions.filter((decision) => decision === null).length;
  const hasReviewer = Boolean(byId("clinician").value.trim());
  const send = byId("text-on-approval").checked;
  const needsPhone = send && !validPhone("approval-phone");
  const verification = currentVerification();
  const open = openFindings(verification);
  const reconciliation = prescriptionReconciliation(verification);
  const wrongLanguage = verification?.findings.some(finding => finding.kind === "language"
    && (finding.order_index === null || decisions[finding.order_index] === "keep"));
  const unsupported = (verification?.findings || []).some(finding =>
    finding.kind === "unsupported_order" && decisions[finding.order_index] === "keep");
  byId("approval-recipient").hidden = !send;
  byId("approval-phone").disabled = !send;
  byId("order-count").textContent = visit?.draft?.status === "approved"
    ? decisions.length + " kept"
    : undecided ? undecided + (undecided === 1 ? " needs a decision" : " need a decision") : "";
  byId("approve").disabled = busy || undecided > 0 || !hasReviewer || needsPhone
    || !verification || unsupported || open.length > 0 || wrongLanguage
    || reconciliation.recheck || reconciliation.keptWithheld || reconciliation.directions.length > 0;
  byId("approval-hint").textContent = reconciliation.directions.length
    ? "Edit the flagged instructions and save & verify, or dismiss an incorrect mark."
    : reconciliation.keptWithheld ? "Drop the prescription documented as withheld."
    : reconciliation.recheck ? "Verify the Plan and patient instructions again after dropping this prescription."
    : wrongLanguage ? "Correct the flagged language mismatch and verify again."
    : !verification
    ? "Save and verify this draft before approving."
    : unsupported ? "Drop proposed orders without source support."
      : open.some(item => item.severity === "blocking") ? "Drop the affected order or record an override reason."
        : open.length ? "Answer or dismiss the open questions."
          : undecided ? "Keep or drop every proposed order before approving."
            : !hasReviewer ? "Enter the reviewer’s name to approve."
              : needsPhone ? "Enter a phone number with the country code." : "";

}

function updateControls() {
  const recording = recorder !== null;
  const processing = processingStep !== null;
  const hasTranscript = Boolean(visit?.transcript && !visit.draft);
  const ready = Boolean(selection && !recording && !processing);
  byId("open-new-visit").disabled = busy || recording;
  byId("back-visits").disabled = busy || recording;
  byId("visit-list").inert = busy || recording;
  byId("visit-title").disabled = busy;
  byId("rename-visit-input").readOnly = busy;
  byId("patient-mode").disabled = Boolean(visit) || busy || recording;
  byId("patient-name").disabled = Boolean(visit) || busy || recording;
  byId("patient-chart").disabled = Boolean(visit) || busy || recording;
  byId("chart-allergies").readOnly = Boolean(visit) || busy;
  byId("chart-medications").readOnly = Boolean(visit) || busy;
  byId("review").inert = busy;
  byId("view-review").disabled = busy;
  byId("back-approved").disabled = busy;
  byId("upload-area").hidden = recording || processing || hasTranscript;
  byId("file").disabled = busy || recording;
  byId("record").hidden = ready || recording || processing || hasTranscript;
  byId("record").disabled = busy || !navigator.mediaDevices || typeof MediaRecorder === "undefined";
  byId("stop").hidden = !recording;
  byId("recording-clock").hidden = processing || hasTranscript;
  byId("waveform").hidden = processing || hasTranscript;
  byId("record-hint").hidden = ready || processing || hasTranscript;
  byId("record-hint").textContent = recording ? "Recording microphone audio" : "Select the microphone to start recording";
  byId("recording-actions").hidden = !ready;
  byId("transcribe").disabled = busy;
  byId("discard").disabled = busy;
  byId("resume").hidden = !hasTranscript || processing;
  byId("generate").disabled = busy;
  byId("processing").hidden = !processing;
  byId("recorder-card").dataset.stage = recording ? "recording" : processing ? "processing" : ready ? "ready" : "idle";
  for (const [index, item] of [...byId("processing").children].entries()) {
    item.dataset.state = index < processingStep ? "done" : index === processingStep ? "active" : "queued";
    if (index === processingStep) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  }
  byId("send-sms").disabled = busy || !validPhone();
  updateApproval();
}

async function work(action, errorNode) {
  if (busy) return;
  busy = true;
  if (errorNode) errorNode.textContent = "";
  updateControls();
  try {
    await action();
  } catch (error) {
    const message = error.message || "Something went wrong. Try again.";
    if (errorNode) errorNode.textContent = message;
    else say(message, true);
  } finally {
    busy = false;
    processingStep = null;
    updateControls();
  }
}

async function copyText(value, button) {
  try {
    await navigator.clipboard.writeText(value);
    say("Copied to clipboard.");
    const label = button.querySelector("span") || button;
    const original = label.textContent;
    button.disabled = true;
    label.textContent = "Copied";
    setTimeout(() => {
      label.textContent = original;
      button.disabled = false;
    }, 1500);
  } catch {
    say("Could not copy to the clipboard.", true);
  }
}

for (const field of fields) {
  const name = field[0].toUpperCase() + field.slice(1);
  const row = create("div", "soap-row");
  const tile = create("span", "soap-tile", name[0]);
  tile.setAttribute("aria-hidden", "true");
  const content = create("div", "soap-content");
  const label = create("label", "field-label", name);
  label.htmlFor = field;
  const editor = create("textarea");
  editor.id = field;
  editor.rows = 2;
  editor.required = true;
  editor.maxLength = 20000;
  content.append(label, editor);
  row.append(tile, content);
  byId("soap-fields").append(row);
  editors.set(field, editor);
}

for (let index = 0; index < 48; index++) {
  const bar = create("span");
  const amplitude = Math.abs(Math.sin(index * 1.27) * Math.cos(index * .43));
  bar.style.height = (10 + 50 * amplitude) + "px";
  bar.style.animationDelay = (-(index % 10) * .09) + "s";
  byId("waveform").append(bar);
}

function setScreen(next, focus = false) {
  if (screen !== next) say("");
  screen = next;
  if (visit) {
    const path = pagePath(visit.id, next);
    if (location.pathname !== path) history.replaceState(null, "", path);
  }
  document.body.dataset.screen = next;
  const approved = visit?.draft?.status === "approved";
  for (const page of ["capture", "review", "approved"]) byId(page).hidden = page !== next;
  byId("steps").hidden = next === "approved" || approved;
  byId("view-review").hidden = next !== "approved";
  byId("back-approved").hidden = next !== "review" || !approved;
  byId("capture-title").hidden = next !== "capture";
  byId("capture-title").textContent = visit?.label || "New visit";
  byId("visit-title-text").textContent = visit?.label || "";
  byId("visit-heading").hidden = false;
  byId("visit-title").hidden = next === "capture";
  byId("rename-visit-form").hidden = true;
  const prior = visit?.previous_visit;
  const priorLink = byId("previous-visit");
  priorLink.hidden = !prior;
  if (prior) {
    priorLink.href = pagePath(prior.id, "review");
    priorLink.textContent = "Prior visit on this chart · " + formatDate(prior.created_at);
  }
  const badge = byId("visit-status");
  badge.hidden = next === "capture";
  badge.textContent = approved ? "Approved · locked" : "Draft · needs review";
  badge.dataset.state = approved ? "completed" : "documentation_pending";
  for (const [index, item] of [...byId("steps").children].entries()) {
    if (index === (next === "capture" ? 0 : 1)) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  }
  if (focus) {
    window.scrollTo(0, 0);
    byId(next === "capture" && !visit ? "patient-name" : "visit-heading").focus();
  }
}

function renderVisitList() {
  const list = byId("visit-list");
  list.replaceChildren();
  if (!visits.length) list.append(create("p", "muted", "No visits yet."));
  for (const item of visits) {
    const button = create("a", "visit-item");
    button.href = pagePath(item.id, pageForVisit(item));
    const title = create("strong", "", item.label);
    const state = create("span", "visit-dot");
    state.dataset.state = item.status;
    state.setAttribute("aria-hidden", "true");
    button.append(title, state, create("span", "visit-meta", [
      item.patient_name || "Unnamed patient", formatDate(item.created_at), visitStates[item.status],
    ].join(" · ")));
    button.setAttribute("aria-current", String(item.id === visit?.id));
    list.append(button);
  }
}

function hasUnsavedWork() {
  const pendingDecisions = visit?.draft?.status === "review_pending" && decisions.some(Boolean);
  return dirty || selection || pendingDecisions;
}

function startVisit() {
  if (busy || recorder) return;
  if (visit) location.assign("/visits/new");
  else byId("patient-name").focus();
}

byId("open-new-visit").addEventListener("click", startVisit);
byId("back-visits").addEventListener("click", startVisit);
byId("view-review").addEventListener("click", () => location.assign(pagePath(visit.id, "review")));
byId("back-approved").addEventListener("click", () => location.assign(pagePath(visit.id, "approved")));

function closeVisitRename() {
  byId("rename-visit-form").hidden = true;
  byId("visit-heading").hidden = false;
  byId("visit-title").focus();
}

byId("visit-title").addEventListener("click", () => {
  const input = byId("rename-visit-input");
  input.value = visit.label;
  input.setCustomValidity("");
  byId("visit-heading").hidden = true;
  byId("rename-visit-form").hidden = false;
  input.focus();
  input.select();
});

byId("rename-visit-input").addEventListener("input", event => event.target.setCustomValidity(""));
byId("rename-visit-input").addEventListener("keydown", event => {
  if (event.key === "Escape" && !busy) {
    event.preventDefault();
    closeVisitRename();
  }
});

byId("rename-visit-form").addEventListener("submit", async event => {
  event.preventDefault();
  const input = byId("rename-visit-input");
  const label = input.value.trim();
  if (!label) {
    input.setCustomValidity("Enter a title for this visit.");
    input.reportValidity();
    return;
  }
  let saved = false;
  await work(async () => {
    const renamed = await api(visitPath(), json("PATCH", { label }));
    visit.label = renamed.label;
    visit.label_edited = renamed.label_edited;
    visits = visits.map(item => item.id === renamed.id ? renamed : item);
    byId("visit-title-text").textContent = renamed.label;
    renderVisitList();
    say("Visit renamed.");
    saved = true;
  });
  if (saved) closeVisitRename();
});

const isReturningPatient = () => document.querySelector('input[name="patient-mode"]:checked').value === "returning";
const allergyText = allergy => allergy.reaction ? allergy.allergen + " · " + allergy.reaction : allergy.allergen;

function updatePatientSelection() {
  const returning = isReturningPatient();
  byId("patient-mode").hidden = Boolean(visit);
  byId("new-patient-fields").hidden = returning || Boolean(visit);
  byId("returning-patient-fields").hidden = !returning || Boolean(visit);
  byId("capture-chart-inputs").hidden = returning && !visit;
  const chosen = patientCharts.find(chart => chart.id === byId("patient-chart").value);
  byId("returning-allergies").textContent = chosen
    ? (chosen.allergies.map(allergyText).join(", ") || "No allergies documented on this chart.")
    : "";
}

byId("patient-mode").addEventListener("change", () => {
  updatePatientSelection();
  if (!isReturningPatient()) return;
  work(async () => {
    patientCharts = await api("/api/patients");
    const select = byId("patient-chart");
    const placeholder = new Option(patientCharts.length ? "Choose a patient" : "No saved patient charts", "");
    select.replaceChildren(placeholder, ...patientCharts.map(chart =>
      new Option(chart.label + " · Last visit " + formatDate(chart.last_visit_at), chart.id)));
    updatePatientSelection();
  });
});
byId("patient-chart").addEventListener("change", updatePatientSelection);

function showVisit(data, reset = false, requestedPage) {
  const firstDraft = !visit?.draft && Boolean(data.draft);
  visit = data;
  const index = visits.findIndex((item) => item.id === visit.id);
  if (index >= 0) visits[index] = data;
  else visits.unshift(data);
  renderVisitList();
  dirty = false;
  if (reset) {
    byId("clinician").value = "";
    resolutions.clear();
    byId("text-on-approval").checked = false;
    byId("approval-phone").value = "";
    clearAudio();
  }
  updatePatientSelection();
  const language = visit.patient_language || visit.preferred_language || "en";
  byId("review-note-language").value = visit.note_language || "en";
  byId("review-patient-language").value = language;
  byId("detected-language").textContent = {
    en: "Detected: English", es: "Detected: Spanish",
    mixed: "Detected: Mixed",
  }[visit.detected_language] || "Recording language unavailable";
  if (visit.detected_language === "mixed") {
    byId("detected-language").textContent += language === "es"
      ? " → patient message in Spanish" : " → patient message in English";
  }
  byId("chart-allergies").value = (visit.allergies || []).map(allergyText).join("\n");
  byId("review-patient").textContent = visit.patient_name || visit.label;
  const allergies = visit.allergies || [];
  byId("saved-allergies").replaceChildren(...(allergies.length ? allergies.map(allergyText)
    : ["No allergies documented."]).map(text => create("li", "", text)));
  byId("chart-medications").value = (visit.current_medications || []).map(med =>
    [med.name, med.dosage, med.frequency].filter(Boolean).join(" ")).join("\n");
  byId("review-language").textContent = language === "es" ? "Español" : "English";
  const draft = visit.draft;
  if (!draft) {
    setScreen("capture", reset);
    return;
  }
  const approved = draft.status === "approved";
  if (reset || firstDraft || approved) {
    decisions = draft.staged_orders.map(() => approved ? "keep" : null);
    byId("approval-error").textContent = "";
  }
  for (const [field, input] of editors) {
    input.value = draft.soap_note[field];
    input.lang = visit.note_language || "en";
    input.readOnly = approved;
  }
  byId("summary").value = draft.visit_summary_for_patient;
  byId("summary").lang = language;
  byId("summary").readOnly = approved;
  byId("change-language").hidden = approved;
  byId("approval").hidden = approved;
  byId("review-locked").hidden = !approved;
  byId("save-allergy").hidden = approved;
  byId("edit-hint").hidden = approved;
  if (!approved && visit.verification) decisions.forEach((decision, index) => {
    if (decision === "drop") chooseOrder(index, "drop");
  });
  renderTranscript();
  renderOrders();
  renderVerification();
  if (approved) renderApproved();
  setScreen(approved && requestedPage !== "review" ? "approved" : "review",
    firstDraft && !reset);
  say("");
}

function openFindings(report) {
  return (report?.findings || []).filter(finding => finding.severity !== "info"
    && !findingDecision(finding));
}

function findingDecision(finding) {
  return resolutions.get(finding.id)
    || visit.verification?.resolutions?.find(item => item.finding_id === finding.id);
}

function setResolution(finding, action, note) {
  resolutions.set(finding.id, { finding_id: finding.id, action, note });
  if (action === "dropped") chooseOrder(finding.order_index, "drop");
  else if (action === "overridden") chooseOrder(finding.order_index, "keep");
  dirty = true;
  renderOrders();
  renderVerification();
  updateApproval();
}

function chooseOrder(index, decision) {
  if (index === null) return;
  decisions[index] = decision;
  for (const finding of visit.verification?.findings || []) {
    if (finding.order_index !== index || finding.area !== "order" || finding.severity === "info") continue;
    if (decision === "drop") {
      resolutions.set(finding.id, { finding_id: finding.id, action: "dropped" });
    } else if (resolutions.get(finding.id)?.action === "dropped") {
      resolutions.delete(finding.id);
    }
  }
  if (decision !== "drop") {
    for (const mark of visit.verification?.prescription_directions || []) {
      if (mark.order_index === index) resolutions.delete(mark.id);
    }
  }
}

function appendEvidence(container, quotes) {
  const quote = quotes[0];
  if (!quote) return;
  const timestamp = orderOffset({ transcript_timestamp_reference: quote });
  const source = makeButton("source-link", timestamp === undefined ? "View source"
    : "View source · " + formatOffset(timestamp), () => showQuote(quote, source));
  for (const evidence of quotes) container.append(create("blockquote", "muted", evidence));
  container.append(source);
}

function findingEntry(finding, approved) {
  const entry = create("details", "verification-check");
  const decision = findingDecision(finding);
  entry.dataset.state = decision ? "resolved" : finding.severity;
  entry.open = !decision && finding.status === "flagged";
  const area = finding.area === "order" && finding.order_index !== null ? "Order " + (finding.order_index + 1)
    : finding.area.replace("_", " ");
  const label = decision
    ? "✓ " + decision.action[0].toUpperCase() + decision.action.slice(1)
      + (decision.note ? " · " + decision.note : "") + " · "
    : finding.kind === "language" ? "⚑ Language · " : finding.status === "supported" ? "✓ " : "⚑ ";
  const timestamp = finding.evidence.length
    ? orderOffset({ transcript_timestamp_reference: finding.evidence[0] }) : undefined;
  const summary = create("summary", "", label + area + " · " + finding.text
    + (timestamp === undefined ? "" : " · " + formatOffset(timestamp)));
  entry.append(summary);
  if (finding.chart_evidence) entry.append(create("p", "muted", "Chart · " + finding.chart_evidence));
  if (finding.question) entry.append(create("p", "", finding.question));
  const quoted = finding.evidence.length ? finding.evidence : (!approved && finding.area === "order"
    && finding.order_index !== null
    ? [visit.draft.staged_orders[finding.order_index]?.transcript_timestamp_reference].filter(Boolean)
    : []);
  appendEvidence(entry, quoted);
  if (decision) {
    if (decision.reviewer) entry.append(create("p", "resolution-status",
      decision.reviewer + " · " + formatDate(decision.resolved_at)));
    if (!approved) summary.append(makeButton("resolution-undo", "Undo", event => {
      event.preventDefault();
      event.stopPropagation();
      resolutions.delete(finding.id);
      if (finding.area === "order" && finding.order_index !== null
        && (finding.severity === "blocking" || decision.action === "dropped")) {
        decisions[finding.order_index] = null;
        for (const other of visit.verification.findings) {
          if (other.order_index === finding.order_index
            && resolutions.get(other.id)?.action === "dropped") {
            resolutions.delete(other.id);
          }
        }
      }
      renderOrders();
      renderVerification();
      updateApproval();
    }));
  } else if (!approved && finding.kind === "language" && finding.order_index === null) {
    entry.append(create("p", "muted", "Edit this section and save & verify again."));
  } else if (!approved && finding.severity !== "info") {
    const blocking = finding.severity === "blocking";
    const actions = create("div", "actions");
    if (blocking && finding.order_index !== null) {
      actions.append(makeButton("", "Drop order", () => setResolution(finding, "dropped")));
    }
    if (!blocking) {
      actions.append(makeButton("", "Not needed", () => setResolution(finding, "dismissed")));
      if (finding.question === "Can you confirm the prescription's duration?") {
        actions.append(makeButton("", "Mark as ongoing", () =>
          setResolution(finding, "answered", "Clinician confirmed ongoing treatment.")));
      }
    }
    const form = create("div", "resolution-form");
    form.hidden = true;
    const reason = create("textarea");
    reason.rows = 2;
    reason.maxLength = 2000;
    reason.setAttribute("aria-label", blocking ? "Override reason" : "Answer");
    const confirm = makeButton("primary", blocking ? "Keep with override" : "Save answer", () =>
      setResolution(finding, blocking ? "overridden" : "answered", reason.value.trim()));
    confirm.disabled = true;
    reason.addEventListener("input", () => { confirm.disabled = reason.value.trim().length < (blocking ? 8 : 1); });
    actions.append(makeButton("", blocking ? "Override with reason" : "Answer", () => {
      form.hidden = !form.hidden;
      if (!form.hidden) reason.focus();
    }));
    form.append(reason, confirm);
    entry.append(actions, form);
  }
  return entry;
}

function renderVerification() {
  const report = currentVerification();
  const approved = visit?.draft?.status === "approved";
  const trace = byId("verification-trace");
  const findings = byId("verification-findings");
  const counts = byId("verification-counts");
  const status = byId("verification-status");
  const metrics = byId("verification-metrics");
  const button = byId("run-verification");
  button.hidden = !visit?.draft || verifying || approved;
  button.textContent = report ? "Re-run" : "Save & verify";
  for (const node of [trace, findings, counts]) node.replaceChildren();
  for (const marker of document.querySelectorAll(".soap-verification")) marker.remove();
  metrics.textContent = "";
  byId("verification-steps").open = verifying;
  const reconciliation = prescriptionReconciliation(report);
  for (const direction of reconciliation.marks) {
    const editor = direction.area === "plan" ? editors.get("plan") : byId("summary");
    const dismissed = findingDecision(direction);
    const warning = create("div", "soap-verification",
      (dismissed ? "Dismissed as not a direction · " : "⚑ Dropped prescription · ") + direction.text);
    if (!dismissed) warning.dataset.state = "blocking";
    warning.append(makeButton("resolution-undo", dismissed ? "Undo" : "Not a direction", () => {
      if (dismissed) {
        resolutions.delete(direction.id);
        renderVerification();
        updateApproval();
      } else {
        setResolution(direction, "not_a_direction");
      }
    }));
    editor.after(warning);
  }
  if (!report || verifying) {
    status.textContent = verifying ? "Verifying…"
      : approved ? "No verification is saved for this visit." : "This draft needs verification.";
    return;
  }
  const supported = report.findings.filter(item => item.status === "supported");
  const resolved = report.findings.filter(item => item.severity !== "info" && findingDecision(item));
  const open = approved ? [] : openFindings(report);
  status.textContent = open.length
    ? open.length + (open.length === 1 ? " item needs your decision" : " items need your decision")
    : reconciliation.directions.length || reconciliation.recheck || reconciliation.keptWithheld
      ? "Review instructions for dropped prescriptions"
      : approved ? "Verification and decisions saved" : "Verified · ready to approve";
  metrics.textContent = report.findings.filter(item => item.kind !== "medication_safety").length
    + " lines and orders checked · " + report.prescription_count + " prescriptions"
    + (report.elapsed_seconds === null ? "" : " · " + report.elapsed_seconds + " s");
  const corrections = report.corrections || [];
  for (const [severity, amount, text] of [
    ["blocking", open.filter(item => item.severity === "blocking").length
      + Number(reconciliation.directions.length > 0 || reconciliation.recheck || reconciliation.keptWithheld), "blocking"],
    ["question", open.filter(item => item.severity === "question").length, "questions"],
    ["info", corrections.length, "fixed"], ["info", supported.length, "supported"],
  ]) {
    const chip = create("span", "badge verification-chip", amount + " " + text);
    chip.dataset.state = severity;
    counts.append(chip);
  }
  if (open.length) findings.append(create("h3", "verification-group", "Needs your decision"));
  for (const finding of [...open].sort((a, b) => Number(b.severity === "blocking")
    - Number(a.severity === "blocking"))) findings.append(findingEntry(finding, approved));
  const dismissedMarks = approved ? (report.prescription_directions || []).filter(item => findingDecision(item)) : [];
  if (resolved.length || dismissedMarks.length) findings.append(create("h3", "verification-group", "Resolved"));
  for (const finding of resolved) findings.append(findingEntry(finding, approved));
  for (const mark of dismissedMarks) {
    const decision = findingDecision(mark);
    findings.append(create("p", "resolution-status",
      "Not a direction · " + mark.text + " · " + decision.reviewer));
  }
  if (corrections.length) findings.append(create("h3", "verification-group", "Fixed automatically"));
  for (const correction of corrections) {
    const entry = create("details", "verification-check");
    entry.append(create("summary", "", "✓ Corrected · " + correction.area.replace("_", " ")));
    entry.append(create("p", "muted", "Before · " + correction.before));
    entry.append(create("p", "", "After · " + correction.after));
    const evidence = supported.find(item => item.area === correction.area && item.text === correction.after);
    if (evidence) appendEvidence(entry, evidence.evidence);
    if (!approved) entry.append(makeButton("", "Undo fix", () => work(async () => {
      showVisit(await api(visitPath("/verification/undo"), json("POST", {
        revision: visit.draft.updated_at, correction_id: correction.id,
      })));
      say("Original wording restored. The question is open.");
    })));
    findings.append(entry);
  }
  const supportedGroup = create("details", "verification-group");
  supportedGroup.append(create("summary", "", "✓ " + supported.length + " supported checks"));
  for (const item of supported) supportedGroup.append(findingEntry(item, approved));
  findings.append(supportedGroup);
  byId("verification-steps-label").textContent = "How the verifier checked · " + report.trace.length + " steps";
  trace.append(...report.trace.map(step => create("li", "", step)));
  for (const [field, editor] of editors) {
    if (reconciliation.directions.some(item => item.area === field)) continue;
    const sectionBlocking = open.filter(item => item.severity === "blocking"
      && (item.area === field || (field === "plan" && item.area === "order"))).length;
    const sectionQuestions = open.filter(item => item.area === field
      && item.severity === "question").length;
    const fixes = corrections.filter(item => item.area === field);
    const marker = create("div", "soap-verification",
      sectionBlocking ? "⚑ " + sectionBlocking + " blocking"
        : sectionQuestions ? "⚑ " + sectionQuestions + " open"
          : report.findings.some(item => item.area === field && item.status === "supported") ? "✓ Supported" : "");
    if (sectionBlocking) marker.dataset.state = "blocking";
    else if (sectionQuestions) marker.dataset.state = "question";
    for (const fix of fixes) marker.append(create("p", "muted", "Fixed · " + fix.after));
    editor.after(marker);
  }
}

const sentenceSegmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });

function transcriptLines() {
  const timeline = visit.transcript_segments;
  if (!timeline?.length) return sentenceSegmenter.segment(visit.transcript || "");
  let offset = 0;
  return timeline.map(({ text, start, language }) => {
    const line = { segment: text, index: offset, start: start ?? undefined, language };
    offset += text.length + 1; // The saved transcript joins audio segments with a newline.
    return line;
  });
}

function formatOffset(seconds) {
  const time = Math.max(0, Math.floor(seconds));
  const minutes = String(Math.floor(time / 60)).padStart(2, "0");
  return minutes + ":" + String(time % 60).padStart(2, "0");
}

function renderTranscript(match) {
  const container = byId("transcript");
  container.replaceChildren();
  byId("transcript-timing").hidden = Boolean(visit.transcript_segments?.some(segment => segment.start !== null));
  for (const { segment, index, start: timestamp, language } of transcriptLines()) {
    const line = create("p", "transcript-line");
    const words = create("span", "transcript-words", segment);
    if (language) words.lang = language;
    if (timestamp !== undefined) {
      line.append(create("span", "transcript-time", formatOffset(timestamp)));
    }
    line.append(words);
    if (match) {
      const start = Math.max(0, match.index - index);
      const end = Math.min(segment.length, match.index + match[0].length - index);
      if (start < end) {
        line.classList.add("source-active");
        words.replaceChildren(segment.slice(0, start), create("mark", "", segment.slice(start, end)), segment.slice(end));
      }
    }
    container.append(line);
  }
}

function matchQuote(quote) {
  const tokens = quote.match(/[\p{L}\p{M}\p{N}_]+/gu);
  const boundary = "[^\\p{L}\\p{M}\\p{N}_]";
  const query = tokens && new RegExp("(?<![\\p{L}\\p{M}\\p{N}_])" + tokens.join(boundary + "+") + "(?![\\p{L}\\p{M}\\p{N}_])", "iu");
  return query?.exec(visit.transcript || "");
}

function orderOffset(order) {
  const match = matchQuote(order.transcript_timestamp_reference);
  if (!match) return undefined;
  for (const { segment, index, start } of transcriptLines()) {
    if (match.index >= index && match.index < index + segment.length) return start;
  }
  return undefined;
}

function showQuote(quote, button) {
  const match = matchQuote(quote);
  renderTranscript(match);
  for (const source of byId("orders").querySelectorAll(".source-link")) {
    source.setAttribute("aria-pressed", String(Boolean(match) && source === button));
  }
  const active = byId("transcript").querySelector("mark");
  if (active) {
    active.scrollIntoView({ block: "nearest" });
    say("Supporting quote highlighted in the transcript.");
  } else {
    say("The supporting quote could not be located in the transcript.", true);
  }
}

byId("copy-transcript").addEventListener("click", (event) => copyText(visit.transcript, event.currentTarget));

function renderOrders() {
  const list = byId("orders");
  list.replaceChildren();
  if (!decisions.length) list.append(create("p", "muted", "No proposed orders."));
  const approved = visit.draft.status === "approved";
  visit.draft.staged_orders.forEach((order, index) => {
    const card = create("article", "order-card");
    const heading = create("div", "order-heading");
    const state = create("span", "badge");
    heading.append(create("h3", "", order.description), state);
    const blocking = currentVerification()?.findings.find(item => item.order_index === index
      && item.severity === "blocking");
    if (blocking && decisions[index] !== "drop") {
      heading.append(create("span", "badge verification-blocking", blocking.chart_evidence || "Blocking"));
    }

    const actions = create("div", "order-actions");
    const offset = orderOffset(order);
    const label = offset === undefined ? "View source" : "View source · " + formatOffset(offset);
    const source = makeButton("source-link", label, () => showQuote(order.transcript_timestamp_reference, source));
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    icon.setAttribute("class", "icon");
    icon.setAttribute("aria-hidden", "true");
    use.setAttribute("href", "#icon-eye");
    icon.append(use);
    source.prepend(icon);
    source.setAttribute("aria-controls", "transcript");
    source.setAttribute("aria-pressed", "false");
    actions.append(source);
    const drop = makeButton("", "Drop", () => choose("drop"));
    const keep = makeButton("", "Keep", () => choose("keep"));
    if (!approved) actions.append(drop, keep);

    function paint() {
      const decision = decisions[index];
      state.textContent = decision === "keep" ? "Keep" : decision === "drop" ? "Dropped" : "Needs decision";
      state.dataset.state = decision || "documentation_pending";
      card.dataset.decision = decision || "pending";
      drop.setAttribute("aria-pressed", String(decision === "drop"));
      keep.setAttribute("aria-pressed", String(decision === "keep"));
      updateApproval();
    }
    function choose(decision) {
      chooseOrder(index, decision);
      dirty = true;
      renderOrders();
      renderVerification();
      updateApproval();
    }

    card.addEventListener("keydown", (event) => {
      if (event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) return;
      const key = event.key.toLowerCase();
      if (key === "s") source.click();
      else if (!approved && key === "k") keep.click();
      else if (!approved && key === "d") drop.click();
      else return;
      event.preventDefault();
    });
    card.append(heading, create("p", "muted", order.icd10_rationale), actions);
    list.append(card);
    paint();
  });
}

function renderApproved() {
  const { draft } = visit;
  byId("approved-by").textContent = "Approved by " + visit.approved_by;
  byId("approved-at").textContent = formatDate(visit.approved_at) + " · Approved version saved";
  const note = byId("approved-soap");
  note.replaceChildren();
  for (const field of fields) {
    const name = field[0].toUpperCase() + field.slice(1);
    const row = create("div", "soap-row");
    const tile = create("span", "soap-tile", name[0]);
    tile.setAttribute("aria-hidden", "true");
    const content = create("div", "soap-content");
    content.append(create("h3", "field-label", name), create("p", "approved-field", draft.soap_note[field]));
    const copy = makeButton("copy-section", "Copy", () => copyText(draft.soap_note[field], copy));
    copy.setAttribute("aria-label", "Copy " + name);
    row.append(tile, content, copy);
    note.append(row);
  }

  const list = byId("kept-orders");
  list.replaceChildren();
  byId("kept-count").textContent = draft.staged_orders.length + " kept";
  if (!draft.staged_orders.length) list.append(create("p", "muted", "No orders to place."));
  for (const order of draft.staged_orders) {
    const card = create("article", "kept-order");
    card.append(create("h3", "", order.description), create("p", "muted", order.icd10_rationale));
    const offset = orderOffset(order);
    const source = create("span", "kept-source", offset === undefined
      ? "Source · timing unavailable" : "Source · " + formatOffset(offset));
    if (offset !== undefined) source.dataset.timed = "true";
    card.append(source, create("blockquote", "muted", order.transcript_timestamp_reference));
    list.append(card);
  }

  const messages = byId("sms-preview");
  const language = visit.patient_language || visit.preferred_language || "en";
  messages.lang = language;
  byId("care-team").lang = language;
  byId("care-team").textContent = language === "es" ? "Su equipo de atención" : "Your care team";
  messages.replaceChildren(...draft.visit_summary_for_patient.split(/\n\s*\n/).filter(Boolean)
    .map((paragraph) => create("p", "message", paragraph)));
  byId("sms-phone").value = "";
  byId("sms-error").textContent = "";
  byId("sms-result").textContent = "";
  const textStatus = sessionStorage.getItem(textStatusKey(visit.id));
  if (textStatus === "submitted") byId("sms-result").textContent = "Submitted to Twilio.";
  else if (textStatus) byId("sms-error").textContent = "Visit approved. Text could not be confirmed. " + textStatus;
}

byId("copy-note").addEventListener("click", (event) => {
  const note = fields.map((field) => field[0].toUpperCase() + field.slice(1) + "\n" + visit.draft.soap_note[field]);
  copyText(note.join("\n\n"), event.currentTarget);
});
byId("copy-summary").addEventListener("click", (event) => {
  copyText(visit.draft.visit_summary_for_patient, event.currentTarget);
});

function clearAudio() {
  selection = null;
  byId("file").value = "";
  byId("elapsed").textContent = "00:00";
  byId("audio-name").textContent = audioLabel();
}

function selectAudio(file) {
  if (!file || busy || recorder) return;
  const extension = file.name.split(".").at(-1).toLowerCase();
  if (!audioFormats.has(extension) || !file.size || file.size > maxAudioBytes) {
    byId("file").value = "";
    say("Choose a supported audio file up to " + audioLimit() + ".", true);
    return;
  }
  selection = file;
  byId("elapsed").textContent = "00:00";
  byId("audio-name").textContent = file.name;
  say("Audio ready. Select Transcribe and draft.");
  updateControls();
}

byId("file").addEventListener("change", () => selectAudio(byId("file").files[0]));
const dropZone = byId("drop-zone");
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (!busy) dropZone.dataset.dragging = "true";
});
dropZone.addEventListener("dragleave", () => delete dropZone.dataset.dragging);
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  delete dropZone.dataset.dragging;
  selectAudio(event.dataTransfer.files[0]);
});
byId("discard").addEventListener("click", () => {
  clearAudio();
  say("");
  updateControls();
  byId("record").focus();
});

byId("record").addEventListener("click", () => work(async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  try {
    const type = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"]
      .find((candidate) => MediaRecorder.isTypeSupported(candidate));
    if (!type) throw new Error("This browser cannot record a supported format. Choose an audio file.");
    const active = new MediaRecorder(stream, { mimeType: type });
    const chunks = [];
    let failed = false;
    let size = 0;
    active.addEventListener("dataavailable", ({ data }) => {
      size += data.size;
      if (size > maxAudioBytes) {
        if (active.state === "recording") active.stop();
      } else if (data.size) {
        chunks.push(data);
      }
    });
    active.addEventListener("error", () => { failed = true; });
    active.addEventListener("stop", () => {
      stream.getTracks().forEach((track) => track.stop());
      clearInterval(recordingTimer);
      recorder = null;
      if (size > maxAudioBytes) {
        say("The recording exceeded " + audioLimit() + ". Choose a shorter recording.", true);
      } else if (!failed && chunks.length) {
        const extension = { "audio/webm": "webm", "audio/mp4": "m4a", "audio/ogg": "ogg" }[type.split(";")[0]];
        selection = new File(chunks, "Microphone recording." + extension, { type });
        byId("file").value = "";
        byId("audio-name").textContent = selection.name;
        say("Recording ready. Select Transcribe and draft.");
      } else {
        say("The microphone recording could not be saved.", true);
      }
      updateControls();
    });
    active.start(1000);
    recorder = active;
    const started = performance.now();
    byId("elapsed").textContent = "00:00";
    recordingTimer = setInterval(() => {
      const seconds = Math.floor((performance.now() - started) / 1000);
      const minutes = String(Math.floor(seconds / 60)).padStart(2, "0");
      byId("elapsed").textContent = minutes + ":" + String(seconds % 60).padStart(2, "0");
    }, 250);
    say("Recording microphone audio…");
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    throw error;
  }
}));
byId("stop").addEventListener("click", () => {
  if (recorder?.state === "recording") recorder.stop();
});

function setProgress(step, message) {
  processingStep = step;
  say(message);
  updateControls();
}

async function submitRecording(file, extension) {
  const headers = { "X-Audio-Format": extension, "Content-Type": "application/octet-stream" };
  const path = visitPath("/recording");
  if (audioChunkBytes && file.size > audioChunkBytes) {
    const uploadId = crypto.randomUUID();
    const parts = Math.ceil(file.size / audioChunkBytes);
    const query = "upload_id=" + uploadId;
    for (let part = 0; part < parts; part++) {
      setProgress(0, `Uploading recording (${part + 1} of ${parts})…`);
      const offset = part * audioChunkBytes;
      await api(`${path}/chunks/${part}?${query}&total_bytes=${file.size}`, {
        method: "PUT", headers, body: file.slice(offset, offset + audioChunkBytes),
      });
    }
    setProgress(0, "Transcribing the recording…");
    return api(`${path}/chunks/complete?${query}`, { method: "POST", headers });
  }
  setProgress(0, "Transcribing the recording…");
  return api(path, { method: "PUT", headers, body: file });
}

async function generateDraft() {
  setProgress(2, "Drafting SOAP note, patient instructions, and orders…");
  showVisit(await api(visitPath("/draft"), { method: "POST" }));
  await verifyCurrentDraft(true);
}

async function verifyCurrentDraft(repair = false) {
  verifying = true;
  renderVerification();
  try {
    const options = json("POST", {
      revision: visit.draft.updated_at,
      repair,
      dropped_prescriptions: droppedPrescriptions(),
    });
    const response = await fetch(visitPath("/verify"), {
      ...options,
      headers: {
        ...options.headers,
        "X-Escriba-Workspace": "1",
        Accept: "application/x-ndjson",
      },
    });
    if (!response.ok || !response.body) throw new Error("Verification could not start.");
    let buffer = "";
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let checked;
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          const event = JSON.parse(line);
          if (event.error) throw new Error(event.error);
          if (event.step) byId("verification-trace").append(create("li", "", event.step));
          if (event.visit) checked = event.visit;
        }
      }
    } finally {
      await reader.cancel();
    }
    if (!checked) throw new Error("Verification did not finish.");
    const checkedMarks = new Set(checked.verification?.prescription_directions?.map(item => item.id));
    for (const [id, decision] of resolutions) {
      if (decision.action !== "not_a_direction" || !checkedMarks.has(id)) resolutions.delete(id);
    }
    showVisit(checked);
  } finally {
    verifying = false;
    renderVerification();
    updateApproval();
  }
}

const chartLines = (id) => byId(id).value.split("\n").map(line => line.trim()).filter(Boolean);

byId("transcribe").addEventListener("click", () => work(async () => {
  if (!selection || recorder) return;
  if (!visit) {
    const patientId = byId("patient-chart").value;
    const patientName = byId("patient-name").value.trim();
    if (isReturningPatient() && !patientId) {
      byId("patient-chart").focus();
      throw new Error("Choose the returning patient’s saved chart.");
    }
    visit = await api("/api/visits", json("POST", {
      time_zone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      ...(isReturningPatient() ? { patient_id: patientId } : {
        patient_name: patientName || null,
        allergies: chartLines("chart-allergies").map(allergen => ({ allergen })),
        current_medications: chartLines("chart-medications").map(name => ({ name })),
      }),
    }));
    updatePatientSelection();
    byId("capture-title").textContent = visit.label;
    history.replaceState(null, "", pagePath(visit.id, "capture"));
    visits.unshift(visit);
    renderVisitList();
  }
  const extension = selection.name.split(".").at(-1).toLowerCase();
  const data = await submitRecording(selection, extension);
  clearAudio();
  showVisit(data);
  await generateDraft();
}));
byId("generate").addEventListener("click", () => work(generateDraft));

function validateReview() {
  for (const input of [...editors.values(), byId("summary")]) {
    if (input.value.trim()) continue;
    input.focus();
    say("Complete the empty field before saving or approving.", true);
    return false;
  }
  return true;
}

async function saveReview(approve, reverify = true) {
  const kept = decisions.flatMap((decision, index) => decision === "keep" ? [index] : []);
  const body = {
    revision: visit.draft.updated_at,
    soap_note: Object.fromEntries(fields.map((field) => [field, editors.get(field).value])),
    visit_summary_for_patient: byId("summary").value,
    clinician: approve ? byId("clinician").value.trim() : null,
    kept_orders: kept,
    resolutions: approve ? [...resolutions.values()].filter(resolution =>
      resolution.action !== "not_a_direction" || prescriptionReconciliation(currentVerification())
        .marks.some(mark => mark.id === resolution.finding_id)) : [],
    text_on_approval_to: approve && byId("text-on-approval").checked
      ? normalizedPhone("approval-phone") : null,
  };
  const updated = await api(visitPath("/review"), json("PUT", body));
  if (approve) {
    if (updated.summary_text_sent || updated.summary_text_error) {
      sessionStorage.setItem(textStatusKey(updated.id), updated.summary_text_sent
        ? "submitted" : updated.summary_text_error);
    }
    dirty = false;
    decisions = [];
    busy = false;
    location.assign(pagePath(updated.id, "approved"));
  } else {
    showVisit(updated);
    if (reverify) {
      await verifyCurrentDraft();
      say("Draft saved and verified.");
    }
  }
}

byId("review").addEventListener("input", event => {
  if (event.target.closest("#save-allergy")) return;
  dirty = true;
  if (event.target.id !== "summary" && !fields.includes(event.target.id)) return;
  renderVerification();
  updateApproval();
});
byId("save-allergy").addEventListener("submit", event => {
  event.preventDefault();
  const allergen = byId("allergen-name").value.trim();
  const reaction = byId("allergy-reaction").value.trim() || null;
  if (!allergen || !validateReview()) return;
  work(async () => {
    if (!matchesSavedDraft()) await saveReview(false, false);
    const updated = await api(visitPath("/allergies"), json("POST", { allergen, reaction }));
    byId("save-allergy").reset();
    resolutions.clear();
    showVisit(updated);
    if (!updated.verification) await verifyCurrentDraft();
    say("Allergy saved to the patient chart.");
  });
});
byId("clinician").addEventListener("input", updateApproval);
byId("text-on-approval").addEventListener("change", updateApproval);
byId("approval-phone").addEventListener("input", updateApproval);
byId("save").addEventListener("click", () => {
  if (validateReview()) work(() => saveReview(false), byId("approval-error"));
});
byId("change-language").addEventListener("click", event => {
  const controls = byId("language-controls");
  controls.hidden = !controls.hidden;
  event.currentTarget.setAttribute("aria-expanded", String(!controls.hidden));
});
byId("apply-language").addEventListener("click", () => {
  if (!validateReview()) return;
  const noteLanguage = byId("review-note-language").value;
  const patientLanguage = byId("review-patient-language").value;
  work(async () => {
    if (!matchesSavedDraft()) await saveReview(false, false);
    const updated = await api(visitPath("/language"), json("PUT", {
      revision: visit.draft.updated_at,
      note_language: noteLanguage,
      patient_language: patientLanguage,
    }));
    resolutions.clear();
    showVisit(updated);
    byId("language-controls").hidden = true;
    byId("change-language").setAttribute("aria-expanded", "false");
    say("Output languages updated and the draft verified.");
  });
});
byId("run-verification").addEventListener("click", () => {
  if (!validateReview()) return;
  work(() => matchesSavedDraft() ? verifyCurrentDraft() : saveReview(false));
});
byId("approval").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!byId("approve").disabled && validateReview()) work(() => saveReview(true), byId("approval-error"));
});
document.addEventListener("keydown", (event) => {
  if (screen !== "review" || event.key !== "Enter" || !(event.metaKey || event.ctrlKey)) return;
  if (byId("approval").hidden || byId("approve").disabled) return;
  event.preventDefault();
  byId("approval").requestSubmit();
});
window.addEventListener("beforeunload", (event) => {
  if (hasUnsavedWork() || recorder || busy) event.preventDefault();
});
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  if (visit) location.reload();
  else say("");
});

byId("sms-phone").addEventListener("input", () => {
  byId("sms-error").textContent = "";
  byId("sms-result").textContent = "";
  updateControls();
});
byId("sms-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!validPhone() || visit?.draft?.status !== "approved") return;
  const phone = normalizedPhone();
  work(async () => {
    const button = byId("send-sms");
    byId("sms-result").textContent = "";
    button.textContent = "Sending…";
    byId("sms-phone").readOnly = true;
    try {
      await api(visitPath("/text-summary"), json("POST", { phone_number: phone }));
      sessionStorage.setItem(textStatusKey(visit.id), "submitted");
      byId("sms-phone").value = "";
      byId("sms-result").textContent = "Submitted to Twilio · ••• " + phone.slice(-4);
    } finally {
      button.textContent = "Send text";
      byId("sms-phone").readOnly = false;
    }
  }, byId("sms-error"));
});

work(async () => {
  const config = await api("/api/visits/config");
  maxAudioBytes = config.max_audio_bytes;
  audioChunkBytes = config.audio_chunk_bytes;
  byId("audio-name").textContent = audioLabel();
  byId("demo-notice").hidden = !config.hosted_demo;
  byId("sms-form").hidden = config.hosted_demo;
  byId("text-on-approval-option").hidden = config.hosted_demo;
  const path = /^\/visits\/([^/]+)\/(capture|review|approved)$/.exec(location.pathname);
  if (path) showVisit(await api("/api/visits/" + path[1]), true, path[2]);
  visits = await api("/api/visits");
  renderVisitList();
  if (visit?.draft?.status === "review_pending" && !visit.verification) await verifyCurrentDraft();
});
