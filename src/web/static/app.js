/* delta-chat web UI.
 *
 * Nothing here computes anything about the drawings. Every delta, box,
 * severity and colour arrives from /api/jobs/<id>/payload, which is
 * `payload.build_payload` -- the same function that produces the JSON
 * embedded in the downloadable report.html. This file only decides how to
 * show it.
 *
 * Two things worth knowing before editing:
 *
 * 1. Delta boxes are normalized [0,1] against the page (see
 *    src/canonical/model.py), so they are laid out as plain CSS percentages
 *    inside a container that sits exactly over the canvas. That is why zoom
 *    and devicePixelRatio never touch box coordinates: pdf.js changes how
 *    many pixels fill the canvas, not how wide the canvas is on screen.
 *
 * 2. A citation chip resolves through the server's CitationResolver, never
 *    by parsing ids here. If `resolved` is null the chip is rendered dead
 *    rather than guessing a location -- a chip that highlights the wrong
 *    valve is worse than no chip.
 */
import * as pdfjsLib from "/static/vendor/pdfjs/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.min.mjs";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const SEVERITY_ORDER = { critical: 3, high: 2, medium: 1, low: 0 };
const ZOOM_STEPS = [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8];
// Browsers refuse canvases past roughly this dimension, and a 5100px-wide
// D-size sheet at 8x zoom would sail past it. Beyond the cap the page still
// scales up; it just stops gaining detail.
const MAX_CANVAS_PX = 8192;

const state = {
  health: null,
  jobId: null,
  payload: null,
  files: { a: null, b: null },
  sheet: null,
  zoom: 1,
  selected: null,
  filters: { search: "", kinds: new Set(), cascade: false },
  panes: { a: null, b: null },
  syncing: false,
  pollTimer: null,
  elapsedTimer: null,
};

/* ══════════════════════════════════════════════════════════ utilities ══ */

const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON body */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

function toast(message, ms = 2200) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

function showScreen(name) {
  for (const id of ["landing", "progress", "refused", "errored", "workspace"]) {
    $("#" + id).hidden = id !== name;
  }
}

function banner(html, kind = "warn") {
  const el = document.createElement("div");
  el.className = `banner ${kind}`;
  el.innerHTML = `<span>${html}</span>
    <button class="close" title="Dismiss">×</button>`;
  $(".close", el).onclick = () => el.remove();
  $("#banners").append(el);
}

/* ═══════════════════════════════════════════════════════════ landing ══ */

function setupLanding() {
  for (const zone of $$(".dropzone")) {
    const side = zone.dataset.side;
    const input = $("input", zone);

    input.onchange = () => input.files[0] && pickFile(side, input.files[0]);
    zone.ondragover = (e) => { e.preventDefault(); zone.classList.add("dragover"); };
    zone.ondragleave = () => zone.classList.remove("dragover");
    zone.ondrop = (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      if (file) pickFile(side, file);
    };
  }

  $("#btn-analyse").onclick = submitUpload;
  $("#btn-new").onclick = resetToLanding;
  $("#btn-refusal-back").onclick = resetToLanding;
  $("#btn-error-back").onclick = resetToLanding;
  $("#btn-force").onclick = forceJob;
}

function pickFile(side, file) {
  state.files[side] = file;
  const zone = $(`.dropzone[data-side="${side}"]`);
  zone.classList.add("filled");
  $(".dz-file", zone).textContent = file.name;
  $("#upload-error").hidden = true;
  $("#btn-analyse").disabled = !(state.files.a && state.files.b);
}

async function submitUpload() {
  const body = new FormData();
  body.append("a", state.files.a);
  body.append("b", state.files.b);
  $("#btn-analyse").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body });
    startJob(job);
  } catch (e) {
    $("#upload-error").textContent = e.message;
    $("#upload-error").hidden = false;
    $("#btn-analyse").disabled = false;
  }
}

function renderSamples() {
  const samples = state.health.samples ?? [];
  if (!samples.length) return;
  $("#samples").hidden = false;
  $("#samples-row").innerHTML = "";
  for (const s of samples) {
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = s.label;
    btn.onclick = async () => {
      try {
        startJob(await api(`/api/jobs/sample/${s.key}`, { method: "POST" }));
      } catch (e) { toast(e.message); }
    };
    $("#samples-row").append(btn);
  }
}

function resetToLanding() {
  clearInterval(state.pollTimer);
  clearInterval(state.elapsedTimer);
  state.jobId = null;
  state.payload = null;
  state.files = { a: null, b: null };
  state.panes = { a: null, b: null };
  state.selected = null;
  $("#banners").innerHTML = "";
  $("#stats").hidden = true;
  $("#btn-new").hidden = true;
  // The workspace DOM is reused across comparisons, so anything not rebuilt
  // by openWorkspace has to be cleared here or it bleeds into the next pair.
  $("#chatlog").innerHTML = "";
  $("#jsonview").textContent = "";
  $("#jsonview")._raw = "";
  $("#search").value = "";
  $("#show-cascade").checked = false;
  state.filters = { search: "", kinds: new Set(), cascade: false };
  state.zoom = 1;
  $("#zoomlevel").textContent = "100%";
  $("#pair-title").textContent = "Compare two revisions of a P&ID";
  $("#pair-subtitle").textContent = "Select revision A and revision B, then analyse.";
  for (const zone of $$(".dropzone")) {
    zone.classList.remove("filled");
    $(".dz-file", zone).textContent = "";
    $("input", zone).value = "";
  }
  $("#btn-analyse").disabled = true;
  $("#upload-error").hidden = true;
  showScreen("landing");
}

/* ══════════════════════════════════════════════════════ job progress ══ */

function startJob(job) {
  state.jobId = job.job_id;
  $("#pair-title").textContent = `${job.label_a} → ${job.label_b}`;
  $("#pair-subtitle").textContent = "Comparing…";
  renderStages(job);
  showScreen("progress");

  clearInterval(state.elapsedTimer);
  const started = Date.now();
  state.elapsedTimer = setInterval(() => {
    $("#elapsed").textContent = `${((Date.now() - started) / 1000).toFixed(1)}s elapsed`;
  }, 100);

  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(poll, 350);
  poll();
}

function renderStages(job) {
  const doneIdx = job.stage ? job.stages.findIndex((s) => s.key === job.stage) : -1;
  $("#stagelist").innerHTML = job.stages.map((s, i) => {
    const cls = i < doneIdx ? "done" : i === doneIdx ? "active" : "";
    const mark = i < doneIdx ? "✓" : "";
    return `<li class="${cls}"><span class="marker">${mark}</span>${escapeHtml(s.label)}</li>`;
  }).join("");
}

async function poll() {
  let job;
  try {
    job = await api(`/api/jobs/${state.jobId}`);
  } catch (e) {
    clearInterval(state.pollTimer);
    return showError(e.message);
  }
  renderStages(job);

  if (job.status === "done") {
    clearInterval(state.pollTimer);
    clearInterval(state.elapsedTimer);
    openWorkspace(job);
  } else if (job.status === "refused") {
    clearInterval(state.pollTimer);
    clearInterval(state.elapsedTimer);
    showRefusal(job);
  } else if (job.status === "error") {
    clearInterval(state.pollTimer);
    clearInterval(state.elapsedTimer);
    showError(job.error);
  }
}

function showRefusal(job) {
  const p = job.precheck ?? {};
  $("#refusal-reason").textContent = p.reason ?? "The two files could not be matched.";
  $("#rf-drawno-a").textContent = p.drawing_no_a ?? "— none found —";
  $("#rf-drawno-b").textContent = p.drawing_no_b ?? "— none found —";
  $("#rf-equip-a").textContent = p.equipment_a ?? "— none found —";
  $("#rf-equip-b").textContent = p.equipment_b ?? "— none found —";
  showScreen("refused");
}

function showError(message) {
  $("#error-message").textContent = message ?? "unknown error";
  showScreen("errored");
}

async function forceJob() {
  try {
    startJob(await api(`/api/jobs/${state.jobId}/force`, { method: "POST" }));
  } catch (e) { toast(e.message); }
}

/* ═════════════════════════════════════════════════════════ workspace ══ */

async function openWorkspace(job) {
  state.payload = await api(`/api/jobs/${state.jobId}/payload`);
  const p = state.payload;

  $("#pair-title").textContent = `${p.pid_a} → ${p.pid_b}`;
  $("#pair-subtitle").textContent =
    `${p.summary.n_primary} change${p.summary.n_primary === 1 ? "" : "s"} found in ${(p.elapsed_ms / 1000).toFixed(1)}s`;
  $("#label-a").textContent = p.pid_a;
  $("#label-b").textContent = p.pid_b;
  $("#btn-new").hidden = false;

  renderStats(p.summary);
  renderPrecheckBanners(p.precheck, job.forced);
  // Set before the first renderRows so rows do not briefly label every
  // change with a sheet prefix and then re-render without it.
  state.sheet = p.sheets[0]?.number ?? 1;
  state.filters.kinds = new Set(Object.keys(p.kind_label));
  renderKindChips();
  renderSheetTabs();
  renderRows();
  renderDownloads();
  renderSuggestions();
  setupChatAvailability();

  showScreen("workspace");

  // pdf.js needs the panes laid out before it can size a canvas to them.
  await Promise.all([loadPdf("a"), loadPdf("b")]);
  await selectSheet(state.sheet);
  loadJsonView();
}

function renderStats(s) {
  const tiles = [{ n: s.n_primary, label: "changes", cls: "" }];
  for (const [sev, n] of Object.entries(s.severity_counts)) {
    if (n > 0) tiles.push({ n, label: sev, cls: sev });
  }
  if (s.n_cascade > 0) tiles.push({ n: s.n_cascade, label: "knock-on", cls: "" });
  if (s.n_semantic_null > 0) tiles.push({ n: s.n_semantic_null, label: "no-op", cls: "" });

  $("#stats").innerHTML = tiles.map((t) =>
    `<div class="stat ${t.cls}"><span class="n">${t.n}</span><span class="label">${t.label}</span></div>`
  ).join("");
  $("#stats").hidden = false;
}

function renderPrecheckBanners(precheck, forced) {
  if (!precheck) return;
  if (forced && !precheck.is_pair) {
    // The override must not erase the doubt: this stays for the session.
    banner(`<strong>Overridden:</strong> these were not recognised as the same
      drawing (${escapeHtml(precheck.reason)}). Treat every change below with
      that in mind.`);
  } else if (precheck.weak_identity) {
    // cmd_run prints this to stderr; a GUI that dropped it would be quietly
    // claiming more confidence than the pipeline has.
    banner(`<strong>Weak match:</strong> ${escapeHtml(precheck.reason)}. The
      comparison ran, but the two files were matched on limited evidence.`);
  }
}

/* ── pdf.js panes ─────────────────────────────────────────────────────── */

async function loadPdf(side) {
  const paneEl = $(`.pane[data-side="${side}"]`);
  const pdf = await pdfjsLib.getDocument(`/api/jobs/${state.jobId}/pdf/${side}`).promise;
  state.panes[side] = {
    pdf,
    el: paneEl,
    viewport: $(".pane-viewport", paneEl),
    pageEl: $(".page", paneEl),
    canvas: $("canvas", paneEl),
    overlay: $(".overlay", paneEl),
    empty: $(".pane-empty", paneEl),
    renderTask: null,
  };
}

/** Mirrors scroll position as a FRACTION, not in pixels: the two revisions
 *  can render at slightly different sizes, and matching raw pixel offsets
 *  would drift them apart the further you pan.
 *
 *  Bound once at boot against the static DOM, not per job -- binding it in
 *  loadPdf would stack a new listener on the same element every time the
 *  user starts another comparison. */
function setupPaneScrollSync() {
  for (const side of ["a", "b"]) {
    const src = $(`.pane[data-side="${side}"] .pane-viewport`);
    src.addEventListener("scroll", () => {
      if (state.syncing || !$("#sync-panes").checked) return;
      const other = state.panes[side === "a" ? "b" : "a"];
      if (!other) return;
      state.syncing = true;
      const dst = other.viewport;
      const fx = src.scrollLeft / Math.max(1, src.scrollWidth - src.clientWidth);
      const fy = src.scrollTop / Math.max(1, src.scrollHeight - src.clientHeight);
      dst.scrollLeft = fx * (dst.scrollWidth - dst.clientWidth);
      dst.scrollTop = fy * (dst.scrollHeight - dst.clientHeight);
      requestAnimationFrame(() => { state.syncing = false; });
    });
  }
}

async function renderPane(side) {
  const pane = state.panes[side];
  if (!pane) return;
  const sheetInfo = state.payload.sheets.find((s) => s.number === state.sheet);
  const exists = side === "a" ? sheetInfo?.has_a : sheetInfo?.has_b;

  if (!exists || state.sheet > pane.pdf.numPages) {
    pane.pageEl.hidden = true;
    pane.empty.hidden = false;
    return;
  }
  pane.pageEl.hidden = false;
  pane.empty.hidden = true;
  pane.pageEl.style.width = `${state.zoom * 100}%`;

  const page = await pane.pdf.getPage(state.sheet);
  const base = page.getViewport({ scale: 1 });
  const cssWidth = pane.pageEl.clientWidth;

  // Render above CSS size for crispness on HiDPI, but never past what a
  // canvas can actually allocate.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let scale = (cssWidth / base.width) * dpr;
  if (base.width * scale > MAX_CANVAS_PX) scale = MAX_CANVAS_PX / base.width;

  const viewport = page.getViewport({ scale });
  const canvas = pane.canvas;
  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  canvas.style.aspectRatio = `${base.width} / ${base.height}`;

  if (pane.renderTask) pane.renderTask.cancel();
  pane.renderTask = page.render({ canvasContext: canvas.getContext("2d"), viewport });
  try {
    await pane.renderTask.promise;
  } catch (e) {
    if (e?.name !== "RenderingCancelledException") throw e;
  }
}

/* ── boxes ────────────────────────────────────────────────────────────── */

function renderBoxes() {
  for (const side of ["a", "b"]) {
    const pane = state.panes[side];
    if (!pane) continue;
    pane.overlay.innerHTML = "";
    const key = side === "a" ? "box_a" : "box_b";
    for (const rec of visibleRecords()) {
      if (rec.sheet !== state.sheet || !rec[key]) continue;
      pane.overlay.append(boxDiv(rec, rec[key]));
    }
  }
  if (state.selected) markSelected(state.selected);
}

/** Percentages, not pixels -- the whole reason zoom needs no box maths. */
function boxDiv(rec, box) {
  const d = document.createElement("div");
  d.className = "box";
  d.dataset.did = rec.did;
  d.style.left = `${box[0] * 100}%`;
  d.style.top = `${box[1] * 100}%`;
  d.style.width = `${(box[2] - box[0]) * 100}%`;
  d.style.height = `${(box[3] - box[1]) * 100}%`;
  d.style.color = state.payload.kind_color[rec.kind] ?? "#888";
  d.title = rec.description;
  d.onclick = (e) => { e.stopPropagation(); selectDelta(rec.did); };
  return d;
}

async function selectDelta(did, flash = false) {
  const rec = state.payload.deltas.find((r) => r.did === did);
  if (!rec) return;
  if (rec.sheet !== state.sheet) await selectSheet(rec.sheet);
  state.selected = did;
  markSelected(did, flash);
}

function markSelected(did, flash = false) {
  for (const el of $$(".box.active, .row.active")) el.classList.remove("active");
  const targets = $$(`[data-did="${CSS.escape(did)}"]`);
  for (const el of targets) {
    el.classList.add("active");
    if (el.classList.contains("box")) {
      el.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
      if (flash) {
        el.classList.remove("flash");
        void el.offsetWidth;                    // restart the animation
        el.classList.add("flash");
      }
    }
  }
  const row = $(`.row[data-did="${CSS.escape(did)}"]`, $("#rows"));
  if (row) row.scrollIntoView({ block: "nearest" });
}

/* ── sheets & zoom ────────────────────────────────────────────────────── */

function renderSheetTabs() {
  const sheets = state.payload.sheets;
  $("#sheettabs").hidden = sheets.length < 2;
  $("#sheettabs").innerHTML = sheets.map((s) =>
    `<button data-sheet="${s.number}">Sheet ${s.number}</button>`).join("");
  for (const btn of $$("#sheettabs button")) {
    btn.onclick = () => selectSheet(Number(btn.dataset.sheet));
  }
}

async function selectSheet(n) {
  state.sheet = n;
  for (const btn of $$("#sheettabs button")) {
    btn.classList.toggle("active", Number(btn.dataset.sheet) === n);
  }
  await Promise.all([renderPane("a"), renderPane("b")]);
  // Rows first: renderRows replaces the list wholesale, which would drop the
  // .active class that renderBoxes' markSelected had just restored.
  renderRows();
  renderBoxes();
}

async function setZoom(z) {
  state.zoom = Math.min(8, Math.max(0.5, z));
  $("#zoomlevel").textContent = `${Math.round(state.zoom * 100)}%`;
  await Promise.all([renderPane("a"), renderPane("b")]);
}

function setupZoom() {
  const step = (dir) => {
    const i = ZOOM_STEPS.findIndex((z) => z > state.zoom + 1e-6);
    const next = dir > 0
      ? ZOOM_STEPS[i === -1 ? ZOOM_STEPS.length - 1 : i]
      : ZOOM_STEPS[Math.max(0, (i === -1 ? ZOOM_STEPS.length : i) - 2)];
    setZoom(next ?? state.zoom);
  };
  $("#zoom-in").onclick = () => step(1);
  $("#zoom-out").onclick = () => step(-1);
  $("#zoom-fit").onclick = () => setZoom(1);
}

/* ═══════════════════════════════════════════════════════ changes tab ══ */

function visibleRecords() {
  const f = state.filters;
  const q = f.search.trim().toLowerCase();
  return state.payload.deltas
    .filter((r) => f.kinds.has(r.kind))
    .filter((r) => f.cascade || !r.is_cascade)
    .filter((r) => !q ||
      r.description.toLowerCase().includes(q) ||
      (r.zone ?? "").toLowerCase().includes(q) ||
      r.element_type.toLowerCase().includes(q) ||
      r.did.toLowerCase().includes(q))
    .sort((a, b) =>
      (SEVERITY_ORDER[b.severity] ?? 0) - (SEVERITY_ORDER[a.severity] ?? 0) ||
      a.sheet - b.sheet ||
      a.did.localeCompare(b.did));
}

function renderKindChips() {
  const counts = {};
  for (const r of state.payload.deltas) counts[r.kind] = (counts[r.kind] ?? 0) + 1;

  $("#kindchips").innerHTML = Object.entries(state.payload.kind_label)
    .filter(([kind]) => counts[kind])
    .map(([kind, label]) => `
      <label class="chip" data-kind="${kind}">
        <span class="dot" style="background:${state.payload.kind_color[kind]}"></span>
        <span>${escapeHtml(label)}</span>
        <span class="count">${counts[kind]}</span>
      </label>`).join("");

  for (const chip of $$("#kindchips .chip")) {
    chip.onclick = () => {
      const kind = chip.dataset.kind;
      if (state.filters.kinds.has(kind)) state.filters.kinds.delete(kind);
      else state.filters.kinds.add(kind);
      chip.classList.toggle("off", !state.filters.kinds.has(kind));
      renderRows();
      renderBoxes();
    };
  }
}

function renderRows() {
  const records = visibleRecords();
  const total = state.payload.deltas.length;
  $("#tabcount").textContent = records.length;

  if (!records.length) {
    $("#rows").innerHTML = `<div class="empty-rows">No changes match these filters.</div>`;
  } else {
    $("#rows").innerHTML = records.map(rowHtml).join("");
    for (const row of $$("#rows .row")) {
      row.onclick = (e) => {
        if (e.target.classList.contains("rawtoggle")) {
          const pre = $("pre.raw", row);
          pre.hidden = !pre.hidden;
          e.target.textContent = pre.hidden ? "show raw data" : "hide raw data";
          return;
        }
        selectDelta(row.dataset.did);
      };
    }
  }
  $("#rowsfoot").textContent = `${records.length} of ${total} shown`;
}

function rowHtml(r) {
  const color = state.payload.kind_color[r.kind] ?? "#888";
  const onThisSheet = r.sheet === state.sheet;
  const located = r.box_a || r.box_b;
  return `
  <div class="row ${r.is_cascade ? "cascade" : ""}" data-did="${escapeHtml(r.did)}">
    <div class="top">
      <span class="kind-dot" style="background:${color}"></span>
      <span class="sev-badge ${r.severity}">${r.severity}</span>
      ${r.semantic_null ? `<span class="noop" title="${escapeHtml(r.semantic_null_reason ?? "")}">≈ no-op</span>` : ""}
      <span class="loc">${onThisSheet ? "" : `Sheet ${r.sheet} · `}zone ${escapeHtml(r.zone)}</span>
    </div>
    <div class="desc">${escapeHtml(r.description || state.payload.kind_label[r.kind])}</div>
    ${r.is_cascade ? `<div class="note">knock-on effect of ${escapeHtml(r.primary_did ?? "another change")}</div>` : ""}
    ${located ? "" : `<div class="note">no exact location — zone only</div>`}
    <div class="rawtoggle">show raw data</div>
    <pre class="raw mono" hidden>${escapeHtml(JSON.stringify(r, null, 2))}</pre>
  </div>`;
}

function setupFilters() {
  $("#search").oninput = (e) => {
    state.filters.search = e.target.value;
    renderRows();
    renderBoxes();
  };
  $("#show-cascade").onchange = (e) => {
    state.filters.cascade = e.target.checked;
    renderRows();
    renderBoxes();
  };
}

/* ═══════════════════════════════════════════════════════════ ask tab ══ */

function setupChatAvailability() {
  const enabled = state.health.llm_configured;
  $("#chatbox").hidden = !enabled;
  $("#chat-disabled").hidden = enabled;
}

function renderSuggestions() {
  const worst = [...state.payload.deltas]
    .filter((d) => !d.is_cascade)
    .sort((a, b) => (SEVERITY_ORDER[b.severity] ?? 0) - (SEVERITY_ORDER[a.severity] ?? 0))[0];

  const qs = ["What changed in this revision?", "Are any of these changes safety-critical?"];
  if (worst) qs.push(`Tell me more about the change in zone ${worst.zone}.`);

  // Rebuilt rather than filled: the intro block lives inside #chatlog, which
  // is emptied both on the first question and when a new comparison starts.
  const intro = document.createElement("div");
  intro.className = "chat-intro";
  intro.innerHTML = `
    <p>Ask about this revision in plain language. Every answer is grounded in
       the two drawings and the changes found between them — click a citation
       to jump to it.</p>
    <div class="suggestions">
      ${qs.map((q) => `<button class="suggestion">${escapeHtml(q)}</button>`).join("")}
    </div>`;
  $("#chatlog").innerHTML = "";
  $("#chatlog").append(intro);
  for (const btn of $$(".suggestion", intro)) {
    btn.onclick = () => { $("#question").value = btn.textContent.trim(); ask(); };
  }
}

function setupChat() {
  $("#btn-ask").onclick = ask;
  $("#question").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  };
}

async function ask() {
  const input = $("#question");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  $(".chat-intro")?.remove();

  appendMsg("q", escapeHtml(question));
  const pending = appendMsg("a thinking", "Searching the drawings…");
  $("#btn-ask").disabled = true;

  try {
    const res = await api(`/api/jobs/${state.jobId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    pending.remove();
    renderAnswer(res);
  } catch (e) {
    pending.remove();
    appendMsg("a refused", `<span class="refused-label">Could not answer</span>${escapeHtml(e.message)}`);
  } finally {
    $("#btn-ask").disabled = false;
  }
}

function appendMsg(cls, html) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.innerHTML = html;
  $("#chatlog").append(el);
  $("#chatlog").scrollTop = $("#chatlog").scrollHeight;
  return el;
}

function renderAnswer(res) {
  if (res.refused) {
    // chat.py leaves raw model output in `text` on every refusal path, so
    // `reason` is the only thing safe to present as the outcome.
    appendMsg("a refused",
      `<span class="refused-label">Not answered</span>${escapeHtml(res.reason ?? "")}`);
    return;
  }
  // Citations are stashed per-message rather than keyed by delta id: a
  // citation to a raw element ([A:1:F-7:el_…]) resolves to a real box but
  // has no delta behind it, so it is navigable without being selectable.
  const el = appendMsg("a", citationHtml(res.text, res.citations));
  for (const chip of $$(".cite", el)) {
    if (chip.classList.contains("dead")) continue;
    const cite = res.citations[Number(chip.dataset.i)];
    chip.onclick = () => focusCitation(cite);
  }
}

/** Replaces each inline `[A:1:F-7:el_…]` marker with a numbered chip. Uses
 *  the literal `raw` string the server echoed back rather than re-parsing,
 *  so the two can never disagree about what a marker was. */
function citationHtml(text, citations) {
  let html = escapeHtml(text);
  citations.forEach((c, i) => {
    const r = c.resolved;
    const title = r
      ? `${r.description ?? ""} (sheet ${r.sheet})`
      : "this reference could not be located on the drawing";
    const chip = `<span class="cite ${r ? "" : "dead"}" data-i="${i}"
                    title="${escapeHtml(title)}">${i + 1}</span>`;
    html = html.replace(escapeHtml(c.raw), chip);
  });
  return html;
}

/** A delta citation selects the delta (lighting up its row and both boxes).
 *  An element citation has no delta and no sidebar row, so it gets a
 *  transient marker instead -- still navigable, but not pretending to be a
 *  finding. */
async function focusCitation(cite) {
  const r = cite.resolved;
  if (!r) return;
  if (r.did) return selectDelta(r.did, true);

  if (r.sheet !== state.sheet) await selectSheet(r.sheet);
  for (const side of ["a", "b"]) {
    const box = side === "a" ? r.box_a : r.box_b;
    const pane = state.panes[side];
    if (!box || !pane) continue;
    for (const old of $$(".box.transient", pane.overlay)) old.remove();
    const d = boxDiv({ did: `cite_${cite.id}`, kind: "modify", description: r.description }, box);
    d.classList.add("transient", "active", "flash");
    pane.overlay.append(d);
    d.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
    setTimeout(() => d.remove(), 4000);
  }
}

/* ══════════════════════════════════════════════════════════ data tab ══ */

function renderDownloads() {
  const items = [
    ["report.json", "JSON"],
    ["report.md", "Markdown"],
    ["report.html", "Standalone report"],
    ["markup_a.pdf", "Annotated A"],
    ["markup_b.pdf", "Annotated B"],
  ];
  $("#downloads").innerHTML = items.map(([artifact, label]) =>
    `<a href="/api/jobs/${state.jobId}/download/${artifact}" download>${label}</a>`).join("");
}

async function loadJsonView() {
  const res = await fetch(`/api/jobs/${state.jobId}/download/report.json`);
  const text = JSON.stringify(await res.json(), null, 2);
  $("#jsonview")._raw = text;
  $("#jsonview").innerHTML = highlightJson(text);
}

/** Escapes only the three characters that can break out of a text node.
 *  Deliberately NOT escapeHtml: that turns `"` into `&quot;`, after which
 *  the string pattern below matches nothing and the whole view renders
 *  unhighlighted. Quotes need escaping in attribute values, not in text. */
const escapeText = (s) =>
  String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function highlightJson(text) {
  return escapeText(text).replace(
    /("(?:\\.|[^"\\])*"\s*:?)|(\b-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(\btrue\b|\bfalse\b)|(\bnull\b)/g,
    (m, str, num, bool, nul) => {
      if (str) return `<span class="j-${str.trimEnd().endsWith(":") ? "key" : "str"}">${str}</span>`;
      if (num) return `<span class="j-num">${num}</span>`;
      if (bool) return `<span class="j-bool">${bool}</span>`;
      return `<span class="j-null">${nul}</span>`;
    });
}

function setupJsonTab() {
  $("#btn-copy-json").onclick = async () => {
    try {
      await navigator.clipboard.writeText($("#jsonview")._raw ?? "");
      toast("Copied to clipboard");
    } catch { toast("Could not copy — select the text instead"); }
  };
}

/* ══════════════════════════════════════════════════════════════ chrome ══ */

function setupTabs() {
  for (const btn of $$(".tabbtn")) {
    btn.onclick = () => {
      for (const b of $$(".tabbtn")) b.classList.toggle("active", b === btn);
      for (const p of $$(".tabpanel")) p.classList.toggle("active", p.dataset.tab === btn.dataset.tab);
    };
  }
}

function setupTheme() {
  const saved = localStorage.getItem("delta-chat-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#btn-theme").onclick = () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const current = document.documentElement.dataset.theme || (dark ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("delta-chat-theme", next);
  };
}

/* ════════════════════════════════════════════════════════════════ boot ══ */

async function main() {
  setupLanding();
  setupTabs();
  setupTheme();
  setupFilters();
  setupChat();
  setupZoom();
  setupJsonTab();
  setupPaneScrollSync();

  try {
    state.health = await api("/api/health");
    renderSamples();
  } catch {
    state.health = { llm_configured: false, samples: [] };
    banner("<strong>Cannot reach the server.</strong> Is it still running?");
  }
}

main();
