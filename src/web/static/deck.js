const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];

const STAGE_PROGRESS = {empty:0, frames:30, transcribed:70, captioned:70, embedded:100};
const NEEDS = {
  ingest:      {video:true,  run:false, ingest:true,  transcribe:false},
  transcribe:  {video:true,  run:false, ingest:false, transcribe:true},
  summarize:   {video:false, run:true,  ingest:false, transcribe:false},
  reconstruct: {video:false, run:true,  ingest:false, transcribe:false},
  assemble:    {video:false, run:false, ingest:false, transcribe:false},
};

let RUNS = [];
let SELECTED = null;      // slug
let SNAPSHOT = null;      // detailed snapshot of SELECTED
let BUSY = false;
let MODELS = {all: [], vision: [], text: []};

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function fmtNum(n) {
  n = Number(n) || 0;
  return n < 1000 ? String(n) : (n/1000).toFixed(1) + "k";
}
function fmtBytes(n) {
  n = Number(n) || 0;
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n/1024).toFixed(0) + " KB";
  return (n/1048576).toFixed(1) + " MB";
}
function fmtTime(epoch) {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toISOString().replace("T", " ").slice(0, 16);
}

function tickClock() { $("#clock").textContent = new Date().toTimeString().slice(0,8); }
setInterval(tickClock, 1000); tickClock();

/* ── nav ── */
function go(name) {
  $$(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + name));
  location.hash = name;
}
$$(".nav-btn").forEach(b => b.addEventListener("click", () => go(b.dataset.view)));

/* ── API ── */
async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function post(path, body) {
  return api(path, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
}

/* ── models + roles ── */
function endpointQuery() {
  return "?" + new URLSearchParams({backend: $("#f-backend").value, base_url: $("#f-baseurl").value});
}
async function loadRoles() {
  try {
    const roles = await api("/api/roles" + endpointQuery());
    const v = roles.caption || {}, t = roles.text || {};
    const vb = $("#badge-vision");
    vb.textContent = "◉ " + (v.model || "—");
    vb.className = "badge" + (v.vision_ok === false ? " bad" : "");
    vb.title = `Vision role — ${v.provider} @ ${v.base_url}` +
      (v.vision_ok === false ? " — NOT vision-capable, captions/OCR will fail" : "");
    $("#badge-text").textContent = "▤ " + (t.model || "—");
    $("#badge-text").title = `Text role — ${t.provider} @ ${t.base_url}`;
    if (!$("#f-baseurl").value) $("#f-baseurl").value = v.base_url || "";
    return roles;
  } catch (e) { return null; }
}

let endpointRevision = 0;
async function loadBackend() {
  const revision = ++endpointRevision;
  try {
    const d = await api("/api/backend" + endpointQuery());
    if (revision !== endpointRevision) return;
    const el = $("#backend-status");
    if (d.reachable) {
      MODELS = {all: d.models || [], vision: d.vision_models || [], text: d.text_models || []};
      el.innerHTML = `<span style="color:var(--green)">●</span> ${esc(d.provider)} reachable at ${esc(d.base_url)} — ${MODELS.all.length} model(s)`;
      await fillModelSelects(revision);
    } else {
      MODELS = {all: [], vision: [], text: []};
      $("#f-vision").innerHTML = ""; $("#f-text").innerHTML = "";
      el.innerHTML = `<span style="color:var(--amber)">●</span> ${esc(d.provider)} offline at ${esc(d.base_url)}${d.detail ? " — " + esc(d.detail) : ""}`;
    }
  } catch (e) {
    $("#backend-status").textContent = "endpoint probe failed";
  }
}

async function fillModelSelects(revision) {
  const roles = await loadRoles();
  if (revision !== endpointRevision) return;
  const opt = (list, current, note) => {
    const seen = new Set();
    const items = [];
    if (!list.length) return '<option value="">No compatible served model</option>';
    for (const m of list) if (!seen.has(m)) { seen.add(m); items.push([m, m + note(m)]); }
    return items.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join("");
  };
  const vision = $("#f-vision"), text = $("#f-text");
  const curV = roles?.caption?.model, curT = roles?.text?.model;
  // Vision role: only offer models that can actually see. The capability guard
  // aborts on a text-only pick anyway, so don't let the UI suggest one.
  vision.innerHTML = opt(MODELS.vision, curV, () => " (vision)");
  text.innerHTML = opt(MODELS.all, curT, m => MODELS.vision.includes(m) ? " (vision)" : " (text)");
  if (MODELS.vision.includes(curV)) vision.value = curV;
  if (MODELS.all.includes(curT)) text.value = curT;
}

/* ── runs ── */
async function loadRuns() {
  const d = await api("/api/runs");
  RUNS = d.runs || [];
  $("#nav-run-count").textContent = RUNS.length || "";
  renderRunTable();
  renderRunProgress();
  renderStageTrack();
  fillRunSelects();
  if (!SELECTED && RUNS.length) selectRun(RUNS[0].slug);
  return d;
}

function fillRunSelects() {
  const html = RUNS.map(r =>
    `<option value="${esc(r.slug)}">${esc(r.slug)} — ${r.frames} frames, ${r.captions} captions</option>`
  ).join("");
  for (const sel of [$("#q-run"), $("#f-run")]) {
    const keep = sel.value;
    sel.innerHTML = html || `<option value="">no runs found</option>`;
    if (keep && RUNS.some(r => r.slug === keep)) sel.value = keep;
    else if (SELECTED) sel.value = SELECTED;
  }
}

function renderRunTable(filter="") {
  const q = filter.trim().toLowerCase();
  const rows = RUNS.filter(r => !q || r.slug.toLowerCase().includes(q) || r.stage.includes(q));
  const tb = $("#run-tbody");
  $("#run-empty").style.display = rows.length ? "none" : "block";
  tb.innerHTML = rows.map(r => `
    <tr data-slug="${esc(r.slug)}" class="${r.slug === SELECTED ? "sel" : ""}">
      <td class="mono">${esc(r.slug)}</td>
      <td><span class="status-pill st-${esc(r.stage)}">${esc(r.stage)}</span></td>
      <td class="mono">${r.frames}</td>
      <td class="mono">${r.captions}</td>
      <td class="mono">${r.ocr}</td>
      <td class="mono">${r.outputs.length}</td>
      <td class="mono dim">${r.has_chromadb ? "yes" : "—"}</td>
      <td class="mono dim">${fmtTime(r.modified)}</td>
    </tr>`).join("");
  tb.querySelectorAll("tr").forEach(tr =>
    tr.addEventListener("click", () => selectRun(tr.dataset.slug)));
}

function renderRunProgress() {
  const box = $("#run-progress");
  if (!RUNS.length) { box.innerHTML = `<div class="empty">NO RUNS FOUND</div>`; return; }
  box.innerHTML = RUNS.slice(0, 12).map(r => {
    const pct = STAGE_PROGRESS[r.stage] ?? 0;
    return `<div class="risk-row" data-slug="${esc(r.slug)}">
      <div class="risk-rank">${r.frames}f</div>
      <div class="risk-name">${esc(r.slug)}</div>
      <div class="risk-bar"><i style="width:${pct}%"></i></div>
      <div class="risk-score">${esc(r.stage)}</div>
    </div>`;
  }).join("");
  box.querySelectorAll(".risk-row").forEach(row =>
    row.addEventListener("click", () => { selectRun(row.dataset.slug); go("runs"); }));
}

function renderStageTrack() {
  const counts = {};
  for (const r of RUNS) counts[r.stage] = (counts[r.stage] || 0) + 1;
  const stages = ["frames","captioned","embedded","transcribed","empty"];
  $("#stage-track").innerHTML = stages.map(s =>
    `<div class="stage-pill ${counts[s] ? "active" : ""}"><div class="sn">${s}</div><div class="sv">${counts[s] || 0}</div></div>`
  ).join("");
}

function renderKPIs() {
  const totals = RUNS.reduce((a, r) => ({
    frames: a.frames + r.frames, captions: a.captions + r.captions,
    ocr: a.ocr + r.ocr, outputs: a.outputs + r.outputs.length,
    vector: a.vector + (r.has_chromadb ? 1 : 0),
  }), {frames:0, captions:0, ocr:0, outputs:0, vector:0});

  const cards = [
    {label:"Runs", value:String(RUNS.length), sub:`under ./data/`, cls:"accent"},
    {label:"Frames", value:fmtNum(totals.frames), sub:"extracted keyframes", cls:""},
    {label:"Captions", value:fmtNum(totals.captions), sub:"vision-model reads", cls:"accent"},
    {label:"OCR frames", value:fmtNum(totals.ocr), sub:"verbatim transcriptions", cls:""},
    {label:"Artifacts", value:String(totals.outputs), sub:"files in output/", cls:""},
    {label:"Vector DBs", value:String(totals.vector), sub:"searchable collections", cls: totals.vector ? "accent" : "warn"},
  ];
  $("#kpis").innerHTML = cards.map(c => `
    <div class="panel kpi ${c.cls}">
      <div class="k-label">${c.label}</div>
      <div class="k-value">${esc(c.value)}</div>
      <div class="k-sub">${esc(c.sub)}</div>
    </div>`).join("");
}

async function selectRun(slug) {
  if (!slug) return;
  SELECTED = slug;
  renderRunTable($("#run-search").value);
  $("#frames-run").textContent = slug;
  $("#art-run").textContent = slug;
  for (const sel of [$("#q-run"), $("#f-run")]) if (RUNS.some(r => r.slug === slug)) sel.value = slug;
  try {
    SNAPSHOT = await api("/api/run?slug=" + encodeURIComponent(slug));
  } catch (e) { SNAPSHOT = null; }
  renderFrames();
  renderArtifactChips();
}

/* ── frames ── */
function renderFrames() {
  const grid = $("#frame-grid"), empty = $("#frames-empty");
  const frames = SNAPSHOT?.frames_list || [];
  $("#nav-frame-count").textContent = frames.length || "";
  if (!frames.length) {
    grid.innerHTML = "";
    empty.style.display = "block";
    empty.textContent = SELECTED ? "THIS RUN HAS NO FRAMES" : "SELECT A RUN IN THE REGISTER";
    return;
  }
  empty.style.display = "none";
  const caps = new Map((SNAPSHOT?.captions_preview || []).map(c => [c.frame, c]));
  grid.innerHTML = frames.map((f, i) => {
    const cap = caps.get(f.name);
    const src = `/api/frame/${encodeURIComponent(SELECTED)}/${encodeURIComponent(f.name)}`;
    return `<div class="frame-card" data-i="${i}">
      <img loading="lazy" src="${src}" alt="${esc(f.name)}">
      <div class="fc-meta"><b>${esc(cap?.timestamp_str || f.name.replace(/\.\w+$/, ""))}</b><span>${cap ? cap.chars + " ch" : fmtBytes(f.size)}</span></div>
    </div>`;
  }).join("");
  grid.querySelectorAll(".frame-card").forEach(card =>
    card.addEventListener("click", () => openFrame(Number(card.dataset.i))));
}

function openFrame(i) {
  const f = (SNAPSHOT?.frames_list || [])[i];
  if (!f) return;
  const cap = (SNAPSHOT?.captions_preview || []).find(c => c.frame === f.name);
  const src = `/api/frame/${encodeURIComponent(SELECTED)}/${encodeURIComponent(f.name)}`;
  $("#modal-body").innerHTML = `
    <h2>${esc(f.name)}</h2>
    <div class="chips" style="margin-top:10px">
      <span class="tag">${esc(SELECTED)}</span>
      ${cap?.timestamp_str ? `<span class="tag amber">${esc(cap.timestamp_str)}</span>` : ""}
      <span class="tag violet">${fmtBytes(f.size)}</span>
    </div>
    <img class="frame-full" src="${src}" alt="${esc(f.name)}">
    <div class="m-section"><div class="s-label">Caption${cap ? ` (${cap.chars} chars)` : ""}</div>
      <div class="s-body">${cap ? esc(cap.preview) + (cap.chars > cap.preview.length ? "\n…" : "") : "No caption for this frame yet — run the ingest pipeline."}</div>
    </div>`;
  $("#modal-overlay").classList.add("open");
}
$("#modal-close").onclick = () => $("#modal-overlay").classList.remove("open");
$("#modal-overlay").addEventListener("click", e => { if (e.target.id === "modal-overlay") e.currentTarget.classList.remove("open"); });

/* ── artifacts ── */
function renderArtifactChips() {
  const outputs = SNAPSHOT?.outputs || [];
  $("#nav-art-count").textContent = outputs.length || "";
  const box = $("#art-chips");
  if (!outputs.length) {
    box.innerHTML = `<span class="tag red">no artifacts in output/</span>`;
    $("#art-title").textContent = "No artifact opened";
    $("#art-body").textContent = SELECTED
      ? "This run has no output/ files yet. Run transcribe, reconstruct or summarize."
      : "Select a run in the register, then pick an artifact above.";
    return;
  }
  box.innerHTML = outputs.map(n => `<button class="btn ghost small" data-art="${esc(n)}">${esc(n)}</button>`).join("");
  box.querySelectorAll("[data-art]").forEach(b =>
    b.addEventListener("click", () => openArtifact(b.dataset.art)));
}

async function openArtifact(name) {
  $("#art-title").textContent = name;
  $("#art-body").textContent = "loading…";
  try {
    const d = await api(`/api/artifact?slug=${encodeURIComponent(SELECTED)}&name=${encodeURIComponent(name)}`);
    $("#art-title").textContent = `${name} — ${fmtBytes(d.size)}${d.truncated ? " (truncated)" : ""}`;
    $("#art-body").textContent = d.text;
  } catch (e) {
    $("#art-body").textContent = "Could not read artifact: " + e.message;
  }
}

/* ── events / jobs ── */
function renderEvents(events) {
  const box = $("#event-log");
  if (!events || !events.length) { box.innerHTML = `<div class="empty">NO EVENTS YET</div>`; return; }
  box.innerHTML = events.slice(-120).reverse().map(e => {
    const at = (e.at || "").slice(11, 19);
    const step = e.step ? ` [${e.step}/${e.steps}]` : "";
    return `<div class="ev ${esc(e.kind)}"><span class="t">${esc(at)}</span><b>${esc(e.kind)}${step}</b>${esc(e.note || "")}</div>`;
  }).join("");
}

function renderJobs(jobs) {
  const tb = $("#job-tbody");
  if (!jobs || !jobs.length) {
    tb.innerHTML = `<tr><td colspan="6" class="dim" style="text-align:center">NO JOBS RUN YET</td></tr>`;
    return;
  }
  tb.innerHTML = jobs.map(j => {
    const detail = j.error || (j.result ? summarizeResult(j) : (j.progress?.stage || "—"));
    const cls = j.status === "done" ? "done" : j.status === "running" ? "running" : "error";
    return `<tr>
      <td class="mono">${esc(j.id)}</td>
      <td class="mono dim">${esc(j.pipeline)}</td>
      <td><span class="status-pill st-${cls}">${esc(j.status)}</span></td>
      <td class="mono dim">${esc((j.started_at||"").replace("T"," ").slice(0,19))}</td>
      <td class="mono dim">${j.elapsed_seconds != null ? j.elapsed_seconds + "s" : "—"}</td>
      <td class="dim">${esc(String(detail).slice(0, 160))}</td>
    </tr>`;
  }).join("");
}

function summarizeResult(job) {
  const r = job.result || {};
  if (job.pipeline === "ingest") return `${r.num_frames ?? "?"} frames → ${r.run_slug ?? ""}`;
  if (job.pipeline === "transcribe") return `${r.frames_with_text ?? "?"}/${r.frames_selected ?? "?"} frames with text`;
  if (job.pipeline === "summarize") return `${r.chars ?? 0} char summary`;
  if (job.pipeline === "reconstruct") return `${r.folders ?? 0} folder(s)`;
  return JSON.stringify(r).slice(0, 120);
}

/* ── pipeline form ── */
function applyNeeds() {
  const needs = NEEDS[$("#f-pipeline").value] || {};
  $$("[data-need]").forEach(el => {
    el.style.display = needs[el.dataset.need] ? "" : "none";
  });
}
$("#f-pipeline").addEventListener("change", applyNeeds);

async function loadVideos() {
  try {
    const d = await api("/api/videos?folder=./input");
    const sel = $("#f-video-pick");
    sel.innerHTML = `<option value="">— pick from ./input —</option>` +
      (d.videos || []).map(v => `<option value="${esc(v.path)}">${esc(v.name)} (${fmtBytes(v.size)})</option>`).join("");
  } catch (e) { /* folder may not exist */ }
}
$("#f-video-pick").addEventListener("change", e => { if (e.target.value) $("#f-video").value = e.target.value; });

async function startRun() {
  const pipeline = $("#f-pipeline").value;
  const body = {
    pipeline,
    backend: $("#f-backend").value,
    base_url: $("#f-baseurl").value.trim() || null,
    vision_model: $("#f-vision").value || null,
    text_model: $("#f-text").value || null,
  };
  if (["ingest", "transcribe"].includes(pipeline) && !body.vision_model) {
    alert("This endpoint has no recognized vision model. Choose Spark or serve a vision model in oMLX.");
    return;
  }
  if (["summarize", "reconstruct"].includes(pipeline) && !body.text_model) {
    alert("Choose an endpoint with a served text model first.");
    return;
  }
  const needs = NEEDS[pipeline] || {};
  if (needs.video) {
    body.video_path = $("#f-video").value.trim();
    if (!body.video_path) { alert("A video path is required for " + pipeline); return; }
  }
  if (needs.run) {
    body.run_slug = $("#f-run").value;
    if (!body.run_slug && pipeline === "summarize") { alert("Select a run folder"); return; }
  }
  if (pipeline === "ingest") {
    body.strategy = $("#f-strategy").value;
    body.fps = parseFloat($("#f-fps").value) || 1.0;
    body.caption_max_tokens = parseInt($("#f-captiontokens").value) || null;
  }
  if (pipeline === "transcribe") {
    body.sample_fps = parseFloat($("#f-samplefps").value) || 2.0;
    body.cleanup = $("#f-cleanup").checked;
    body.deterministic = $("#f-deterministic").checked;
  }
  try {
    $("#run-btn").disabled = true;
    await post("/api/run", body);
    go("overview");
    await poll();
  } catch (e) {
    alert(e.message || String(e));
    $("#run-btn").disabled = false;
  }
}
$("#run-btn").addEventListener("click", startRun);

/* ── search ── */
async function doSearch() {
  const query = $("#q-query").value.trim();
  if (!query) { alert("Enter a question"); return; }
  const slug = $("#q-run").value;
  if (!slug) { alert("No run selected — ingest a video first"); return; }
  $("#q-btn").disabled = true;
  $("#q-status").innerHTML = `<span class="spinner"></span> searching…`;
  $("#q-results").innerHTML = "";
  $("#q-summary-panel").style.display = "none";
  try {
    const d = await post("/api/search", {
      query, run_slug: slug,
      top_k: parseInt($("#q-topk").value) || 10,
      summarize: $("#q-summarize").checked,
    });
    const results = d.results || [];
    $("#q-status").textContent = `${results.length} hit(s) in ${esc(slug)}`;
    if (d.summary) {
      $("#q-summary").textContent = d.summary;
      $("#q-summary-panel").style.display = "";
    }
    $("#q-results").innerHTML = results.length ? results.map(r => `
      <div class="result-card">
        <div class="rc-head">
          <span class="tag amber">${esc(r.timestamp_str || "?")}</span>
          <span class="tag">score ${(Number(r.score)||0).toFixed(3)}</span>
          ${r.frame_id != null ? `<span class="tag violet">frame ${r.frame_id}</span>` : ""}
        </div>
        <div class="rc-body">${esc(r.caption || "(no caption)")}</div>
      </div>`).join("") : `<div class="empty">NO MATCHES</div>`;
  } catch (e) {
    $("#q-status").innerHTML = `<span style="color:var(--red)">✗ ${esc(e.message)}</span>`;
  } finally {
    $("#q-btn").disabled = false;
  }
}
$("#q-btn").addEventListener("click", doSearch);
$("#q-query").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

/* ── poll ── */
function applyStatus(health, jobs, events) {
  const wasBusy = BUSY;
  BUSY = !!health.busy;
  const badge = $("#badge-status"), dot = $("#live-dot");
  const job = (jobs.jobs || []).find(j => j.id === health.active_job_id);

  if (BUSY) {
    badge.textContent = "RUNNING";
    badge.className = "badge warn";
    dot.classList.remove("off");
    const p = job?.progress || {};
    const pct = p.steps ? ` <b style="color:var(--cyan)">${Math.round(100 * p.step / p.steps)}%</b>` : "";
    $("#live-line").innerHTML =
      `<span class="spinner"></span> ${esc(job?.pipeline || "job")} · ${esc(p.stage || "starting")}${pct}` +
      (p.note ? ` · ${esc(String(p.note).slice(0,120))}` : "") +
      (health.current?.slug ? ` · ${esc(health.current.slug)}` : "");
    $("#run-btn").disabled = true;
    $("#run-btn").textContent = "RUNNING…";
  } else {
    const last = (jobs.jobs || [])[0];
    const failed = last && last.status === "error";
    badge.textContent = failed ? "ERROR" : "READY";
    badge.className = failed ? "badge bad" : "badge ok";
    dot.classList.add("off");
    if (failed) {
      $("#live-line").innerHTML = `<span style="color:var(--red)">✗ ${esc(last.error || "job failed")}</span>`;
    } else if (last) {
      $("#live-line").textContent = `${last.pipeline} ${last.status} in ${last.elapsed_seconds ?? "?"}s — ${summarizeResult(last)}`;
    } else {
      $("#live-line").textContent = "⟳ no job running";
    }
    $("#run-btn").disabled = false;
    $("#run-btn").textContent = "▶ RUN PIPELINE";
  }
  renderEvents(events.events);
  renderJobs(jobs.jobs);

  // A finished job changes what's on disk — rescan, and follow the run it made.
  if (wasBusy && !BUSY) {
    const slug = (jobs.jobs || [])[0]?.result?.run_slug;
    loadRuns().then(() => { if (slug) selectRun(slug); });
  }
}

async function poll() {
  try {
    const [health, jobs, events] = await Promise.all([
      api("/api/health"), api("/api/jobs"), api("/api/events?limit=120"),
    ]);
    applyStatus(health, jobs, events);
    renderKPIs();
  } catch (e) { console.warn(e); }
}

/* ── wiring ── */
$("#run-search").addEventListener("input", e => renderRunTable(e.target.value));
$("#refresh-runs").addEventListener("click", () => loadRuns());
$("#frames-reload").addEventListener("click", () => SELECTED && selectRun(SELECTED));
$("#art-reload").addEventListener("click", () => SELECTED && selectRun(SELECTED));
$("#f-backend").addEventListener("change", () => {
  $("#f-baseurl").value = $("#f-backend").value === "vllm"
    ? "http://192.168.86.44:8000/v1" : "http://127.0.0.1:8000/v1";
  loadBackend();
});
$("#f-baseurl").addEventListener("change", loadBackend);

(async function init() {
  applyNeeds();
  if (location.hash) {
    const n = location.hash.slice(1);
    if ($(`[data-view="${n}"]`)) go(n);
  }
  await Promise.all([loadRuns(), loadVideos(), loadBackend()]);
  renderKPIs();
  await poll();
  setInterval(poll, 1500);
})();
