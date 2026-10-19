// KAL detection console - browser side. Polls /api/frame and draws frames, boxes, tracks and charts.
const $ = (id) => document.getElementById(id);
const api = (url, body) => fetch(url, body === undefined ? {} : {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
}).then((r) => r.json());

let CAT = null, lastK = -1, polling = null;
const COL = { det: "#ff9c28", trk: "#46e178", tent: "#ffc43c", tru: "#5ac8e6" };
const imgs = { rgb: new Image(), lwir: new Image() };

// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
  if (b.dataset.tab === "evidence") loadEvidence();
  if (b.dataset.tab === "results") loadResults();
});

// ---------------------------------------------------------------- catalogue and controls
async function init() {
  CAT = await api("/api/catalogue");
  $("scenario").innerHTML = CAT.scenarios.map((s) =>
    `<option value="${s.id}">${s.id.replace(/_/g, " ")} (${s.lighting})</option>`).join("");
  const order = (m) => (m.version === "v2" ? 0 : 1) * 10 + (m.deployable ? 0 : 1);
  const models = [...CAT.models].sort((a, b) => order(a) - order(b));
  const label = (m) => `${m.name}${m.version !== "v2" ? " [" + m.version + "]" : ""}${m.deployable ? "" : " (reference camera)"}`;
  $("model").innerHTML = models.map((m) => `<option value="${m.id}">${label(m)}</option>`).join("");
  $("evModel").innerHTML = models.filter((m) => m.version === "v2").map((m) => `<option value="${m.id}">${label(m)}</option>`).join("");
  $("thrModes").innerHTML = Object.entries(CAT.threshold_modes).map(([k, v], i) =>
    `<label class="toggle"><input type="radio" name="thr" value="${k}" ${i === 0 ? "checked" : ""}> ${v}</label>`).join("");
  document.querySelectorAll("input[name=thr]").forEach((r) => r.onchange = () => {
    $("thrSlider").disabled = r.value !== "custom"; push({ threshold_mode: r.value });
  });
  $("scenario").onchange = showInfo; $("model").onchange = showInfo; showInfo();
  $("start").onclick = start; $("stop").onclick = stop;
  $("thrSlider").oninput = () => { $("thrValue").textContent = `threshold ${(+$("thrSlider").value).toFixed(2)}`; push({ threshold: +$("thrSlider").value }); };
  for (const id of ["rgb_on", "lwir_on", "target_free", "show_truth", "loop"]) $(id).onchange = () => push({ [id]: $(id).checked });
  $("thermal_noise").oninput = () => { $("noiseVal").textContent = (+$("thermal_noise").value).toFixed(1) + "x"; push({ thermal_noise: +$("thermal_noise").value }); };
  $("stripes").oninput = () => { $("stripeVal").textContent = (+$("stripes").value).toFixed(1) + "x"; push({ stripes: +$("stripes").value }); };
  drawChart([]);
}

function showInfo() {
  const s = CAT.scenarios.find((x) => x.id === $("scenario").value);
  const m = CAT.models.find((x) => x.id === $("model").value);
  if (s) $("scenarioInfo").textContent = `${s.frames} frames at ${s.fps} fps, ${s.speed_mps} m/s, range ${s.range_m[0]} -> ${s.range_m[1]} m, `
    + `target ${s.size_px[0]} -> ${s.size_px[1]} px, air ${s.air_temp_c} C`;
  if (m) $("modelInfo").textContent = `inputs ${m.bands.join(" + ")} | validation FROC ${fmt(m.val.froc)}, AP50 ${fmt(m.val.ap50)}`;
}
const fmt = (v, d = 3) => (v === null || v === undefined ? "-" : (+v).toFixed(d));

function settings() {
  const mode = document.querySelector("input[name=thr]:checked").value;
  return { threshold_mode: mode, threshold: +$("thrSlider").value, rgb_on: $("rgb_on").checked, lwir_on: $("lwir_on").checked,
    target_free: $("target_free").checked, show_truth: $("show_truth").checked, loop: $("loop").checked,
    thermal_noise: +$("thermal_noise").value, stripes: +$("stripes").value };
}
let pushTimer = null;
function push(change) { clearTimeout(pushTimer); pushTimer = setTimeout(() => api("/api/settings", change), 120); }

async function start() {
  $("chip").className = "chip search"; $("chip").textContent = "STARTING";
  await api("/api/start", { scenario: $("scenario").value, model: $("model").value, settings: settings() });
  lastK = -1;
  if (!polling) polling = setInterval(poll, 120);
}
async function stop() {
  await api("/api/stop", {}); clearInterval(polling); polling = null;
  $("chip").className = "chip idle"; $("chip").textContent = "IDLE";
}

// ---------------------------------------------------------------- live frames
async function poll() {
  const r = await api(`/api/frame?after=${lastK}`);
  if (!r.running || !r.fresh) return;
  const f = r.frame; lastK = f.k;
  await Promise.all(["rgb", "lwir"].map((b) => new Promise((ok) => { imgs[b].onload = ok; imgs[b].src = f.images[b]; })));
  const showTruth = r.settings.show_truth && f.truth;
  for (const b of ["rgb", "lwir"]) drawView(b, f, showTruth);
  telemetry(r);
}

function box(ctx, b, color, dash = [], w = 2, pad = 4) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash);
  ctx.strokeRect(b[0] - pad, b[1] - pad, b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad); ctx.restore();
}
function label(ctx, x, y, text, color) {
  ctx.font = "bold 13px Consolas, monospace"; ctx.fillStyle = "rgba(0,0,0,.65)";
  const w = ctx.measureText(text).width + 8; ctx.fillRect(x - 4, y - 14, w, 17);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}

function drawView(b, f, showTruth) {
  const cv = $(`cv_${b}`), ctx = cv.getContext("2d");
  ctx.drawImage(imgs[b], 0, 0, 640, 512);
  if (showTruth) box(ctx, f.truth.box, COL.tru, [3, 3], 1.5, 7);
  for (const d of f.detections) { box(ctx, d.box, COL.det, [], 2, 4); label(ctx, d.box[0] - 4, d.box[1] - 8, `KAL ${d.score.toFixed(2)}`, COL.det); }
  for (const t of f.tracks) {
    if (t.confirmed) { box(ctx, t.box, COL.trk, [], 2, 9); label(ctx, t.box[0] - 9, t.box[3] + 24, `TRK ${t.id}${t.coasting ? " coast" : ""}`, COL.trk); }
    else box(ctx, t.box, COL.tent, [4, 3], 1.5, 7);
  }
  // magnifier on the confirmed track, else the best detection, else (truth shown) the truth box
  const conf = f.tracks.find((t) => t.confirmed);
  const src = conf ? conf.box : (f.detections[0] ? f.detections[0].box : (showTruth ? f.truth.box : null));
  const mg = $(`mag_${b}`), mc = mg.getContext("2d");
  mg.style.display = src ? "block" : "none";
  if (!src) return;
  const cx = (src[0] + src[2]) / 2, cy = (src[1] + src[3]) / 2;
  const win = Math.max(24, 2.4 * Math.max(src[2] - src[0], src[3] - src[1]));
  mc.imageSmoothingEnabled = false;
  mc.drawImage(imgs[b], cx - win / 2, cy - win / 2, win, win, 0, 0, 150, 150);
  mc.fillStyle = COL.det; mc.font = "bold 12px Consolas"; mc.fillText(`x${(150 / win).toFixed(0)}`, 5, 14);
}

function telemetry(r) {
  const f = r.frame, s = r.stats;
  $("clock").textContent = `T+${f.t.toFixed(1)} s`;
  $("fps").textContent = `${s.fps.toFixed(1)} fps (video 25)`;
  const conf = f.tracks.some((t) => t.confirmed), coast = f.tracks.some((t) => t.confirmed && t.coasting > 0);
  const chip = $("chip");
  if (f.band_missing) { chip.className = "chip warn"; chip.textContent = "MODEL NEEDS THAT CAMERA"; }
  else if (conf && !coast) { chip.className = "chip track"; chip.textContent = f.target_present ? "TRACK CONFIRMED" : "FALSE TRACK"; }
  else if (coast) { chip.className = "chip coast"; chip.textContent = "COASTING"; }
  else { chip.className = "chip search"; chip.textContent = f.detections.length ? "TENTATIVE" : "SEARCHING"; }
  const fc = s.first_confirm;
  $("tFirst").textContent = fc ? `${fc.t.toFixed(1)} s, ${Math.round(fc.range_m)} m, ${fc.size_px.toFixed(1)} px` : "-";
  $("tHeld").textContent = fc ? `${Math.round(100 * s.tracked_fraction)} % of frames` : "-";
  $("tFalse").textContent = s.false_tracks;
  $("tDets").textContent = f.detections.length;
  $("tRange").textContent = f.truth ? `${Math.round(f.truth.range_m)} m` : "no target (empty sky)";
  $("tSize").textContent = f.truth ? `${f.truth.size_px.toFixed(1)} px` : "-";
  $("tFrame").textContent = `${f.k + 1} / ${f.n}`;
  $("tBands").textContent = r.model.bands.join(" + ");
  $("tThr").textContent = f.threshold.toFixed(2);
  $("tMs").textContent = `${Math.round(f.detector_ms)} ms`;
  $("thrValue").textContent = `threshold ${f.threshold.toFixed(2)}`;
  drawChart(s.history);
}

function drawChart(h) {
  const cv = $("chart"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.fillStyle = "#0b0f13"; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#222c35"; ctx.lineWidth = 1;
  for (const y of [0.25, 0.5, 0.75]) { ctx.beginPath(); ctx.moveTo(0, H - y * H); ctx.lineTo(W, H - y * H); ctx.stroke(); }
  if (!h.length) return;
  const x = (i) => (i / Math.max(1, 299)) * W, y = (v) => H - 4 - v * (H - 8);
  ctx.strokeStyle = "#ff9c28"; ctx.setLineDash([6, 4]); ctx.beginPath();
  h.forEach((p, i) => (i ? ctx.lineTo(x(i), y(p.thr)) : ctx.moveTo(x(i), y(p.thr)))); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = "#5ac8e6"; ctx.lineWidth = 1.5; ctx.beginPath();
  h.forEach((p, i) => (i ? ctx.lineTo(x(i), y(p.best)) : ctx.moveTo(x(i), y(p.best)))); ctx.stroke();
  ctx.fillStyle = "#46e178"; h.forEach((p, i) => { if (p.on) ctx.fillRect(x(i) - 1, H - 6, 3, 5); });
}

// ---------------------------------------------------------------- held-out evidence
let EV = [];
async function loadEvidence() {
  const r = await api("/api/evidence/scenes");
  EV = r.scenes;
  if (!EV.length) { $("evMsg").textContent = "Available after the M1 detector v2 evaluation (out/m1/det2/test_lock.json)."; return; }
  $("evMsg").textContent = "Pick a held-out test scene.";
  const show = () => {
    const f = $("evFilter").value;
    $("evList").innerHTML = EV.filter((s) => !f || s.lighting === f || s.size_bin === f).map((s) =>
      `<li data-id="${s.id}">${s.id} ${s.lighting.padEnd(5)} ${String(s.size_px).padStart(5)} px ${String(s.range_m).padStart(5)} m</li>`).join("");
    document.querySelectorAll("#evList li").forEach((li) => li.onclick = () => runEvidence(li));
  };
  $("evFilter").onchange = show; show();
}
async function runEvidence(li) {
  document.querySelectorAll("#evList li").forEach((x) => x.classList.toggle("sel", x === li));
  $("evResult").textContent = "Running saved weights...";
  const r = await api("/api/evidence/run", { scene: li.dataset.id, model: $("evModel").value });
  for (const b of ["rgb", "lwir"]) {
    const im = new Image();
    im.onload = () => {
      const ctx = $(`ev_${b}`).getContext("2d"); ctx.drawImage(im, 0, 0, 640, 512);
      for (const t of r.truth) box(ctx, t, "#46e178", [], 1.5, 3);
      for (const d of r.detections) { box(ctx, d.box, COL.det, [], 2, 6); label(ctx, d.box[0] - 6, d.box[1] - 10, d.score.toFixed(2), COL.det); }
    };
    im.src = r.images[b];
  }
  const m = r.matches_saved_evaluation;
  $("evResult").innerHTML = `Green = label, orange = prediction at the frozen 0.01 threshold (${r.threshold.toFixed(2)}). `
    + (m === null ? "No stored evaluation to compare." : m ? "<b style='color:#46e178'>Identical to the stored evaluation.</b>"
      : "<b style='color:#f05046'>Differs from the stored evaluation.</b>");
}

// ---------------------------------------------------------------- results
async function loadResults() {
  const r = await api("/api/results");
  if (!r.available) { $("resBody").innerHTML = "<h2>Held-out results</h2><p class='muted'>Not yet available: they appear after the M1 detector v2 evaluation. The compact pilot results are in out/m1/pilot/results.md.</p>"; return; }
  const rows = r.rows.map((x) => `<tr><td>${x.model}</td><td>${x.bands.join(" + ")}</td><td>${fmt(x.froc)} [${fmt(x.froc_ci[0])}, ${fmt(x.froc_ci[1])}]</td>`
    + `<td>${fmt(x.pd)} (${x.pd_tp}/${x.n})</td><td>${fmt(x.fa, 4)}</td><td>${fmt(x.ap50)}</td></tr>`).join("");
  const cmp = Object.entries(r.comparisons).map(([k, v]) =>
    `<li>${k.replace(/_/g, " ")}: FROC ${v.froc.diff >= 0 ? "+" : ""}${fmt(v.froc.diff)} [${fmt(v.froc.lo)}, ${fmt(v.froc.hi)}]</li>`).join("");
  $("resBody").innerHTML = `<h2>Held-out results (${r.n_scenes} fresh test scenes)</h2>
    <table><tr><th>Model</th><th>Inputs</th><th>FROC [95% CI]</th><th>Detection @ 0.01 FA/frame</th><th>FA/frame measured</th><th>AP50</th></tr>${rows}</table>
    <p class="muted">Teacher gate on validation: ${r.gate.passed ? "passed" : "not passed (no distillation claim)"}.</p>
    <h3>Paired differences (scene bootstrap)</h3><ul>${cmp}</ul>
    <p class="muted">Full report: out/m1/det2/results.md. Synthetic scenes only; one training seed.</p>`;
}

init();
