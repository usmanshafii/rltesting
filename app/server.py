"""HTTP side of the detection console: JSON endpoints and the single-page UI (127.0.0.1 only)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from app import engine

STATIC = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=None)
STATE = dict(run=None)
LOCK = threading.Lock()
DEFAULTS = dict(threshold_mode="fa01", threshold=0.5, rgb_on=True, lwir_on=True, show_truth=False,
                thermal_noise=1.0, stripes=1.0, target_free=False, loop=True)


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.get("/api/catalogue")
def catalogue():
    rep = engine.ROOT / "out/m1/det2/results.md"
    return jsonify(scenarios=engine.scenarios(), models=engine.models(), threshold_modes=engine.THRESHOLD_MODES,
                   defaults=DEFAULTS, evidence=bool(engine.evidence_scenes()), results_available=rep.exists())


@app.post("/api/start")
def start():
    body = request.get_json(force=True)
    scn = next(s for s in engine.scenarios() if s["id"] == body["scenario"])
    mdl = next(m for m in engine.models() if m["id"] == body["model"])
    settings = dict(DEFAULTS, **body.get("settings", {}))
    with LOCK:
        if STATE["run"] is not None:
            STATE["run"].stop()
        STATE["run"] = engine.LiveRun(scn["path"], mdl, settings)
        STATE["scenario"], STATE["model"] = scn, mdl
    return jsonify(ok=True, scenario=scn, model=mdl)


@app.post("/api/settings")
def settings():
    run = STATE["run"]
    if run is None:
        return jsonify(ok=False, error="not running"), 400
    run.update(request.get_json(force=True))
    return jsonify(ok=True, settings=run.settings)


@app.post("/api/stop")
def stop():
    with LOCK:
        if STATE["run"] is not None:
            STATE["run"].stop()
            STATE["run"] = None
    return jsonify(ok=True)


@app.get("/api/frame")
def frame():
    run = STATE["run"]
    if run is None:
        return jsonify(running=False)
    snap = run.snapshot(int(request.args.get("after", -1)))
    if snap is None:
        return jsonify(running=True, fresh=False)
    return jsonify(running=True, fresh=True, scenario=STATE.get("scenario"), model=STATE.get("model"), **snap)


@app.get("/api/evidence/scenes")
def evidence_scenes():
    return jsonify(scenes=engine.evidence_scenes())


@app.post("/api/evidence/run")
def evidence_run():
    body = request.get_json(force=True)
    mdl = next(m for m in engine.models() if m["id"] == body["model"])
    return jsonify(engine.evidence_run(body["scene"], mdl))


@app.get("/api/results")
def results():
    p = engine.ROOT / "out/m1/det2/report.json"
    if not p.exists():
        return jsonify(available=False)
    rep = json.loads(p.read_text())
    rows = []
    for name, r in rep["cohorts"]["fresh_test"]["models"].items():
        o = r["operating_points"]["threshold_fa01"]
        rows.append(dict(model=name, bands=r["bands"], froc=r["froc"], froc_ci=[r["froc_ci95"]["lo"], r["froc_ci95"]["hi"]],
                         pd=o["pd_centre"], pd_tp=o["pd_tp"], n=o["n"], fa=o["fa_per_target_free_frame"], ap50=r["ap50"]))
    return jsonify(available=True, n_scenes=rep["cohorts"]["fresh_test"]["n_scenes"], rows=rows,
                   gate=rep["gate_on_validation"], comparisons=rep["comparisons"]["fresh_test"])
