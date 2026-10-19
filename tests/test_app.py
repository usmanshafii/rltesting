"""Live detection console (app/): API, frame encoding and the live pipeline on a rendered scenario."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

pytest.importorskip("flask")
torch = pytest.importorskip("torch")
SCENARIO = ROOT / "out/video_v2/sensor/plan.json"
V1 = ROOT / "out/m1/cnn/runs/rgb_lwir/best.pt"


def test_jpeg_data_url_and_threshold_modes():
    from app import engine
    url = engine._jpeg(np.linspace(0, 1, 64 * 80).reshape(64, 80, 1))
    assert url.startswith("data:image/jpeg;base64,") and len(url) > 500
    assert {"fa01", "fa05", "f1", "custom"} == set(engine.THRESHOLD_MODES)


def test_api_catalogue_and_idle_frame():
    from app.server import app
    c = app.test_client()
    cat = c.get("/api/catalogue").get_json()
    assert {"scenarios", "models", "threshold_modes", "defaults"} <= set(cat)
    assert c.get("/api/frame").get_json() == {"running": False}
    assert c.get("/").status_code == 200 and b"KAL DETECTION CONSOLE" in c.get("/").data


@pytest.mark.skipif(not (SCENARIO.exists() and V1.exists()), reason="rendered scenario or saved detector not present")
def test_live_run_produces_detections_and_tracks_without_truth_input():
    from app import engine
    info = next(m for m in engine.models() if m["id"] == "v1:rgb_lwir")
    run = engine.LiveRun(str(SCENARIO.parent), info, dict(threshold_mode="f1", loop=False))
    try:
        snap = None
        for _ in range(240):
            snap = run.snapshot()
            if snap and snap["frame"]["k"] >= 4:
                break
            time.sleep(0.5)
        assert snap is not None, "no frame produced"
        f = snap["frame"]
        assert f["images"]["rgb"].startswith("data:image/jpeg") and f["threshold"] == pytest.approx(info["thresholds"]["f1"])
        assert all(d["score"] >= f["threshold"] for d in f["detections"])
        run.update(dict(rgb_on=False))                  # band switched off -> the model still runs (presence flags)
        assert run.settings["rgb_on"] is False
    finally:
        run.stop()
