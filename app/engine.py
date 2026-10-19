"""Live simulation engine of the KAL detection console (M1 MVP).

Every frame is produced live from the rendered scene layers, exactly as in the experiment:

    rendered sky + target layers (render/track_sequence.py; the only precomputed part)
      -> haze, LWIR water vapour, optical blur                     (data/composite.py, render/psf.py)
      -> camera: RGB shot/read noise; LWIR stripes, drift, noise   (StreamCamera, tools/m1_det2_eval.py)
      -> 8-bit video -> destripe + standardise                     (models/cnn_detector.preprocess)
      -> CNN detector -> frozen threshold -> multi-target tracker  (eval/tracking.py)

The detector never sees labels. Truth boxes are drawn only when "show truth" is on, for evaluation.
Two threads: one makes camera frames, the other detects and tracks, so the slower stage sets the rate.
"""
from __future__ import annotations

import base64
import io
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

THRESHOLD_MODES = {"fa01": "Low false alarm (0.01 per frame, frozen on validation)",
                   "fa05": "Balanced (0.05 per frame, frozen on validation)",
                   "f1": "Best F1 on validation", "custom": "Custom"}


# ============================================================================ catalogue
def scenarios():
    """Rendered flight sequences: out/m1/det2/flights/* plus the day clip of the video."""
    out = []
    for d in sorted((ROOT / "out/m1/det2/flights").glob("*")) + [ROOT / "out/video_v2/sensor"]:
        p = d / "plan.json"
        if not p.exists():
            continue
        plan = json.loads(p.read_text())
        f0, f1 = plan["frames"][0], plan["frames"][-1]
        name = d.name if d.name != "sensor" else "day_approach"
        out.append(dict(id=name, path=str(d), lighting=plan["lighting"], frames=len(plan["frames"]), fps=plan["fps"],
                        speed_mps=plan["speed_mps"], range_m=[round(f0["range_m"]), round(f1["range_m"])],
                        size_px=[round(f0["size_px"], 1), round(f1["size_px"], 1)],
                        air_temp_c=round(plan["air_temp_c"], 1)))
    return out


LIVE_BANDS = {"rgb", "lwir"}          # what the live video renders: the deployable cameras (RGB + uncooled LWIR)


def models():
    """Saved detectors that run on the live video's cameras: M1 runs whose bands are RGB and/or uncooled LWIR (the
    four-sensor teacher and the cooled-LWIR / MWIR references need cameras the video does not render), then the
    superseded v1 baseline if present."""
    import torch
    found = []
    for p in sorted((ROOT / "out/m1/det2/runs").glob("*/best.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if set(ck["config"]["bands"]) <= LIVE_BANDS:
            found.append(_model_info(p, ck, "v2"))
    v1 = ROOT / "out/m1/cnn/runs/rgb_lwir/best.pt"
    if v1.exists():
        ck = torch.load(v1, map_location="cpu", weights_only=False)
        found.append(_model_info(v1, ck, "v1 (superseded)"))
    return found


def _model_info(path, ck, version):
    thr = {"fa01": ck.get("threshold_fa01", ck.get("threshold_fa")), "fa05": ck.get("threshold_fa05", ck.get("threshold_f1")),
           "f1": ck.get("threshold_f1")}
    name = ck.get("name", path.parent.name)
    return dict(id=f"{version.split()[0]}:{name}", name=name, version=version, path=str(path), bands=ck["config"]["bands"],
                deployable=set(ck["config"]["bands"]) <= LIVE_BANDS, thresholds=thr,
                val=dict(froc=ck.get("val_froc"), ap50=ck.get("val_ap50")), step=ck.get("step"))


# ============================================================================ scene source
class Scenario:
    """A rendered flight sequence. radiance(k) composes frame k live, exactly as the video tests do.
    (A 1/4-resolution sky shortcut was measured at up to 32 mK error in LWIR for little speed gain,
    so it is not offered.)"""

    def __init__(self, path):
        from kal_video import Clip
        self.clip = Clip(Path(path))
        self.plan = self.clip.plan
        self.n = len(self.plan["frames"])
        self.H, self.W = self.clip.H, self.clip.W

    def radiance(self, k, band, target=True):
        from data import composite as cp
        from kal_video import reproject
        from render import psf
        from thermal import atmosphere
        c = self.clip
        meta = self.plan["frame_meta"][k]
        sky = reproject(c.pano[band]["sky"], c.pano_cam, c.camera(k))
        objects = cp.load_layers(str(c.dir / "layers" / band / f"f{k:04d}.npz"))[1] if target else []
        s0 = self.plan["psf"][band]["sigma0"]
        w = meta["extra"]["weather"]
        wv = atmosphere.scene_params(meta, c.cfgs["render"]["weather"].get("lwir_water_vapour")) if band == "lwir" else None
        img = cp.composite(dict(sky=sky, psf_sigma0=s0), objects, cp.extinction(w["visibility_km"], w["band_ratio"][band]),
                           w.get("aerosol_scale_height_m"), wv=wv)
        return psf.gaussian_blur(img, psf.extra_sigma(meta["extra"]["psf"][band]["nominal"], s0))

    def truth(self, k):
        f = self.plan["frames"][k]
        b = f["bbox_px"]
        return dict(box=[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], range_m=f["range_m"], size_px=f["size_px"],
                    t=f["t"], heading_deg=f.get("heading_deg"), elevation_deg=f.get("elevation_deg"))


# ============================================================================ detector
class Detector:
    def __init__(self, info):
        from models import cnn_detector as cd
        self.cd = cd
        self.info = info
        self.model, self.ck = cd.load(info["path"])
        self.version = self.ck.get("preprocess")

    def __call__(self, frames, present):
        bands = self.model.bands
        inp = self.cd.preprocess({b: frames[b] for b in bands}, self.version)
        pres = None
        if self.model.presence:
            pres = {b: (1 if present.get(b, True) else 0) for b in bands}
        elif not all(present.get(b, True) for b in bands):
            return None                            # a band this model needs is switched off
        return self.cd.predict(self.model, inp, present=pres)


# ============================================================================ live run
class LiveRun:
    def __init__(self, scenario_path, model_info, settings):
        import torch
        from eval.tracking import MultiTracker
        from m1_det2 import cfgs
        from m1_det2_eval import StreamCamera
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4) // 2))
        self.scn = Scenario(scenario_path)
        self.det = Detector(model_info)
        self.settings = dict(settings)
        self.cam = StreamCamera(cfgs()["sensors"], 2026, self.scn.H, self.scn.W)
        self.base = dict(self.cam.p)
        self.fixed = dict(col=self.cam.col.copy(), pix=self.cam.pix.copy(), gain=self.cam.gain.copy())
        self.apply_camera_settings()
        self.tracker = MultiTracker()
        self.q = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.latest = None
        self.stats = dict(frames=0, first_confirm=None, frames_on_target=0, frames_after_first=0, false_track_ids=[],
                          fps=0.0, history=[])
        self.running = True
        self.k = 0
        self.t_last = time.time()
        threading.Thread(target=self._produce, daemon=True).start()
        threading.Thread(target=self._consume, daemon=True).start()

    # -------------------------------------------------------------------- settings
    def apply_camera_settings(self):
        """Scale the thermal noise and stripe amplitudes of the same camera (fixed patterns kept)."""
        n, s = float(self.settings.get("thermal_noise", 1.0)), float(self.settings.get("stripes", 1.0))
        p = self.cam.p
        p["netd_k"] = self.base["netd_k"] * n
        for k in ("row_k", "col_t_k", "col_drift_k"):
            if k in self.base:
                p[k] = self.base[k] * s
        self.cam.col = self.fixed["col"] * s
        self.cam.gain = self.fixed["gain"] * s

    def update(self, settings):
        with self.lock:
            self.settings.update(settings)
            self.apply_camera_settings()

    def stop(self):
        self.running = False

    # -------------------------------------------------------------------- threads
    def _produce(self):
        while self.running:
            k = self.k
            if k >= self.scn.n:
                if self.settings.get("loop", True):
                    self._reset_tracks()
                    k = self.k = 0
                else:
                    time.sleep(0.2)
                    continue
            self.k = k + 1
            with self.lock:
                target = not bool(self.settings.get("target_free", False))
            fm = self.scn.plan["frame_meta"][k]
            rad = {b: self.scn.radiance(k, b, target) for b in ("rgb", "lwir")}
            frames = dict(rgb=self.cam.rgb(rad["rgb"], fm["camera"]["exposure_ms"]),
                          lwir=self.cam.lwir(rad["lwir"][..., 0], fm["conditions"]["air_temp_c"] + 273.15))
            self.q.put((k, frames, target))

    def _reset_tracks(self):
        from eval.tracking import MultiTracker
        self.tracker = MultiTracker()
        self.stats.update(first_confirm=None, frames_on_target=0, frames_after_first=0, false_track_ids=[], history=[])

    def _threshold(self):
        mode = self.settings.get("threshold_mode", "fa01")
        if mode == "custom":
            return float(self.settings.get("threshold", 0.5))
        t = self.det.info["thresholds"].get(mode)
        return float(t if t is not None else 0.5)

    def _consume(self):
        while self.running:
            try:
                k, frames, target = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.lock:
                present = dict(rgb=bool(self.settings.get("rgb_on", True)), lwir=bool(self.settings.get("lwir_on", True)))
                thr = self._threshold()
            disp = dict(rgb=frames["rgb"] if present["rgb"] else np.zeros_like(frames["rgb"]),
                        lwir=frames["lwir"] if present["lwir"] else np.zeros_like(frames["lwir"]))
            t0 = time.time()
            pred = self.det(frames, present)
            dt = time.time() - t0
            dets = [] if pred is None else [(b, s) for b, s in zip(pred["boxes"], pred["scores"]) if s >= thr]
            best = 0.0 if pred is None or not pred["scores"] else float(max(pred["scores"]))
            tracks = self.tracker.step(dets)
            truth = self.scn.truth(k)
            on = self._score_tracks(tracks, truth, target, k)
            now = time.time()
            fps = 1.0 / max(1e-3, now - self.t_last)
            self.t_last = now
            st = self.stats
            st["frames"] += 1
            st["fps"] = 0.8 * st["fps"] + 0.2 * fps if st["fps"] else fps
            st["history"] = (st["history"] + [dict(k=k, t=truth["t"], best=best, thr=thr, det=bool(dets), on=on)])[-300:]
            frame = dict(k=k, n=self.scn.n, t=truth["t"], threshold=thr, band_missing=pred is None,
                         detections=[dict(box=[float(v) for v in b], score=float(s)) for b, s in dets],
                         tracks=[dict(id=t.id, box=[float(v) for v in t.box()], confirmed=t.confirmed, coasting=t.misses,
                                      score=float(t.score)) for t in tracks],
                         truth=truth if target else None, target_present=target, detector_ms=dt * 1000,
                         images=dict(rgb=_jpeg(disp["rgb"]), lwir=_jpeg(disp["lwir"] ** 0.8)))
            with self.lock:
                self.latest = frame

    def _score_tracks(self, tracks, truth, target, k):
        st = self.stats
        tb = truth["box"]
        tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
        tol = max(4.0, 0.5 * max(tb[2] - tb[0], tb[3] - tb[1]))
        on = False
        for t in tracks:
            if not t.confirmed:
                continue
            if target and math.hypot(t.cx - tcx, t.cy - tcy) <= tol:
                on = True
            elif t.id not in st["false_track_ids"]:
                st["false_track_ids"].append(t.id)
        if target:
            if on and st["first_confirm"] is None:
                st["first_confirm"] = dict(k=k, t=truth["t"], range_m=truth["range_m"], size_px=truth["size_px"])
            if st["first_confirm"] is not None:
                st["frames_after_first"] += 1
                st["frames_on_target"] += on
        return on

    def snapshot(self, after=-1):
        with self.lock:
            f = self.latest
            if f is None or f["k"] == after:
                return None
            st = self.stats
            return dict(frame=f, stats=dict(frames=st["frames"], fps=st["fps"], first_confirm=st["first_confirm"],
                                            tracked_fraction=st["frames_on_target"] / max(1, st["frames_after_first"]),
                                            false_tracks=len(st["false_track_ids"]), history=st["history"]),
                        settings=self.settings)


def _jpeg(img, q=88):
    a = np.asarray(img, np.float32)
    a = a if a.ndim == 3 else a[..., None]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, -1)
    buf = io.BytesIO()
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ============================================================================ held-out evidence
def _test_cohort(m):
    """The held-out test scenes: 'test' (four-sensor M1), 'fresh_test' in older manifests."""
    return m["cohorts"].get("test", m["cohorts"].get("fresh_test", []))


def evidence_scenes():
    """Held-out test scenes of the M1 detector (only after its evaluation exists)."""
    man = ROOT / "out/m1/det2/manifest.json"
    if not man.exists() or not (ROOT / "out/m1/det2/test_lock.json").exists():
        return []
    m = json.loads(man.read_text())
    out = []
    for sid in _test_cohort(m):
        meta = json.loads((ROOT / "out/m1/dataset/meta" / f"{sid}.json").read_text())
        o = meta["extra"]["objects"][0]
        out.append(dict(id=sid, lighting=meta["conditions"]["lighting"], size_px=round(o["size_px"], 1), size_bin=o["size_bin"],
                        range_m=round(o["range_m"]), aspect_deg=round(o.get("aspect_deg", -1))))
    return out


def evidence_run(sid, model_info):
    """Rebuild the exact test frame of a scene from its rendered layers (same seeds as the evaluation),
    run the saved weights, and check the result against the stored evaluation."""
    import m1_det2 as D
    det = Detector(model_info)
    meta = D.meta_of(sid)
    frames, boxes, cov = D._scene_frames(meta, D.TEST_PRESENT[0], False, 0, absent=False)
    from models import cnn_detector as cd
    inp = {b: frames[b] for b in det.model.bands}                    # already preprocessed (v2)
    masks = cd.coverage_masks(cov, inp[det.model.bands[0]].shape[:2], det.model.bands) if det.model.presence else None
    pred = cd.predict(det.model, inp, masks=masks)
    stored, match = None, None
    pp = ROOT / "out/m1/det2/predictions.json"
    if pp.exists():
        allp = json.loads(pp.read_text())
        for r in allp.get("test", allp.get("fresh_test", {})).get(model_info["name"], {}).get("present", []):
            if r["scene"] == sid:
                stored = r["preds"]
                match = bool(np.allclose(np.asarray(stored["scores"]), np.asarray(pred["scores"]), atol=1e-5))
    thr = model_info["thresholds"]["fa01"]
    disp = {b: _jpeg(_display(frames[b])) for b in ("rgb", "lwir")}
    return dict(scene=sid, truth=boxes, threshold=thr, matches_saved_evaluation=match,
                detections=[dict(box=b, score=s) for b, s in zip(pred["boxes"], pred["scores"]) if s >= thr],
                all_scores=pred["scores"][:10], images=disp)


def _display(x):
    x = np.clip(np.asarray(x, np.float32) * 10.0, -6, 18)
    return (x + 6) / 24.0
