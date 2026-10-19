"""KAL detection video: sensor frames -> trained detector -> tracker -> console -> MP4.

    python tools/kal_video.py sensor  [--model out/m1/cnn/runs/student/best.pt]
    python tools/kal_video.py console
    python tools/kal_video.py final

sensor   composes every frame of render/track_sequence.py exactly like the data loader's evaluation
         path (haze, LWIR water vapour, PSF; data/dataset.py), applies the repository sensor models with
         one set of camera parameters for the whole clip (fixed LWIR column/pixel pattern, fresh
         temporal noise, smoothed AGC), runs the saved CNN on the full 640 x 512 RGB + LWIR frames
         and links detections into a track. The detector never sees labels, SWIR or the plan.
console  lays out the EO zoom view (tools/render_kal_video.py) and both sensor feeds with the
         detector's boxes. Detection boxes come from the detector only; truth is used afterwards for
         the clip statistics on the end card.
final    title, cinematic shots, console and end card -> out/video/KAL_detection_simulation.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target")]

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

from data import composite as cp
from noise import lwir_fpn, rgb_noise
from render import placement as pl
from render import psf
from render.sampling import load_configs
from thermal import atmosphere

OUT = ROOT / "out" / "video"
VW, VH = 1920, 1080
BG = (9, 12, 15)
INK = (232, 236, 239)
MUTED = (140, 152, 160)
DIM = (82, 92, 100)
ORANGE = (255, 156, 40)
GREEN = (70, 225, 120)
AMBER = (255, 196, 60)
RED = (240, 80, 70)
CYAN = (90, 200, 230)


def filmic(im, seed, grain=1.8, vignette=0.10):
    """Camera-like finish for rendered shots: fine luminance grain and a slight vignette."""
    a = np.asarray(im, np.float32)
    H, W = a.shape[:2]
    rng = np.random.default_rng(seed)
    a = a + rng.normal(0, grain, (H, W, 1)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    a = a * (1 - vignette * np.clip(r2 - 0.25, 0, None)[..., None] / 1.75)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def font(size, bold=False, mono=False):
    names = (["consolab.ttf" if bold else "consola.ttf"] if mono else []) + (
        ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"])
    for n in names:
        p = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / n
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


# ============================================================================ sensor composition
class SensorSim:
    """noise/rgb_noise.py and noise/lwir_fpn.py with one camera for the whole clip."""

    def __init__(self, S, seed, W, H):
        self.S = S
        rng = np.random.default_rng([seed, 77])
        self.p_rgb = rgb_noise.sample_params(rng, S["rgb_noise"])
        self.p_lw = lwir_fpn.sample_params(rng, S["lwir_noise"])
        self.col = rng.standard_normal(W) * self.p_lw["col_k"]
        self.pix = rng.standard_normal((H, W)) * self.p_lw["pix_k"]
        self.agc = None
        self.disp = None

    def rgb(self, L, rng, exp_ms):
        return rgb_noise.apply(L, rng, self.S["rgb_noise"], exp_ms, params=self.p_rgb)[0]

    def lwir(self, L, rng, T_air_k):
        """Returns (detector input: percentile AGC as in the data loader, display: the same with gamma 0.8)."""
        p, cfg = self.p_lw, self.S["lwir_noise"]
        cpk = p["counts_per_k"] * lwir_fpn._ref_scale(T_air_k)
        c = lwir_fpn.to_counts(L, T_air_k, p) + (self.col[None, :] + self.pix) * cpk
        c = c + rng.standard_normal(c.shape) * p["netd_k"] * cpk
        c = np.clip(np.round(c), 0, 2 ** int(cfg["bits"]) - 1)
        lo, hi = np.percentile(c, [p["agc_lo"], p["agc_hi"]])
        self.agc = (lo, hi) if self.agc is None else (0.9 * self.agc[0] + 0.1 * lo, 0.9 * self.agc[1] + 0.1 * hi)
        det = np.clip((c - self.agc[0]) / max(self.agc[1] - self.agc[0], 1.0), 0.0, 1.0)
        # display: the same AGC output the detector receives, with a mild gamma for the monitor
        disp = det ** 0.8
        return det.astype(np.float32), disp.astype(np.float32)


def reproject(img, pano, cam):
    """Sample a panorama layer (H_p x W_p x C) along the rays of camera `cam` (pixel centres)."""
    u = np.arange(cam.W) + 0.5
    v = np.arange(cam.H) + 0.5
    U, V = np.meshgrid(u, v)
    d = np.stack([(U - cam.W / 2) / cam.f, -(V - cam.H / 2) / cam.f, -np.ones_like(U)], -1) @ cam.R.T
    pc = d @ pano.R
    x = pano.W / 2 + pano.f * pc[..., 0] / -pc[..., 2] - 0.5
    y = pano.H / 2 - pano.f * pc[..., 1] / -pc[..., 2] - 0.5
    img = img if img.ndim == 3 else img[..., None]
    return np.stack([ndi.map_coordinates(img[..., c], [y, x], order=1, mode="nearest") for c in range(img.shape[-1])], -1)


class Clip:
    def __init__(self, sensor_dir: Path, seed=2026):
        self.dir = sensor_dir
        self.plan = json.loads((sensor_dir / "plan.json").read_text())
        self.cfgs = load_configs(overlay="configs/m1.yaml")
        c = self.plan["camera"]
        self.W, self.H = c["width"], c["height"]
        P = self.plan["panorama"]
        self.pano_cam = pl.Camera(P["width"], P["height"], P["hfov_deg"], c["position"], P["azimuth_deg"], P["elevation_deg"])
        self.pano = {b: cp.load_layers(str(sensor_dir / f"pano_{b}.npz"))[0] for b in ("rgb", "lwir")
                     if (sensor_dir / f"pano_{b}.npz").exists()}
        self.sim = SensorSim(self.cfgs["sensors"], seed, self.W, self.H)
        self.rng = np.random.default_rng([seed, 5])

    def camera(self, k):
        f = self.plan["frames"][k]
        c = self.plan["camera"]
        return pl.Camera(self.W, self.H, c["hfov_deg"], c["position"], f["cam_az"], f["cam_el"])

    def radiance(self, k, band):
        meta = self.plan["frame_meta"][k]
        cam = self.camera(k)
        sky = reproject(self.pano[band]["sky"], self.pano_cam, cam)
        objects = cp.load_layers(str(self.dir / "layers" / band / f"f{k:04d}.npz"))[1]
        s0 = self.plan["psf"][band]["sigma0"]
        w = meta["extra"]["weather"]
        wv = atmosphere.scene_params(meta, self.cfgs["render"]["weather"].get("lwir_water_vapour")) if band == "lwir" else None
        img = cp.composite(dict(sky=sky, psf_sigma0=s0), objects, cp.extinction(w["visibility_km"], w["band_ratio"][band]),
                           w.get("aerosol_scale_height_m"), wv=wv)
        return psf.gaussian_blur(img, psf.extra_sigma(meta["extra"]["psf"][band]["nominal"], s0))

    def frame(self, k):
        meta = self.plan["frame_meta"][k]
        rgb = self.sim.rgb(self.radiance(k, "rgb"), self.rng, meta["camera"]["exposure_ms"])
        lw_det, lw_disp = self.sim.lwir(self.radiance(k, "lwir")[..., 0], self.rng, meta["conditions"]["air_temp_c"] + 273.15)
        return dict(rgb=rgb, lwir=lw_det[..., None]), dict(rgb=rgb, lwir=lw_disp)


class Tracker:
    """One-target track from detections: confirm on 3 hits in 5 frames, alpha-beta update, coast up
    to 15 frames. Display logic only; it does not feed back into detection."""

    def __init__(self):
        self.state = None            # (cx, cy, w, h, vx, vy)
        self.hist = []
        self.misses = 0
        self.confirmed = False

    def step(self, dets):
        pred = None
        if self.state is not None:
            cx, cy, w, h, vx, vy = self.state
            pred = (cx + vx, cy + vy, w, h)
        best = None
        for b, s in dets:
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if pred is not None:
                gate = max(24.0, 2.5 * max(pred[2], pred[3]))
                if math.hypot(c[0] - pred[0], c[1] - pred[1]) > gate:
                    continue
            if best is None or s > best[1]:
                best = (b, s)
        if best is not None:
            b, s = best
            cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
            if self.state is None:
                self.state = (cx, cy, w, h, 0.0, 0.0)
            else:
                px, py, pw, ph = pred
                rx, ry = cx - px, cy - py
                vx, vy = self.state[4] + 0.15 * rx, self.state[5] + 0.15 * ry
                self.state = (px + 0.6 * rx, py + 0.6 * ry, 0.6 * w + 0.4 * pw, 0.6 * h + 0.4 * ph, vx, vy)
            self.misses = 0
            self.hist.append(1)
        else:
            self.hist.append(0)
            if self.state is not None:
                self.misses += 1
                cx, cy, w, h, vx, vy = self.state
                self.state = (cx + vx, cy + vy, w, h, vx, vy)
                if self.misses > 15:
                    self.state, self.confirmed = None, False
        if sum(self.hist[-5:]) >= 3:
            self.confirmed = True
        return dict(confirmed=self.confirmed and self.state is not None, detected=best is not None,
                    box=None if self.state is None else [self.state[0] - self.state[2] / 2, self.state[1] - self.state[3] / 2,
                                                         self.state[0] + self.state[2] / 2, self.state[1] + self.state[3] / 2],
                    score=None if best is None else float(best[1]), coasting=self.misses)


def cmd_sensor(a):
    import torch
    from models import cnn_detector as cd
    torch.set_num_threads(2)
    sd = Path(a.sensor)
    clip = Clip(sd)
    model, ck = cd.load(a.model)
    thr = float(ck[a.threshold])
    n = len(clip.plan["frames"])
    out = sd / "composed"
    for b in ("rgb", "lwir"):
        (out / b).mkdir(parents=True, exist_ok=True)
    trk = Tracker()
    rows = []
    for k in range(n):
        det_in, disp = clip.frame(k)
        frames = {b: cd.standardise(det_in[b]) for b in model.bands}
        p = cd.predict(model, frames)
        dets = [(b, s) for b, s in zip(p["boxes"], p["scores"]) if s >= thr]
        st = trk.step(dets)
        Image.fromarray((np.clip(disp["rgb"], 0, 1) * 255 + 0.5).astype(np.uint8)).save(out / "rgb" / f"{k:04d}.png")
        Image.fromarray((np.clip(disp["lwir"], 0, 1) * 255 + 0.5).astype(np.uint8)).save(out / "lwir" / f"{k:04d}.png")
        f = clip.plan["frames"][k]
        rows.append(dict(k=k, t=f["t"], preds=p, threshold=thr, track=st, truth=f["bbox_px"], range_m=f["range_m"],
                         size_px=f["size_px"]))
        if k % 25 == 0:
            print(f"frame {k}/{n}: {len(dets)} detections >= {thr:.2f}, best "
                  f"{max(p['scores']) if p['scores'] else 0:.2f}, track {'ON' if st['confirmed'] else 'off'}", flush=True)
    (sd / "detections.json").write_text(json.dumps(dict(model=str(a.model), threshold_key=a.threshold, threshold=thr,
                                                         bands=model.bands, checkpoint=ck.get("name"), frames=rows)), encoding="utf-8")
    stats(sd)


def stats(sd: Path):
    """Clip statistics against the truth box, computed after detection (not shown to the detector)."""
    from eval.nwd import iou_matrix
    d = json.loads((sd / "detections.json").read_text())
    hit, ious, first = 0, [], None
    fp = 0
    for r in d["frames"]:
        t = r["truth"]
        tb = np.array([[t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]]])
        P = [(b, s) for b, s in zip(r["preds"]["boxes"], r["preds"]["scores"]) if s >= d["threshold"]]
        ok = False
        for b, s in P:
            iou = float(iou_matrix(np.array([b]), tb)[0, 0])
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            inside = t["x"] - 2 <= c[0] <= t["x"] + t["w"] + 2 and t["y"] - 2 <= c[1] <= t["y"] + t["h"] + 2
            if iou >= 0.5 or inside:
                ok = True
                ious.append(iou)
            else:
                fp += 1
        if ok:
            hit += 1
            if first is None:
                first = r
    n = len(d["frames"])
    s = dict(frames=n, detected_frames=hit, detected_fraction=hit / n, false_positive_boxes=fp,
             false_positives_per_frame=fp / n, mean_iou_of_hits=float(np.mean(ious)) if ious else None,
             first_detection=dict(t=first["t"], range_m=first["range_m"], size_px=first["size_px"]) if first else None,
             note="hit = prediction at the frozen threshold with IoU >= .5 or centre within the truth box (+2 px); "
                  "truth is used only here, after detection")
    (sd / "clip_stats.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))
    return s


# ============================================================================ console layout
def panel_title(d, x, y, text, sub=None):
    d.text((x, y), text, fill=INK, font=font(17, True))
    if sub:
        w = d.textlength(text, font=font(17, True))
        d.text((x + w + 12, y + 2), sub, fill=MUTED, font=font(14))


def corners(d, box, color, L=10, w=2):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x0 + L, y0), (x0, y0, x0, y0 + L), (x1, y0, x1 - L, y0), (x1, y0, x1, y0 + L),
                             (x0, y1, x0 + L, y1), (x0, y1, x0, y1 - L), (x1, y1, x1 - L, y1), (x1, y1, x1, y1 - L)):
        d.line((ax, ay, bx, by), fill=color, width=w)


def sensor_panel(canvas, img8, x, y, w, h, preds, thr, track, label, sub, zoom_corner="br"):
    d = ImageDraw.Draw(canvas)
    im = Image.fromarray(img8).convert("RGB")
    sx, sy = w / im.width, h / im.height
    canvas.paste(im.resize((w, h), Image.Resampling.BILINEAR), (x, y))
    d.rectangle((x - 1, y - 1, x + w, y + h), outline=(40, 48, 55))
    panel_title(d, x, y - 28, label, sub)
    shown = [(b, s) for b, s in zip(preds["boxes"], preds["scores"]) if s >= thr]
    for b, s in shown:
        bx = (x + b[0] * sx, y + b[1] * sy, x + b[2] * sx, y + b[3] * sy)
        pad = 6
        box = (bx[0] - pad, bx[1] - pad, bx[2] + pad, bx[3] + pad)
        corners(d, box, ORANGE, L=7, w=2)
        d.text((box[0], box[1] - 19), f"KAL {s:.2f}", fill=ORANGE, font=font(14, True, True))
    # magnifier on the detection (or the coasting track) - display only
    src = shown[0][0] if shown else (track["box"] if track.get("box") else None)
    if src is not None:
        cx, cy = (src[0] + src[2]) / 2, (src[1] + src[3]) / 2
        win = max(28, int(max(src[2] - src[0], src[3] - src[1]) * 2.2))
        x0 = int(np.clip(cx - win / 2, 0, im.width - win))
        y0 = int(np.clip(cy - win / 2, 0, im.height - win))
        Z = 150
        crop = im.crop((x0, y0, x0 + win, y0 + win)).resize((Z, Z), Image.Resampling.NEAREST)
        zx, zy = (x + w - Z - 8, y + h - Z - 8) if zoom_corner == "br" else (x + 8, y + h - Z - 8)
        canvas.paste(crop, (zx, zy))
        d.rectangle((zx - 1, zy - 1, zx + Z, zy + Z), outline=ORANGE if shown else AMBER, width=1)
        d.text((zx + 4, zy + 2), f"x{Z / win:.0f}", fill=ORANGE if shown else AMBER, font=font(12, True, True))


def map_box(box, cam_s, cam_z):
    """Sensor-frame box -> zoom-frame box through the shared viewpoint (rays, no depth needed)."""
    pts = []
    for u, v in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
        r = cam_s.ray(u, v)
        uu, vv, z = cam_z.project(cam_z.C + r[None] * 1000.0)
        pts.append((float(uu[0]), float(vv[0])))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def sparkline(d, x, y, w, h, scores, thr, k, n):
    d.rectangle((x, y, x + w, y + h), outline=(36, 44, 50))
    yt = y + h - thr * h
    for xx in range(x, x + w, 8):
        d.line((xx, yt, xx + 4, yt), fill=DIM, width=1)
    d.text((x + w - 70, yt - 16), f"thr {thr:.2f}", fill=MUTED, font=font(12, mono=True))
    pts = []
    for i in range(k + 1):
        s = scores[i]
        pts.append((x + i / (n - 1) * w, y + h - (s if s is not None else 0) * h))
    for (ax, ay), (bx, by), i in zip(pts, pts[1:], range(1, k + 1)):
        col = GREEN if (scores[i] or 0) >= thr else AMBER
        d.line((ax, ay, bx, by), fill=col, width=2)
    d.text((x, y - 20), "Detector confidence (best box per frame)", fill=MUTED, font=font(13))


def cmd_console(a):
    sd = Path(a.sensor)
    hd = Path(a.hero)
    plan = json.loads((sd / "plan.json").read_text())
    det = json.loads((sd / "detections.json").read_text())
    zp = json.loads((hd / "zoom_plan.json").read_text())
    out = OUT / "console"
    out.mkdir(parents=True, exist_ok=True)
    n = len(plan["frames"])
    c = plan["camera"]
    ck = json.loads((Path(det["model"]).parent / "training.json").read_text())
    thr = float(ck[a.threshold])
    first, last = a.first, min(a.last, n) if a.last else n
    # the tracker is display logic: recompute it at the chosen threshold, warmed up from frame 0
    trk, tracks = Tracker(), []
    for r in det["frames"]:
        tracks.append(trk.step([(b, sc) for b, sc in zip(r["preds"]["boxes"], r["preds"]["scores"]) if sc >= thr]))
    best = [max(r["preds"]["scores"]) if r["preds"]["scores"] else 0.0 for r in det["frames"]]
    vis = plan["weather"]["visibility_km"]
    for k in range(first, last):
        path = out / f"{k:04d}.png"
        if path.exists() and not a.overwrite:
            continue
        f, r, z = plan["frames"][k], det["frames"][k], zp["frames"][k]
        canvas = Image.new("RGB", (VW, VH), BG)
        d = ImageDraw.Draw(canvas)
        # header
        d.rectangle((0, 0, VW, 56), fill=(14, 19, 23))
        d.text((24, 12), "KAL DETECTION", fill=INK, font=font(26, True))
        d.text((236, 19), "simulation  |  RGB + thermal sensor head  |  AI detector  |  zoom camera confirmation", fill=MUTED, font=font(16))
        d.text((VW - 520, 10), f"T+{f['t']:05.2f} s", fill=INK, font=font(24, True, True))
        d.text((VW - 360, 17), f"{plan['lighting'].upper()}  CLEAR  VIS {vis:.0f} km  AIR {plan['air_temp_c']:.0f} C",
               fill=MUTED, font=font(15))
        # zoom view
        zx, zy, zw, zh = 16, 96, 1280, 720
        zimg = Image.open(hd / "zoom" / f"{k:04d}.png").convert("RGB")
        if zimg.size != (zw, zh):
            zimg = zimg.resize((zw, zh), Image.Resampling.LANCZOS)
        canvas.paste(filmic(zimg, k), (zx, zy))
        panel_title(d, zx, zy - 30, "ZOOM CAMERA", f"visual confirmation  |  field of view {z['hfov']:.2f} deg")
        cam_s = pl.Camera(c["width"], c["height"], c["hfov_deg"], c["position"], f["cam_az"], f["cam_el"])
        cam_z = pl.Camera(zw, zh, z["hfov"], zp["camera_position"], z["az"], z["el"])
        tr = tracks[k]
        shown = [(b, s) for b, s in zip(r["preds"]["boxes"], r["preds"]["scores"]) if s >= thr]
        # zoom-view reticle
        cxz, cyz = zx + zw / 2, zy + zh / 2
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d.line((cxz + dx * 18, cyz + dy * 18, cxz + dx * 40, cyz + dy * 40), fill=(200, 210, 215), width=1)
        if tr["box"] is not None and tr["confirmed"]:
            bz = map_box(tr["box"] if not shown else shown[0][0], cam_s, cam_z)
            pad = 18
            box = (zx + bz[0] - pad, zy + bz[1] - pad, zx + bz[2] + pad, zy + bz[3] + pad)
            col = GREEN if shown else AMBER
            corners(d, box, col, L=26, w=3)
            lab = f"TRK 01  KAL  {shown[0][1]:.2f}" if shown else f"TRK 01  COAST {tr['coasting']}"
            d.rectangle((box[0], box[1] - 30, box[0] + d.textlength(lab, font=font(18, True, True)) + 16, box[1] - 4), fill=(0, 0, 0))
            d.text((box[0] + 8, box[1] - 28), lab, fill=col, font=font(18, True, True))
        az_true, el_true = pl.azimuth_elevation(np.asarray(f["target_matrix"])[:3, 3] - np.asarray(c["position"]))
        d.text((zx + 16, zy + zh - 34), f"RANGE {f['range_m']:5.0f} m   AZ {az_true:6.1f}   EL {el_true:5.1f}   "
               f"SPEED {plan['speed_mps'] * 3.6:.0f} km/h", fill=INK, font=font(18, True, True))
        # status chip
        if tr["confirmed"] and shown:
            chip, ccol = "TRACKING", GREEN
        elif shown:
            chip, ccol = "DETECTED", GREEN
        elif tr["confirmed"]:
            chip, ccol = "COASTING", AMBER
        else:
            chip, ccol = "SEARCHING", AMBER
        d.rounded_rectangle((zx + zw - 190, zy + 14, zx + zw - 16, zy + 50), radius=6, fill=(0, 0, 0), outline=ccol, width=2)
        d.text((zx + zw - 175, zy + 18), chip, fill=ccol, font=font(20, True, True))
        # sensor feeds
        sx, sw, sh = 1320, 584, 467
        rgb8 = np.asarray(Image.open(sd / "composed" / "rgb" / f"{k:04d}.png"))
        lw8 = np.asarray(Image.open(sd / "composed" / "lwir" / f"{k:04d}.png"))
        sensor_panel(canvas, rgb8, sx, 96, sw, sh, r["preds"], thr, tr, "DAYLIGHT CAMERA", "640 x 512  |  detector input")
        sensor_panel(canvas, lw8, sx, 96 + sh + 44, sw, sh, r["preds"], thr, tr, "THERMAL CAMERA", "LWIR 8-14 um, white-hot  |  detector input")
        # telemetry and confidence
        ty = 836
        d.text((16, ty), "DETECTOR", fill=MUTED, font=font(13, True))
        d.text((16, ty + 18), "Neural network on daylight + thermal frames", fill=INK, font=font(17, True))
        d.text((16, ty + 42), f"trained only on synthetic data  |  score threshold {thr:.2f} set on validation scenes", fill=MUTED, font=font(15))
        d.text((16, ty + 76), "TARGET", fill=MUTED, font=font(13, True))
        d.text((16, ty + 94), f"{f['size_px']:.1f} px on the sensor  |  bank {f['bank_deg']:+.0f} deg  |  "
               f"{f['range_m'] / 1000:.2f} km", fill=INK, font=font(17, True))
        sparkline(d, 560, ty + 24, 700, 150, best, thr, k, n)
        d.text((16, VH - 30), "Simulation: rendered scene with modelled camera noise. Boxes are the detector's own output (truth is not drawn). "
               "Camera pointing is scripted.", fill=DIM, font=font(13))
        canvas.save(path)
        if k % 50 == 0:
            print(f"console {k}/{n}", flush=True)


# ============================================================================ cards and final cut
def card(path, lines, sub=None, bg=BG):
    im = Image.new("RGB", (VW, VH), bg)
    d = ImageDraw.Draw(im)
    y = VH // 2 - 30 * len(lines)
    for i, (txt, size, bold, col) in enumerate(lines):
        w = d.textlength(txt, font=font(size, bold))
        d.text(((VW - w) / 2, y), txt, fill=col, font=font(size, bold))
        y += int(size * 1.5)
    if sub:
        w = d.textlength(sub, font=font(16))
        d.text(((VW - w) / 2, VH - 70), sub, fill=DIM, font=font(16))
    im.save(path)


def upscale_dir(src: Path, dst: Path, overlay=None):
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    for i, p in enumerate(files):
        q = dst / p.name
        if q.exists():
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (VW, VH):
            im = im.resize((VW, VH), Image.Resampling.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
        if overlay:
            d = ImageDraw.Draw(im)
            d.text((40, VH - 64), overlay[0], fill=(235, 238, 240), font=font(26, True))
            d.text((40, VH - 32), overlay[1], fill=(200, 205, 210), font=font(16))
        im.save(q)
    return len(files)


def encode(frames_glob_dir: Path, fps, out: Path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_glob_dir / "%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def cmd_final(a):
    sd, hd = Path(a.sensor), Path(a.hero)
    plan = json.loads((sd / "plan.json").read_text())
    fps = int(round(plan["fps"]))
    st = json.loads((sd / "clip_stats.json").read_text())
    det = json.loads((sd / "detections.json").read_text())
    seq = OUT / "sequence"
    if seq.exists():
        for p in seq.glob("*.png"):
            p.unlink()
    seq.mkdir(parents=True, exist_ok=True)
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    k = 0

    def add(img_path, repeat=1):
        nonlocal k
        for _ in range(repeat):
            dst = seq / f"{k:05d}.png"
            try:
                os.link(img_path, dst)
            except OSError:
                Image.open(img_path).save(dst)
            k += 1

    card(work / "title.png", [("KAL", 96, True, INK), ("Multi-spectral drone detection  -  M1 simulation", 30, False, MUTED),
                              ("RGB + LWIR sensor head, trained CNN detector, EO zoom confirmation", 22, False, DIM)],
         sub="Synthetic scene rendered with the project pipeline. MD-550 / L550E engine model. Not real flight footage.")
    add(work / "title.png", int(2.5 * fps))
    for shot, ov in (("intro", ("KAL in flight", "presentation model: dimensions from the project geometry; MD-550 / L550E engine, 44 m/s")),
                     ("flyby", ("Low pass over the range", "ground observer, 1 ms shutter: propeller blades nearly frozen (stroboscopic alias)"))):
        if (hd / shot).exists():
            n = upscale_dir(hd / shot, work / shot, ov)
            for p in sorted((work / shot).glob("*.png")):
                add(p)
    card(work / "console_card.png", [("Detection", 64, True, INK),
                                     ("The S1 sensor head stares, then tracks; a CNN trained on synthetic RGB + LWIR data", 24, False, MUTED),
                                     ("finds the KAL in every frame it can, without SWIR, labels or the flight plan", 24, False, MUTED)])
    add(work / "console_card.png", int(2.0 * fps))
    for p in sorted((OUT / "console").glob("*.png")):
        add(p)
    fd = st.get("first_detection") or {}
    card(work / "end.png", [("Clip result (computed against simulation truth after detection)", 26, True, INK),
                            (f"Detected in {st['detected_frames']} of {st['frames']} frames ({100 * st['detected_fraction']:.0f} %)", 30, True, GREEN),
                            (f"First detection at {fd.get('range_m', float('nan')):.0f} m ({fd.get('size_px', float('nan')):.1f} px)   |   "
                             f"false boxes per frame {st['false_positives_per_frame']:.3f}", 24, False, MUTED),
                            ("Held-out test results, method and limitations: out/m1/cnn/results.md", 20, False, DIM)],
         sub=f"Detector {det['checkpoint']} (saved weights); one synthetic clip is a demonstration, not an accuracy measurement.")
    add(work / "end.png", int(4.0 * fps))
    out = OUT / "KAL_detection_simulation.mp4"
    encode(seq, fps, out)
    print(f"wrote {out} ({k} frames, {k / fps:.1f} s)")


def cmd_short(a):
    """10-second cut: the sunlit chase shot, then the detection console, with crossfades."""
    hd = Path(a.hero)
    plan = json.loads((Path(a.sensor) / "plan.json").read_text())
    fps = int(round(plan["fps"]))
    seq = OUT / "short_seq"
    if seq.exists():
        for p in seq.glob("*.png"):
            p.unlink()
    seq.mkdir(parents=True, exist_ok=True)
    intro = sorted((hd / "intro").glob("*.png"))
    cons = [OUT / "console" / f"{k:04d}.png" for k in range(a.first, a.last)]
    frames = []
    for i, p in enumerate(intro):
        im = Image.open(p).convert("RGB")
        if im.size != (VW, VH):
            im = im.resize((VW, VH), Image.Resampling.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
        im = filmic(im, 10_000 + i)
        d = ImageDraw.Draw(im)
        d.text((40, 32), "KAL  |  simulated flight", fill=(225, 228, 230), font=font(22, True))
        frames.append(im)
    fade = int(0.5 * fps)
    for j, p in enumerate(cons):
        im = Image.open(p).convert("RGB")
        if j < fade and frames:
            im = Image.blend(frames[len(frames) - fade + j] if len(frames) >= fade else frames[-1], im, (j + 1) / fade)
        frames.append(im)
    for k, im in enumerate(frames):
        im.save(seq / f"{k:05d}.png")
    out = OUT / "KAL_detection_10s.mp4"
    encode(seq, fps, out)
    print(f"wrote {out} ({len(frames)} frames, {len(frames) / fps:.1f} s)")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sensor", "console", "final", "stats", "short"])
    ap.add_argument("--out", default=str(OUT), help="folder for console frames and videos")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--sensor", default=str(OUT / "sensor"))
    ap.add_argument("--hero", default=str(OUT / "hero"))
    ap.add_argument("--model", default=str(ROOT / "out/m1/cnn/runs/student/best.pt"))
    ap.add_argument("--threshold", default="threshold_fa", choices=["threshold_fa", "threshold_f1"],
                    help="sensor: threshold stored with the detections; console: operating point shown")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    OUT = Path(a.out)
    if a.command == "sensor":
        cmd_sensor(a)
    elif a.command == "console":
        cmd_console(a)
    elif a.command == "stats":
        stats(Path(a.sensor))
    elif a.command == "short":
        cmd_short(a)
    else:
        cmd_final(a)


if __name__ == "__main__":
    main()
