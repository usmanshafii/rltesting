try: import blenderproc as bproc  # noqa: E401,E701  first code line, as in render/scene.py
except Exception: bproc = None  # noqa: E701
__doc__ = """Pan-tilt tracking sequence for the detection video (tools/kal_video.py), rendered through the
same scene sampler, signatures and per-object crop renders as the dataset (render/scene.py).

    set KAL_CONFIG_OVERLAY=configs/m1.yaml
    blender -b -P render/track_sequence.py -- --out out/video/sensor --lighting day

What is simulated
  * one scene sampled exactly as for the M1 dataset (render/sampling.py): sun, sky, weather, target
    variant and per-band signatures; S1 narrow camera (8.78 deg, 640 x 512), RGB + LWIR;
  * the KAL in level flight at the protected default airspeed (44 m/s) along a gentle S-turn with
    coordinated bank (tan(bank) = v * turn rate / g), approaching and passing the sensor site;
  * a pan-tilt head that stares at a cue for the first seconds, then follows the aircraft with a
    first-order lag and small pointing jitter. The pointing is SCRIPTED from the simulated flight
    path; it is not closed-loop tracking on detector output.

How it is rendered
  * the open-sky background depends only on ray direction, so it is rendered once per band as a
    low-resolution pinhole panorama and resampled for every camera pose (exact up to the sky's own
    smoothness, which is far below one sensor pixel of variation);
  * the target is rendered per frame and band as a border crop at 2x / 4x with motion blur, with
    the camera at that frame's pose (render/scene.py render_objects), stored as layers;
  * haze, PSF, sensor noise and display are applied afterwards in Python, as in data/dataset.py.

Outputs: plan.json (per-frame pose, camera, label box, range), pano_<band>.npz, layers/<band>/fNNNN.npz.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(ROOT, "target"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bpy  # noqa: E402,F401
import numpy as np  # noqa: E402

from data import composite as cp  # noqa: E402
from render import bpy_util as bu  # noqa: E402
from render import placement as pl  # noqa: E402
from render import sampling  # noqa: E402
from render import scene as rs  # noqa: E402
from render import sequence as sq  # noqa: E402

G = 9.81


def _log(msg):
    print(msg, flush=True)


def flight_path(C, az_c, a):
    """Level S-turn approach. Returns per-frame (t, position, heading, bank, velocity)."""
    n = int(round(a.duration * a.fps))
    u = pl.direction(az_c, 0.0)                         # horizontal look direction of the cue
    w = np.array([u[1], -u[0], 0.0])                     # to its right
    p = C + a.start_m * u + a.offset_m * w + np.array([0, 0, a.alt_m])
    head0 = (az_c + 180.0) % 360.0                       # flying back towards the site
    amp, period = a.weave_deg, a.weave_s
    dt = 1.0 / a.fps
    out = []
    sub = 8
    for k in range(n):
        t = k * dt
        heading = head0 + amp * math.sin(2 * math.pi * t / period)
        rate = math.radians(amp) * 2 * math.pi / period * math.cos(2 * math.pi * t / period)
        bank = math.degrees(math.atan(a.speed * rate / G))
        v = a.speed * pl.direction(heading, 0.0)
        out.append(dict(t=t, position=p.copy(), heading=heading, bank=bank, velocity=v))
        for s in range(sub):                              # integrate the heading change within the frame
            ts = t + (s + 0.5) * dt / sub
            hs = head0 + amp * math.sin(2 * math.pi * ts / period)
            p = p + a.speed * pl.direction(hs, 0.0) * dt / sub
    return out


def pointing(path, C, a, rng):
    """Pan-tilt angles per frame: stare at a cue, then follow with lag tau and jitter."""
    az, el = [], []
    cue_az, cue_el = pl.azimuth_elevation(path[0]["position"] - C)
    cue_az, cue_el = cue_az + a.cue_err[0], cue_el + a.cue_err[1]
    cur = np.array([cue_az, cue_el])
    jit = np.zeros(2)
    alpha = 1 - math.exp(-1.0 / (a.fps * a.tau_s))
    for f in path:
        tgt = np.array(pl.azimuth_elevation(f["position"] - C))
        if f["t"] >= a.stare_s:
            d = tgt - cur
            d[0] = (d[0] + 180) % 360 - 180
            cur = cur + alpha * d
        jit = 0.85 * jit + 0.15 * rng.normal(0, a.jitter_deg, 2)
        az.append(float((cur[0] + jit[0]) % 360))
        el.append(float(cur[1] + jit[1]))
    return az, el


def panorama_camera(cams, f_px, scale):
    """Pinhole camera at the common centre that contains every frame's corner rays."""
    dirs = []
    for c in cams:
        for u, v in ((0, 0), (c.W, 0), (0, c.H), (c.W, c.H), (c.W / 2, 0), (c.W / 2, c.H)):
            dirs.append(c.ray(u, v))
    dirs = np.array(dirs)
    m = dirs.mean(0)
    az, el = pl.azimuth_elevation(m)
    f = f_px / scale
    probe = pl.Camera(100, 100, 2 * math.degrees(math.atan(50 / f)), cams[0].C, az, el)
    pc = dirs @ probe.R
    x = f * pc[:, 0] / -pc[:, 2]
    y = f * pc[:, 1] / -pc[:, 2]
    W = int(2 * (np.abs(x).max() + 8)) // 2 * 2
    H = int(2 * (np.abs(y).max() + 8)) // 2 * 2
    return pl.Camera(W, H, math.degrees(2 * math.atan(W / 2 / f)), cams[0].C, az, el)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lighting", default="day", choices=("day", "dusk", "night"))
    ap.add_argument("--start", type=int, default=7000, help="first scene index searched for the base scene")
    ap.add_argument("--cam-el", type=float, default=6.0)
    ap.add_argument("--speed", type=float, default=None, help="m/s; default: render.yaml target airspeed default")
    ap.add_argument("--duration", type=float, default=26.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--start-m", dest="start_m", type=float, default=1450.0)
    ap.add_argument("--offset-m", dest="offset_m", type=float, default=170.0)
    ap.add_argument("--alt-m", dest="alt_m", type=float, default=150.0)
    ap.add_argument("--weave-deg", dest="weave_deg", type=float, default=10.0)
    ap.add_argument("--weave-s", dest="weave_s", type=float, default=22.0)
    ap.add_argument("--stare-s", dest="stare_s", type=float, default=2.5)
    ap.add_argument("--tau-s", dest="tau_s", type=float, default=0.35)
    ap.add_argument("--jitter-deg", dest="jitter_deg", type=float, default=0.012)
    ap.add_argument("--cue-err", dest="cue_err", type=float, nargs=2, default=(1.6, 0.9))
    ap.add_argument("--bands", default="rgb,lwir")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args(argv)
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    cfgs = sampling.load_configs()
    if a.speed is None:
        a.speed = float(cfgs["render"]["target"]["airspeed_mps"]["default"]) if isinstance(
            cfgs["render"]["target"].get("airspeed_mps"), dict) else 44.0
    if bproc is not None:
        bproc.init()
    bands = tuple(b for b in a.bands.split(",") if b)
    sq_args = argparse.Namespace(start=a.start, lighting=a.lighting, cam_el=a.cam_el)
    idx, spec, objs = sq.find_scene(cfgs, sq_args)
    cam0 = objs["camera"]
    C = cam0.C
    rng = np.random.default_rng([idx, 31])
    path = flight_path(C, cam0.azimuth, a)
    az, el = pointing(path, C, a, rng)
    cams = [pl.Camera(cam0.W, cam0.H, cam0.hfov, C, az[k], el[k]) for k in range(len(path))]
    sun = pl.direction(spec["sun"]["azimuth_deg"], spec["sun"]["elevation_deg"]) if spec["lighting"] != "night" else None
    frames, plan = [], []
    pitch = 2.0                                          # level-flight attitude, nose slightly up (assumed)
    for k, (f, cam) in enumerate(zip(path, cams)):
        objs["camera"] = cam
        R = pl.object_rotation(f["heading"], pitch, f["bank"])
        pose = dict(heading_deg=f["heading"], pitch_deg=pitch, roll_deg=f["bank"])
        e = sq.frame_entry(cfgs, spec, objs, f["position"], R, f["velocity"], pose, f["t"])
        if e is None:
            raise RuntimeError(f"frame {k}: target behind the camera")
        bx = e["bbox_blur_px"]
        if bx["x"] < 2 or bx["y"] < 2 or bx["x"] + bx["w"] > cam.W - 2 or bx["y"] + bx["h"] > cam.H - 2:
            raise RuntimeError(f"frame {k} t={f['t']:.2f}: target leaves the sensor frame {bx}")
        low = min(pl.azimuth_elevation(cam.ray(u, cam.H))[1] for u in (0, cam.W / 2, cam.W))
        if low < 1.5:
            raise RuntimeError(f"frame {k}: the frame reaches {low:.1f} deg elevation (terrain); raise --alt-m or --cam-el")
        sun_sep = math.degrees(math.acos(float(np.clip(sun @ cam.forward, -1, 1)))) if sun is not None else 180.0
        frames.append(e)
        plan.append(dict(k=k, t=f["t"], cam_az=az[k], cam_el=el[k], cam_matrix=pl.matrix_world(cam.R, cam.C).tolist(),
                         target_matrix=e["matrix_world"], velocity=e["velocity_mps"], range_m=e["range_m"],
                         bbox_px=e["bbox_px"], bbox_blur_px=e["bbox_blur_px"], size_px=e["size_px"],
                         size_bin=e["size_bin"], heading_deg=f["heading"], bank_deg=f["bank"], pitch_deg=pitch,
                         elevation_deg=e["elevation_deg"], agl_m=e["agl_m"], sun_separation_deg=sun_sep,
                         crop_px=e["crop_px"], supersample=e["supersample"]))
    pano = panorama_camera(cams, cam0.f, 6.0)
    tv = spec["target_variant"]
    summary = dict(base_scene=idx, lighting=spec["lighting"], thermal_state=spec["thermal_state"],
                   air_temp_c=spec["air_temp_c"], weather=spec["weather"], sun=spec["sun"], moon=spec["moon"],
                   psf=spec["psf"], camera=dict(width=cam0.W, height=cam0.H, hfov_deg=cam0.hfov, f_px=cam0.f,
                                                position=C.tolist(), exposure_ms=spec["camera"]["exposure_ms"],
                                                sensor_width_mm=spec["camera"]["sensor_width_mm"]),
                   panorama=dict(width=pano.W, height=pano.H, hfov_deg=pano.hfov, azimuth_deg=pano.azimuth,
                                 elevation_deg=pano.elevation, matrix=pl.matrix_world(pano.R, pano.C).tolist()),
                   speed_mps=a.speed, fps=a.fps, duration_s=a.duration, args=vars(a),
                   variant=dict(span_m=tv["span_m"], length_m=tv["length_m"], params=tv["params"]),
                   signature_model=spec["signature_model"], frames=plan,
                   frame_meta=[sq.frame_meta(spec, fr, f"f{k:04d}", argparse.Namespace(name="track", fps=a.fps), bands, k)
                               for k, fr in enumerate(frames)])
    with open(os.path.join(a.out, "plan.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    r0, r1 = plan[0]["range_m"], plan[-1]["range_m"]
    _log(f"base scene {idx} ({spec['lighting']}), {len(plan)} frames, range {r0:.0f} -> {r1:.0f} m, "
         f"size {plan[0]['size_px']:.1f} -> {plan[-1]['size_px']:.1f} px, AGL {min(p['agl_m'] for p in plan):.0f} m+, "
         f"sun separation >= {min(p['sun_separation_deg'] for p in plan):.0f} deg, panorama {pano.W}x{pano.H}")
    if a.plan_only:
        return

    t0 = time.time()
    calib = rs.calibrate(spec, cfgs)
    objs["camera"] = cams[0]
    spec["objects"] = [frames[0]]
    spec["camera"]["matrix_world"] = plan[0]["cam_matrix"]
    h = rs.build(spec, objs, cfgs, calib)
    tobj = h.labelled[0][1]
    exp_s = spec["camera"]["exposure_ms"] / 1000.0
    tmp = os.path.join(a.out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    lens_narrow = h.cam.data.lens
    for band in bands:
        rs.apply_band(band, h, cfgs)
        os.makedirs(os.path.join(a.out, "layers", band), exist_ok=True)
        pano_path = os.path.join(a.out, f"pano_{band}.npz")
        if not os.path.exists(pano_path):
            saved = {k: spec["camera"][k] for k in ("width", "height", "hfov_deg", "supersample_background")}
            spec["camera"].update(width=pano.W, height=pano.H, hfov_deg=pano.hfov, supersample_background=1)
            h.cam.data.lens = pl.focal_mm(spec["camera"]["sensor_width_mm"], pano.hfov)
            bu.set_matrix(h.cam, pl.matrix_world(pano.R, pano.C))
            # The panorama's lower corners reach below the horizon, which no sensor frame does
            # (checked above): hide the terrain so the panorama holds the sky radiance only.
            env, h.env_objs = h.env_objs, []          # scene._visibility would un-hide them
            for o in env:
                o.hide_render = True
            layer = rs.render_background(band, spec, h, cfgs, tmp, _log)
            h.env_objs = env
            for o in env:
                o.hide_render = False
            if "ground" in layer:
                raise RuntimeError("terrain inside the panorama; the open-sky resampling assumes sky only")
            cp.save_layers(pano_path, layer, [], half=(band != "lwir"))
            spec["camera"].update(saved)
            h.cam.data.lens = lens_narrow
        tb = time.time()
        for k, (fr, cam) in enumerate(zip(frames, cams)):
            lpath = os.path.join(a.out, "layers", band, f"f{k:04d}.npz")
            if os.path.exists(lpath):
                continue
            bu.set_matrix(h.cam, pl.matrix_world(cam.R, cam.C))
            bu.set_matrix(tobj, fr["matrix_world"])
            bu.linear_motion(tobj, fr["matrix_world"], fr["velocity_mps"], exp_s)
            rs.update_rotation(h, spec, fr["t_s"])
            objects = rs.render_objects(band, spec, h, cfgs, tmp, lambda m: None, [fr])
            cp.save_layers(lpath, dict(psf_sigma0=spec["psf"][band]["sigma0"]), objects, half=(band != "lwir"))
            if k % 25 == 0 or k == len(frames) - 1:
                _log(f"  {band} frame {k + 1}/{len(frames)} ({fr['size_px']:.1f} px @ {fr['range_m']:.0f} m) "
                     f"{(time.time() - tb) / (k + 1):.2f} s/frame")
    _log(f"TRACK_SEQUENCE_DONE {len(frames)} frames in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
