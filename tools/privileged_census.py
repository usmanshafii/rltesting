"""Signal-regime census for the privileged-teacher question (docs/privileged_teacher_case.md).

    python tools/privileged_census.py                       per-scene contrast-to-noise -> .../data/census.json
    python tools/privileged_census.py summary               the tables of the document -> .../census_summary.json
    python tools/privileged_census.py --zoom 2 2 --tag z2   the same with the teacher's zoom cameras set 2x narrower
                                                            (configs/sensors.yaml teacher_zoom) -> census_z2.json

census: for every rendered M1 scene outside the sealed test split, in each of the four cameras' network-input frames
(0 where a camera does not see the target), with the loader as configured (the previous sensor models since the noise
verification), one evaluation noise draw per scene (seed 17, as the validation frames):
  * the contrast-to-noise CNR of tools/m1_det2._cnr (box mean against a 2-px ring), kept for comparison. It is not a
    noise-referenced measure for small targets: without any sensor noise it barely changes, because the ring holds the
    target's own blur and the background (2026-10-18 check, docs/privileged_teacher_case.md);
  * the matched-filter detectability d' (_dprime): the target's exact signal (the frame with minus the frame without
    the target, same noise draw) against the spread of the same filter over the target-free frame, split into sensor
    noise and the scene's own structure (clutter) with a second, independent noise draw;
plus the target's size, range, lighting, visibility and clouds. No model is run and no test-split scene is read.

summary: the regime tables (student = best of RGB and uncooled LWIR; privileged = best of cooled LWIR and MWIR), on
the CNR (D >= 3, F 1-3, N < 1, the reporting definitions of tools/m1_det2_eval) and on d' (D >= 5: a Gaussian matched
filter finds the target with probability 0.5 near 0.01 false alarms per 640 x 512 frame; F 2-5; N < 2; definitions),
the camera ratios, the ideal-fusion bound (independent noise: sqrt of the summed squares), what N-frame integration
of a static scene would give (only the sensor noise averages: d'_N = 1 / sqrt(clutter^2 + noise^2 / N)), and the render
time of each band from the scenes' own timings.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]
import m1_det2 as D  # noqa: E402

DATA = ROOT / "docs" / "reference" / "privileged" / "data"
CENSUS = DATA / "census.json"
SUMMARY = DATA / "census_summary.json"


def _paths(tag):
    return (CENSUS, SUMMARY) if not tag else (DATA / f"census_{tag}.json", DATA / f"census_{tag}_summary.json")


def _zoom(factor, cameras):
    """Switch the loader-only teacher zoom on for this process (tools/m1_det2's configs)."""
    s = D.cfgs()["sensors"]
    s["teacher_zoom"] = dict(s.get("teacher_zoom") or {}, enabled=True, factor=[float(v) for v in factor],
                             cameras=list(cameras))
DETECTABLE, FAINT, MAX_PX = 3.0, 1.0, 16.0
from eval.detectability import DETECTABLE as D_DETECT, FAINT as D_FAINT, NONE, d_after as _d_after  # noqa: E402
from eval.detectability import dprime as _dprime, signal_class as _dclass  # noqa: E402


def census(tag=None):
    DATA.mkdir(parents=True, exist_ok=True)
    out, _ = _paths(tag)
    src = ROOT / D.SOURCE
    rows = json.loads(out.read_text()) if out.exists() else {}
    ids = [f"s{i:06d}" for i in range(D.N_POP) if D.scene_done(src, i)]
    t0 = time.time()
    made = 0
    for sid in ids:
        if sid in rows and "dprime" in rows[sid]:
            continue
        meta = D.meta_of(sid)
        if meta["split"] == "test":                 # M1's test split stays sealed
            continue
        fr, boxes, cov = D._scene_frames(meta, 17, False, 0, absent=False)
        fa, _, _ = D._scene_frames(meta, 17, False, 0, absent=True)
        fa2, _, _ = D._scene_frames(meta, 1017, False, 0, absent=True)
        c = D._cnr(fr, boxes, cov)
        dp = {b: (_dprime(fr[b], fa[b], fa2[b], boxes[0], cover=cov[b]) if boxes and D.centre_inside(boxes[0], cov[b])
                  else dict(NONE))
              for b in D.BANDS}
        t = meta["target"]
        bx = boxes[0] if boxes else None
        rows[sid] = dict(split=meta["split"], cnr={b: (v[0] if v else 0.0) for b, v in c.items()}, dprime=dp,
                         size_px=(max(bx[2] - bx[0], bx[3] - bx[1]) if bx else None), size_bin=t.get("size_bin"),
                         range_m=t.get("range_m"), lighting=meta["conditions"]["lighting"],
                         visibility_km=meta["conditions"]["visibility_km"], clouds=meta["conditions"].get("clouds"),
                         hfov_deg=meta["camera"]["hfov_deg"], coverage={b: list(v) for b, v in cov.items()},
                         zoom={b: (D.scene_camera(meta, b) or {}).get("teacher_zoom", 1.0) for b in ("lwir_cooled", "mwir")})
        made += 1
        if made % 10 == 0:
            out.write_text(json.dumps(rows, indent=1))
            print(f"{time.strftime('%H:%M:%S')} {made} scenes ({(time.time() - t0) / made:.1f} s each)", flush=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"{time.strftime('%H:%M:%S')} done: {len(rows)} scenes", flush=True)


def _cls(c):
    return "D" if c >= DETECTABLE else ("F" if c >= FAINT else "N")


def _regimes(sel):
    stu = np.array([max(r["cnr"]["rgb"], r["cnr"]["lwir"]) for r in sel])
    pri = np.array([max(r["cnr"]["lwir_cooled"], r["cnr"]["mwir"]) for r in sel])
    table = {f"student {s} / privileged {p}": int(sum(1 for a, b in zip(stu, pri) if _cls(a) == s and _cls(b) == p))
             for s in "DFN" for p in "DFN"}
    fs = np.array([np.hypot(r["cnr"]["rgb"], r["cnr"]["lwir"]) for r in sel])
    ft = np.array([np.sqrt(sum(r["cnr"][b] ** 2 for b in D.BANDS)) for r in sel])
    lw = np.array([r["cnr"]["lwir"] for r in sel])
    out = dict(n=len(sel), table=table,
               privileged_better=float(np.mean(pri > stu)), privileged_2x=float(np.mean(pri >= 2 * np.maximum(stu, 1e-3))),
               student_detectable=float(np.mean(stu >= DETECTABLE)),
               student_faint_privileged_detectable=float(np.mean((stu >= FAINT) & (stu < DETECTABLE) & (pri >= DETECTABLE))),
               student_none_privileged_detectable=float(np.mean((stu < FAINT) & (pri >= DETECTABLE))),
               nothing_anywhere=float(np.mean(np.maximum(stu, pri) < FAINT)),
               ideal_fusion_ratio_median=float(np.median(ft / np.maximum(fs, 1e-3))),
               ideal_fusion_teacher_detectable_student_faint=float(np.mean((ft >= DETECTABLE) & (fs >= FAINT) & (fs < DETECTABLE))))
    for b in ("lwir_cooled", "mwir"):
        v = np.array([r["cnr"][b] for r in sel])
        seen = v > 0
        out[f"{b}_over_lwir_median_where_seen"] = float(np.median(v[seen] / np.maximum(lw[seen], 1e-3))) if seen.any() else None
        out[f"{b}_not_seeing_target"] = float(np.mean(~seen))
    return out


def _dregimes(sel, frames=5):
    def g(r, b):
        return r["dprime"][b]["d"]
    stu = np.array([max(g(r, "rgb"), g(r, "lwir")) for r in sel])
    pri = np.array([max(g(r, "lwir_cooled"), g(r, "mwir")) for r in sel])
    table = {f"student {a} / privileged {b}": int(sum(1 for x, y in zip(stu, pri) if _dclass(x) == a and _dclass(y) == b))
             for a in "DFN" for b in "DFN"}
    fs = np.array([np.hypot(g(r, "rgb"), g(r, "lwir")) for r in sel])
    ft = np.array([np.sqrt(sum(g(r, b) ** 2 for b in D.BANDS)) for r in sel])
    tn = np.array([max(_d_after(r["dprime"]["rgb"], frames), _d_after(r["dprime"]["lwir"], frames)) for r in sel])
    lw = np.array([g(r, "lwir") for r in sel])
    shares = {b: [r["dprime"][b]["noise_share"] for r in sel if r["dprime"][b]["noise_share"] is not None] for b in D.BANDS}
    out = dict(n=len(sel), table=table, median={b: float(np.median([g(r, b) for r in sel])) for b in D.BANDS},
               noise_share_median={b: (float(np.median(v)) if v else None) for b, v in shares.items()},
               student_detectable=float(np.mean(stu >= D_DETECT)),
               privileged_better=float(np.mean(pri > stu)), privileged_2x=float(np.mean(pri >= 2 * np.maximum(stu, 1e-3))),
               student_faint_privileged_detectable=float(np.mean((stu >= D_FAINT) & (stu < D_DETECT) & (pri >= D_DETECT))),
               student_none_privileged_detectable=float(np.mean((stu < D_FAINT) & (pri >= D_DETECT))),
               fusion_student_detectable=float(np.mean(fs >= D_DETECT)), fusion_teacher_detectable=float(np.mean(ft >= D_DETECT)),
               fusion_ratio_median=float(np.median(ft / np.maximum(fs, 1e-3))),
               frames=frames, temporal_student_detectable=float(np.mean(tn >= D_DETECT)),
               temporal_bound_student_detectable=float(np.mean(stu * np.sqrt(frames) >= D_DETECT)))
    for b in ("lwir_cooled", "mwir"):
        v = np.array([g(r, b) for r in sel])
        seen = v > 0
        out[f"{b}_over_lwir_median_where_seen"] = float(np.median(v[seen] / np.maximum(lw[seen], 1e-3))) if seen.any() else None
        out[f"{b}_not_seeing_target"] = float(np.mean(~seen))
    return out


def summary(tag=None):
    src, dest = _paths(tag)
    rows = json.loads(src.read_text())
    R = list(rows.values())
    small = [r for r in R if (r["size_px"] or 1e9) <= MAX_PX]
    res = dict(scenes=len(R), splits={s: sum(1 for r in R if r["split"] == s) for s in ("train", "val")},
               lighting={k: sum(1 for r in R if r["lighting"] == k) for k in sorted({r["lighting"] for r in R})},
               median_cnr={b: float(np.median([r["cnr"][b] for r in R])) for b in D.BANDS},
               all=_regimes(R), small=_regimes(small),
               small_day=_regimes([r for r in small if r["lighting"] == "day"]),
               small_night=_regimes([r for r in small if r["lighting"] != "day"]))
    if all("dprime" in r for r in R):
        res.update(dprime_all=_dregimes(R), dprime_small=_dregimes(small),
                   dprime_small_day=_dregimes([r for r in small if r["lighting"] == "day"]),
                   dprime_small_night=_dregimes([r for r in small if r["lighting"] != "day"]))
    ratio, noise_ratio = [], []                     # cooled / uncooled: observed CNR vs the cameras' noise alone
    for sid, r in rows.items():
        if r["cnr"]["lwir_cooled"] > 0 and r["cnr"]["lwir"] > 0.3:
            cams = D.meta_of(sid)["extra"]["cameras"]["cameras"]
            u, c = cams["lwir_uncooled"], cams["lwir_cooled"]
            ratio.append(r["cnr"]["lwir_cooled"] / r["cnr"]["lwir"])
            noise_ratio.append(u["netd_mk"] / np.hypot(c["netd_mk"], 1000.0 * c.get("pixel_fpn_k", 0.0)))
    res["cooled_vs_uncooled"] = dict(n=len(ratio), observed_cnr_ratio_median=float(np.median(ratio)),
                                     noise_only_ratio_median=float(np.median(noise_ratio)),
                                     implied_contrast_ratio=float(np.median(ratio) / np.median(noise_ratio)))
    times = {"day": [], "night": []}                 # every rendered scene, test split included: render timings only
    for i in range(D.N_POP):
        if D.scene_done(ROOT / D.SOURCE, i):
            m = D.meta_of(f"s{i:06d}")
            times["day" if m["conditions"]["lighting"] == "day" else "night"].append(m["extra"]["timings_s"])
    res["render_time_s"] = {}
    for k, ts in list(times.items()) + [("all", times["day"] + times["night"])]:
        tot = sum(sum(t.values()) for t in ts)
        priv = sum(t["lwir_cooled"] + t["mwir"] for t in ts)
        res["render_time_s"][k] = dict(n=len(ts), mean_per_scene=tot / len(ts), privileged_share=priv / tot,
                                       student_only_scenes_per_four_band_scene=tot / (tot - priv))
    DATA.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="census", choices=["census", "summary"])
    ap.add_argument("--zoom", type=float, nargs=2, metavar=("LO", "HI"), help="teacher zoom factor range")
    ap.add_argument("--cameras", nargs="+", default=["lwir_cooled", "mwir"], help="zoomed cameras")
    ap.add_argument("--tag", default=None, help="file suffix for this setting")
    a = ap.parse_args()
    if a.zoom:
        _zoom(a.zoom, a.cameras)
    summary(a.tag) if a.command == "summary" else census(a.tag)
