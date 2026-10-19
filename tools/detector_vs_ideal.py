"""How far is the trained detector from the ideal observer? (docs/detection_gain_plan.md, section 2)

Joins the signal census (tools/privileged_census.py: matched-filter d' of each target in the student's cameras, noise
draw seed 17, the previous sensor models) with the noise verification's detections on the same scenes and the same
noise draw (tools/noise_ablation.py: the `previous`-trained rgb_lwir models on `previous` test frames, seed 17 present
frames). For each target, a detection is the best centre-matched score at or above the model's own validation
threshold for 0.01 false alarms per frame. A logistic fit of detection against log d' gives the d' at which the trained
detector finds half the targets (d'50). An ideal matched filter needs d' of about 5 for the same false-alarm budget
(eval/detectability.py).

    python tools/detector_vs_ideal.py      -> docs/reference/privileged/data/detector_vs_ideal.json
    python tools/detector_vs_ideal.py projections -> .../projections.json: detection each model's fitted curve predicts
                                              for 1 frame, 5 frames (static scene / moving-target bound) and the
                                              teacher's fused signal (estimates)
    python tools/detector_vs_ideal.py ap   -> .../baseline_ap.json: the same models' test AP (IoU 0.5, 0.5:0.95, targets
                                              up to 16 px, with target-free frames, and with centre matching)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

ABL = ROOT / "out" / "m1" / "noise_ablation"
DATA = ROOT / "docs" / "reference" / "privileged" / "data"


def _fit(x, y):
    """Logistic fit of y (0/1) against x = log d' by damped Newton steps from zero (the step is halved until the
    likelihood rises); returns (a, b) of p = 1 / (1 + exp(-(a + b x)))."""
    X = np.c_[np.ones_like(x), x]

    def loglik(w):
        z = np.clip(X @ w, -30, 30)
        return float(np.sum(y * z - np.log1p(np.exp(z))))
    w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        t, base = 1.0, loglik(w)
        while loglik(w + t * step) < base and t > 1e-6:
            t /= 2
        w = w + t * step
        if np.abs(t * step).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def main():
    import torch
    census = json.loads((DATA / "census.json").read_text())
    split = json.loads((ABL / "split.json").read_text())
    scores = json.loads((ABL / "scores.json").read_text())
    out = dict(models={}, note=__doc__.splitlines()[0])
    for seed, name in ((29, "rgb_lwir"), (31, "rgb_lwir_s31")):
        key = f"{seed}|previous|previous"
        if key not in scores:
            continue
        ck = torch.load(ABL / "previous" / "runs" / name / "best.pt", map_location="cpu", weights_only=False)
        meta = ck.get("meta", ck)
        t = float(meta["threshold_fa01"])
        best = np.asarray(scores[key]["best"], float)
        rows = []
        for sid, s in zip(split["test"], best):
            r = census.get(sid)
            if r is None:
                continue
            d = max(r["dprime"]["rgb"]["d"], r["dprime"]["lwir"]["d"])
            rows.append((sid, d, int(s >= t), r["size_px"]))
        d = np.array([x[1] for x in rows])
        hit = np.array([x[2] for x in rows], float)
        ok = d > 0
        a, b = _fit(np.log(d[ok]), hit[ok])
        d50 = float(np.exp(-a / b)) if b > 0 else None
        bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
        out["models"][name] = dict(
            threshold_fa01=t, n=len(rows), detected=int(hit.sum()),
            ideal_detectable=int(np.sum(d >= 5.0)), dprime50=d50, logistic=dict(a=a, b=b),
            by_dprime={f"{lo:g}-{hi:g}": dict(n=int(np.sum((d >= lo) & (d < hi))),
                                               detected=int(hit[(d >= lo) & (d < hi)].sum())) for lo, hi in bins},
            scenes=[dict(scene=s, dprime=round(x, 2), detected=h, size_px=z) for s, x, h, z in rows])
        print(f"{name}: {int(hit.sum())}/{len(rows)} detected at its 0.01 FA/frame threshold; ideal observer: "
              f"{int(np.sum(d >= 5))}/{len(rows)}; d'50 = {d50:.1f}" if d50 else f"{name}: fit failed")
        for k, v in out["models"][name]["by_dprime"].items():
            print(f"   d' {k:>8s}: {v['detected']}/{v['n']}")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "detector_vs_ideal.json").write_text(json.dumps(out, indent=1))


def baseline_ap():
    """AP of the same models on the same test frames: IoU 0.5 (the standard), 0.5:0.95, targets up to 16 px, with one
    target-free frame per scene, and with the project's centre match (eval/nwd.centre_ok_matrix)."""
    import torch
    import m1_det2 as D
    from eval import metrics
    from models import cnn_detector as cd
    out_dir = ABL / "previous"
    test = json.loads((ABL / "split.json").read_text())["test"]
    cfg = D.eval_cfg()
    metas = {s: D.meta_of(s) for s in test}
    torch.set_num_threads(D.threads())
    res = {}
    for name in ("rgb_lwir", "rgb_lwir_s31"):
        model, _ = cd.load(out_dir / "runs" / name / "best.pt")
        pres = D.records(out_dir, model, test, "p", 1, None, metas)
        absn = D.records(out_dir, model, test, "a", 1, None, metas)
        side = lambda r: max(r["gts"][0]["box"][2] - r["gts"][0]["box"][0], r["gts"][0]["box"][3] - r["gts"][0]["box"][1])  # noqa: E731
        small = [r for r in pres if r["gts"] and side(r) <= 16]
        res[name] = dict(ap50=metrics.ap(pres, cfg=cfg), ap50_95=metrics.ap_50_95(pres, cfg=cfg),
                         ap50_small=metrics.ap(small, cfg=cfg), n_small=len(small), n=len(pres),
                         ap50_with_target_free=metrics.ap(pres + absn, cfg=cfg),
                         ap_centre=metrics.ap(pres, kind="centre", thr=0.0, cfg=cfg),
                         ap_centre_small=metrics.ap(small, kind="centre", thr=0.0, cfg=cfg))
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
    (DATA / "baseline_ap.json").write_text(json.dumps(res, indent=1))


def projections(frames=5):
    """Detection at 0.01 false alarms per frame predicted by each model's fitted detection curve from the census d'
    (one frame), after averaging `frames` frames of a static scene (sensor noise only), the moving-target bound
    (sqrt(frames) on all of d'), and the teacher's fused four-camera d' (the ceiling of what privileged information
    could give). Estimates: they hold the network's efficiency fixed."""
    from eval.detectability import d_after
    det = json.loads((DATA / "detector_vs_ideal.json").read_text())["models"]
    rows = list(json.loads((DATA / "census.json").read_text()).values())
    out = {}
    for name, m in det.items():
        a, b = m["logistic"]["a"], m["logistic"]["b"]

        def pd(d):
            return float(np.mean(np.where(d > 0, 1 / (1 + np.exp(-np.clip(a + b * np.log(np.maximum(d, 1e-6)), -30, 30))), 0)))
        for title, sel in (("all", rows), ("small", [r for r in rows if (r["size_px"] or 1e9) <= 16])):
            d1 = np.array([max(r["dprime"]["rgb"]["d"], r["dprime"]["lwir"]["d"]) for r in sel])
            dn = np.array([max(d_after(r["dprime"]["rgb"], frames), d_after(r["dprime"]["lwir"], frames)) for r in sel])
            te = np.array([np.sqrt(sum(r["dprime"][x]["d"] ** 2 for x in ("rgb", "lwir", "lwir_cooled", "mwir"))) for r in sel])
            out[f"{name} {title}"] = dict(one_frame=pd(d1), five_frames_static=pd(dn), five_frames_moving_bound=pd(d1 * np.sqrt(frames)),
                                          teacher_signal=pd(te), n=len(sel))
            print(name, title, {k: round(v, 2) for k, v in out[f"{name} {title}"].items() if k != "n"})
    note = " ".join(projections.__doc__.split())
    (DATA / "projections.json").write_text(json.dumps(dict(note=note, frames=frames, projections=out), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1:]
    baseline_ap() if cmd == ["ap"] else projections() if cmd == ["projections"] else main()
