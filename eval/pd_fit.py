"""Detection probability against apparent size as a fitted curve, so every test target informs every
estimate instead of only the targets in one slice.

    python eval/pd_fit.py --eval out/m1/eval --data out/m1/dataset

Per model and lighting (day, night, all): logistic regression of detection at the operating point
(centre match, eval/metrics.py) on log2(size_px), with the aspect group as an offset (beam is the
reference; head <= 30 deg, tail >= 150 deg):

    logit P = a + b log2(size) + c_head [head] + c_tail [tail]

px50 is the apparent size at which P = 0.5, per aspect group. The aspect effect is reported as the
px50 ratio head/beam and tail/beam: a ratio of 1.5 means a head-on target needs 1.5x the pixels
(1/1.5 of the range) for the same detection probability. 95 % intervals by bootstrap over scenes.
A small ridge penalty regularises separation by size. A non-positive slope or numerical overflow
has no finite increasing-size crossing. CSV values are left blank with an explicit status when a
crossing cannot be estimated; missing aspect groups and extrapolations are also labelled.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

import numpy as np  # noqa: E402

GROUPS = ("beam", "head", "tail")


def aspect_group(a: float) -> str | None:
    if a is None or a < 0:
        return None
    return "head" if a <= 30 else "tail" if a >= 150 else "beam"


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 100) -> np.ndarray:
    """Newton-Raphson with an L2 penalty on every coefficient except the intercept (column 0)."""
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p) - pen * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def design(size_px, groups) -> np.ndarray:
    s = np.log2(np.maximum(np.asarray(size_px, float), 1e-3))
    g = np.asarray(groups)
    return np.column_stack([np.ones_like(s), s, (g == "head").astype(float), (g == "tail").astype(float)])


def px50(w: np.ndarray, group: str) -> float:
    a = w[0] + (w[2] if group == "head" else w[3] if group == "tail" else 0.0)
    b = w[1]
    if b <= 1e-6:
        return float("inf")
    with np.errstate(over="ignore", under="ignore"):
        return float(np.exp2(-a / b))


def summarise(dets: list, n_boot: int = 500, seed: int = 0) -> dict:
    """dets: dicts with scene, size_px, detected, group. Returns px50 per group, ratios, slope, CIs."""
    dets = [d for d in dets if d["group"] in GROUPS]
    quantities = [f"px50_{g}" for g in GROUPS] + ["ratio_head_beam", "ratio_tail_beam", "slope_per_doubling"]
    counts = {g: int(sum(d["group"] == g for d in dets)) for g in GROUPS}
    if len(dets) < 10 or len({d["detected"] for d in dets}) < 2:
        status = ("insufficient_targets" if len(dets) < 10 else
                  "all_detected" if all(d["detected"] for d in dets) else "no_detections")
        return dict({k: dict(value=float("nan"), lo=float("nan"), hi=float("nan"), status=status)
                     for k in quantities}, n_targets=counts)
    scenes = np.array([d["scene"] for d in dets])
    X = design([d["size_px"] for d in dets], [d["group"] for d in dets])
    y = np.array([d["detected"] for d in dets], float)

    def stats(ix):
        if len(np.unique(y[ix])) < 2:
            return {k: float("nan") for k in quantities}
        w = fit_logistic(X[ix], y[ix])
        out = {f"px50_{g}": px50(w, g) for g in GROUPS}
        present = {dets[i]["group"] for i in ix}
        for g in GROUPS:
            if g not in present:
                out[f"px50_{g}"] = float("nan")
        for g in ("head", "tail"):
            beam = out["px50_beam"]
            out[f"ratio_{g}_beam"] = out[f"px50_{g}"] / beam if np.isfinite(beam) and beam > 0 else float("nan")
        out["slope_per_doubling"] = float(w[1])
        return out
    base = stats(np.arange(len(y)))
    uniq = np.unique(scenes)
    idx_of = {s: np.nonzero(scenes == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    boots = [stats(np.concatenate([idx_of[s] for s in rng.choice(uniq, len(uniq))])) for _ in range(n_boot)]
    res = {}
    for k, v in base.items():
        b = np.array([x[k] for x in boots], float)
        b = b[np.isfinite(b)]
        lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) > 0.8 * n_boot else (float("nan"), float("nan")))
        status = "ok"
        needed = ([k[5:]] if k.startswith("px50_") else
                  [k.split("_")[1], "beam"] if k.startswith("ratio_") else [])
        if any(counts[g] == 0 for g in needed):
            status = "missing_aspect_group"
        elif not np.isfinite(v):
            status = "no_finite_crossing"
        elif k.startswith("px50_") and not min(d["size_px"] for d in dets) <= v <= max(d["size_px"] for d in dets):
            status = "extrapolated"
        elif not np.isfinite(lo) or not np.isfinite(hi):
            status = "unstable_interval"
        res[k] = dict(value=v, lo=float(lo), hi=float(hi), status=status)
    res["n_targets"] = counts
    return res


def run(eval_dir: str, data: str, out: str, cfg: dict, n_boot: int = 500) -> list:
    import torch
    from data.dataset import load_metas
    from eval import metrics as M
    records = torch.load(os.path.join(eval_dir, "records.pt"), weights_only=False)
    metas = {m["scene_id"]: m for m in load_metas(data)}
    rows = []
    for model, rec in records.items():
        dets = M.detections_per_target(rec, cfg, "centre", 0.0)
        for d in dets:
            m = metas.get(d["scene"], {})
            d["group"] = aspect_group(M.target_aspect_deg(m)) if m else None
            d["lighting"] = (m.get("conditions") or {}).get("lighting")
        for lighting in ("day", "night", "all"):
            sel = [d for d in dets if lighting == "all" or d["lighting"] == lighting]
            res = summarise(sel, n_boot)
            for k, v in res.items():
                if k == "n_targets":
                    continue
                rows.append(dict(model=model, lighting=lighting, quantity=k,
                                 value=v["value"] if np.isfinite(v["value"]) else None,
                                 lo=v["lo"] if np.isfinite(v["lo"]) else None,
                                 hi=v["hi"] if np.isfinite(v["hi"]) else None, status=v["status"],
                                 n_head=res["n_targets"]["head"], n_beam=res["n_targets"]["beam"],
                                 n_tail=res["n_targets"]["tail"]))
    os.makedirs(out, exist_ok=True)
    if rows:
        with open(os.path.join(out, "px50.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval/evaluate.py output folder (records.pt)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    from train.engine import load_all_configs
    cfg = load_all_configs()["train"]["eval"]
    rows = run(a.eval, a.data, a.out or a.eval, cfg, a.boot)
    for r in rows:
        if r["quantity"].startswith(("px50", "ratio")):
            print(f"{r['model']:>10s} {r['lighting']:>5s} {r['quantity']:<15s} "
                  f"{r['value']} [{r['lo']}, {r['hi']}] {r['status']}")


if __name__ == "__main__":
    main()
