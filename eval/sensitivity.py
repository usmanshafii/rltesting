"""Recall against every sampled input: the measured margin of each uncertain modelling value.

    python eval/sensitivity.py --eval out/m1/eval --data out/m1/dataset --out out/m1/eval/sensitivity

Reads the per-scene records written by eval/evaluate.py (records.pt) and each scene's metadata. For
every model and every input recorded in the metadata (sky humidity and emittance, LWIR water-vapour
transmission, skin and engine temperatures, propeller coverage, RGB albedo and SWIR reflectance of the
paint, sun and camera elevation, range, aspect, ...), the test
targets are split into quantiles within coarse size groups, and recall at the operating point (centre
match) is reported per quantile with a Wilson 95% interval. The swing (highest minus lowest quantile
recall) says how much that input moves the result: a large swing means its evidence grade matters and
is worth measuring; a flat line suggests that input has little association within the sampled range.
Some inputs share a profile or are derived from others; swings are associations, not causal effects.

Categorical draws (engine power-setting profile, SWIR paint family, engine temperature cap acted, lighting)
are split the same way into sensitivity_categories.csv.
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

from data.dataset import load_metas  # noqa: E402
from eval import metrics as M  # noqa: E402

SIZE_GROUPS = {"2-8": (2.0, 8.0), "8-32": (8.0, 32.0), "32+": (32.0, 1e9)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def _get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def scene_inputs(meta: dict) -> dict:
    """Numeric inputs drawn for the scene's target, flattened. Missing values are skipped."""
    ex = meta.get("extra", {})
    sig = ex.get("signatures", {})
    tgt = next((o for o in ex.get("objects", []) if o.get("cls") == "target"), {})
    lw, rgb = _get(sig, "target", "lwir", default={}), _get(sig, "target", "rgb", default={})
    sw = _get(sig, "target", "swir", default={})
    sky = _get(sig, "environment", "lwir", "sky", default={})
    v = dict(
        range_m=tgt.get("range_m"), aspect_deg=tgt.get("aspect_deg"),
        target_elevation_deg=tgt.get("elevation_deg"), camera_elevation_deg=_get(ex, "camera", "elevation_deg"),
        air_temp_c=_get(meta, "conditions", "air_temp_c"), sun_elevation_deg=_get(ex, "sun", "elevation_deg"),
        visibility_km=_get(meta, "conditions", "visibility_km"),
        sky_rh=sky.get("rh_effective"), sky_vapour_pressure_hpa=sky.get("vapour_pressure_hpa"),
        sky_emissivity_zenith=sky.get("emissivity_zenith"), sky_dT_at_camera_k=sky.get("dT_at_camera_k"),
        skin_dT_lower_k=_get(lw, "skin", "dT"), skin_dT_upper_k=_get(lw, "skin", "dT_upper"),
        skin_emissivity=_get(lw, "skin", "emissivity"),
        engine_head_dT_k=_get(lw, "engine_head", "dT"), engine_barrel_dT_k=_get(lw, "engine_barrel", "dT"),
        engine_case_dT_k=_get(lw, "engine_case", "dT"), exhaust_stub_dT_k=_get(lw, "exhaust_stub", "dT"),
        engine_head_emissivity=_get(lw, "engine_head", "emissivity"),
        exhaust_stub_emissivity=_get(lw, "exhaust_stub", "emissivity"),
        engine_visible_cm2=_get(tgt, "hot_visible_cm2", "engine_total"),
        engine_hot_visible_cm2=_get(tgt, "hot_visible_cm2", "hot_total"),
        exhaust_visible_cm2=_get(tgt, "hot_visible_cm2", "exhaust_stub"),
        prop_alpha=_get(rgb, "propeller_disc", "alpha"),
        paint_albedo=_get(rgb, "skin", "albedo"), paint_coat=_get(rgb, "skin", "coat"),
        swir_paint_reflectance=_get(sw, "skin", "reflectance"),
        lwir_wv_transmission=tgt.get("lwir_wv_transmission"),
        prop_rpm=_get(ex, "rotation", "rpm"), rgb_blade_sweep_deg=_get(ex, "rotation", "sweep_deg", "rgb"),
    )
    return {k: float(x) for k, x in v.items() if isinstance(x, (int, float)) and not isinstance(x, bool)}


def scene_categories(meta: dict) -> dict:
    """Categorical draws to split results by: engine power-setting profile, SWIR paint family, whether an
    engine temperature cap acted (only when caps are configured), lighting."""
    ex = meta.get("extra", {})
    sig = ex.get("signatures", {})
    lw, sw = _get(sig, "target", "lwir", default={}), _get(sig, "target", "swir", default={})
    capped = [_get(lw, s, "capped") for s in ("engine_head", "engine_barrel", "exhaust_stub")]
    v = dict(thermal_profile=_get(lw, "engine_head", "thermal_profile"),
             swir_paint_family=_get(sw, "skin", "reflectance_family"),
             engine_cap_acted=None if all(c is None for c in capped) else ("yes" if any(capped) else "no"),
             lighting=_get(meta, "conditions", "lighting"))
    return {k: str(x) for k, x in v.items() if x is not None}


def category_table(dets: list, cats: dict, model: str, min_n: int = 5) -> list:
    """Recall per category value within each size group, with Wilson intervals and the swing."""
    rows = []
    for g, (lo, hi) in SIZE_GROUPS.items():
        sel = [d for d in dets if lo <= d["size_px"] < hi and d["scene"] in cats]
        for name in sorted({k for d in sel for k in cats[d["scene"]]}):
            groups = {}
            for d in sel:
                if name in cats[d["scene"]]:
                    groups.setdefault(cats[d["scene"]][name], []).append(bool(d["detected"]))
            groups = {k: v for k, v in groups.items() if len(v) >= min_n}
            if len(groups) < 2:
                continue
            rec = []
            for value, ys in sorted(groups.items()):
                n, kk = len(ys), int(sum(ys))
                lo_ci, hi_ci = wilson(kk, n)
                rec.append(dict(model=model, size_group=g, input=name, value=value, n=n, recall=kk / n,
                                ci_lo=lo_ci, ci_hi=hi_ci))
            swing = max(t["recall"] for t in rec) - min(t["recall"] for t in rec)
            separated = max(t["ci_lo"] for t in rec) > min(t["ci_hi"] for t in rec)
            for t in rec:
                t.update(swing=float(swing), ci_separated=bool(separated))
            rows += rec
    return rows


def table(dets: list, inputs: dict, model: str, n_q: int = 5, min_n: int = 8) -> list:
    rows = []
    for g, (lo, hi) in SIZE_GROUPS.items():
        sel = [d for d in dets if lo <= d["size_px"] < hi and d["scene"] in inputs]
        names = sorted({k for d in sel for k in inputs[d["scene"]]})
        for name in names:
            xs = [(inputs[d["scene"]][name], d["detected"]) for d in sel if name in inputs[d["scene"]]]
            if len(xs) < 2 * min_n:
                continue
            x = np.array([a for a, _ in xs])
            y = np.array([b for _, b in xs], bool)
            q = min(n_q, max(2, len(x) // min_n))
            edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
            if len(edges) < 3:
                continue
            k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
            rec = []
            for j in range(len(edges) - 1):
                m = k == j
                n, kk = int(m.sum()), int(y[m].sum())
                lo_ci, hi_ci = wilson(kk, n)
                rec.append(dict(model=model, size_group=g, input=name, quantile=j, x_lo=float(edges[j]),
                                x_hi=float(edges[j + 1]), x_mean=float(x[m].mean()) if n else float("nan"),
                                n=n, recall=kk / n if n else float("nan"), ci_lo=lo_ci, ci_hi=hi_ci))
            r = [t["recall"] for t in rec if t["n"]]
            swing = float(max(r) - min(r)) if r else float("nan")
            separated = max(t["ci_lo"] for t in rec if t["n"]) > min(t["ci_hi"] for t in rec if t["n"])
            for t in rec:
                t.update(swing=swing, ci_separated=bool(separated))
            rows += rec
    return rows


def plot(rows: list, out_dir: str, top: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    key = {}
    for r in rows:
        key[(r["model"], r["size_group"], r["input"])] = r["swing"]
    best = sorted(key.items(), key=lambda kv: -np.nan_to_num(kv[1]))[:top]
    for (model, g, name), sw in best:
        rs = [r for r in rows if (r["model"], r["size_group"], r["input"]) == (model, g, name)]
        x = np.array([r["x_mean"] for r in rs])
        y = np.array([r["recall"] for r in rs])
        lo, hi = np.array([r["ci_lo"] for r in rs]), np.array([r["ci_hi"] for r in rs])
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(name)
        ax.set_ylabel("recall (centre match)")
        ax.set_title(f"{model}, {g} px: swing {sw:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sens_{model}_{g}_{name}.png"), dpi=130)
        plt.close(fig)


def run(eval_dir: str, data: str, out: str, cfg: dict) -> list:
    import torch
    records = torch.load(os.path.join(eval_dir, "records.pt"), weights_only=False)
    metas = {m["scene_id"]: m for m in load_metas(data)}
    inputs = {sid: scene_inputs(m) for sid, m in metas.items()}
    cats = {sid: scene_categories(m) for sid, m in metas.items()}
    rows, crow = [], []
    for model, rec in records.items():
        dets = M.detections_per_target(rec, cfg, "centre", 0.0)
        rows += table(dets, inputs, model)
        crow += category_table(dets, cats, model)
    os.makedirs(out, exist_ok=True)
    for filename, data_rows, fields in (
        ("sensitivity_categories.csv", crow,
         ["model", "size_group", "input", "value", "n", "recall", "ci_lo", "ci_hi", "swing", "ci_separated"]),
        ("sensitivity.csv", rows,
         ["model", "size_group", "input", "quantile", "x_lo", "x_hi", "x_mean", "n", "recall", "ci_lo", "ci_hi", "swing", "ci_separated"])):
        with open(os.path.join(out, filename), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data_rows[0]) if data_rows else fields)
            w.writeheader()
            w.writerows(data_rows)
    if rows:
        plot(rows, out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval/evaluate.py output folder (records.pt)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from train.engine import load_all_configs
    cfg = load_all_configs()["train"]["eval"]
    out = a.out or os.path.join(a.eval, "sensitivity")
    rows = run(a.eval, a.data, out, cfg)
    swings = sorted({(r["model"], r["size_group"], r["input"], r["swing"], r["ci_separated"]) for r in rows},
                    key=lambda t: -np.nan_to_num(t[3]))
    print("largest recall swings (model, size group, input, swing, quantile CIs separated):")
    for t in swings[:15]:
        print(f"  {t[0]:>10s} {t[1]:>5s} px  {t[2]:<26s} {t[3]:.2f}  {'yes' if t[4] else 'no'}")
    path = os.path.join(out, "sensitivity_categories.csv")
    if os.path.exists(path):
        with open(path) as f:
            crow = list(csv.DictReader(f))
        cs = sorted({(r["model"], r["size_group"], r["input"], float(r["swing"]), r["ci_separated"]) for r in crow},
                    key=lambda t: -t[3])
        print("categorical splits (model, size group, input, swing, CIs separated):")
        for t in cs[:15]:
            print(f"  {t[0]:>10s} {t[1]:>5s} px  {t[2]:<26s} {t[3]:.2f}  {'yes' if t[4] == 'True' else 'no'}")


if __name__ == "__main__":
    main()
