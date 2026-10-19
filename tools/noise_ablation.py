"""Does the camera model help or hurt? Train the same detector on four noise models and test each on all of them and on
a camera none of them models (docs/sensor_noise.md, "Verification").

    python tools/noise_ablation.py prepare --min-scenes 200     snapshot the rendered scenes, split, one manifest per variant
    python tools/noise_ablation.py cache --variant camera        that variant's camera frames (training, validation, test)
    python tools/noise_ablation.py train --variant camera [--seed 29]
    python tools/noise_ablation.py evaluate                      the cross matrix -> out/m1/noise_ablation/results.md
    python tools/noise_ablation.py report --seeds 29 31          the tables again from the saved scores (no inference)
    python tools/noise_ablation.py evaluate --only mixed --seeds 29 31   add one training variant to the saved results
    python tools/noise_ablation.py all [--seeds 29 31]           everything; each variant in its own process

The detector is `rgb_lwir` of tools/m1_det2.py (the student's two cameras), with identical seeds, crops and budget for
every variant. Noise models (each used for training and for testing):

  gaussian   additive white Gaussian noise at the sensor's own level (median-level shot + read noise; NETD) with a
             linear display stretch: the plain-augmentation baseline (Carlson et al. 2018's "additive Gaussian")
  previous   the models before 2026-10-17: heteroscedastic shot + read noise (Brooks et al. 2019) for RGB; NETD, pixel
             and column fixed pattern and stripe terms (He et al. 2018 style) for LWIR; linear display stretch
  physics    the new sensor physics without the camera's processing: Poisson shot, Tukey-lambda read, row noise, Bayer
             mosaic (Wei et al. 2020); cubic column pattern, shading, non-uniformities, defect replacement for LWIR;
             linear display stretch, no noise reduction, edge enhancement, filters or chromatic aberration
  camera     the new default: physics + the camera's processing (noise reduction acting with gain, edge enhancement,
             chromatic aberration; histogram AGC with detail enhancement, filters)

Training only (added 2026-10-17 after the first results, which hinged on cloud false alarms):

  mixed      every training frame (and each validation scene) taken at random from one of the four models' caches
             (hard links, no new frames): randomisation over the camera models instead of choosing one

Test only:

  foreign    a camera none of the four models: the previous sensor, then a mild blur (sigma 0-0.8 px), a random
             display gamma (0.8-1.25) and JPEG compression (quality 60-85) of the 8-bit video

Scenes: only scenes outside M1's test cohort, whose test split stays sealed for the M1 evaluation. The ablation's own
split: test = the population's validation-split scenes plus every third training-split scene; validation = every
eighth of the remaining training-split scenes; training = the rest. Results: out/m1/noise_ablation/results.md and .json.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

import numpy as np  # noqa: E402

import m1_det2 as D  # noqa: E402

OUT = ROOT / "out" / "m1" / "noise_ablation"
NOISE_MODELS = ("gaussian", "previous", "physics", "camera")
TRAIN_VARIANTS = NOISE_MODELS + ("mixed",)
TEST_VARIANTS = NOISE_MODELS + ("foreign",)
REFERENCES = ("camera", "mixed")                 # the candidates compared with every other training noise
MODEL = "rgb_lwir"
TRAINING = dict(steps=2000, realisations=2, absent_realisations=1, val_every=250,
                hard_negatives=dict(steps=[400, 800, 1200, 1600], frames=24, min_score=0.15, share=0.33, pool_max=600))
UNUSED_BANDS = ("lwir_cooled", "mwir")          # rgb_lwir never reads them: their frames are left empty here


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ the variants
def _gaussian_sensor(band, cam, meta, rng, S):
    """Additive white Gaussian noise at the sensor's own level, linear display stretch."""
    from noise import lwir_fpn, rgb_noise
    from thermal.planck_lut import band_lut
    if band == "rgb":
        cfg = dict(S["rgb_noise"], read_noise_e=[cam["read_noise_e"]] * 2)
        exp_ms = meta["camera"]["exposure_ms"]

        def f(x):
            p = rgb_noise.sample_params(rng, cfg)
            e_open = np.clip(np.asarray(x, np.float32), 0, None) * (p["e_per_unit_ms"] * exp_ms)
            light, gain = rgb_noise.auto_exposure(float(np.median(e_open.mean(-1))), p["full_well"], p["ae_target"], cfg["max_gain"])
            e = e_open * light
            sigma = math.sqrt(max(float(np.median(e)), 0.0) + p["read_e"] ** 2)          # one level for the whole frame
            e = e + rng.standard_normal(e.shape).astype(np.float32) * sigma
            q = 2 ** int(cfg["bits"]) - 1
            raw = np.round(np.clip(e * gain / p["full_well"], 0.0, 1.0) * q) / q
            return (raw ** (1.0 / cfg["gamma"])).astype(np.float32)
        return f
    lut = band_lut(tuple(cam["band_um"]))
    t_air = meta["conditions"]["air_temp_c"] + 273.15
    cfg = dict(S["lwir_noise"], netd_mk=[cam["netd_mk"]] * 2)

    def g(x):
        p = lwir_fpn.sample_params(rng, cfg)
        c = lwir_fpn.to_counts(x, t_air, p, lut)
        cpk = p["counts_per_k"] * lwir_fpn._ref_scale(t_air, lut)
        c = c + rng.standard_normal(c.shape).astype(np.float32) * p["netd_k"] * cpk
        c = np.clip(np.round(c), 0, 2 ** int(cfg["bits"]) - 1)
        lo, hi = np.percentile(c, [p["agc_lo"], p["agc_hi"]])
        return np.clip((c - lo) / max(hi - lo, 1.0), 0.0, 1.0).astype(np.float32)
    return g


def _foreign_post(y, rng):
    """An unmodelled camera's output stage: mild blur, a display gamma and JPEG compression of the 8-bit video."""
    from PIL import Image
    from scipy import ndimage
    y = np.asarray(y, np.float32)
    s = float(rng.uniform(0.0, 0.8))
    if s > 0.05:
        y = np.stack([ndimage.gaussian_filter(y[..., c], s) for c in range(y.shape[-1])], -1) if y.ndim == 3 \
            else ndimage.gaussian_filter(y, s)
    y = np.clip(y, 0.0, 1.0) ** float(rng.uniform(0.8, 1.25))
    img = Image.fromarray((np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(rng.integers(60, 86)))
    out = np.asarray(Image.open(io.BytesIO(buf.getvalue())), np.float32) / 255.0
    return out.reshape(y.shape)


def variant_configs(name):
    """The configs of one noise variant (tools/m1_det2's configs with the uncooled stripe terms)."""
    from render.sampling import load_configs
    C = copy.deepcopy(load_configs(overlay=D.OVERLAY))
    C["sensors"]["lwir_noise"].update(D.LWIR_REALISM)
    pipes = C["sensors"].get("camera_pipelines") or {}
    if name in ("previous", "gaussian", "foreign"):
        C["sensors"].pop("camera_pipelines", None)
    elif name == "physics":
        rgb = pipes["rgb"]
        rgb.update(nr_level=[0, 0], aperture_level=[0, 0], chromatic_aberration=[0.0, 0.0])
        for cam in ("lwir_uncooled", "lwir_cooled", "mwir"):
            pipes[cam]["filter_attenuation"] = [0.0, 0.0]
            pipes[cam]["agc"]["histogram_prob"] = 0.0
    elif name not in ("camera", "mixed"):            # mixed trains on cached frames of all four models
        raise ValueError(name)
    if "camera_pipelines" in C["sensors"]:
        pipes["enabled"] = True                      # off by default since the verification; these variants use them
    return C


def setup_variant(name):
    """Configure tools/m1_det2 (its configs and the loader's camera models) for one noise variant, in this process."""
    from data import dataset as ds_mod
    C = variant_configs(name)
    D._CFG["c"] = C
    original_sensor = ds_mod.SceneDataset._camera_sensor
    if name == "gaussian":
        ds_mod.SceneDataset._camera_sensor = lambda self, band, cam, meta, rng: _gaussian_sensor(band, cam, meta, rng, self.cfgs["sensors"])
    elif name == "foreign":
        def foreign(self, band, cam, meta, rng):
            f = original_sensor(self, band, cam, meta, rng)
            return lambda x: _foreign_post(f(x), rng)
        ds_mod.SceneDataset._camera_sensor = foreign
    original_band = ds_mod.SceneDataset.band_image

    def band_image(self, meta, band, rng, vis_km, ratio, reg):
        if band in UNUSED_BANDS:
            return np.zeros((self.H, self.W, 1), np.float32)
        return original_band(self, meta, band, rng, vis_km, ratio, reg)
    ds_mod.SceneDataset.band_image = band_image


# ============================================================================ steps
def prepare(min_scenes, limit=None, steps=None):
    src = ROOT / D.SOURCE
    done = [i for i in range(D.N_POP) if D.scene_done(src, i)][:limit]
    if len(done) < min_scenes:
        raise ValueError(f"only {len(done)} scenes rendered; the ablation needs {min_scenes}")
    metas = {f"s{i:06d}": D.read(src / "meta" / f"s{i:06d}.json") for i in done}
    tr = sorted(s for s, m in metas.items() if m["split"] == "train")
    va = sorted(s for s, m in metas.items() if m["split"] == "val")
    test = sorted(va + tr[::3])                                         # the M1 test split is never used here
    rest = [s for s in tr if s not in set(test)]
    val = rest[::8]
    train = [s for s in rest if s not in set(val)]
    split = dict(train=train, val=val, test=test, scenes=len(done), sealed_m1_test_scenes_excluded=len(metas) - len(tr) - len(va))
    OUT.mkdir(parents=True, exist_ok=True)
    D.atomic_json(OUT / "split.json", split)
    for v in TEST_VARIANTS:
        out = OUT / v
        if (out / "manifest.json").exists():
            continue
        D.manifest(out, scenes=set(train + val + test), smoke=True)
        m = D.read(out / "manifest.json")
        m["splits"] = dict(train=train, val=val)
        m["cohorts"] = dict(test=test)
        m["training"] = dict(m["training"], **TRAINING)
        if steps:                                                      # mechanics checks only
            m["training"].update(steps=steps, val_every=max(1, steps // 2),
                                 hard_negatives=dict(TRAINING["hard_negatives"], steps=[max(1, steps // 3)], frames=2))
        m["experiment"] = f"noise-ablation-{v}"
        D.atomic_json(out / "manifest.json", m)
    log(f"ablation split: {len(train)} train, {len(val)} validation, {len(test)} test scenes (of {len(done)} rendered; "
        f"the M1 test split is excluded)")


def cache(variant):
    if variant == "mixed":
        mix_cache()
        return
    setup_variant(variant)
    parts = {"test"} if variant == "foreign" else {"train", "val", "test"}
    D.cache(OUT / variant, parts)


def _link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mix_cache(seed=20261017):
    """The mixed training set from the four models' caches: each training frame file, and each validation scene, from
    one model drawn at random (hard links; nothing is re-made). The draws are saved in mixed/mix.json."""
    split = D.read(OUT / "split.json")
    dst = OUT / "mixed"
    for part in ("train", "eval"):
        (dst / "cache" / part).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    picks = {}
    stems = [f"{sid}_r{k}" for sid in split["train"] for k in range(TRAINING["realisations"])]
    stems += [f"{sid}_a{j}" for sid in split["train"] for j in range(TRAINING["absent_realisations"])]
    for stem in sorted(stems):
        v = NOISE_MODELS[int(rng.integers(len(NOISE_MODELS)))]
        picks[f"train/{stem}"] = v
        for ext in (".npy", ".json"):
            src = OUT / v / "cache" / "train" / (stem + ext)
            if src.exists():
                _link(src, dst / "cache" / "train" / src.name)
            elif ext == ".npy":
                raise FileNotFoundError(f"{src}: build the {v} cache first")
    for sid in split["val"]:
        v = NOISE_MODELS[int(rng.integers(len(NOISE_MODELS)))]
        picks[f"eval/{sid}"] = v
        src_dir = OUT / v / "cache" / "eval"
        for src in [src_dir / f"{sid}.json"] + sorted(src_dir.glob(f"{sid}_*.npy")):
            _link(src, dst / "cache" / "eval" / src.name)
    m = D.read(OUT / "camera" / "manifest.json")
    m["experiment"] = "noise-ablation-mixed"
    D.atomic_json(dst / "manifest.json", m)
    D.atomic_json(dst / "mix.json", dict(seed=seed, picks=picks))
    counts = {v: sum(1 for p in picks.values() if p == v) for v in NOISE_MODELS}
    log(f"mixed cache: {len(picks)} draws {counts}")


def train(variant, seed):
    setup_variant(variant)
    name = MODEL if seed == D.TRAIN_SEED else f"{MODEL}_s{seed}"
    D.SPECS[name] = D.SPECS[MODEL]
    D.TRAIN_SEED = int(seed)
    D.train(OUT / variant, [name])


def evaluate(seeds, only=None):
    """The cross matrix. With `only`, just those training variants are run and added to the saved scores and rows."""
    import torch
    import m1_det2_eval as E
    from models import cnn_detector as cd
    split = D.read(OUT / "split.json")
    test = split["test"]
    cfg = D.eval_cfg()
    budgets = D.TRAINING["fa_budgets"]
    metas = {s: D.meta_of(s) for s in test}
    torch.set_num_threads(D.threads())
    arrays, rows = {}, {}
    if only:
        arrays, rows = load_scores(), D.read(OUT / "results.json")["rows"]
    for seed in seeds:
        for tv in TRAIN_VARIANTS:
            if only and tv not in only:
                continue
            name = MODEL if seed == D.TRAIN_SEED else f"{MODEL}_s{seed}"
            path = OUT / tv / "runs" / name / "best.pt"
            if not path.exists():
                continue
            model, ck = cd.load(path)
            for sv in TEST_VARIANTS:
                pres = D.records(OUT / sv, model, test, "p", len(D.TEST_PRESENT), None, metas)
                absn = D.records(OUT / sv, model, test, "a", len(D.TEST_ABSENT), None, metas)
                best, absent = E.scene_arrays(pres, absn, cfg)
                froc, pd = E.froc_from_arrays(best, absent, budgets)
                t = float(ck["threshold_fa01"])
                n_abs = sum(len(a) for a in absent)
                fa = float(sum(int(np.sum(a >= t)) for f in absent for a in f)) / max(1, n_abs)
                key = (seed, tv, sv)
                arrays[key] = (best, absent)
                rows[f"{seed}|{tv}|{sv}"] = dict(seed=seed, train=tv, test=sv, froc=froc, froc_ci95=E.froc_ci(best, absent, budgets, 500),
                                                 pd_at=pd, pd_val_threshold=float(np.mean(best >= t)), fa_per_frame_val_threshold=fa,
                                                 val_froc=ck["val_froc"], step=ck["step"])
                log(f"seed {seed} train {tv:8s} test {sv:8s}: FROC {froc:.3f}  Pd@thr {np.mean(best >= t):.3f}  FA/frame {fa:.4f}")
    save_scores(arrays)
    report(arrays, rows, seeds, {k: (len(v) if isinstance(v, list) else v) for k, v in split.items()})


def save_scores(arrays):
    """Every model's per-scene scores on every test set (scores.json), so the analysis can be redone without inference."""
    raw = {f"{s}|{tv}|{sv}": dict(best=[float(x) for x in b], absent=[[[float(x) for x in f] for f in a] for a in ab])
           for (s, tv, sv), (b, ab) in arrays.items()}
    tmp = OUT / "scores.json.tmp"
    tmp.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")
    tmp.replace(OUT / "scores.json")


def load_scores():
    arrays = {}
    for k, v in D.read(OUT / "scores.json").items():
        s, tv, sv = k.split("|")
        arrays[(int(s), tv, sv)] = (np.asarray(v["best"], float), [[np.asarray(f, float) for f in a] for a in v["absent"]])
    return arrays


def report(arrays, rows, seeds, split_sizes):
    """Paired differences per seed and pooled over the seeds -> results.json and results.md."""
    import m1_det2_eval as E
    budgets = D.TRAINING["fa_budgets"]
    diffs = {}
    for seed in seeds:
        for ref in REFERENCES:
            for sv in TEST_VARIANTS:
                for tv in TRAIN_VARIANTS:
                    a, b = (seed, ref, sv), (seed, tv, sv)
                    if tv != ref and tv not in REFERENCES[:REFERENCES.index(ref)] and a in arrays and b in arrays:
                        diffs[f"{seed}|{ref}-{tv}|{sv}"] = E.paired_froc(arrays[a], arrays[b], budgets, 1000)
    pool = None
    if len(seeds) > 1:
        try:
            pool = pooled(arrays, seeds, budgets)
        except Exception as e:                       # the per-seed results are still written
            log(f"pooled analysis failed: {e!r}")
    res = dict(split=split_sizes, rows=rows, diffs=diffs, pooled=pool,
               training=D.read(OUT / "camera" / "manifest.json")["training"], model=MODEL)
    D.atomic_json(OUT / "results.json", D.clean(res))
    write_results(res, seeds)


def pooled(arrays, seeds, budgets, n_boot=1000, rng_seed=0):
    """Seed-averaged FROC, one scene bootstrap shared by every model, seed and test set (added 2026-10-17, before
    any test result was computed): each reference (camera, mixed) minus each other training noise on each test set,
    and how often each training noise has the highest worst-case FROC over the five test sets. Also (added after the
    first results) the leave-one-scene-out range of each model's mean and worst-case FROC: how far one test scene
    moves them."""
    import m1_det2_eval as E
    n = len(next(iter(arrays.values()))[0])

    def table(k):
        froc = {key: E.froc_from_arrays(b[k], [ab[i] for i in k], budgets)[0] for key, (b, ab) in arrays.items()}
        out = {}
        for tv in TRAIN_VARIANTS:
            for sv in TEST_VARIANTS:
                v = [froc[(s, tv, sv)] for s in seeds if (s, tv, sv) in froc]
                if len(v) == len(seeds):
                    out[(tv, sv)] = float(np.mean(v))
        return out

    def worst(t):
        return {tv: min(t[(tv, sv)] for sv in TEST_VARIANTS) for tv in TRAIN_VARIANTS
                if all((tv, sv) in t for sv in TEST_VARIANTS)}

    rng = np.random.default_rng(rng_seed)
    base = table(np.arange(n))
    boots = [table(rng.integers(0, n, n)) for _ in range(n_boot)]
    diffs = {}
    for ref in REFERENCES:
        for sv in TEST_VARIANTS:
            for tv in TRAIN_VARIANTS:
                if tv == ref or tv in REFERENCES[:REFERENCES.index(ref)] or (ref, sv) not in base or (tv, sv) not in base:
                    continue
                d = [t[(ref, sv)] - t[(tv, sv)] for t in boots]
                lo, hi = np.percentile(d, [2.5, 97.5])
                diffs[f"{ref}-{tv}|{sv}"] = dict(diff=base[(ref, sv)] - base[(tv, sv)], lo=float(lo), hi=float(hi),
                                                 significant=bool(lo > 0 or hi < 0))
    wins = {tv: 0.0 for tv in worst(base)}
    for t in boots:
        w = worst(t)
        top = [tv for tv in w if w[tv] == max(w.values())]
        for tv in top:
            wins[tv] += 1.0 / len(top)

    def mean(t):
        return {tv: float(np.mean([t[(tv, sv)] for sv in TEST_VARIANTS])) for tv in worst(t)}
    loo = [table(np.delete(np.arange(n), i)) for i in range(n)]
    jack = {tv: dict(mean=[min(mean(t)[tv] for t in loo), max(mean(t)[tv] for t in loo)],
                     worst=[min(worst(t)[tv] for t in loo), max(worst(t)[tv] for t in loo)]) for tv in worst(base)}
    return dict(seeds=list(seeds), resamples=n_boot, n_scenes=n, diffs=diffs, worst=worst(base), mean=mean(base),
                best_worst_case_share={tv: v / n_boot for tv, v in wins.items()}, leave_one_scene_out=jack)


def write_results(res, seeds):
    sp = res["split"]
    L = ["# Noise-model verification (rgb_lwir, the student's cameras)", "",
         f"{sp['train']} training, {sp['val']} validation and {sp['test']} test scenes, none from M1's sealed test split; "
         f"{res['training']['steps']} steps of {res['training'].get('batch', 16)} crops per model; FROC = mean detection probability at 0.005-0.05 false alarms per "
         "target-free frame of the test set itself (each test set has its own false-alarm thresholds).", ""]
    for seed in seeds:
        L += [f"## Seed {seed}: FROC [95% CI] (rows: trained on; columns: tested on)", "",
              "| Trained on | " + " | ".join(TEST_VARIANTS) + " | mean | worst |", "|---|" + "---:|" * (len(TEST_VARIANTS) + 2)]
        for tv in TRAIN_VARIANTS:
            cells, vals = [], []
            for sv in TEST_VARIANTS:
                r = res["rows"].get(f"{seed}|{tv}|{sv}")
                if r is None:
                    cells.append("-")
                    continue
                vals.append(r["froc"])
                cells.append(f"{r['froc']:.3f} [{r['froc_ci95']['lo']:.2f}, {r['froc_ci95']['hi']:.2f}]")
            if vals:
                L.append(f"| {tv} | " + " | ".join(cells) + f" | {np.mean(vals):.3f} | {np.min(vals):.3f} |")
        L += ["", "Operating point (each model's own validation threshold for 0.01 false alarms per frame): Pd / measured "
              "false alarms per target-free frame:", "", "| Trained on | " + " | ".join(TEST_VARIANTS) + " |",
              "|---|" + "---:|" * len(TEST_VARIANTS)]
        for tv in TRAIN_VARIANTS:
            cells = []
            for sv in TEST_VARIANTS:
                r = res["rows"].get(f"{seed}|{tv}|{sv}")
                cells.append("-" if r is None else f"{r['pd_val_threshold']:.2f} / {r['fa_per_frame_val_threshold']:.3f}")
            L.append(f"| {tv} | " + " | ".join(cells) + " |")
        L += ["", "Paired FROC differences, reference (camera, mixed) minus the other model (same test scenes, 1,000 "
              "scene-bootstrap resamples):", ""]
        for k, v in res["diffs"].items():
            s, pair, sv = k.split("|")
            if int(s) == seed:
                L.append(f"- {pair} on {sv}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" + (" (significant)" if v["significant"] else ""))
        L.append("")
    if len(seeds) > 1:
        L += [f"## Seeds {', '.join(str(s) for s in seeds)}: mean FROC (spread = half the range between seeds)", "",
              "| Trained on | " + " | ".join(TEST_VARIANTS) + " | mean | worst |", "|---|" + "---:|" * (len(TEST_VARIANTS) + 2)]
        for tv in TRAIN_VARIANTS:
            cells, means = [], []
            for sv in TEST_VARIANTS:
                v = [res["rows"][f"{s}|{tv}|{sv}"]["froc"] for s in seeds if f"{s}|{tv}|{sv}" in res["rows"]]
                if not v:
                    cells.append("-")
                    continue
                means.append(float(np.mean(v)))
                cells.append(f"{np.mean(v):.3f} ± {(max(v) - min(v)) / 2:.3f}")
            if means:
                L.append(f"| {tv} | " + " | ".join(cells) + f" | {np.mean(means):.3f} | {np.min(means):.3f} |")
        L.append("")
    P = res.get("pooled")
    if P:
        L += [f"## Seeds {', '.join(str(s) for s in P['seeds'])} pooled: each reference minus each other training noise "
              f"(seed-averaged FROC, {P['resamples']:,} scene-bootstrap resamples shared by all models)", ""]
        for k, v in P["diffs"].items():
            pair, sv = k.split("|")
            L.append(f"- {pair} on {sv}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" + (" (significant)" if v["significant"] else ""))
        J = P.get("leave_one_scene_out", {})
        L += ["", "| Trained on | worst-case FROC | mean FROC | share of resamples with the highest worst case | "
              "worst case, one scene left out (min-max) | mean, one scene left out (min-max) |", "|---|---:|---:|---:|---:|---:|"]
        for tv in P["worst"]:
            j = J.get(tv)
            jj = f" {j['worst'][0]:.3f}-{j['worst'][1]:.3f} | {j['mean'][0]:.3f}-{j['mean'][1]:.3f} |" if j else " - | - |"
            L.append(f"| {tv} | {P['worst'][tv]:.3f} | {P['mean'][tv]:.3f} | {P['best_worst_case_share'][tv]:.2f} |" + jj)
        L.append("")
    L += ["## How to read this", "",
          "- The diagonal is each model on its own kind of frames; off the diagonal, on another camera model. A real camera "
          "is none of these models; `foreign` adds processing none of them has (blur, gamma, JPEG).",
          "- 'No deterioration' = the camera-trained model is not worse than the others on their own frames; 'gain' = it is "
          "better on the camera and foreign frames. The worst column is the robustness figure that matters for an unknown "
          "real camera.",
          "- Synthetic test frames cannot show the gain on real images; the published sim-to-real gains of physically "
          "modelled camera effects are in docs/sensor_noise.md."]
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {OUT / 'results.md'}")


def run_all(min_scenes, seeds, limit=None, steps=None):
    """Every step, each variant in its own process (the variants patch the loader)."""
    if not (OUT / "split.json").exists():
        prepare(min_scenes, limit, steps)
    py = [sys.executable, "-u", str(Path(__file__)), "--out", str(OUT)]
    for v in TEST_VARIANTS + ("mixed",):              # mixed links the four caches, so it comes last
        subprocess.run(py + ["cache", "--variant", v], check=True)
    for seed in seeds:
        for v in TRAIN_VARIANTS:
            subprocess.run(py + ["train", "--variant", v, "--seed", str(seed)], check=True)
    subprocess.run(py + ["evaluate", "--seeds"] + [str(s) for s in seeds], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "cache", "train", "evaluate", "report", "all"])
    ap.add_argument("--variant", choices=TEST_VARIANTS + ("mixed",))
    ap.add_argument("--seed", type=int, default=D.TRAIN_SEED)
    ap.add_argument("--seeds", type=int, nargs="*", default=[D.TRAIN_SEED])
    ap.add_argument("--min-scenes", type=int, default=200)
    ap.add_argument("--out", default=None, help="results folder (default out/m1/noise_ablation)")
    ap.add_argument("--limit-scenes", type=int, default=None, help="prepare: only the first N rendered scenes (checks)")
    ap.add_argument("--steps", type=int, default=None, help="prepare: training steps (checks only)")
    ap.add_argument("--only", nargs="*", choices=TRAIN_VARIANTS, default=None,
                    help="evaluate: only these training variants, added to the saved scores")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    D.keep_awake(True)
    try:
        if a.command == "prepare":
            prepare(a.min_scenes, a.limit_scenes, a.steps)
        elif a.command == "cache":
            cache(a.variant)
        elif a.command == "train":
            if a.variant not in TRAIN_VARIANTS:
                ap.error("train needs a training variant")
            train(a.variant, a.seed)
        elif a.command == "evaluate":
            evaluate(a.seeds, a.only)
        elif a.command == "report":                  # redo the tables from scores.json, without inference
            old = D.read(OUT / "results.json")
            report(load_scores(), old["rows"], a.seeds, old["split"])
        else:
            run_all(a.min_scenes, a.seeds, a.limit_scenes, a.steps)
    finally:
        D.keep_awake(False)


if __name__ == "__main__":
    main()
