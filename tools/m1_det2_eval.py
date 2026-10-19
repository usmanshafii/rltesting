"""M1 detector, four-sensor revision: held-out evaluation (once) and false alarms on video.

    python tools/m1_det2.py evaluate     all seven runs frozen -> test_lock.json -> report.json, results.md
    python tools/m1_det2.py sequences    after evaluate: confirmed false tracks on target-free sky video
                                         and detection/track latency on rendered flight sequences

The FROC score summarises detection probability at fixed false-alarm rates measured on whole target-free
frames of the same scenes; the frozen validation thresholds give the operating points (does the
false-alarm rate measured on test match the 0.01 budget set on validation?). Intervals resample whole
scenes (a scene's present frame and its target-free frames together). Results are also split by whether
the cooled-LWIR and MWIR cameras saw the target (a zoom camera may not cover the uncooled camera's field).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import m1_det2 as D

NAMES = D.ORDER
# The same weights with sensors switched off: the two-band models trained with band dropout lose one band; the teacher
# (trained on every sensor subset) is run on each sensor alone and on the student's two sensors.
STUDENT_VARIANTS = {"@lwir_only": dict(rgb=0, lwir=1), "@rgb_only": dict(rgb=1, lwir=0)}
TEACHER_VARIANTS = dict({f"@{b}_only": {x: int(x == b) for x in D.BANDS} for b in D.BANDS},
                        **{"@rgb_lwir": dict(rgb=1, lwir=1, lwir_cooled=0, mwir=0)})
VARIANTS = {"rgb_lwir": STUDENT_VARIANTS, "student": STUDENT_VARIANTS, "student_ctrl": STUDENT_VARIANTS,
            "teacher": TEACHER_VARIANTS}


# ============================================================================ per-scene arrays
def scene_arrays(pres, absn, cfg):
    """best[i]: highest centre-matched score in scene i's present frame; absent[i]: list of score arrays
    of its target-free frames (same scene order)."""
    best = D.best_match_scores(pres, cfg)
    by = {}
    for r in absn:
        by.setdefault(r["scene"], []).append(np.asarray(r["preds"]["scores"], float))
    return best, [by.get(r["scene"], []) for r in pres]


def froc_from_arrays(best, absent, budgets):
    frames = [f for a in absent for f in a]
    scores = np.sort(np.concatenate(frames + [np.zeros(0)]))[::-1]
    pd = {}
    for b in budgets:
        allowed = int(math.floor(b * len(frames)))
        t = 0.01 if allowed >= len(scores) else float(np.nextafter(scores[allowed], 2.0))
        pd[str(b)] = float(np.mean(best >= t)) if len(best) else float("nan")
    return float(np.mean(list(pd.values()))), pd


def froc_ci(best, absent, budgets, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        k = rng.integers(0, len(best), len(best))
        vals.append(froc_from_arrays(best[k], [absent[i] for i in k], budgets)[0])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), resamples=n_boot)


def paired_froc(a, b, budgets, n_boot=1000, seed=0):
    """FROC(a) - FROC(b) on the same scenes, paired scene bootstrap."""
    (ba, aa), (bb, ab) = a, b
    base = froc_from_arrays(ba, aa, budgets)[0] - froc_from_arrays(bb, ab, budgets)[0]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ba), len(ba))
        d.append(froc_from_arrays(ba[k], [aa[i] for i in k], budgets)[0] - froc_from_arrays(bb[k], [ab[i] for i in k], budgets)[0])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=float(base), lo=float(lo), hi=float(hi), significant=bool(lo > 0 or hi < 0), n_scenes=len(ba))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ============================================================================ summary of one model
def summarise(pres, absn, ck, cfg, budgets):
    from eval import metrics
    from eval.pd_fit import aspect_group
    from eval.pd_fit import summarise as pd_summarise
    best, absent = scene_arrays(pres, absn, cfg)
    score, pd_at = froc_from_arrays(best, absent, budgets)
    n_abs = sum(len(a) for a in absent)
    all_abs = np.concatenate([f for a in absent for f in a] + [np.zeros(0)])
    res = dict(n_scenes=len(pres), n_absent_frames=n_abs, froc=score, froc_ci95=froc_ci(best, absent, budgets), pd_at_fa=pd_at,
               ap50=metrics.ap(pres, cfg=cfg), ap50_ci95=metrics.ap_interval(pres, cfg), ap50_95=metrics.ap_50_95(pres, cfg=cfg))
    res["operating_points"] = {}
    for key in ("threshold_fa01", "threshold_fa05", "threshold_f1"):
        t = float(ck[key])
        k = int(np.sum(best >= t))
        op = metrics.operating_point(pres, dict(cfg, score_thresh=t))
        res["operating_points"][key] = dict(threshold=t, pd_centre=k / max(1, len(best)), pd_tp=k, n=len(best),
                                            pd_ci95=wilson(k, len(best)), fa_per_target_free_frame=float(np.sum(all_abs >= t)) / max(1, n_abs),
                                            iou_recall=op["recall"], iou_tp=op["tp"], precision_present_frames=op["precision"])
    t = float(ck["threshold_fa01"])
    det = best >= t
    res["bins"] = {}
    for b in sorted({o["size_bin"] for r in pres for o in r["gts"]}):
        ix = [i for i, r in enumerate(pres) if r["gts"][0]["size_bin"] == b]
        res["bins"][b] = dict(n=len(ix), pd_centre_fa01=float(det[ix].mean()), tp=int(det[ix].sum()),
                              ap50=metrics.ap(pres, bin_name=b, cfg=cfg))
    groups = {"day": lambda r: r["conditions"]["lighting"] == "day", "night": lambda r: r["conditions"]["lighting"] == "night",
              "head": lambda r: aspect_group(r["target_aspect_deg"]) == "head",
              "beam": lambda r: aspect_group(r["target_aspect_deg"]) == "beam",
              "tail": lambda r: aspect_group(r["target_aspect_deg"]) == "tail",
              "cooled_sees": lambda r: r["target_in_view"].get("lwir_cooled", True),
              "cooled_misses": lambda r: not r["target_in_view"].get("lwir_cooled", True),
              "mwir_sees": lambda r: r["target_in_view"].get("mwir", True),
              "mwir_misses": lambda r: not r["target_in_view"].get("mwir", True)}
    res["slices"] = {}
    for g, f in groups.items():
        ix = [i for i, r in enumerate(pres) if f(r)]
        if ix:
            res["slices"][g] = dict(n=len(ix), pd_centre_fa01=float(det[ix].mean()), tp=int(det[ix].sum()),
                                    froc=froc_from_arrays(best[ix], [absent[i] for i in ix], budgets)[0])
    dets = [dict(scene=r["scene"], size_px=r["gts"][0]["size_px"], detected=bool(det[i]),
                 group=aspect_group(r["target_aspect_deg"]), lighting=r["conditions"]["lighting"]) for i, r in enumerate(pres)]
    res["px50"] = {L: pd_summarise([d for d in dets if L == "all" or d["lighting"] == L], n_boot=500)
                   for L in ("all", "day", "night")}
    return res, (best, absent)


# ============================================================================ evaluate
def evaluate(out):
    import torch
    from eval import metrics
    from models import cnn_detector as cd
    m = D.read(out / "manifest.json")
    cfg = D.eval_cfg()
    budgets = m["training"]["fa_budgets"]
    paths = {n: out / "runs" / n / "best.pt" for n in NAMES}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        raise ValueError(f"freeze all runs before opening the test cohorts; missing {missing}")
    if not (out / "cache" / "eval" / f"{m['cohorts']['test'][0]}_p0.npy").exists():
        raise FileNotFoundError("build the test cache first: python tools/m1_det2.py cache --part test")
    lock = dict(manifest_sha256=D.sha(out / "manifest.json"), weights={n: D.sha(p) for n, p in paths.items()},
                evaluation_settings=cfg, metrics=m["metrics"])
    if (out / "test_lock.json").exists() and D.read(out / "test_lock.json") != lock:
        raise ValueError("test cohorts were already opened with different weights or settings")
    D.atomic_json(out / "test_lock.json", lock)
    torch.set_num_threads(D.threads())
    cks = {n: torch.load(p, map_location="cpu", weights_only=False) for n, p in paths.items()}
    gate = dict(rule=m["metrics"]["gate"], teacher_val_froc=cks["teacher"]["val_froc"],
                baseline_val_froc=cks["rgb_lwir"]["val_froc"], teacher_val_ap50=cks["teacher"]["val_ap50"],
                baseline_val_ap50=cks["rgb_lwir"]["val_ap50"])
    gate["passed"] = bool(gate["teacher_val_froc"] > gate["baseline_val_froc"] and gate["teacher_val_ap50"] > gate["baseline_val_ap50"])
    report = dict(scope="M1 four-sensor preliminary evaluation: synthetic open-sky scenes in Lahore conditions (station "
                        "hours of NOAA ISD 416400), RGB + LWIR uncooled + LWIR cooled + MWIR cameras of each scene's rig, "
                        "compact CNN (0.37 M parameters), one training seed",
                  evaluation_settings=cfg, metrics=m["metrics"], gate_on_validation=gate, cohorts={}, comparisons={},
                  coverage=m["sensors"]["coverage"])
    preds = {}
    for cohort in ("test",):
        sids = m["cohorts"][cohort]
        metas = {s: D.meta_of(s) for s in sids}
        report["cohorts"][cohort] = dict(n_scenes=len(sids), note=m["cohort_notes"][cohort], models={})
        preds[cohort], arrays = {}, {}
        for name in NAMES:
            model, ck = cd.load(paths[name])
            variants = {"": None}
            variants.update(VARIANTS.get(name, {}))
            for tag, present in variants.items():
                pres = D.records(out, model, sids, "p", len(D.TEST_PRESENT), present, metas)
                absn = D.records(out, model, sids, "a", len(D.TEST_ABSENT), present, metas)
                res, arr = summarise(pres, absn, ck, cfg, budgets)
                res.update(bands=ck["bands"] if present is None else [b for b, v in present.items() if v],
                           checkpoint=str(paths[name].relative_to(D.ROOT)), checkpoint_sha256=D.sha(paths[name]),
                           best_step=ck["step"], val_froc=ck["val_froc"], val_ap50=ck["val_ap50"])
                if tag == "":
                    un, _ = cd.load(paths[name].with_name("untrained.pt"))
                    up = D.records(out, un, sids, "p", 1, None, metas)
                    ua = D.records(out, un, sids, "a", len(D.TEST_ABSENT), None, metas)
                    res["untrained"] = dict(froc=froc_from_arrays(*scene_arrays(up, ua, cfg), budgets)[0], ap50=metrics.ap(up, cfg=cfg))
                report["cohorts"][cohort]["models"][name + tag] = res
                preds[cohort][name + tag] = dict(present=pres, absent=absn)
                arrays[name + tag] = arr
                D.log(f"{cohort} {name + tag}: FROC {res['froc']:.3f}, Pd@FA0.01 {res['operating_points']['threshold_fa01']['pd_centre']:.3f}, "
                      f"AP50 {res['ap50']:.3f}")
        P = {k: v["present"] for k, v in preds[cohort].items()}
        pairs = [("student", "rgb_lwir"), ("student", "student_ctrl"), ("student_ctrl", "rgb_lwir"),
                 ("teacher", "rgb_lwir"), ("rgb_lwir", "lwir_only"), ("student@lwir_only", "lwir_only"),
                 ("lwir_cooled_only", "lwir_only"), ("mwir_only", "lwir_only")]
        report["comparisons"][cohort] = {
            f"{a}_minus_{b}": dict(froc=paired_froc(arrays[a], arrays[b], budgets),
                                   ap50=metrics.bootstrap_diff(P[a], P[b], lambda rr: metrics.ap(rr, cfg=cfg)))
            for a, b in pairs}
        report["cohorts"][cohort]["regimes"] = regimes(out, sids, preds[cohort], cks, cfg)
    D.atomic_json(out / "report.json", D.clean(report))
    (out / "predictions.json").write_text(json.dumps(D.clean(preds)), encoding="utf-8")
    write_results(out, m, report)
    learning_curves(out)
    gallery(out, m, preds, cks, cfg)
    D.log(f"wrote {out / 'results.md'}")


REGIME_MODELS = ("lwir_only", "lwir_cooled_only", "mwir_only", "rgb_lwir", "teacher", "teacher@rgb_lwir", "student",
                 "student_ctrl", "student@lwir_only")
# The regimes use the matched-filter detectability d' (eval/detectability.py: D >= 5, F 2-5, N < 2, definitions). The
# box-against-ring contrast-to-noise used before 2026-10-18 is not noise-referenced for small targets (its ring holds
# the target's own blur), so it is reported beside d' for comparison only.
from eval.detectability import DETECTABLE, FAINT, NONE, dprime, signal_class  # noqa: E402

REGIME_MAX_PX = 16.0              # the small-target subset, reported separately (most detection-range limits are here)


REGIME_CASES = {
    "student sensors detectable": lambda c: max(c["rgb"], c["lwir"]) >= DETECTABLE,
    "student faint, privileged detectable": lambda c: FAINT <= max(c["rgb"], c["lwir"]) < DETECTABLE
    and max(c["lwir_cooled"], c["mwir"]) >= DETECTABLE,
    "all faint (fusion of sub-threshold signals)": lambda c: max(c.values()) < DETECTABLE and max(c.values()) >= FAINT,
    "student no signal, privileged detectable": lambda c: max(c["rgb"], c["lwir"]) < FAINT
    and max(c["lwir_cooled"], c["mwir"]) >= DETECTABLE,
    "no signal in any sensor": lambda c: max(c.values()) < FAINT,
    "one sensor detectable": lambda c: sum(v >= DETECTABLE for v in c.values()) == 1,
    "two sensors detectable": lambda c: sum(v >= DETECTABLE for v in c.values()) == 2,
    "three or four detectable": lambda c: sum(v >= DETECTABLE for v in c.values()) >= 3,
}


def scene_detectability(out, sid, meta=None) -> dict:
    """d' of the scene's target in each band's cached test frames: p0 and a0 share noise seed 17, so p0 - a0 is the
    target's exact signal; a1 (seed 1017) splits sensor noise from clutter. 0 where a camera does not see the target."""
    meta = meta or D.meta_of(sid)
    boxes = [o["box"] for o in D.gt_objects(meta)]
    cov = D.coverage_boxes(meta)
    p0, a0, a1 = (np.load(out / "cache" / "eval" / f"{sid}_{k}.npy", mmap_mode="r") for k in ("p0", "a0", "a1"))
    res = {}
    for b, ch in D.EVAL_CH.items():
        if not boxes or not D.centre_inside(boxes[0], cov[b]):
            res[b] = dict(NONE)
            continue
        res[b] = dprime(*(np.asarray(f[..., ch], np.float32) for f in (p0, a0, a1)), boxes[0], cover=cov[b])
    return res


def _cases(dv, members_of, cohort_preds, cks, cfg):
    cases = {}
    for case, rule in REGIME_CASES.items():
        members = {s for s in members_of if rule(dv[s])}
        row = dict(n=len(members))
        for name in REGIME_MODELS:
            if name not in cohort_preds:
                continue
            pres = [r for r in cohort_preds[name]["present"] if r["scene"] in members]
            t = float(cks[name.split("@")[0]]["threshold_fa01"])
            row[name] = dict(tp=int(np.sum(D.best_match_scores(pres, cfg) >= t)) if pres else 0, n=len(pres))
        cases[case] = row
    return cases


def regimes(out, sids, cohort_preds, cks, cfg):
    """The teacher-student design's cases (docs/m1_v2.md): each test scene is classed by its target's matched-filter
    detectability d' in each sensor's network-input frame (eval/detectability.py; 0 where that camera does not see it);
    per case, detections of each model at its own 0.01 false-alarm threshold (centre match), for all targets and for
    those up to REGIME_MAX_PX, and the count of every sensor pattern (R L C M)."""
    size = {r["scene"]: r["gts"][0]["size_px"] for r in next(iter(cohort_preds.values()))["present"]}
    small = [s for s in sids if size.get(s, 1e9) <= REGIME_MAX_PX]
    det = {s: scene_detectability(out, s) for s in sids}
    dv = {s: {b: v["d"] for b, v in det[s].items()} for s in sids}
    patterns = {}
    for s in sids:
        key = " ".join(signal_class(dv[s][b]) for b in ("rgb", "lwir", "lwir_cooled", "mwir"))
        patterns[key] = patterns.get(key, 0) + 1
    cnr = {}
    for s in sids:
        info = D.read(out / "cache" / "eval" / f"{s}.json")
        cnr[s] = {b: v[0] for b, v in info["cnr"]["p0"].items()} if "cnr" in info and "p0" in info["cnr"] else {}
    return dict(thresholds=dict(measure="matched-filter d'", detectable=DETECTABLE, faint=FAINT, max_size_px=REGIME_MAX_PX),
                scenes=len(sids), small_scenes=len(small), cases=_cases(dv, sids, cohort_preds, cks, cfg),
                cases_small=_cases(dv, small, cohort_preds, cks, cfg),
                patterns=dict(sorted(patterns.items(), key=lambda kv: -kv[1])),
                dprime={s: {b: round(v["d"], 3) for b, v in det[s].items()} for s in sids},
                noise_share={s: {b: (None if v["noise_share"] is None else round(v["noise_share"], 3))
                                 for b, v in det[s].items()} for s in sids},
                cnr_for_comparison=cnr)


def write_results(out, m, report):
    L = ["# M1 four-sensor preliminary evaluation - held-out results", "",
         f"Frozen split: {len(m['splits']['train'])} training scenes, {len(m['splits']['val'])} validation scenes. "
         "Synthetic open-sky scenes in Lahore conditions (a station hour per scene), one target per present frame, one "
         f"training seed. Each test scene adds {len(D.TEST_ABSENT)} target-free frames (same sky, independent sensor "
         "noise) to measure false alarms on whole frames.", "",
         f"**Primary metric.** {m['metrics']['primary']}. A detection counts when {m['metrics']['detection_match']}.", ""]
    g = report["gate_on_validation"]
    L += [f"**Teacher gate (validation):** four-sensor teacher FROC {g['teacher_val_froc']:.3f} vs baseline "
          f"{g['baseline_val_froc']:.3f}; AP50 {g['teacher_val_ap50']:.3f} vs {g['baseline_val_ap50']:.3f} -> "
          f"**{'passed' if g['passed'] else 'FAILED (no distillation claim)'}**.", ""]
    cov = report.get("coverage", {}).get("test", {})
    if cov:
        L += ["**Camera coverage (test scenes).** " + "; ".join(
            f"{b.replace('_', ' ')}: covers the whole frame in {c['whole_grid']}/{c['scenes']}, sees the target in "
            f"{c['target_in_view']}/{c['scenes']}" for b, c in cov.items()), ""]
    for cohort, c in report["cohorts"].items():
        L += [f"## {cohort.replace('_', ' ')} ({c['n_scenes']} scenes)", "",
              "| Model | Inputs | FROC [95% CI] | Pd @ 0.01 FA/frame (val thr) | FA/frame measured | AP50 [95% CI] | AP50:95 | IoU recall @F1 | Untrained FROC / AP50 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, r in c["models"].items():
            o = r["operating_points"]["threshold_fa01"]
            f1 = r["operating_points"]["threshold_f1"]
            un = f"{r['untrained']['froc']:.3f} / {r['untrained']['ap50']:.3f}" if "untrained" in r else "-"
            L.append(f"| {name} | {' + '.join(r['bands'])} | {r['froc']:.3f} [{r['froc_ci95']['lo']:.3f}, {r['froc_ci95']['hi']:.3f}] | "
                     f"{o['pd_centre']:.3f} ({o['pd_tp']}/{o['n']}) | {o['fa_per_target_free_frame']:.4f} | "
                     f"{r['ap50']:.3f} [{r['ap50_ci95']['lo']:.3f}, {r['ap50_ci95']['hi']:.3f}] | {r['ap50_95']:.3f} | "
                     f"{f1['iou_recall']:.3f} ({f1['iou_tp']}/{f1['n']}) | {un} |")
        L += ["", "Paired differences (same scenes, 1,000 scene-bootstrap resamples):", ""]
        for k, v in report["comparisons"][cohort].items():
            L.append(f"- {k}: FROC {v['froc']['diff']:+.3f} [{v['froc']['lo']:+.3f}, {v['froc']['hi']:+.3f}]; "
                     f"AP50 {v['ap50']['diff']:+.3f} [{v['ap50']['lo']:+.3f}, {v['ap50']['hi']:+.3f}]")
        first = next(iter(c["models"].values()))
        L += ["", "Detection at 0.01 FA/frame by apparent size (centre match):", "",
              "| Model | " + " | ".join(sorted(first["bins"])) + " |", "|---|" + "---:|" * len(first["bins"])]
        for name, r in c["models"].items():
            if "@" in name:
                continue
            L.append(f"| {name} | " + " | ".join(f"{v['tp']}/{v['n']}" for _, v in sorted(r["bins"].items())) + " |")
        cols = [("day", "day"), ("night", "night"), ("head", "head-on"), ("beam", "beam"), ("tail", "tail-on"),
                ("cooled_sees", "cooled sees target"), ("cooled_misses", "cooled misses"), ("mwir_sees", "MWIR sees"),
                ("mwir_misses", "MWIR misses")]
        L += ["", "| Model | " + " | ".join(t for _, t in cols) + " |", "|---|" + "---:|" * len(cols)]
        for name, r in c["models"].items():
            if "@" in name:
                continue
            s = r["slices"]
            L.append(f"| {name} | " + " | ".join(f"{s[k]['tp']}/{s[k]['n']}" if k in s else "-" for k, _ in cols) + " |")
        rg = c["regimes"]
        cols = [m for m in REGIME_MODELS if any(m in row for row in rg["cases"].values())]
        L += ["", f"Signal regimes ({rg['scenes']} scenes; {rg['small_scenes']} with targets up to {rg['thresholds']['max_size_px']:g} px). "
              "The target's matched-filter detectability d' in each sensor's frame (eval/detectability.py): "
              f"detectable >= {rg['thresholds']['detectable']:g}, faint {rg['thresholds']['faint']:g}-{rg['thresholds']['detectable']:g}, "
              "none below; 0 where a camera does not see it. Detections at each model's 0.01 false-alarm threshold "
              "(all targets / targets up to the size limit):", "",
              "| Case | n | " + " | ".join(cols) + " |", "|---|---:|" + "---:|" * len(cols)]
        for case, row in rg["cases"].items():
            small = rg.get("cases_small", {}).get(case, {})
            L.append(f"| {case} | {row['n']} / {small.get('n', '-')} | " + " | ".join(
                (f"{row[k]['tp']}/{row[k]['n']}" + (f" ({small[k]['tp']}/{small[k]['n']})" if k in small else "")) if k in row else "-"
                for k in cols) + " |")
        L += ["", "Sensor patterns in the test scenes (RGB, LWIR uncooled, LWIR cooled, MWIR; D detectable, F faint, N none): "
              + ", ".join(f"{k}: {v}" for k, v in rg["patterns"].items())]
        L.append("")
    L += ["## How to read this", "",
          "- FROC and 'FA/frame measured' use the target-free frames of the same test scenes; 'Pd @ 0.01' uses the "
          "threshold frozen on validation, so 'FA/frame measured' shows whether that budget held on new scenes.",
          "- AP50 needs IoU >= .5, which a one-pixel error fails on 2-4 px boxes; it is reported, not the headline.",
          "- `@lwir_only` / `@rgb_only`: the same weights with the other band switched off (modality dropout); "
          "`teacher@<sensor>_only` and `teacher@rgb_lwir`: the teacher (trained on every sensor subset) with only those sensors.",
          "- teacher needs cooled LWIR and MWIR cameras at inference; it is a reference, not the deployed model. "
          "student - student_ctrl is the part of a student gain that the privileged cameras can be credited with.",
          "- lwir_cooled_only and mwir_only include the scenes where their zoom camera could not see the target "
          "(counted as misses); the 'sees' / 'misses' columns separate them.",
          "- Intervals resample scenes only; one training seed, synthetic priors and the sim-to-real gap are not in them.",
          "- Frame-level false alarms are not a video false-alarm rate: see sequences.md for confirmed tracks."]
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def learning_curves(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for name in NAMES:
        p = out / "runs" / name / "history.json"
        if not p.exists():
            continue
        hist = D.read(p)
        val = [r for r in hist if "val_froc" in r]
        ax[0].plot([r["step"] for r in val], [r["val_froc"] for r in val], marker="o", ms=3, label=name)
        k = 50
        loss = np.convolve([r["loss"] for r in hist], np.ones(k) / k, mode="valid")
        ax[1].plot(np.arange(len(loss)) + k, loss, label=name)
    ax[0].set(xlabel="Training step (batch 16)", ylabel="Validation FROC", ylim=(0, 1), title="Checkpoint selection: validation only")
    ax[1].set(xlabel="Training step", ylabel="Training loss (50-step mean)", yscale="log", title="Training loss")
    ax[0].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)


# ============================================================================ figures
def _display(img):
    """Standardised frame -> uint8 RGB for figures (display only)."""
    x = np.asarray(img, np.float32)
    x = np.clip(x * 10.0, -6, 18)
    x = (x + 6) / 24.0
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, -1)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def draw(frames, record, threshold, title, path):
    from PIL import Image, ImageDraw, ImageFont
    bands = [b for b in D.BANDS if b in frames]
    W, H = 640, 512
    canvas = Image.new("RGB", (W * len(bands), H + 320), (14, 18, 26))
    d = ImageDraw.Draw(canvas)
    fp = Path("C:/Windows/Fonts/segoeui.ttf")
    font = ImageFont.truetype(str(fp), 15) if fp.exists() else ImageFont.load_default()
    for col, b in enumerate(bands):
        arr = _display(frames[b])
        canvas.paste(Image.fromarray(arr), (W * col, 40))
        d.text((W * col + 10, 12), f"{b.upper()}  green = label, orange = saved-weight prediction (score >= {threshold:.2f})", fill="white", font=font)
        for o in record["gts"]:
            x0, y0, x1, y1 = o["box"]
            d.rectangle((x0 + W * col - 1, y0 + 39, x1 + W * col + 1, y1 + 41), outline=(50, 230, 110), width=1)
        for box, s in zip(record["preds"]["boxes"], record["preds"]["scores"]):
            if s < threshold:
                continue
            x0, y0, x1, y1 = box
            d.rectangle((x0 + W * col - 3, y0 + 37, x1 + W * col + 3, y1 + 43), outline=(255, 150, 40), width=2)
            d.text((x0 + W * col, max(40, y0 + 18)), f"{s:.2f}", fill=(255, 150, 40), font=font)
        if record["gts"]:
            g = record["gts"][0]["box"]
            cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            size = max(64, int(max(g[2] - g[0], g[3] - g[1]) * 1.6))
            l, t = int(np.clip(cx - size / 2, 0, W - size)), int(np.clip(cy - size / 2, 0, H - size))
            crop = Image.fromarray(arr).crop((l, t, l + size, t + size)).resize((256, 256), Image.Resampling.NEAREST)
            canvas.paste(crop, (W * col + 12, H + 56))
            k = 256 / size

            def box_in(bx, color, w):
                x0, y0, x1, y1 = (np.array(bx) - [l, t, l, t]) * k
                d.rectangle((W * col + 12 + x0, H + 56 + y0, W * col + 12 + x1, H + 56 + y1), outline=color, width=w)
            box_in(g, (50, 230, 110), 1)
            for box, s in zip(record["preds"]["boxes"], record["preds"]["scores"]):
                if s >= threshold:
                    box_in(box, (255, 150, 40), 2)
            seen = "" if record.get("target_in_view", {}).get(b, True) else "\nthis camera does not see the target"
            d.text((W * col + 290, H + 60), f"Enlarged at the label (display only)\nscene {record['scene_id']}\n"
                   f"size bin {record['gts'][0]['size_bin']} px\nrange {record['gts'][0]['range_m']:.0f} m{seen}",
                   fill=(220, 225, 230), font=font)
    d.text((10, H + 300), title, fill=(160, 170, 180), font=font)
    canvas.save(path)


def gallery(out, m, preds, cks, cfg):
    """First two successes and first two failures by scene ID in the test cohort, at the frozen 0.01 threshold (rule
    fixed before evaluation). Panels: the model's own cameras."""
    dest = out / "gallery"
    dest.mkdir(exist_ok=True)
    lines = ["# Test successes and failures", "",
             "Rule (fixed before evaluation): per model, the first two successes and first two failures in ascending "
             "scene ID, at the model's validation 0.01 false-alarm threshold; success = a prediction whose centre "
             "matches the label. Aggregate numbers are in results.md.", ""]
    for name in ("student", "rgb_lwir", "lwir_only", "teacher"):
        recs = sorted(preds["test"][name]["present"], key=lambda r: r["scene"])
        t = float(cks[name]["threshold_fa01"])
        best = D.best_match_scores(recs, cfg)
        got = {True: [], False: []}
        for r, b in zip(recs, best):
            if len(got[bool(b >= t)]) < 2:
                got[bool(b >= t)].append(r)
        for ok, rr in got.items():
            for r in rr:
                fr = D.eval_frames(out, r["scene"], cks[name]["bands"], "p", 0)
                fn = f"{name}_{'success' if ok else 'failure'}_{r['scene']}.png"
                draw(fr, dict(r, scene_id=r["scene"]), t,
                     f"{name} | test | {'SUCCESS' if ok else 'FAILURE'} | saved weights", dest / fn)
                lines.append(f"- {name} {'success' if ok else 'failure'}: [{fn}]({fn})")
    (dest / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================ video: confirmed tracks
class StreamCamera:
    """One physical RGB + LWIR camera pair for a video: fixed sensor parameters and fixed LWIR patterns
    (column offsets and gains, pixel non-uniformity), a column drift profile that wanders slowly
    (AR(1), ~8 s at 25 fps), and fresh temporal, row and temporal-column noise every frame. The same
    parameter ranges as training (D.LWIR_REALISM); per-frame percentile AGC as in training."""

    def __init__(self, S, seed, H, W, fps=25.0):
        from noise import lwir_fpn, rgb_noise
        self.lw, self.rn = lwir_fpn, rgb_noise
        self.S = S
        rng = np.random.default_rng([seed, 77])
        self.p_rgb = rgb_noise.sample_params(rng, S["rgb_noise"])
        self.p = lwir_fpn.sample_params(rng, S["lwir_noise"])
        self.col = rng.standard_normal(W) * self.p["col_k"]
        self.pix = rng.standard_normal((H, W)) * self.p["pix_k"]
        self.gain = rng.standard_normal(W) * self.p.get("col_gain", 0.0)
        self.knots = rng.standard_normal((5, W)) * self.p.get("col_drift_k", 0.0)
        self.rho = math.exp(-1.0 / (8.0 * fps))
        self.rng = np.random.default_rng([seed, 5])
        y = np.linspace(0.0, 4.0, H)
        self.i0 = np.minimum(np.floor(y).astype(int), 3)
        self.t = (y - self.i0)[:, None]

    def rgb(self, L, exp_ms):
        return self.rn.apply(L, self.rng, self.S["rgb_noise"], exp_ms, params=self.p_rgb)[0]

    def lwir(self, L, T_air_k):
        p, r = self.p, self.rng
        H, W = L.shape
        cpk = p["counts_per_k"] * self.lw._ref_scale(T_air_k)
        c0 = self.lw.to_counts(L, T_air_k, p)
        c = c0 + (self.col[None, :] + self.pix) * cpk + self.gain[None, :] * (c0 - p["offset"])
        self.knots = self.rho * self.knots + math.sqrt(1 - self.rho ** 2) * r.standard_normal(self.knots.shape) * p.get("col_drift_k", 0.0)
        c = c + (self.knots[self.i0] * (1 - self.t) + self.knots[self.i0 + 1] * self.t) * cpk
        c = c + (r.standard_normal(H) * p.get("row_k", 0.0) * cpk)[:, None] + (r.standard_normal(W) * p.get("col_t_k", 0.0) * cpk)[None, :]
        c = c + r.standard_normal((H, W)) * p["netd_k"] * cpk
        c = np.clip(np.round(c), 0, 2 ** int(self.S["lwir_noise"]["bits"]) - 1).astype(np.float32)
        lo, hi = np.percentile(c, [p["agc_lo"], p["agc_hi"]])
        return np.clip((c - lo) / max(hi - lo, 1.0), 0.0, 1.0).astype(np.float32)[..., None]


def scene_radiance(meta, band, C):
    """Noise-free radiance of a population scene without its target (data/dataset.py minus noise): its layered thermal
    atmosphere and the nominal PSF of the band's rig camera on the grid. The video camera noise (StreamCamera) is a
    generic commodity RGB + LWIR pair on the grid, not each scene's rig camera."""
    from data import camera_emulation as ce
    from data import composite as cp
    from data.dataset import scene_camera
    from render import psf
    from thermal import atmosphere
    from thermal import atmosphere_bands as atb
    layer, _ = cp.load_layers(str(D.ROOT / D.SOURCE / meta["modalities"][band]["layers"]))
    w = meta["extra"]["weather"]
    gas = atb.gas_table(meta, band)
    wv = (atmosphere.scene_params(meta, C["render"]["weather"].get("lwir_water_vapour"))
          if band == "lwir" and gas is None else None)
    img = cp.composite(layer, [], cp.extinction(w["visibility_km"], w["band_ratio"][band]), w.get("aerosol_scale_height_m"),
                       wv=wv, gas=gas)
    sig = meta["extra"]["psf"][band]["nominal"]
    cam = scene_camera(meta, band)
    if cam is not None:
        sig = ce.grid_sigma(sig, cam["pixel_scale"], layer["psf_sigma0"])
    return psf.gaussian_blur(img, psf.extra_sigma(sig, layer["psf_sigma0"]))


def _detect(model, ck, frames):
    from models import cnn_detector as cd
    inp = cd.preprocess({b: frames[b] for b in model.bands}, ck.get("preprocess"))
    p = cd.predict(model, inp)
    t = float(ck["threshold_fa01"])
    return [(b, s) for b, s in zip(p["boxes"], p["scores"]) if s >= t]


def sequences(out, models=None, n_scenes=20, n_frames=150, max_flight_frames=None, flights_dir=None):
    """Confirmed false tracks on target-free sky video (test scenes; after evaluate), and track
    latency on rendered flight sequences (out/m1/det2/flights/*/plan.json and out/video_v2/sensor)."""
    import torch
    from eval.tracking import MultiTracker
    from models import cnn_detector as cd
    if not (out / "test_lock.json").exists():
        raise ValueError("run evaluate first: the target-free video uses test scenes")
    m = D.read(out / "manifest.json")
    torch.set_num_threads(D.threads())
    models = models or ["rgb_lwir", "student", "student_ctrl", "lwir_only"]
    C = D.cfgs()
    S = C["sensors"]
    res = dict(settings=dict(tracker="confirm 3 of 5 frames, gate max(8 px, 2.5 x size), delete after 10 misses",
                             threshold="each model's validation 0.01 threshold", fps=25, n_scenes=n_scenes, n_frames=n_frames),
               target_free={}, flights={})
    sids = m["cohorts"]["test"][:n_scenes]
    for name in models:
        model, ck = cd.load(out / "runs" / name / "best.pt")
        dets_total, confirmed, frames_total = 0, 0, 0
        for i, sid in enumerate(sids):
            meta = D.meta_of(sid)
            L = {b: scene_radiance(meta, b, C) for b in ("rgb", "lwir")}
            cam = StreamCamera(S, 9000 + i, *L["lwir"].shape[:2])
            trk = MultiTracker()
            for k in range(n_frames):
                fr = dict(rgb=cam.rgb(L["rgb"], meta["camera"]["exposure_ms"]),
                          lwir=cam.lwir(L["lwir"][..., 0], meta["conditions"]["air_temp_c"] + 273.15))
                d = _detect(model, ck, fr)
                dets_total += len(d)
                trk.step(d)
            confirmed += len(trk.confirmed_ids)
            frames_total += n_frames
            D.log(f"target-free video {name} {i + 1}/{len(sids)} {sid}: {len(trk.confirmed_ids)} confirmed false tracks")
        minutes = frames_total / 25.0 / 60.0
        up = 3.0 if confirmed == 0 else confirmed + 1.96 * math.sqrt(confirmed) + 1.0
        res["target_free"][name] = dict(frames=frames_total, minutes=minutes, frame_false_alarms=dets_total,
                                        frame_fa_per_frame=dets_total / frames_total, confirmed_false_tracks=confirmed,
                                        confirmed_per_minute=confirmed / minutes, approx_95_upper_per_minute=up / minutes)
    fdir = Path(flights_dir) if flights_dir else out / "flights"
    flights = sorted(fdir.glob("*/plan.json")) + [D.ROOT / "out/video_v2/sensor/plan.json"]
    for plan_p in [p for p in flights if p.exists()]:
        key = plan_p.parent.name if plan_p.parent.name != "sensor" else "video_v2_day"
        res["flights"][key] = flight(plan_p.parent, models, S, out, max_flight_frames)
    D.atomic_json(out / "sequences.json", D.clean(res))
    L = ["# False alarms on video and track latency (M1 detector v2)", "",
         f"Tracker: {res['settings']['tracker']}; threshold: {res['settings']['threshold']}.", "",
         "## Target-free sky video (test scenes, fixed camera patterns, fresh temporal noise)", "",
         "| Model | Minutes of video | Frame false alarms per frame | Confirmed false tracks | per minute (95% upper) |", "|---|---:|---:|---:|---:|"]
    for name, r in res["target_free"].items():
        L.append(f"| {name} | {r['minutes']:.1f} | {r['frame_fa_per_frame']:.4f} | {r['confirmed_false_tracks']} | "
                 f"{r['confirmed_per_minute']:.2f} ({r['approx_95_upper_per_minute']:.2f}) |")
    L += ["", "## Flight sequences (rendered approaches)", "",
          "| Sequence | Model | First confirmed track on the target: time, range, size | Frames tracked after that | False confirmed tracks |",
          "|---|---|---|---:|---:|"]
    for seq, rows in res["flights"].items():
        for name, r in rows.items():
            f = r["first_confirmed"]
            first = f"{f['t']:.1f} s, {f['range_m']:.0f} m, {f['size_px']:.1f} px" if f else "never"
            L.append(f"| {seq} | {name} | {first} | {r['tracked_fraction']:.2f} | {r['false_confirmed_tracks']} |")
    L += ["", "Simulated video: one rendered sky per scene, so the minutes cover few sky types; real clutter (birds, clouds, "
          "terrain) is untested. A zero count bounds the rate; it does not prove it is zero."]
    (out / "sequences.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    D.log(f"wrote {out / 'sequences.md'}")


def flight(seq_dir, models, S, out, max_frames=None):
    from eval.tracking import MultiTracker
    from models import cnn_detector as cd
    from kal_video import Clip
    clip = Clip(seq_dir)
    plan = clip.plan
    n = len(plan["frames"]) if not max_frames else min(max_frames, len(plan["frames"]))
    st = {}
    for name in models:              # same camera seed for every model -> identical noise frames
        model, ck = cd.load(out / "runs" / name / "best.pt")
        st[name] = dict(model=model, ck=ck, cam=StreamCamera(S, 4242, clip.H, clip.W), trk=MultiTracker(),
                        first=None, tracked=0, after=0, on_truth=set())
    for k in range(n):               # radiance once per frame (panorama reprojection is the slow part)
        rad = {b: clip.radiance(k, b) for b in ("rgb", "lwir")}
        fm = plan["frame_meta"][k]
        f = plan["frames"][k]
        tb = f["bbox_px"]
        tcx, tcy = tb["x"] + tb["w"] / 2, tb["y"] + tb["h"] / 2
        tol = max(4.0, 0.5 * max(tb["w"], tb["h"]))
        for name, s in st.items():
            fr = dict(rgb=s["cam"].rgb(rad["rgb"], fm["camera"]["exposure_ms"]),
                      lwir=s["cam"].lwir(rad["lwir"][..., 0], fm["conditions"]["air_temp_c"] + 273.15))
            tracks = s["trk"].step(_detect(s["model"], s["ck"], fr))
            hit = [t for t in tracks if t.confirmed and math.hypot(t.cx - tcx, t.cy - tcy) <= tol]
            s["on_truth"] |= {t.id for t in hit}
            if hit and s["first"] is None:
                s["first"] = dict(k=k, t=f["t"], range_m=f["range_m"], size_px=f["size_px"])
            if s["first"] is not None:
                s["after"] += 1
                s["tracked"] += bool(hit)
        if k % 50 == 0:
            D.log(f"flight {seq_dir.name}: frame {k}/{n}")
    out = {}
    for name, s in st.items():
        out[name] = dict(frames=n, first_confirmed=s["first"], tracked_fraction=s["tracked"] / max(1, s["after"]),
                         false_confirmed_tracks=len(set(s["trk"].confirmed_ids) - s["on_truth"]))
        D.log(f"flight {seq_dir.name} {name}: first confirmed {s['first']}, false tracks {out[name]['false_confirmed_tracks']}")
    return out
