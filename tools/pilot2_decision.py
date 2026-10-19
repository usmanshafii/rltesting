"""The pre-registered decision of pilot 2 (docs/privileged_teacher_case.md, sections 6c and 6d), written before any
pilot-2 result.

Reads the per-seed evaluations of one cohort (scores.json and results.json files: the cloud's cloud_eval/<cohort>/s<seed>
folders, or any evaluate() output), and computes, for each contrast:
- the per-seed differences in detection at 0.01 false alarms per target-free frame, with the threshold taken from the
  cohort's own target-free frames;
- the hierarchical bootstrap 95 % interval [L, U] (seeds resampled, and scenes resampled with every threshold
  recomputed);
- section 6d's decision against the 3-point margin;
- section 6c's grade.
The frozen-threshold detection and its false-alarm rate are reported alongside.

    python tools/pilot2_decision.py --cohort fresh out/m1/privileged_ablation_cloud/cloud_eval/fresh/s*
    -> docs/reference/privileged/data/pilot2_decision_<cohort>.json and .md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "reference" / "privileged" / "data"
MARGIN = 0.03                                     # section 6d
PRIMARY = ("student_g", "rgb_lwir")
CO_PRIMARY = ("student_g", "student_blind_g")
SECONDARY = (("student_cool_g", "student_blind_g"), ("student_mwir_g", "student_blind_g"),
             ("student_ctrl_g", "rgb_lwir"), ("student_g", "student_ctrl_g"), ("student_blind_g", "rgb_lwir"),
             ("teacher", "teacher_blind"), ("teacher_cool", "teacher_blind"), ("teacher_mwir", "teacher_blind"),
             ("student_g", "student"))
N_BOOT = 10000


class Arm:
    """One model and seed: best target score per scene, and the target-free scores per scene."""

    def __init__(self, best, absent):
        self.best = np.asarray(best, float)
        pooled = [np.sort(np.concatenate([np.asarray(f, float) for f in a] + [np.zeros(0)]))[::-1] for a in absent]
        self.draws = np.array([len(a) for a in absent])
        self.count = np.array([len(p) for p in pooled])
        self.top = np.full((len(pooled), 12), -np.inf)        # exact for 0.01 (2-3 allowed) on these frame counts
        for j, p in enumerate(pooled):
            self.top[j, :min(12, len(p))] = p[:12]

    def pd01(self, k):
        vals = np.sort(self.top[k].ravel())[::-1]
        allowed = int(math.floor(0.01 * int(self.draws[k].sum())))
        t = 0.01 if allowed >= int(self.count[k].sum()) else float(np.nextafter(vals[allowed], 2.0))
        return float(np.mean(self.best[k] >= t))


def load(folders):
    scores, rows = {}, {}
    for f in folders:
        f = Path(f)
        scores.update(json.loads((f / "scores.json").read_text()))
        if (f / "results.json").exists():
            rows.update(json.loads((f / "results.json").read_text())["rows"])
    return scores, rows


def decide(L, U):
    """Section 6d."""
    return dict(gain_at_least_3_points=bool(L >= MARGIN), gain_shown=bool(L > 0),
                no_gain_of_3_points=bool(U < MARGIN),
                verdict=("a gain of at least 3 points shown" if L >= MARGIN else
                         "a gain shown, smaller than 3 points or not shown to reach them" if L > 0 else
                         "no gain of 3 points or more" if U < MARGIN else "inconclusive"))


def grade(diffs, L, p_pos):
    """Section 6c's grades, with 'at least 4 of 5 seeds' read as at least 80 % of the seeds."""
    if np.mean(np.asarray(diffs) > 0) >= 0.8 and L > 0:
        return "shown"
    if np.mean(diffs) > 0 and p_pos >= 0.8:
        return "suggestive"
    return "not shown"


def analyse(scores, rows, cohort, n_boot=N_BOOT, seed=20261019):
    labels = sorted({k.split("|", 1)[1] for k in scores})
    seeds_of = {lab: sorted(int(k.split("|")[0]) for k in scores if k.split("|", 1)[1] == lab) for lab in labels}
    n_scenes = len(next(iter(scores.values()))["best"])
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, n_scenes, (n_boot, n_scenes))
    full = np.arange(n_scenes)
    out = dict(cohort=cohort, scenes=n_scenes, margin=MARGIN, n_boot=n_boot, contrasts={})
    for a, b in (PRIMARY, CO_PRIMARY) + SECONDARY:
        if a not in seeds_of or b not in seeds_of:
            continue
        seeds = sorted(set(seeds_of[a]) & set(seeds_of[b]))
        if not seeds:
            continue
        A = {s: Arm(**scores[f"{s}|{a}"]) for s in seeds}
        B = {s: Arm(**scores[f"{s}|{b}"]) for s in seeds}
        d = np.array([A[s].pd01(full) - B[s].pd01(full) for s in seeds])
        per = np.array([[A[s].pd01(k) - B[s].pd01(k) for k in boots] for s in seeds])      # seeds x resamples
        pick = rng.integers(0, len(seeds), (n_boot, len(seeds)))
        hier = per[pick, np.arange(n_boot)[:, None]].mean(1)                              # seeds and scenes resampled
        L, U = (float(x) for x in np.percentile(hier, [2.5, 97.5]))
        row = dict(seeds=seeds, per_seed={int(s): float(x) for s, x in zip(seeds, d)}, mean=float(d.mean()),
                   median=float(np.median(d)), positive_seeds=int((d > 0).sum()), L=L, U=U,
                   p_positive=float(np.mean(hier > 0)), scene_only=[float(x) for x in np.percentile(per.mean(0), [2.5, 97.5])],
                   seed_sd=float(d.std(ddof=1)) if len(d) > 1 else None)
        row.update(decide(L, U), grade_6c=grade(d, L, row["p_positive"]))
        fz = [(rows.get(f"{s}|{a}", {}), rows.get(f"{s}|{b}", {})) for s in seeds]
        if all("pd_frozen01" in x and "pd_frozen01" in y for x, y in fz):
            row["frozen"] = dict(pd_diff=float(np.mean([x["pd_frozen01"] - y["pd_frozen01"] for x, y in fz])),
                                 fa_per_frame_diff=float(np.mean([x["fa_per_frame_frozen01"] - y["fa_per_frame_frozen01"]
                                                                  for x, y in fz])))
        out["contrasts"][f"{a} - {b}"] = row
    return out


def write_md(res, path):
    L = [f"# Pilot 2 decision: cohort {res['cohort']} ({res['scenes']} scenes)", "",
         "Detection at 0.01 false alarms per target-free frame (threshold from the cohort's own target-free frames). "
         "[L, U]: hierarchical bootstrap 95 % interval (seeds and scenes). Margin 3 points (section 6d).", "",
         "| Contrast | Seeds | Mean | Positive seeds | [L, U] | Decision (6d) | Grade (6c) | Frozen: Pd / FA per frame |",
         "|---|---:|---:|---:|---|---|---|---|"]
    for k, r in res["contrasts"].items():
        fz = r.get("frozen")
        fzs = f"{fz['pd_diff']:+.3f} / {fz['fa_per_frame_diff']:+.3f}" if fz else "-"
        L.append(f"| {k} | {len(r['seeds'])} | {r['mean']:+.3f} | {r['positive_seeds']} | [{r['L']:+.3f}, {r['U']:+.3f}] | "
                 f"{r['verdict']} | {r['grade_6c']} | {fzs} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    scores, rows = load(a.folders)
    res = analyse(scores, rows, a.cohort, a.boot)
    name = f"pilot2_decision_{a.cohort}{a.tag}"
    (DATA / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, DATA / f"{name}.md")
    for k in ("student_g - rgb_lwir", "student_g - student_blind_g"):
        if k in res["contrasts"]:
            r = res["contrasts"][k]
            print(f"{k}: mean {r['mean']:+.3f}, [L, U] [{r['L']:+.3f}, {r['U']:+.3f}] over {len(r['seeds'])} seeds -> "
                  f"{r['verdict']} (6c: {r['grade_6c']})")
    print(f"wrote {DATA / name}.json and .md")


if __name__ == "__main__":
    main()
