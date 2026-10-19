"""Hard cases of the standard detector: where B misses, does the privileged student find more?

The analysis plan was written before any result: docs/reference/privileged/data/hardcases_plan.md. This tool reads the
saved per-scene scores of tools/privileged_ablation.py and runs no model.

Selecting B's own misses and comparing the privileged student with that same B would favour the student by
construction. Some of B's misses are chance, and another training run recovers some of them (regression to the mean).
So the primary hard set is cross-fitted: for seed s it is the scenes that B misses in both other seeds, and B_s is
judged on a set it did not choose. Retrained B (the other seeds) and the noise ablation's independently trained
supervised models give the rescue rate that retraining alone would bring.

    python tools/privileged_hardcases.py      -> docs/reference/privileged/data/hardcases_pilot1.json (+ a summary)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "reference" / "privileged" / "data"
OUT = ROOT / "out" / "m1" / "privileged_ablation"
NOISE = ROOT / "out" / "m1" / "noise_ablation"
SEEDS = (29, 31, 37)
ARMS = {"B": "rgb_lwir", "T": "teacher", "P": "student", "M": "student_tmask", "K": "student_ctrl",
        "Tf": "teacher@rgb_lwir"}
CONTRASTS = (("P", "B"), ("P", "K"), ("P", "M"), ("T", "B"), ("K", "B"))
# Pilot 2 (docs/privileged_teacher_case.md 6c): P, K and Pb are the target-gated students of the four-sensor teacher, of B
# and of the blind teacher; P1 is pilot 1's student (seeds 29-37 only).
PRESETS = {
    "pilot1": (ARMS, CONTRASTS),
    "pilot2": ({"B": "rgb_lwir", "T": "teacher", "Tb": "teacher_blind", "Tc": "teacher_cool", "Tm": "teacher_mwir",
                "P": "student_g", "Pb": "student_blind_g", "Pc": "student_cool_g", "Pm": "student_mwir_g",
                "K": "student_ctrl_g", "P1": "student", "Tf": "teacher@rgb_lwir"},
               (("P", "B"), ("P", "Pb"), ("P", "K"), ("Pc", "Pb"), ("Pm", "Pb"), ("Pb", "B"), ("K", "B"), ("T", "B"),
                ("T", "Tb"), ("Tc", "Tb"), ("Tm", "Tb"), ("P", "P1"))),
}
NOISE_VARIANTS = ("previous", "gaussian", "physics", "camera", "mixed")
N_BOOT = 10000
RNG_SEED = 20261019
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


class Scores:
    """One model and seed on the test scenes: the best centre-matched target score per scene and the target-free
    scores of each scene's draws, with the tables the resampled measures need."""

    def __init__(self, best, absent):
        self.best = np.asarray(best, float)
        self.n = len(self.best)
        frames = [[np.asarray(f, float) for f in a] for a in absent]
        self.draws = np.array([len(a) for a in frames])
        pooled = [np.sort(np.concatenate(a + [np.zeros(0)]))[::-1] for a in frames]
        self.count = np.array([len(p) for p in pooled])
        self.top = np.full((self.n, 25), -np.inf)            # exact for budgets up to 0.05 on these frame counts
        for j, p in enumerate(pooled):
            self.top[j, :min(25, len(p))] = p[:25]
        # cnt[j, i]: target-free detections of scene j scoring at or above target i's best score
        self.cnt = np.array([[float((p >= b).sum()) for b in self.best] for p in pooled])
        # jaf[j, i]: over scene j's frames, [target i outscores the frame's highest score] + 0.5 [tie]
        fmax = [[float(f.max()) if len(f) else 0.0 for f in a] for a in frames]
        self.jaf = np.array([[sum(float(b > m) + 0.5 * float(b == m) for m in fm) for b in self.best] for fm in fmax])
        self.scene_max = np.array([p[0] if len(p) else 0.0 for p in pooled])

    def threshold(self, k, budget):
        """tools/m1_det2.fa_threshold on the resampled frames: just above the (allowed + 1)-th highest score."""
        vals = np.sort(self.top[k].ravel())[::-1]
        total, nfr = int(self.count[k].sum()), int(self.draws[k].sum())
        allowed = int(math.floor(budget * nfr))
        return 0.01 if allowed >= total else float(np.nextafter(vals[allowed], 2.0))

    def scene_threshold(self, k, allowed=1):
        m = np.sort(self.scene_max[k])[::-1]
        return float(np.nextafter(m[allowed], 2.0)) if allowed < len(m) else 0.01

    def hits(self, budget=0.01):
        full = np.arange(self.n)
        return self.best >= self.threshold(full, budget)

    def measures(self, k, subset):
        """Detection (0.01, 0.05, scene level), mean log10 false-alarm cost and JAFROC on the resampled scenes of
        `subset` (boolean over scenes); None when the resample holds none of them."""
        sel = subset[k]
        if not sel.any():
            return None
        kk = k[sel]
        b = self.best[kk]
        nfr = float(self.draws[k].sum())
        cost = self.cnt[k][:, kk].sum(0) / nfr
        return dict(pd01=float(np.mean(b >= self.threshold(k, 0.01))), pd05=float(np.mean(b >= self.threshold(k, 0.05))),
                    pd_scene=float(np.mean(b >= self.scene_threshold(k, 1))),
                    log_cost=float(np.mean(np.log10(cost + 1.0 / nfr))),
                    jafroc=float(self.jaf[k][:, kk].sum() / (nfr * len(kk))))


def load(out=OUT, seeds=SEEDS, arms=ARMS):
    sc = json.loads((out / "scores.json").read_text())
    return {(s, a): Scores(sc[f"{s}|{lab}"]["best"], sc[f"{s}|{lab}"]["absent"]) for s in seeds for a, lab in arms.items()
            if f"{s}|{lab}" in sc}


def load_noise(seeds=(29, 31)):
    p = NOISE / "scores.json"
    if not p.exists():
        return {}
    sc = json.loads(p.read_text())
    return {(s, v): Scores(sc[f"{s}|{v}|previous"]["best"], sc[f"{s}|{v}|previous"]["absent"])
            for s in seeds for v in NOISE_VARIANTS if f"{s}|{v}|previous" in sc}


def hard_sets(X, seeds):
    miss = {s: ~X[(s, "B")].hits() for s in seeds}
    h1 = {s: np.logical_and.reduce([miss[o] for o in seeds if o != s]) for s in seeds}
    h1p = {s: np.logical_or.reduce([miss[o] for o in seeds if o != s]) for s in seeds}
    h2 = np.logical_and.reduce([miss[s] for s in seeds])
    return miss, h1, h1p, h2


def contrast_table(X, sets, seeds, boots, pairs=CONTRASTS):
    """Paired differences A - B on each seed's own subset, per seed and as the seed mean; scene bootstrap with
    thresholds and costs recomputed in every resample."""
    full = np.arange(next(iter(X.values())).n)
    cache = {}

    def resampled(s, m):                                   # every resample's measures, once per model and seed
        if (s, m) not in cache:
            cache[(s, m)] = [X[(s, m)].measures(k, sets[s]) for k in boots]
        return cache[(s, m)]

    out = {}
    for a, b in pairs:
        if not all((s, a) in X and (s, b) in X for s in seeds):
            continue
        row = {}
        for m in ("pd01", "pd05", "pd_scene", "log_cost", "jafroc"):
            per, dist = {}, []
            for s in seeds:
                S = sets[s]
                ma, mb = X[(s, a)].measures(full, S), X[(s, b)].measures(full, S)
                if ma is None:
                    continue
                d = np.asarray([np.nan if xa is None else xa[m] - xb[m] for xa, xb in zip(resampled(s, a), resampled(s, b))])
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                per[s] = dict(n=int(S.sum()), a=ma[m], b=mb[m], diff=ma[m] - mb[m], lo=float(lo), hi=float(hi),
                              p_gt0=float(np.nanmean(d > 0)))
                dist.append(d)
            if not per:
                continue
            diffs = np.array([v["diff"] for v in per.values()])
            sm = np.nanmean(np.array(dist), 0)
            n = len(diffs)
            sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
            half = T975.get(n - 1, float("nan")) * sd / math.sqrt(n) if n > 1 else float("nan")
            row[m] = dict(per_seed=per, mean=float(diffs.mean()), scene_ci95=[float(x) for x in np.nanpercentile(sm, [2.5, 97.5])],
                          scene_p_gt0=float(np.nanmean(sm > 0)), seed_sd=sd,
                          seed_t_ci95=[float(diffs.mean() - half), float(diffs.mean() + half)])
        out[f"{a}-{b}"] = row
    return out


def rescue(X, N, miss, seeds, arms=ARMS):
    """In-sample: of B_s's misses, how many each model detects (own thresholds), against retrained B (other seeds) and
    the noise ablation's independently trained supervised models; and of B_s's hits, how many each model loses."""
    out = {}
    for s in seeds:
        M = miss[s]
        hitB = ~M
        row = dict(misses=int(M.sum()), hits=int(hitB.sum()))
        for a in arms:
            if a == "B" or (s, a) not in X:
                continue
            h = X[(s, a)].hits()
            row[a] = dict(rescued=int((h & M).sum()), lost=int((~h & hitB).sum()))
        other = [o for o in seeds if o != s]
        row["B_other_seeds"] = dict(rescued=float(np.mean([(X[(o, "B")].hits() & M).sum() for o in other])),
                                    lost=float(np.mean([(~X[(o, "B")].hits() & hitB).sum() for o in other])))
        ind = [v for (ss, v) in N if ss == s]
        if ind:
            row["independent_supervised"] = {v: dict(rescued=int((N[(s, v)].hits() & M).sum()),
                                                     lost=int((~N[(s, v)].hits() & hitB).sum())) for v in ind}
        out[s] = row
    return out


def describe(sids, census, mask):
    rows = []
    for i in np.where(mask)[0]:
        c = census[sids[i]]
        d = {b: round(c["dprime"][b]["d"], 1) for b in ("rgb", "lwir", "lwir_cooled", "mwir")}
        rows.append(dict(index=int(i), scene=sids[i], size_px=round(c["size_px"], 1), range_m=round(c["range_m"]),
                         lighting=c["lighting"], clouds=c["clouds"], visibility_km=c["visibility_km"], dprime=d,
                         privileged_better=max(d["lwir_cooled"], d["mwir"]) > max(d["rgb"], d["lwir"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot1")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--arms", choices=sorted(PRESETS), default="pilot1")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--scores", default=str(OUT), help="folder with scores.json and split.json (default: pilot 1)")
    ap.add_argument("--no-census", action="store_true", help="the scenes are re-rendered (a thermal arm): no census d', no noise-ablation references")
    a = ap.parse_args()
    arms, pairs = PRESETS[a.arms]
    seeds = tuple(a.seeds)
    folder = Path(a.scores)
    X = load(folder, seeds, arms)
    N = load_noise()
    split = folder / "split.json" if (folder / "split.json").exists() else OUT / "split.json"
    sids = json.loads(split.read_text())["test"]
    census = json.loads((DATA / "census.json").read_text())
    if a.no_census or not all(s in census for s in sids):   # a fresh cohort or a thermal arm: the census and the
        census, N = None, {}                                  # noise ablation describe other frames
    n = len(sids)
    rng = np.random.default_rng(RNG_SEED)
    boots = rng.integers(0, n, (a.boot, n))
    miss, h1, h1p, h2 = hard_sets(X, seeds)
    full = np.arange(n)
    allm = np.ones(n, bool)
    present = [m for m in arms if all((s, m) in X for s in seeds)]
    res = dict(plan="docs/reference/privileged/data/hardcases_plan.md", scores=str(folder), arms=arms, seeds=list(seeds),
               n_boot=a.boot, rng_seed=RNG_SEED,
               sizes=dict(miss={s: int(miss[s].sum()) for s in seeds}, H1={s: int(h1[s].sum()) for s in seeds},
                          H1_plus={s: int(h1p[s].sum()) for s in seeds}, H2=int(h2.sum())))
    # absolute measures per model on each subset (full sample)
    res["levels"] = {}
    for name, sets in (("all", {s: allm for s in seeds}), ("H1", h1), ("H1_plus", h1p), ("H2", {s: h2 for s in seeds})):
        res["levels"][name] = {f"{s}|{m}": X[(s, m)].measures(full, sets[s]) for s in seeds for m in arms if (s, m) in X}
    res["levels"]["H2_independent_supervised"] = {f"{s}|{v}": N[(s, v)].measures(full, h2) for (s, v) in N}
    res["levels"]["H1_independent_supervised"] = {f"{s}|{v}": N[(s, v)].measures(full, h1[s]) for (s, v) in N if s in h1}
    print("contrasts on H1 ...", flush=True)
    res["H1"] = contrast_table(X, h1, seeds, boots, pairs)
    print("contrasts on H1+ ...", flush=True)
    res["H1_plus"] = contrast_table(X, h1p, seeds, boots[:4000], pairs)
    print("contrasts on all scenes ...", flush=True)
    res["all"] = contrast_table(X, {s: allm for s in seeds}, seeds, boots[:4000], pairs)
    res["rescue"] = rescue(X, N, miss, seeds, {m: arms[m] for m in present})
    # mechanism: B's misses that the four-sensor teacher detects (in-sample, and cross-fitted on the other seeds)
    mech = {}
    for s in seeds:
        te_in = X[(s, "T")].hits() & miss[s]
        other = [o for o in seeds if o != s]
        te_cf = np.logical_and.reduce([X[(o, "T")].hits() & miss[o] for o in other])
        mech[s] = {}
        for name, S in (("teacher_found_B_missed_in_sample", te_in), ("teacher_found_B_missed_other_seeds", te_cf)):
            mech[s][name] = dict(n=int(S.sum()), **{m: int((X[(s, m)].hits() & S).sum()) for m in present})
    res["mechanism"] = mech
    if census is not None:
        res["H2_scenes"] = describe(sids, census, h2)
        res["H1_scenes"] = {s: [r["scene"] for r in describe(sids, census, h1[s])] for s in seeds}
    hit_table = {f"{s}|{m}": [int(x) for x in X[(s, m)].hits()] for s in seeds for m in arms if (s, m) in X}
    res["hits01"] = dict(scenes=sids, **hit_table)
    path = DATA / f"hardcases_{a.tag}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
