"""Does privileged information add value beyond ordinary distillation? (docs/privileged_teacher_case.md, section 6)

The compact CNN of tools/m1_det2.py, deployed on RGB + uncooled LWIR, trained in five ways per seed. Every student
sees identical batches (the sampler's draws do not depend on the model):

  rgb_lwir        the standard RGB + uncooled LWIR detector (supervised, band dropout): the baseline and the
                  ordinary distillation teacher
  teacher         RGB + uncooled LWIR + cooled LWIR + MWIR, trained on every sensor subset
  student         distilled from `teacher` with all four sensors fed: the privileged student
  student_tmask   distilled from the same `teacher` with only RGB and uncooled LWIR fed (the cooled LWIR and MWIR
                  frames blanked, their flags 0): the within-teacher control. Same teacher weights, same batches; the
                  privileged cameras are the only difference
  student_ctrl    distilled from `rgb_lwir`: ordinary distillation from a teacher of the student's own sensors

Contrasts, per seed and scene (paired), and pooled over seeds with one scene bootstrap:
  student - student_tmask   what the privileged cameras add at distillation time        (the primary contrast)
  student - student_ctrl    privileged distillation against ordinary distillation         (the conventional one)
  student_ctrl - rgb_lwir   what ordinary distillation alone does
  student - rgb_lwir        the total over the standard detector

Every model is taken at its final step (no checkpoint selection: the noise verification found the selection on few
validation scenes as noisy as the effects). Metrics on whole test frames, centre match (eval/nwd): detection at
0.01 false alarms per target-free frame (threshold from the test set's own target-free frames), the FROC over
0.005-0.05, and AP (centre and IoU 0.5). Results also by the target's d' in the uncooled LWIR (the census,
tools/privileged_census.py): the low thermal-signature end is where the privileged cameras should matter.

Scenes: the noise verification's split of the rendered scenes outside M1's sealed test split (62 training,
9 validation, 55 test). Loader as configured (the previous sensor models).

    python tools/privileged_ablation.py all --seeds 29 31 37     cache, train every model and seed, evaluate
    python tools/privileged_ablation.py evaluate --seeds 29 31 37
    -> out/m1/privileged_ablation/results.md, results.json, scores.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

import numpy as np  # noqa: E402

import m1_det2 as D  # noqa: E402

OUT = ROOT / "out" / "m1" / "privileged_ablation"
SPLIT_FROM = ROOT / "out" / "m1" / "noise_ablation" / "split.json"
CENSUS = ROOT / "docs" / "reference" / "privileged" / "data" / "census.json"
MODELS = ("rgb_lwir", "teacher", "student", "student_tmask", "student_ctrl")      # training order (teachers first)
# Added 2026-10-18 after the first seed's test results (secondary): a teacher with the four-sensor teacher's network and
# subset training that never sees the cooled LWIR or MWIR, and its student. It separates the privileged cameras from
# the teacher's training regime and size.
BLIND = ("teacher_blind", "student_blind")
# Pilot 2 (2026-10-19, docs/privileged_teacher_case.md section 6c): the students are distilled only near the labelled
# targets (m1_det2.KD_TARGET_ONLY), so the teachers' false alarms are not taught. student_blind_g, distilled the same way
# from the blind teacher, is the control for the privileged cameras.
# Each extra sensor's contribution (user, 2026-10-19): teachers with the four-sensor teacher's network, subset training
# and settings, the missing cameras always blank (m1_det2.BLANK_BANDS), so capacity, procedure and tuning are the same;
# and a target-gated student of each. teacher_blind = base two sensors, teacher_cool = + cooled LWIR, teacher_mwir =
# + MWIR, teacher = all four.
PRIVILEGED = ("lwir_cooled", "mwir")
TEACHER_BLANK = {"teacher_blind": PRIVILEGED, "teacher_cool": ("mwir",), "teacher_mwir": ("lwir_cooled",)}
GATED = ("student_g", "student_blind_g", "student_cool_g", "student_mwir_g", "student_ctrl_g")
PILOT2 = ("rgb_lwir", "teacher", "teacher_blind", "teacher_cool", "teacher_mwir") + GATED   # training order
PILOT2_SEEDS = (29, 31, 37, 41, 43)
BASE = {"rgb_lwir": "rgb_lwir", "teacher": "teacher", "student": "student", "student_tmask": "student",
        "student_ctrl": "student_ctrl", "teacher_blind": "teacher", "student_blind": "student",
        "teacher_cool": "teacher", "teacher_mwir": "teacher",
        "student_g": "student", "student_ctrl_g": "student_ctrl", "student_blind_g": "student",
        "student_cool_g": "student", "student_mwir_g": "student"}
TEACHER_OF = {"student_tmask": "teacher", "student_blind": "teacher_blind", "student_blind_g": "teacher_blind",
              "student_cool_g": "teacher_cool", "student_mwir_g": "teacher_mwir"}
STEPS = 2000
TRAINING = dict(steps=STEPS, realisations=2, absent_realisations=1, val_every=STEPS,      # one validation: the final step
                hard_negatives=dict(steps=[400, 800, 1200, 1600], frames=24, min_score=0.15, share=0.33, pool_max=600))
CONTRASTS = (("student", "student_tmask"), ("student", "student_ctrl"), ("student_ctrl", "rgb_lwir"),
             ("student", "rgb_lwir"), ("student_tmask", "rgb_lwir"),
             ("student", "student_blind"), ("student_tmask", "student_blind"), ("student_blind", "student_ctrl"),
             ("student_g", "rgb_lwir"), ("student_g", "student_blind_g"), ("student_g", "student_ctrl_g"),
             ("student_ctrl_g", "rgb_lwir"), ("student_blind_g", "rgb_lwir"), ("student_g", "student"),
             ("student_cool_g", "student_blind_g"), ("student_mwir_g", "student_blind_g"),
             ("student_cool_g", "rgb_lwir"), ("student_mwir_g", "rgb_lwir"),
             ("teacher", "teacher_blind"), ("teacher_cool", "teacher_blind"), ("teacher_mwir", "teacher_blind"))
FED = ("rgb", "lwir")
DPRIME_BINS = ((0.0, 20.0), (20.0, 50.0), (50.0, 1e9))


def log(*a):
    D.log(*a)


FIRST_SEED = 29                                   # its runs keep the base names


def run_name(model, seed):
    return model if seed == FIRST_SEED else f"{model}_s{seed}"


def register(seed):
    """Seed-suffixed specs: each student's teacher is the same seed's teacher; the masked student's teacher is fed
    RGB and uncooled LWIR only; the blind teacher never sees the privileged cameras, and its student distils from it."""
    for model in MODELS + BLIND + tuple(TEACHER_BLANK) + GATED:
        bands, teacher, dropout = D.SPECS[BASE[model]]
        teacher = TEACHER_OF.get(model, teacher)
        D.SPECS[run_name(model, seed)] = (bands, run_name(teacher, seed) if teacher else None, dropout)
        if teacher in TEACHER_BLANK:                 # a teacher is fed the cameras it was trained with
            D.TEACHER_FED[run_name(model, seed)] = [b for b in D.BANDS if b not in TEACHER_BLANK[teacher]]
    D.TEACHER_FED[run_name("student_tmask", seed)] = list(FED)
    for t, blank in TEACHER_BLANK.items():
        D.BLANK_BANDS[run_name(t, seed)] = list(blank)
    D.KD_TARGET_ONLY.update(run_name(m, seed) for m in GATED)


def prepare(smoke=False):
    split = D.read(SPLIT_FROM)
    if smoke:                               # mechanics check: a few scenes, a few steps
        split = dict(split, train=split["train"][:4], val=split["val"][:2], test=split["test"][:3])
    OUT.mkdir(parents=True, exist_ok=True)
    D.atomic_json(OUT / "split.json", split)
    if not (OUT / "manifest.json").exists():
        D.manifest(OUT, scenes=set(split["train"] + split["val"] + split["test"]), smoke=True)
        m = D.read(OUT / "manifest.json")
        m["splits"] = dict(train=split["train"], val=split["val"])
        m["cohorts"] = dict(test=split["test"])
        m["training"] = dict(m["training"], **TRAINING)
        if smoke:
            m["training"].update(steps=20, val_every=20, hard_negatives=dict(TRAINING["hard_negatives"], steps=[10], frames=2))
        m["experiment"] = "privileged-ablation"
        D.atomic_json(OUT / "manifest.json", m)
    log(f"privileged ablation: {len(split['train'])} train, {len(split['val'])} validation, {len(split['test'])} test scenes")


# pilot 2 in two halves, the primary and co-primary contrasts first: (a) the blind teacher and the students of the
# primary contrasts, (b) the one-extra-camera teachers and their students
PILOT2A = ("rgb_lwir", "teacher", "teacher_blind", "student_g", "student_blind_g", "student_ctrl_g")
PILOT2B = ("teacher_cool", "teacher_mwir", "student_cool_g", "student_mwir_g")
ARM_SETS = {"pilot1": MODELS, "blind": BLIND, "pilot2": PILOT2, "pilot2a": PILOT2A, "pilot2b": PILOT2B}
# every model of pilots 1 and 2, each once (finished runs only are evaluated)
EVAL_ALL = MODELS + tuple(TEACHER_BLANK) + ("student_blind",) + GATED


def _lock(name):
    """Wait while another live process holds OUT/.<name>.lock (as m1_det2.render does), so two launches of the same
    queue cannot train the same seed's runs."""
    p = OUT / f".{name}.lock"
    while p.exists():
        try:
            pid = int(p.read_text().split()[0])
        except (ValueError, IndexError, OSError):
            pid = -1
        if pid <= 0 or pid == os.getpid() or not D._pid_alive(pid):
            break
        log(f"process {pid} holds {p.name}; waiting for it to finish ...")
        time.sleep(120)
    p.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M')}")
    return p


def train(seed, blind=False, arms=None):
    register(seed)
    D.TRAIN_SEED = int(seed)                # the sampler's and the initialisation's seed
    lock = _lock(f"train_s{seed}")                 # one trainer per seed; other seeds may train alongside (cloud)
    try:
        D.train(OUT, [run_name(m, seed) for m in ARM_SETS[arms or ("blind" if blind else "pilot1")]])
    finally:
        lock.unlink(missing_ok=True)


def _arrays(model, test, metas, cfg, present=None, src=None):
    import m1_det2_eval as E
    pres = D.records(src or OUT, model, test, "p", len(D.TEST_PRESENT), present, metas)
    absn = D.records(src or OUT, model, test, "a", len(D.TEST_ABSENT), present, metas)
    best, absent = E.scene_arrays(pres, absn, cfg)
    return best, absent, pres, absn


def _save_predictions(dest, seed, label, pres, absn):
    """Every test frame's detections (boxes in pixels, scores), so AP and any match rule can be recomputed later."""
    import gzip
    rows = [dict(frame=r["scene_id"], scene=r["scene"], gts=[o["box"] for o in r["gts"]],
                 boxes=[[round(float(v), 2) for v in b] for b in r["preds"]["boxes"]],
                 scores=[round(float(s), 5) for s in r["preds"]["scores"]]) for r in pres + absn]
    p = dest / "predictions" / f"{seed}_{label.replace('@', '_at_')}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(rows, f, separators=(",", ":"))


def pd_at(best, absent, budget=0.01):
    """Detection probability at `budget` false alarms per target-free frame, threshold from those frames."""
    frames = [f for a in absent for f in a]
    scores = np.sort(np.concatenate(frames + [np.zeros(0)]))[::-1]
    allowed = int(np.floor(budget * len(frames)))
    t = 0.01 if allowed >= len(scores) else float(np.nextafter(scores[allowed], 2.0))
    return float(np.mean(best >= t)) if len(best) else float("nan")


def evaluate(seeds, models=None, src=None, dest=None, save_predictions=False, census_bins=True):
    """Every finished model of `models` (default: pilot 1 and the blind control) on the test cohort whose frames are
    cached under `src` (default: this experiment's test split), results written to `dest` (default: OUT)."""
    import torch
    import m1_det2_eval as E
    from eval import metrics
    from models import cnn_detector as cd
    src, dest = src or OUT, dest or OUT
    dest.mkdir(parents=True, exist_ok=True)
    split = D.read(src / "split.json")
    test = split["test"]
    cfg = D.eval_cfg()
    metas = {s: D.meta_of(s) for s in test}
    torch.set_num_threads(D.threads())
    census = json.loads(CENSUS.read_text())
    # the census describes the standard renders: no d' bins for a thermal arm's re-rendered scenes
    lw = np.array([census[s]["dprime"]["lwir"]["d"] if census_bins and s in census else np.nan for s in test])
    budgets = D.TRAINING["fa_budgets"]
    rows, arrays = {}, {}
    fed_only = {b: float(b in FED) for b in D.BANDS}
    for seed in seeds:
        evals = [(m, m, {b: float(b not in TEACHER_BLANK[m]) for b in D.BANDS} if m in TEACHER_BLANK else None)
                 for m in (models or MODELS + BLIND)]
        evals += [("teacher@rgb_lwir", "teacher", fed_only)]
        for label, model, present in evals:
            path = OUT / "runs" / run_name(model, seed) / "best.pt"
            if not path.exists():
                continue
            net, ck = cd.load(path)
            net.to(torch.device(D.DEVICE))                 # KAL_DEVICE (cpu on the laptop, cuda in the cloud)
            best, absent, pres, absn = _arrays(net, test, metas, cfg, present, src)
            arrays[(seed, label)] = (best, absent)
            froc, _ = E.froc_from_arrays(best, absent, budgets)
            t01 = float(ck["threshold_fa01"])        # frozen on validation: the deployable operating point
            frames = [f for a in absent for f in a]
            rows[f"{seed}|{label}"] = dict(seed=seed, model=label, pd_fa01=pd_at(best, absent), froc=froc,
                                           pd_frozen01=float(np.mean(best >= t01)), threshold_frozen01=t01,
                                           fa_per_frame_frozen01=float(sum(int((f >= t01).sum()) for f in frames) / max(1, len(frames))),
                                           ap_centre=metrics.ap(pres, kind="centre", thr=0.0, cfg=cfg),
                                           ap50=metrics.ap(pres, cfg=cfg))
            if save_predictions:
                _save_predictions(dest, seed, label, pres, absn)
            r = rows[f"{seed}|{label}"]
            log(f"seed {seed} {label:16s}: Pd@0.01 {r['pd_fa01']:.3f}  FROC {froc:.3f}  frozen Pd {r['pd_frozen01']:.3f} "
                f"at {r['fa_per_frame_frozen01']:.3f} FA/frame  AP centre {r['ap_centre']:.3f}  AP50 {r['ap50']:.3f}")
    raw = {f"{s}|{m}": dict(best=[float(x) for x in b], absent=[[[float(x) for x in f] for f in a] for a in ab])
           for (s, m), (b, ab) in arrays.items()}
    (dest / "scores.json").write_text(json.dumps(raw, separators=(",", ":")))
    res = dict(split={k: (len(v) if isinstance(v, list) else v) for k, v in split.items()}, seeds=list(seeds),
               steps=STEPS, test_frames=str(src.relative_to(ROOT)), rows=rows, contrasts=contrasts(arrays, seeds, lw))
    D.atomic_json(dest / "results.json", D.clean(res))
    write_results(res, dest)


def contrasts(arrays, seeds, lw, n_boot=2000, rng_seed=0):
    """Paired differences (A - B) in Pd at 0.01 false alarms per frame and in FROC: per seed, and the seed mean with
    one scene bootstrap shared by every model and seed; also by the target's uncooled-LWIR d' bin. The share of
    resamples with A > B is reported with the 95 % interval."""
    import m1_det2_eval as E
    budgets = D.TRAINING["fa_budgets"]
    n = len(lw)
    rng = np.random.default_rng(rng_seed)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]

    def stat(model, seed, k, which):
        b, ab = arrays[(seed, model)]
        bb, aa = b[k], [ab[i] for i in k]
        return pd_at(bb, aa) if which == "pd" else E.froc_from_arrays(bb, aa, budgets)[0]

    out = {}
    for a, b in CONTRASTS:
        have = [s for s in seeds if (s, a) in arrays and (s, b) in arrays]
        if not have:
            continue
        for which in ("pd", "froc"):
            base = np.arange(n)
            per_seed = {s: stat(a, s, base, which) - stat(b, s, base, which) for s in have}
            dist = [np.mean([stat(a, s, k, which) - stat(b, s, k, which) for s in have]) for k in boots]
            lo, hi = np.percentile(dist, [2.5, 97.5])
            row = dict(mean=float(np.mean(list(per_seed.values()))), per_seed=per_seed, lo=float(lo), hi=float(hi),
                       p_better=float(np.mean(np.asarray(dist) > 0)), seeds=have)
            if which == "pd":                 # detection by the target's uncooled-LWIR d' (the low-signature end)
                row["by_lwir_dprime"] = {}
                for lo_d, hi_d in DPRIME_BINS:
                    idx = np.where((lw >= lo_d) & (lw < hi_d))[0]
                    if not len(idx):
                        continue
                    da = np.mean([np.mean(arrays[(s, a)][0][idx] >= _thr(arrays[(s, a)][1])) for s in have])
                    db = np.mean([np.mean(arrays[(s, b)][0][idx] >= _thr(arrays[(s, b)][1])) for s in have])
                    row["by_lwir_dprime"][f"{lo_d:g}-{hi_d:g}"] = dict(n=int(len(idx)), a=float(da), b=float(db))
            out[f"{a} - {b}|{which}"] = row
    return out


def _thr(absent, budget=0.01):
    frames = [f for a in absent for f in a]
    scores = np.sort(np.concatenate(frames + [np.zeros(0)]))[::-1]
    allowed = int(np.floor(budget * len(frames)))
    return 0.01 if allowed >= len(scores) else float(np.nextafter(scores[allowed], 2.0))


def write_results(res, dest=None):
    dest = dest or OUT
    sp = res["split"]
    L = ["# Does privileged information add value beyond ordinary distillation?", "",
         f"{sp['train']} training, {sp['val']} validation and {sp['test']} test scenes (outside M1's sealed test split); "
         f"{res['steps']} steps per model, final step, seeds {', '.join(str(s) for s in res['seeds'])}. Detection at 0.01 "
         "false alarms per target-free frame (centre match; threshold from the test set's own target-free frames), FROC "
         "over 0.005-0.05, AP with centre match and with IoU 0.5.", "",
         "Frozen: at the threshold frozen on validation for 0.01, with the false-alarm rate that threshold gives on test.", "",
         "| Model | Seed | Pd@0.01 | FROC | Pd frozen | FA/frame frozen | AP (centre) | AP50 (IoU) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["rows"].values():
        fz = (f"{r['pd_frozen01']:.3f} | {r['fa_per_frame_frozen01']:.3f}" if "pd_frozen01" in r else "- | -")
        L.append(f"| {r['model']} | {r['seed']} | {r['pd_fa01']:.3f} | {r['froc']:.3f} | {fz} | {r['ap_centre']:.3f} | "
                 f"{r['ap50']:.3f} |")
    L += ["", "## Contrasts (A - B), seed mean, 95 % scene-bootstrap interval, share of resamples with A > B", "",
          "| Contrast | Metric | Mean | Per seed | 95 % interval | P(A > B) |", "|---|---|---:|---|---|---:|"]
    for k, v in res["contrasts"].items():
        pair, which = k.split("|")
        ps = ", ".join(f"{s}: {d:+.3f}" for s, d in v["per_seed"].items())
        L.append(f"| {pair} | {'Pd@0.01' if which == 'pd' else 'FROC'} | {v['mean']:+.3f} | {ps} | "
                 f"[{v['lo']:+.3f}, {v['hi']:+.3f}] | {v['p_better']:.2f} |")
    L += ["", "## Detection by the target's d' in the uncooled LWIR (seed mean; A / B)", ""]
    for k, v in res["contrasts"].items():
        if k.endswith("|pd") and v.get("by_lwir_dprime"):
            L.append(f"- {k.split('|')[0]}: " + "; ".join(f"d' {b}: {x['a']:.2f} / {x['b']:.2f} (n={x['n']})"
                                                         for b, x in v["by_lwir_dprime"].items()))
    (dest / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {dest / 'results.md'}")


def _free_mb():
    """Free physical memory (Windows), for the log when a step has to be repeated."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullAvailPhys / 2 ** 20)
    except Exception:
        return -1


def _run(cmd, done, tries=4):
    """Run a step in its own process until `done()` holds; a process that dies (2026-10-18: one was ended without a
    traceback) costs only the model it was training, as finished models are skipped."""
    for k in range(tries):
        rc = subprocess.run(cmd).returncode
        if done():
            return
        log(f"step {' '.join(cmd[3:])} ended (code {rc}) before it finished; free memory {_free_mb()} MB; retry {k + 1}")
    raise RuntimeError(f"{' '.join(cmd[3:])} did not finish after {tries} tries")


def run_all(seeds, smoke=False):
    prepare(smoke)
    py = [sys.executable, "-u", str(Path(__file__))] + (["--smoke"] if smoke else [])
    _run(py + ["cache"], lambda: (OUT / "cache_fingerprint.json").exists())
    for seed in seeds:
        _run(py + ["train", "--seed", str(seed)],
             lambda s=seed: all((OUT / "runs" / run_name(m, s) / "best.pt").exists() for m in MODELS))
    _run(py + ["evaluate", "--seeds"] + [str(s) for s in seeds], lambda: (OUT / "results.md").exists())


def run_blind(seeds, after=None):
    """The secondary control after the main run: wait for process `after` to end, then the blind teacher and its
    student for every seed, then the evaluation with them included."""
    import time
    while after and D._pid_alive(after):
        time.sleep(60)
    py = [sys.executable, "-u", str(Path(__file__))]
    for seed in seeds:
        _run(py + ["train", "--seed", str(seed), "--blind"],
             lambda s=seed: all((OUT / "runs" / run_name(m, s) / "best.pt").exists() for m in BLIND))
    stamp = (OUT / "results.md").stat().st_mtime if (OUT / "results.md").exists() else 0
    _run(py + ["evaluate", "--seeds"] + [str(s) for s in seeds],
         lambda: (OUT / "results.md").exists() and (OUT / "results.md").stat().st_mtime > stamp)


# ============================================================================ pilot 2: a fresh test cohort
FRESH_RANGE = (180, 600)          # section 6d: every population scene after the first 180, outside M1's test split


def fresh_dir():
    return OUT / "fresh"


def prepare_fresh(lo=FRESH_RANGE[0], hi=FRESH_RANGE[1]):
    """Test cohort of rendered population scenes lo..hi-1 that this experiment has never used, outside M1's sealed test
    split (the population's train and validation scenes only). Nothing in it was looked at when pilot 2 was designed."""
    used = set(sum((v for v in D.read(OUT / "split.json").values()), []))
    src = D.ROOT / D.SOURCE / "meta"
    ids = sorted(p.stem for p in (src.glob("s*.json")) if lo <= int(p.stem[1:]) < hi
                 and D.read(p)["split"] in ("train", "val") and p.stem not in used)
    if not ids:
        raise FileNotFoundError(f"no rendered scenes {lo}-{hi - 1} outside M1's test split; render them first "
                                f"(python tools/m1_det2.py render --start {lo} --count {hi - lo})")
    out = fresh_dir()
    if not (out / "manifest.json").exists():
        D.manifest(out, scenes=set(ids), smoke=True)          # checks the render contract and every layer
        m = D.read(out / "manifest.json")
        m["splits"], m["cohorts"] = dict(train=[], val=[]), dict(test=ids)
        m["training"] = D.read(OUT / "manifest.json")["training"]
        m["experiment"] = f"privileged-ablation fresh test cohort (population scenes {lo}-{hi - 1})"
        D.atomic_json(out / "manifest.json", m)
    D.atomic_json(out / "split.json", dict(train=[], val=[], test=D.read(out / "manifest.json")["cohorts"]["test"]))
    log(f"fresh test cohort: {len(D.read(out / 'split.json')['test'])} scenes of {lo}-{hi - 1}")


# ============================================================================ pilot 2: the thermal arms (section 6c)
# The 55 test scenes re-rendered with the target's propulsion heat changed and nothing else (tests/test_thermal_arms.py),
# each into its own dataset folder; every model is trained on today's engine only, so these test generalisation.
THERMAL_ARMS = {"low_heat": ("configs/m1_low_heat.yaml", "out/m1/dataset_low_heat"),
                "electric_like": ("configs/m1_electric_like.yaml", "out/m1/dataset_electric_like")}
INTERIM_MODELS = ("rgb_lwir", "teacher", "teacher_blind", "student", "student_ctrl", "student_g", "student_blind_g",
                  "student_ctrl_g")


def thermal_dir(arm):
    return OUT / f"thermal_{arm}"


def _test_indices():
    return [int(s[1:]) for s in D.read(OUT / "split.json")["test"]]


def render_thermal(arm):
    overlay, folder = THERMAL_ARMS[arm]
    _run([sys.executable, "-u", str(D.ROOT / "tools" / "m1_det2.py"), "render", "--overlay", overlay, "--source", folder,
          "--scenes"] + [str(i) for i in _test_indices()],
         lambda: all((D.ROOT / folder / "meta" / f"s{i:06d}.json").exists() for i in _test_indices()))


def prepare_thermal(arm):
    """The thermal arm's test cohort: this experiment's 55 test scenes from the arm's own dataset folder."""
    D.SOURCE = THERMAL_ARMS[arm][1]
    out, test = thermal_dir(arm), D.read(OUT / "split.json")["test"]
    if not (out / "manifest.json").exists():
        D.manifest(out, scenes=set(test), smoke=True)          # checks the arm's render contract and every layer
        m = D.read(out / "manifest.json")
        m["splits"], m["cohorts"] = dict(train=[], val=[]), dict(test=test)
        m["training"] = D.read(OUT / "manifest.json")["training"]
        m["experiment"] = f"privileged-ablation thermal arm {arm} ({THERMAL_ARMS[arm][0]})"
        D.atomic_json(out / "manifest.json", m)
    D.atomic_json(out / "split.json", dict(train=[], val=[], test=test))


def power():
    """Section 6d: the number of cloud seeds from the interim's variance components of detection at 0.01 (between seeds
    and between scenes, pooled over the primary arms; the contrasts' means are not used). A contrast's variance is taken
    as twice an arm's (arms independent: conservative). Writes OUT/pilot2_power.json."""
    src = OUT / "pilot2_interim_test55"
    sc = json.loads((src / "scores.json").read_text())
    arms = ("rgb_lwir", "teacher_blind", "student_g", "student_blind_g", "student_ctrl_g")
    seeds = sorted({int(k.split("|")[0]) for k in sc})
    rng = np.random.default_rng(20261019)
    n = len(sc[f"{seeds[0]}|rgb_lwir"]["best"])
    boots = [rng.integers(0, n, n) for _ in range(1000)]
    var_seed, var_scene = [], []
    for m in arms:
        vals, ses = [], []
        for s_ in seeds:
            b, ab = np.asarray(sc[f"{s_}|{m}"]["best"]), [[np.asarray(f) for f in a] for a in sc[f"{s_}|{m}"]["absent"]]
            vals.append(pd_at(b, ab))
            ses.append(np.std([pd_at(b[k], [ab[i] for i in k]) for k in boots], ddof=1))
        var_seed.append(np.var(vals, ddof=1))
        var_scene.append(np.mean(np.square(ses)))
    vs, vn = float(np.mean(var_seed)), float(np.mean(var_scene))
    fresh_n = len(D.read(fresh_dir() / "split.json")["test"]) if (fresh_dir() / "split.json").exists() else 290
    half = {k: float(1.96 * np.sqrt((2 * vs + 2 * vn * n / fresh_n) / k)) for k in range(20, 41)}
    pick = next((k for k in range(20, 41) if half[k] <= 0.025), 40)
    res = dict(rule="docs/privileged_teacher_case.md 6d", interim_seeds=seeds, arms=arms, scenes_interim=n,
               final_cohort_scenes=fresh_n, var_between_seeds=vs, var_scene_per_seed=vn, half_width_by_seeds=half,
               seeds=pick, seed_list=list(range(101, 101 + pick)), expected_half_width=half[pick])
    D.atomic_json(OUT / "pilot2_power.json", res)
    log(f"power (6d): between-seed SD {vs ** 0.5:.3f}, scene SE {vn ** 0.5:.3f} at {n} scenes -> {pick} seeds "
        f"(expected half-width {half[pick]:.3f} on {fresh_n} scenes)")
    return res


def run_pilot2(seeds, fresh=True, thermal=True, laptop=False):
    """Pilot 2's queue, in the order that answers the primary questions first. Finished work is skipped, so it resumes:
    (1) the primary arms of seeds 29-37, an interim evaluation on the 55 test scenes; (2) the thermal arms rendered,
    cached and evaluated (interim); (3) the primary arms of the other seeds and the one-extra-camera arms of all; (4) the
    fresh cohort rendered and cached; (5) the final evaluations of every cohort and their hard-case analyses."""
    py = [sys.executable, "-u", str(Path(__file__))]
    first = [s for s in seeds if s in (29, 31, 37)]

    def train_all(arms, names, which):
        for seed in which:
            _run(py + ["train", "--seed", str(seed), "--arms", arms],
                 lambda s=seed, n=names: all((OUT / "runs" / run_name(m, s) / "best.pt").exists() for m in n))

    def done(d):
        return lambda: (d / "results.md").exists()

    train_all("pilot2a", PILOT2A, first)
    first_s = [str(s) for s in first]
    _run(py + ["evaluate2", "--interim", "--seeds"] + first_s, done(OUT / "pilot2_interim_test55"))
    try:
        power()
    except Exception as e:                        # noqa: BLE001
        log(f"power step failed: {e}")
    ready = []                                   # a failed render or cache costs its cohort only, not the queue
    if thermal:
        for arm in THERMAL_ARMS:
            try:
                render_thermal(arm)
                _run(py + ["thermal-cache", "--arm", arm], lambda a=arm: (thermal_dir(a) / "cache_fingerprint.json").exists())
                _run(py + ["evaluate-thermal", "--arm", arm, "--interim", "--seeds"] + first_s,
                     done(thermal_dir(arm) / "interim"))
                ready.append(arm)
            except Exception as e:                # noqa: BLE001
                log(f"thermal arm {arm} skipped: {e}")
    if laptop:                                    # 6d: the cloud trains; the laptop renders the final cohort
        try:
            lo, hi = FRESH_RANGE
            _run([sys.executable, "-u", str(D.ROOT / "tools" / "m1_det2.py"), "render", "--start", str(lo), "--count",
                  str(hi - lo)], lambda: all((D.ROOT / D.SOURCE / "meta" / f"s{i:06d}.json").exists() for i in range(lo, hi)))
            _run(py + ["fresh-cache"], lambda: (fresh_dir() / "cache_fingerprint.json").exists())
            power()                               # again, with the final cohort's size
        except Exception as e:                    # noqa: BLE001
            log(f"fresh cohort: {e}")
        return
    train_all("pilot2a", PILOT2A, [s for s in seeds if s not in first])
    train_all("pilot2b", PILOT2B, seeds)
    seeds_s = [str(s) for s in seeds]
    cohorts = [("pilot2_test55", ["evaluate2"], OUT / "pilot2_test55", [])]
    cohorts += [(f"pilot2_{arm}", ["evaluate-thermal", "--arm", arm], thermal_dir(arm), ["--no-census"]) for arm in ready]
    if fresh:
        lo, hi = FRESH_RANGE
        try:
            _run([sys.executable, "-u", str(D.ROOT / "tools" / "m1_det2.py"), "render", "--start", str(lo), "--count",
                  str(hi - lo)], lambda: all((D.ROOT / D.SOURCE / "meta" / f"s{i:06d}.json").exists() for i in range(lo, hi)))
            _run(py + ["fresh-cache"], lambda: (fresh_dir() / "cache_fingerprint.json").exists())
            cohorts.append(("pilot2_fresh", ["evaluate-fresh"], fresh_dir(), []))
        except Exception as e:                    # noqa: BLE001
            log(f"fresh cohort skipped: {e}")
    hc = [sys.executable, "-u", str(D.ROOT / "tools" / "privileged_hardcases.py"), "--arms", "pilot2", "--seeds"] + seeds_s
    for tag, cmd, dest, extra in cohorts:
        _run(py + cmd + ["--seeds"] + seeds_s, done(dest))
        _run(hc + ["--scores", str(dest), "--tag", tag] + extra, lambda t=tag: (DATA_DIR / f"hardcases_{t}.json").exists())


DATA_DIR = CENSUS.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "cache", "train", "evaluate", "all", "blind", "evaluate2", "fresh-prepare",
                                        "fresh-cache", "evaluate-fresh", "thermal-render", "thermal-cache",
                                        "evaluate-thermal", "pilot2", "power"])
    ap.add_argument("--arm", choices=sorted(THERMAL_ARMS), default=None, help="thermal-*: which thermal arm")
    ap.add_argument("--interim", action="store_true", help="evaluate2 / evaluate-thermal: the interim models only")
    ap.add_argument("--no-thermal", action="store_true", help="pilot2: skip the thermal arms")
    ap.add_argument("--laptop", action="store_true", help="pilot2: after the interim, render and cache only (6d: the cloud trains)")
    ap.add_argument("--after", type=int, default=None, help="blind: wait for this process id to end first")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true", help="a mechanics check in out/m1/privileged_ablation_smoke")
    ap.add_argument("--blind", action="store_true", help="train: the blind teacher and its student (secondary control)")
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default=None, help="train: which arms (default pilot1)")
    ap.add_argument("--no-fresh", action="store_true", help="pilot2: skip the fresh test cohort")
    a = ap.parse_args()
    global OUT
    if a.smoke:
        OUT = OUT.parent / "privileged_ablation_smoke"
    seeds = a.seeds or (list(PILOT2_SEEDS) if a.command in ("evaluate2", "evaluate-fresh", "pilot2") else [29, 31, 37])
    D.keep_awake(True)
    try:
        if a.command == "prepare":
            prepare(a.smoke)
        elif a.command == "cache":
            prepare(a.smoke)
            D.cache(OUT, {"train", "val", "test"})
        elif a.command == "train":
            train(a.seed, a.blind, a.arms)
        elif a.command == "evaluate":
            for s in seeds:
                register(s)
            evaluate(seeds)
        elif a.command == "evaluate2":         # pilot 2 on this experiment's test split; pilot 1's files are kept
            for s in seeds:
                register(s)
            name = ("pilot2_interim_test55" if a.interim else "pilot2_test55") if not a.smoke else "pilot2_test"
            evaluate(seeds, INTERIM_MODELS if a.interim else EVAL_ALL, dest=OUT / name, save_predictions=not a.interim)
        elif a.command == "thermal-render":
            render_thermal(a.arm)
        elif a.command == "thermal-cache":
            prepare_thermal(a.arm)
            D.cache(thermal_dir(a.arm), {"test"})
        elif a.command == "evaluate-thermal":
            D.SOURCE = THERMAL_ARMS[a.arm][1]
            for s in seeds:
                register(s)
            dest = thermal_dir(a.arm) / "interim" if a.interim else thermal_dir(a.arm)
            evaluate(seeds, INTERIM_MODELS if a.interim else EVAL_ALL, src=thermal_dir(a.arm), dest=dest,
                     save_predictions=not a.interim, census_bins=False)
        elif a.command == "fresh-prepare":
            prepare_fresh()
        elif a.command == "fresh-cache":
            prepare_fresh()
            D.cache(fresh_dir(), {"test"})
        elif a.command == "evaluate-fresh":
            for s in seeds:
                register(s)
            evaluate(seeds, EVAL_ALL, src=fresh_dir(), dest=fresh_dir(), save_predictions=True)
        elif a.command == "blind":
            run_blind(seeds, a.after)
        elif a.command == "pilot2":
            run_pilot2(seeds, fresh=not a.no_fresh, thermal=not a.no_thermal, laptop=a.laptop)
        elif a.command == "power":
            power()
        else:
            run_all(seeds, a.smoke)
    finally:
        D.keep_awake(False)


if __name__ == "__main__":
    main()
