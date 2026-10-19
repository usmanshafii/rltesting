"""Pilot 2 on a cloud GPU (docs/privileged_teacher_case.md, section 6d): package the inputs on the laptop, train and
evaluate the seeds in the cloud (Kaggle or Colab), and merge the results back.

    laptop:  python tools/cloud_pilot2.py package          -> out/m1/cloud/kal_pilot2_train.zip  (code + training data)
             python tools/cloud_pilot2.py package-cohorts  -> out/m1/cloud/kal_pilot2_cohorts.zip (thermal-arm and fresh
                                                              test frames, once the laptop has rendered and cached them)
    cloud:   python tools/cloud_pilot2.py run --seeds 101-120 --sensor-seeds 101-115 --workers 3
             (cloud/pilot2_cloud.ipynb does this; it trains each seed's arms with tools/privileged_ablation.py on the GPU,
             several seeds side by side, evaluates them on every cohort present, and zips the results)
    laptop:  python tools/cloud_pilot2.py merge <results.zip>   -> out/m1/privileged_ablation_cloud/

The code and settings are the laptop's; only the device differs (KAL_DEVICE=cuda). Every seed's runs are trained and
evaluated by the same worker, so a session that stops costs only the seeds it was working on: run again, and finished
runs and evaluations are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PA = ROOT / "out" / "m1" / "privileged_ablation"
CLOUD = ROOT / "out" / "m1" / "cloud"
COHORTS = {"test55": None, "low_heat": "out/m1/dataset_low_heat", "electric_like": "out/m1/dataset_electric_like",
           "fresh": "out/m1/dataset"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _seeds(spec):
    out = []
    for part in spec:
        for p in str(part).split(","):
            if "-" in p:
                a, b = p.split("-")
                out += list(range(int(a), int(b) + 1))
            elif p:
                out.append(int(p))
    return sorted(set(out))


def _code_files():
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                       capture_output=True, check=True)
    return sorted(f for f in r.stdout.decode("utf-8").split("\0")
                  if f and not f.startswith("out/") and (ROOT / f).is_file())


def _zip(dest, items):
    """items: (path, arcname). Frames are stored (float16 does not compress); the rest is deflated."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    n = 0
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for path, arc in items:
            z.write(path, arc, compress_type=zipfile.ZIP_STORED if path.suffix == ".npy" else zipfile.ZIP_DEFLATED)
            n += 1
    os.replace(tmp, dest)
    log(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB, {n} files)")


def package():
    """The code and pilot 2's training inputs: manifest, split, the training crops, the validation and test frames,
    and the scenes' metadata. Nothing else from out/ (no checkpoints, no results)."""
    items = [(ROOT / f, f"fyp/{f}") for f in _code_files()]
    for f in ("manifest.json", "split.json", "cache_fingerprint.json"):
        items.append((PA / f, f"fyp/out/m1/privileged_ablation/{f}"))
    for sub in ("train", "eval"):
        for p in sorted((PA / "cache" / sub).iterdir()):
            items.append((p, f"fyp/out/m1/privileged_ablation/cache/{sub}/{p.name}"))
    for p in sorted((ROOT / "out/m1/dataset/meta").glob("s*.json")):
        items.append((p, f"fyp/out/m1/dataset/meta/{p.name}"))
    _zip(CLOUD / "kal_pilot2_train.zip", items)


def package_cohorts():
    """The thermal arms' and the fresh cohort's test frames, manifests, splits and scene metadata, as far as the laptop
    has built them."""
    items = []
    for arm, folder in (("low_heat", "out/m1/dataset_low_heat"), ("electric_like", "out/m1/dataset_electric_like")):
        d = PA / f"thermal_{arm}"
        if not (d / "cache_fingerprint.json").exists():
            log(f"thermal arm {arm}: not cached yet, left out")
            continue
        for f in ("manifest.json", "split.json", "cache_fingerprint.json"):
            items.append((d / f, f"fyp/out/m1/privileged_ablation/thermal_{arm}/{f}"))
        for p in sorted((d / "cache" / "eval").iterdir()):
            items.append((p, f"fyp/out/m1/privileged_ablation/thermal_{arm}/cache/eval/{p.name}"))
        for p in sorted((ROOT / folder / "meta").glob("s*.json")):
            items.append((p, f"fyp/{folder}/meta/{p.name}"))
    d = PA / "fresh"
    if (d / "cache_fingerprint.json").exists():
        for f in ("manifest.json", "split.json", "cache_fingerprint.json"):
            items.append((d / f, f"fyp/out/m1/privileged_ablation/fresh/{f}"))
        for p in sorted((d / "cache" / "eval").iterdir()):
            items.append((p, f"fyp/out/m1/privileged_ablation/fresh/cache/eval/{p.name}"))
        ids = json.loads((d / "split.json").read_text())["test"]
        for sid in ids:
            items.append((ROOT / "out/m1/dataset/meta" / f"{sid}.json", f"fyp/out/m1/dataset/meta/{sid}.json"))
    else:
        log("fresh cohort: not cached yet, left out")
    if not items:
        raise SystemExit("nothing to package yet")
    _zip(CLOUD / "kal_pilot2_cohorts.zip", items)


def _present_cohorts():
    have = ["test55"]
    for arm in ("low_heat", "electric_like"):
        if (PA / f"thermal_{arm}" / "cache_fingerprint.json").exists():
            have.append(arm)
    if (PA / "fresh" / "cache_fingerprint.json").exists():
        have.append("fresh")
    return have


def _evaluate(seed, cohort):
    """One seed's models on one cohort, into out/m1/privileged_ablation/cloud_eval/<cohort>/s<seed> (skipped if done)."""
    seeds = [seed]
    dest = PA / "cloud_eval" / cohort / f"s{seed}"
    if (dest / "results.md").exists():
        return
    code = (
        "import sys; sys.path[:0] = [r'%s', r'%s', r'%s']\n"
        "import privileged_ablation as P\n"
        "from pathlib import Path\n"
        "P.OUT = Path(r'%s')\n"
        "seeds = %r; cohort = %r; dest = Path(r'%s')\n"
        "[P.register(s) for s in seeds]\n"
        "if cohort == 'test55':\n"
        "    P.evaluate(seeds, P.EVAL_ALL, dest=dest, save_predictions=True)\n"
        "elif cohort == 'fresh':\n"
        "    P.evaluate(seeds, P.EVAL_ALL, src=P.fresh_dir(), dest=dest, save_predictions=True)\n"
        "else:\n"
        "    P.D.SOURCE = P.THERMAL_ARMS[cohort][1]\n"
        "    P.evaluate(seeds, P.EVAL_ALL, src=P.thermal_dir(cohort), dest=dest, save_predictions=True, census_bins=False)\n"
    ) % (ROOT, ROOT / "target", ROOT / "tools", PA, seeds, cohort, dest)
    r = subprocess.run([sys.executable, "-u", "-c", code])
    if r.returncode != 0:
        log(f"seed {seed}: evaluation on {cohort} failed (code {r.returncode})")


def _results_dir():
    return Path(os.environ.get("KAL_RESULTS_DIR", str(CLOUD)))


def pack_seed(seed):
    """One seed's runs and evaluations, saved as soon as the seed is done (a lost session costs only the seed in
    progress): <results dir>/kal_pilot2_seed_<seed>.zip, replaced when the seed gains a cohort."""
    items = []
    for d in sorted((PA / "runs").glob(f"*_s{seed}")):
        for f in ("best.pt", "untrained.pt", "training.json", "history.json", "validation.json"):
            if (d / f).exists():
                items.append((d / f, f"runs/{d.name}/{f}"))
    for d in sorted((PA / "cloud_eval").glob(f"*/s{seed}")):
        items += [(p, f"cloud_eval/{p.relative_to(PA / 'cloud_eval').as_posix()}") for p in sorted(d.rglob("*")) if p.is_file()]
    if items:
        _zip(_results_dir() / f"kal_pilot2_seed_{seed}.zip", items)


def _worker(k, seeds, sensor, cohorts, smoke=False):
    """Train, evaluate and save one worker's seeds, one seed at a time (run in its own process by `run`)."""
    global PA
    if smoke:
        PA = PA.parent / "privileged_ablation_smoke"
    py = [sys.executable, "-u", str(ROOT / "tools" / "privileged_ablation.py")] + (["--smoke"] if smoke else [])
    for s in seeds:
        for arms in (["pilot2a"] + (["pilot2b"] if s in sensor else [])):
            for attempt in range(3):
                r = subprocess.run(py + ["train", "--seed", str(s), "--arms", arms])
                if r.returncode == 0:
                    break
                log(f"worker {k}: seed {s} {arms} ended with code {r.returncode}; retry {attempt + 1}")
        for c in cohorts:
            _evaluate(s, c)
        pack_seed(s)


def run(seeds, sensor, workers, cohorts=None, smoke=False):
    import torch
    env = dict(os.environ)
    env.setdefault("KAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    env.setdefault("KAL_THREADS", "1")
    cohorts = cohorts or _present_cohorts()
    log(f"device {env['KAL_DEVICE']}, {workers} workers, seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}), "
        f"one-extra-camera arms for {len(sensor)} seeds, cohorts {cohorts}")
    procs = []
    for k in range(workers):
        mine = seeds[k::workers]
        if not mine:
            continue
        code = ("import sys; sys.path[:0] = [r'%s']\nimport cloud_pilot2 as C\nC._worker(%d, %r, %r, %r, %r)\n"
                % (ROOT / "tools", k, mine, sorted(sensor), cohorts, smoke))
        procs.append(subprocess.Popen([sys.executable, "-u", "-c", code], env=env))
    for p in procs:
        p.wait()
    pack()


def pack():
    """Every cloud run (checkpoints, training records) and evaluation, into one zip for the laptop."""
    stamp = time.strftime("%Y%m%d_%H%M")
    items = []
    for d in sorted((PA / "runs").iterdir()):
        for f in ("best.pt", "untrained.pt", "training.json", "history.json", "validation.json"):
            if (d / f).exists():
                items.append((d / f, f"runs/{d.name}/{f}"))
    ev = PA / "cloud_eval"
    if ev.exists():
        for p in sorted(ev.rglob("*")):
            if p.is_file():
                items.append((p, f"cloud_eval/{p.relative_to(ev).as_posix()}"))
    _zip(_results_dir() / f"kal_pilot2_results_{stamp}.zip", items)


def resume(zips):
    """Before a cloud session: unpack earlier sessions' seed and result zips into this session's experiment folder,
    so finished runs and evaluations are skipped."""
    for z in sorted(zips):
        with zipfile.ZipFile(z) as f:
            f.extractall(PA)
        log(f"resumed from {z}")


def merge(zips):
    """Unpack cloud result zips into out/m1/privileged_ablation_cloud (runs and cloud_eval) on the laptop."""
    dest = ROOT / "out" / "m1" / "privileged_ablation_cloud"
    for z in sorted(zips):
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        log(f"merged {z} into {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["package", "package-cohorts", "run", "pack", "resume", "merge"])
    ap.add_argument("--seeds", nargs="*", default=["101-120"])
    ap.add_argument("--sensor-seeds", nargs="*", default=["101-115"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cohorts", nargs="*", default=None, choices=list(COHORTS))
    ap.add_argument("--smoke", action="store_true", help="run / pack: a mechanics check in out/m1/privileged_ablation_smoke")
    ap.add_argument("zips", nargs="*")
    a = ap.parse_args()
    global PA
    if a.smoke:
        PA = PA.parent / "privileged_ablation_smoke"
    if a.command == "package":
        package()
    elif a.command == "package-cohorts":
        package_cohorts()
    elif a.command == "run":
        run(_seeds(a.seeds), set(_seeds(a.sensor_seeds)), a.workers, a.cohorts, a.smoke)
    elif a.command == "pack":
        pack()
    elif a.command == "resume":
        resume([Path(z) for z in a.zips])
    else:
        merge([Path(z) for z in a.zips])


if __name__ == "__main__":
    main()
