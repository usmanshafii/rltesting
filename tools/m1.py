"""M1-lite entry point (two bands, full detector). Superseded for M1 by tools/m1_det2.py, the four-sensor preliminary
evaluation (docs/m1_v2.md); kept for reference. Its RGB + LWIR render contract does not match the four-band population
in out/m1/dataset, so generate_dataset refuses to mix them. Uses the existing renderer, loader, detector and evaluator.

    python tools/m1.py preflight
    python tools/m1.py smoke
    python tools/m1.py render
    python tools/m1.py audit
    python tools/m1.py package
    python tools/m1.py train --device cuda
    python tools/m1.py evaluate --device cuda

Smoke outputs are isolated under <out>/smoke*, never accepted as M1 results.
"""
from __future__ import annotations

import argparse
import csv
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target")]

from data.dataset import load_metas
from data.provenance import atomic_json, check_scene, dataset_identity, render_contract
from render.sampling import assign_splits, load_configs, load_yaml

MODELS = ("rgb_only", "lwir_only")
BANDS = ("rgb", "lwir")


def configs(smoke=False):
    cfg = load_configs(overlay=str(ROOT / "configs/m1.yaml"))
    cfg["train"] = load_yaml(str(ROOT / "configs/train.yaml"))
    cfg["train"]["train"]["epochs"] = 60
    if smoke:
        cfg["render"]["render"]["engine_samples"] = 16
        cfg["render"]["render"]["adaptive_min_samples"] = 16
        cfg["train"]["train"].update(batch=2, workers=0, eval_every=1)
        cfg["train"]["eval"]["bootstrap"] = 30
    return cfg


def audit(data, cfg, smoke=False, decode=True):
    """Validate the complete M1 contract without redrawing or changing the scene distribution."""
    from data.composite import load_layers
    import numpy as np
    data = Path(data)
    n = cfg["render"]["dataset"]["n_scenes"]
    expected = 8 if smoke else n
    metas = load_metas(str(data))
    errors = []
    expected_ids = {f"s{i:06d}" for i in range(expected)}
    ids = {m["scene_id"] for m in metas}
    if ids != expected_ids or len(metas) != expected:
        errors.append(f"expected {expected} scenes: missing {sorted(expected_ids - ids)[:12]}, "
                      f"unexpected {sorted(ids - expected_ids)[:12]}")
    contract = render_contract(cfg, n, BANDS)
    splits = assign_splits(n, cfg["render"]["dataset"]["splits"], cfg["render"]["dataset"]["seed0"])
    counts = collections.defaultdict(collections.Counter)
    for m in metas:
        sid = m["scene_id"]
        try:
            idx = int(sid[1:])
            if not 0 <= idx < n or m["split"] != splits[idx]:
                raise ValueError("split differs from the fixed 600-scene assignment")
            if not check_scene(m, data, contract):
                raise ValueError("missing layer files")
            c = m["conditions"]
            if c["background"] != "open_sky" or c["lighting"] not in ("day", "night") or c["weather"] != "clear":
                raise ValueError("conditions are outside M1")
            if not 20 <= c["visibility_km"] <= 60 or m["negatives"]:
                raise ValueError("M1 needs clear air (20-60 km) and no negatives")
            if not np.isclose(m["camera"]["hfov_deg"], 8.7834) or not 5 <= m["extra"]["camera"]["elevation_deg"] <= 25:
                raise ValueError("M1 camera differs")
            objs = m["extra"]["objects"]
            if len(objs) != 1 or objs[0]["cls"] != "target":
                raise ValueError("expected exactly one target")
            t = m["target"]
            if t["size_bin"] != cfg["render"]["size_bins"]["names"][idx % 6]:
                raise ValueError("wrong size bin")
            b = t["bbox_px"]
            if not (0 <= b["x"] < b["x"] + b["w"] <= m["camera"]["width"] and
                    0 <= b["y"] < b["y"] + b["h"] <= m["camera"]["height"]):
                raise ValueError("box is outside the frame")
            for band in BANDS:
                path = (data / m["modalities"][band]["layers"]).resolve()
                if not path.is_relative_to(data.resolve()):
                    raise ValueError("layer path escapes the dataset")
                if decode:
                    layer, objects = load_layers(str(path))
                    if len(objects) != 1 or layer["sky"].shape[:2] != (512, 640):
                        raise ValueError(f"invalid {band} layer dimensions/object count")
                    for arr in [layer["sky"], objects[0]["P"], objects[0]["A"]]:
                        if not np.isfinite(arr).all():
                            raise ValueError(f"non-finite {band} pixels")
            for key, val in (("split", m["split"]), ("bin", t["size_bin"]), ("lighting", c["lighting"]),
                             ("split_x_bin", f"{m['split']}:{t['size_bin']}"),
                             ("split_x_lighting", f"{m['split']}:{c['lighting']}")):
                counts[key][val] += 1
        except (KeyError, ValueError, OSError, IndexError) as exc:
            errors.append(f"{sid}: {exc}")
    result = dict(status="passed" if not errors else "failed", purpose="smoke_only" if smoke else "m1",
                  expected_scenes=expected, n_scenes=len(metas), counts=dict(counts), errors=errors,
                  dataset_sha256=dataset_identity(metas), render_contract=contract,
                  decoded_layers=decode)
    atomic_json(data / "audit.json", result)
    if errors:
        raise ValueError("Dataset audit failed:\n" + "\n".join(errors[:20]))
    return result


def command(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], cwd=ROOT, check=True)


def preflight(out):
    import torch
    from tools.generate_dataset import find_blender
    try:
        blender = find_blender()
    except FileNotFoundError:
        blender = None
    cfg = configs()
    result = dict(python=sys.version, torch=str(torch.__version__), cuda=torch.cuda.is_available(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  blender=blender, overlay=cfg["overlay"], models=list(MODELS), bands=list(BANDS),
                  n_scenes=600, epochs_per_model=60,
                  signature_model=cfg["signatures"]["model_revision"])
    atomic_json(out / "preflight.json", result)
    print(json.dumps(result, indent=2))
    return result


def train(data, runs, cfg, args, smoke=False):
    from train.engine import train_model
    audit(data, cfg, smoke)
    tc = cfg["train"]["train"]
    tc.update(device=args.device, workers=args.workers, batch=args.batch or (2 if smoke else 8))
    if not smoke and args.device == "auto":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU. Transfer the package to a GPU, or explicitly select --device cpu.")
    for name in MODELS:
        train_model(name, str(data), str(runs), cfg, epochs=2 if smoke else 60,
                    max_iters=2 if smoke else None, resume=True,
                    gate_epoch=10 if name == "rgb_only" and not smoke else None,
                    continue_after_gate=args.continue_after_gate)


def evaluate(data, runs, out, cfg, device, smoke=False, backend="repository"):
    from eval.evaluate import evaluate as run_eval
    from eval.pd_fit import run as fit
    from eval.sensitivity import run as sensitivity
    from tools.m1_report import report
    audit(data, cfg, smoke)
    if backend == "repository":
        import torch
        for name in MODELS:
            saved = json.loads((runs / name / "config.json").read_text())
            current_data = dataset_identity([m for m in load_metas(str(data)) if m["split"] in ("train", "val")])
            if saved["provenance"]["dataset_sha256"] != current_data:
                raise ValueError(f"{name}: training and evaluation datasets differ")
            state = torch.load(runs / name / "last.pt", map_location="cpu", weights_only=False)
            if state["epoch"] + 1 < (2 if smoke else 60) or (not smoke and saved.get("max_iters")):
                raise ValueError(f"{name}: incomplete training cannot produce a final M1 report")
    else:
        from data.provenance import config_identity
        manifest = json.loads((runs.parent / "yolo/export_manifest.json").read_text())
        expected = dict(dataset_sha256=dataset_identity(load_metas(str(data))),
                        config_sha256=config_identity(cfg), bands=list(BANDS),
                        splits=["train", "val", "test"], noise_seed=cfg["train"]["train"]["seed"])
        if manifest["status"] != "complete" or manifest["contract"] != expected:
            raise ValueError("fallback export does not match this dataset/configuration")
        for name in MODELS:
            with (runs / name / "results.csv").open(newline="") as f:
                if len(list(csv.DictReader(f))) < 60:
                    raise ValueError(f"{name}: fallback training has not completed 60 epochs")
    run_eval([str(runs / m) for m in MODELS], str(data), "test", str(out), device, 2, cfg, backend=backend)
    fit(str(out), str(data), str(out), cfg["train"]["eval"], n_boot=30 if smoke else 500)
    sensitivity(str(out), str(data), str(out / "sensitivity"), cfg["train"]["eval"])
    report(data, out, cfg, smoke=smoke)


def package(data, out, cfg, smoke=False):
    """Working-tree source + minimal lossless dataset; no Git/private settings or stale outputs."""
    audit(data, cfg, smoke)
    files = {}
    for folder in ("configs", "data", "eval", "models", "noise", "render", "target", "thermal", "tools", "train", "tests", "assets"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".yaml", ".npy", ".png", ".json"):
                files[path.relative_to(ROOT).as_posix()] = path
    for rel in ("README.md", "LICENSE.md", "AGENTS.md", "CLAUDE.md", "requirements.txt", "requirements-ml.txt", "docs/archive/m1_lite_plan.md", "docs/archive/m1_runbook.md"):
        files[rel] = ROOT / rel
    for path in (ROOT / "docs").rglob("*.md"):
        files[path.relative_to(ROOT).as_posix()] = path
    for path in ROOT.glob("FYP_*.md"):
        files[path.name] = path
    data_prefix = "out/m1/smoke/" if smoke else "out/m1/dataset/"
    for folder in ("meta", "layers", "params"):
        for path in (data / folder).rglob("*"):
            if path.is_file():
                files[data_prefix + path.relative_to(data).as_posix()] = path
    files[data_prefix + "audit.json"] = data / "audit.json"
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}
    destination = out / ("m1_smoke_gpu.zip" if smoke else "m1_gpu.zip")
    temp = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
        z.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    os.replace(temp, destination)
    atomic_json(out / ("smoke_bundle_manifest.json" if smoke else "bundle_manifest.json"), manifest)
    print(f"Wrote {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"bundle checksum mismatch: {name}")
    print(f"Verified {len(manifest)} bundled files")
    return len(manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["preflight", "smoke", "render", "audit", "train", "evaluate", "package", "export", "verify-bundle", "status"])
    p.add_argument("--out", type=Path, default=ROOT / "out/m1")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--continue-after-gate", action="store_true")
    p.add_argument("--backend", choices=("repository", "ultralytics"), default="repository", help="evaluation backend")
    p.add_argument("--skip-render", action="store_true", help="smoke: use already rendered/bundled scenes without Blender")
    p.add_argument("--smoke-data", action="store_true", help="package: create a separate smoke-only handoff")
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ["KAL_CONFIG_OVERLAY"] = str(ROOT / "configs/m1.yaml")
    if a.skip_render and a.stage != "smoke":
        p.error("--skip-render is only for the smoke stage")
    if a.smoke_data and a.stage != "package":
        p.error("--smoke-data is only for packaging")
    smoke = a.stage == "smoke" or a.smoke_data
    cfg = configs(smoke)
    data, runs = a.out / ("smoke" if smoke else "dataset"), a.out / ("smoke_runs" if smoke else "runs")
    if a.stage == "preflight":
        preflight(a.out)
    elif a.stage == "verify-bundle":
        verify_bundle(ROOT)
    elif a.stage == "status":
        print(f"Production scenes: {len(list((data / 'meta').glob('*.json')))}/600")
        for name in MODELS:
            path = runs / name / "history.json"
            print(f"{name}: " + (f"{len(json.loads(path.read_text()))} saved epochs" if path.exists() else "not started"))
        print("Production bundle: " + ("available" if (a.out / "m1_gpu.zip").exists() else "not yet available"))
    elif a.stage in ("smoke", "render"):
        extra = ["--count", "8", "--samples", "16"] if smoke else []
        if not a.skip_render:
            command("tools/generate_dataset.py", "--out", data, "--n", "600", "--chunk", "8" if smoke else "25",
                    "--bands", "rgb,lwir", "--jobs", a.jobs, "--threads", a.threads, *extra)
        result = audit(data, cfg, smoke)
        print(json.dumps(result["counts"], indent=2))
        command("tools/preview_dataset.py", "--data", data, "--n", "8", "--out",
                a.out / ("smoke_preview.png" if smoke else "preview.png"))
        if smoke:
            train(data, runs, cfg, a, smoke=True)
            evaluate(data, runs, a.out / "smoke_eval", cfg, a.device, smoke=True)
        else:
            package(data, a.out, cfg)
    elif a.stage == "audit":
        print(json.dumps(audit(data, cfg), indent=2))
    elif a.stage == "train":
        train(data, runs, cfg, a)
    elif a.stage == "evaluate":
        if a.backend == "ultralytics":
            runs = a.out / "ultralytics"
        evaluate(data, runs, a.out / ("eval" if a.backend == "repository" else "eval_ultralytics"), cfg, a.device,
                 backend=a.backend)
    elif a.stage == "package":
        package(data, a.out, cfg, smoke)
    elif a.stage == "export":
        audit(data, cfg)
        command("tools/export_yolo.py", "--data", data, "--out", a.out / "yolo", "--bands", "rgb,lwir")


if __name__ == "__main__":
    main()
