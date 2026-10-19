"""Evaluate trained models on identical test scenes (spec 8).

    python eval/evaluate.py --data out/dataset --split test \
        --runs out/runs/rgb_only out/runs/lwir_only out/runs/rgb_lwir out/runs/teacher out/runs/student

Writes <out>/report.json (per model: overall / per bin / per condition metrics and Pd curves), the
paired-bootstrap comparisons that answer the research question, and the plots.

Negative results are results: if the teacher does not beat the RGB + LWIR baseline, or the
student does not beat it, the report says the teacher's privileged sensors (LWIR cooled and MWIR; CLAUDE.md sensor
set rule 5) gave no transferable benefit under these assumptions (spec 8).
"""
import argparse
import json
import os
import sys
import hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.dataset import SceneDataset  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import plots  # noqa: E402
from train.engine import device_of, load_all_configs, load_ckpt  # noqa: E402

COMPARE = [("teacher", "rgb_lwir"), ("student", "rgb_lwir"), ("rgb_lwir", "rgb_only"), ("rgb_lwir", "lwir_only"),
           ("lwir_only", "rgb_only"),                               # the M1 question
           ("lwir_cooled_only", "lwir_only"), ("mwir_only", "lwir_only")]   # each teacher sensor against the student's


def _json(x):
    if isinstance(x, dict):
        return {k: _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def predict_yolo_records(model, ds, bands, cfg, device):
    """Fallback predictions in original pixel coordinates on the exact exported 8-bit images."""
    from tools.export_yolo import to_png
    from data.dataset import CLASSES
    records = []
    for i in range(len(ds)):
        s = ds.load(i)
        image = np.asarray(to_png(s["images"][bands[0]]))
        # Ultralytics array inputs use OpenCV BGR; its file loader expands monochrome to 3 channels.
        image = np.repeat(image[..., None], 3, axis=-1) if image.ndim == 2 else image[..., ::-1].copy()
        result = model.predict(source=image, imgsz=max(ds.W, ds.H), conf=cfg["conf"],
                               iou=cfg["nms_iou"], max_det=cfg["max_det"], device=str(device), verbose=False)[0]
        b = result.boxes
        meta = s["meta"]
        records.append(dict(scene_id=s["scene_id"],
                            preds=dict(boxes=b.xyxy.cpu().numpy(), scores=b.conf.cpu().numpy(),
                                       labels=b.cls.cpu().numpy().astype(int)),
                            gts=[dict(o, cls_name=CLASSES[o["label"]]) for o in s["objects"]],
                            conditions=meta["conditions"], camera=meta["camera"], visibility_km=s["visibility_km"],
                            target_aspect_deg=M.target_aspect_deg(meta)))
    return records


def evaluate(runs, data, split="test", out=os.path.join(ROOT, "out", "eval"), device="auto", batch=4, cfgs=None,
             backend="repository"):
    cfgs = cfgs or load_all_configs()
    E = cfgs["train"]["eval"]
    dev = device_of(device)
    os.makedirs(out, exist_ok=True)
    reports, records = {}, {}
    provenance = {}
    for run in runs:
        if backend == "ultralytics":
            from ultralytics import YOLO
            if run.endswith(".pt"):
                raise ValueError("fallback --runs must be rgb_only/lwir_only run folders containing weights/best.pt")
            name = os.path.basename(os.path.normpath(run))
            if name not in ("rgb_only", "lwir_only"):
                raise ValueError("fallback run names must be rgb_only and lwir_only")
            ck = os.path.join(run, "weights", "best.pt")
            model = YOLO(ck)
            bands, meta = [name.split("_")[0]], {}
        elif backend == "repository":
            ck = run if run.endswith(".pt") else os.path.join(run, "best.pt")
            model, meta = load_ckpt(ck, dev)
            bands = model.bands
            name = os.path.basename(os.path.dirname(ck))
            from data.provenance import config_identity
            trained_hash = (meta.get("provenance") or {}).get("config_sha256")
            if trained_hash and trained_hash != config_identity(cfgs):
                raise ValueError(f"{name}: evaluation configs differ from the effective training configs")
        else:
            raise ValueError(f"unknown backend: {backend}")
        if name in records:
            raise ValueError(f"duplicate model/run name: {name}")
        ds = SceneDataset(data, split, bands, cfgs, train=False, seed=cfgs["train"]["train"]["seed"])
        print(f"{name}: {len(ds)} {split} scenes, bands {bands}")
        if not len(ds):
            raise ValueError(f"no {split} scenes under {data}")
        with open(ck, "rb") as f:
            checkpoint_hash = hashlib.sha256(f.read()).hexdigest()
        provenance[name] = dict(checkpoint=os.path.abspath(ck), epoch_number=meta.get("epoch", -1) + 1 or None,
                                checkpoint_sha256=checkpoint_hash, backend=backend,
                                training=meta.get("provenance"))
        rec = (M.predict_records(model, ds, bands, E, dev, batch) if backend == "repository" else
               predict_yolo_records(model, ds, bands, E, dev))
        records[name] = rec
        reports[name] = M.summary(rec, E)
        o = reports[name]["overall"]
        print(f"  AP50 {o['ap50']:.3f}  AP50:95 {o['ap50_95']:.3f}  AP(centre) {o.get('ap_centre', float('nan')):.3f}  "
              f"FPPI {o['op_iou50']['fppi']:.3f}")
        for b, v in reports[name]["bins"].items():
            print(f"    {b:>6s} px  AP50 {v['ap50']:.3f}  AP(centre) {v.get('ap_centre', float('nan')):.3f}  "
                  f"recall {v['op_iou50']['recall']:.3f}  n {v['op_iou50']['n_gt']}")
    comparisons = {}
    for a_, b_ in COMPARE:
        if a_ in records and b_ in records:
            c = {}
            for key, fn in (("ap50", lambda r: M.ap(r, "iou", 0.5, cfg=E)),
                            ("ap_centre", lambda r: M.ap(r, "centre", 0.0, cfg=E))):
                c[key] = M.bootstrap_diff(records[a_], records[b_], fn, min(E["bootstrap"], 300))
            for b in E["tiny_bins"]:
                c[f"ap_centre_{b}"] = M.bootstrap_diff(records[a_], records[b_],
                                                       lambda r, b=b: M.ap(r, "centre", 0.0, b, cfg=E), min(E["bootstrap"], 300))
            comparisons[f"{a_}_vs_{b_}"] = c
    verdict = []
    for key in ("teacher_vs_rgb_lwir", "student_vs_rgb_lwir"):
        if key in comparisons:
            d = comparisons[key]["ap_centre"]
            if d["significant"] and d["diff"] > 0:
                verdict.append(f"{key}: +{d['diff']:.3f} AP(centre), 95% CI [{d['lo']:.3f}, {d['hi']:.3f}] - benefit")
            else:
                verdict.append(f"{key}: {d['diff']:+.3f} AP(centre), 95% CI [{d['lo']:.3f}, {d['hi']:.3f}] - "
                               "no significant transferable benefit from the teacher's privileged sensors "
                               "under these assumptions")
    for v in verdict:
        print(v)
    from data.provenance import dataset_identity, atomic_json
    report = _json(dict(split=split, backend=backend, models=reports, comparisons=comparisons, verdict=verdict,
                       provenance=dict(runs=provenance, test_metadata_sha256=dataset_identity(ds.metas),
                                       overlay=cfgs.get("overlay"), eval_config=E,
                                       noise_seed=cfgs["train"]["train"]["seed"])))
    atomic_json(os.path.join(out, "report.json"), report)
    hfovs = cfgs["sensors"]["hfov_deg"]
    if any((m.get("extra") or {}).get("cameras") for m in ds.metas):     # camera rig: each scene's grid HFOV is drawn
        hf = np.array([m["camera"]["hfov_deg"] for m in ds.metas], float)
        hfovs = sorted({round(float(np.percentile(hf, q)), 2) for q in (10, 50, 90)})
    plots.all_plots(reports, out, tuple(E["span_range_m"]), hfovs=hfovs)
    torch.save(records, os.path.join(out, "records.pt"))
    print("wrote", out)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "out", "dataset"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="+", required=True, help="run folders (containing best.pt) or checkpoint files")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "eval"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--backend", choices=("repository", "ultralytics"), default="repository")
    a = ap.parse_args()
    evaluate(a.runs, a.data, a.split, a.out, a.device, a.batch, backend=a.backend)


if __name__ == "__main__":
    main()
