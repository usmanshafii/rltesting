"""Ultralytics fallback (M1): write each band of a rendered dataset as 8-bit PNGs + YOLO labels.

    python tools/export_yolo.py --data out/m1/dataset --out out/m1/yolo --bands rgb,lwir
    yolo detect train data=out/m1/yolo/lwir/lwir.yaml model=yolo11n.pt imgsz=640 epochs=60 seed=0

Only needed if the repository detector (train/train_baseline.py) cannot be trained in time.
Images come from the repository loader (data/dataset.py: compositing, haze, sensor noise, gain
control) with train=False, so every scene's noise is fixed by its seed and identical to what
eval/evaluate.py sees. Training images are therefore exported once: this path loses the fresh noise
per epoch and the loader's scale augmentation. Ultralytics' finest stride is 8 px, so the 2-8 px bins
are expected to be weak; report that rather than tuning around it.

Per band: <out>/<band>/images/<split>/<scene>.png, labels/<split>/<scene>.txt (class cx cy w h,
normalised), <band>.yaml with absolute paths; plus <out>/index.csv (scene, split, lighting, aspect,
size) for slicing the Ultralytics predictions afterwards.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import json
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from data.dataset import CLASSES, SceneDataset, load_metas  # noqa: E402
from data.provenance import atomic_json, config_identity, dataset_identity  # noqa: E402
from eval.metrics import target_aspect_deg  # noqa: E402
from render.sampling import load_configs  # noqa: E402


def yolo_lines(boxes, labels, W: int, H: int) -> list[str]:
    out = []
    for (x0, y0, x1, y1), c in zip(boxes, labels):
        x0, x1 = np.clip([x0, x1], 0, W)
        y0, y1 = np.clip([y0, y1], 0, H)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        out.append(f"{int(c)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out


def to_png(img: np.ndarray) -> Image.Image:
    a = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(a[..., 0], "L") if a.shape[-1] == 1 else Image.fromarray(a, "RGB")


def export(data: str, out: str, bands, splits=("train", "val", "test"), seed: int = 0, cfgs=None) -> dict:
    cfgs = cfgs or load_configs()
    if not bands or set(bands) - {"rgb", "lwir", "lwir_cooled", "mwir", "swir"} or len(set(bands)) != len(bands):
        raise ValueError("invalid or duplicate bands")
    metas = [m for m in load_metas(data) if m["split"] in splits]
    if not metas:
        raise ValueError("no scenes to export")
    contract = dict(dataset_sha256=dataset_identity(metas), config_sha256=config_identity(cfgs),
                    bands=list(bands), splits=list(splits), noise_seed=seed)
    manifest = Path(out) / "export_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["contract"] != contract:
            raise ValueError("existing YOLO export belongs to different data/settings; use a fresh directory")
    elif Path(out).exists() and any(Path(out).iterdir()):
        raise ValueError("existing YOLO directory has no export provenance; use a fresh directory")
    atomic_json(manifest, dict(status="in_progress", contract=contract))
    W, H = cfgs["sensors"]["image"]["width"], cfgs["sensors"]["image"]["height"]
    counts, index = {}, []
    for split in splits:
        ds = SceneDataset(data, split, bands, cfgs, train=False, seed=seed)
        for i in range(len(ds)):
            s = ds.load(i)
            m, sid = s["meta"], s["scene_id"]
            for b in bands:
                img_dir = os.path.join(out, b, "images", split)
                lab_dir = os.path.join(out, b, "labels", split)
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(lab_dir, exist_ok=True)
                to_png(s["images"][b]).save(os.path.join(img_dir, f"{sid}.png"))
                with open(os.path.join(lab_dir, f"{sid}.txt"), "w") as f:
                    f.write("\n".join(yolo_lines(s["boxes"], s["labels"], W, H)) + "\n")
            t = m["target"]
            index.append(dict(scene=sid, split=split, lighting=m["conditions"]["lighting"],
                              aspect_deg=round(target_aspect_deg(m), 2), size_px=round(max(t["bbox_px"]["w"], t["bbox_px"]["h"]), 3),
                              size_bin=t["size_bin"], range_m=round(t["range_m"], 1)))
            counts[split] = counts.get(split, 0) + 1
    for b in bands:
        base = os.path.abspath(os.path.join(out, b)).replace(os.sep, "/")
        with open(os.path.join(out, b, f"{b}.yaml"), "w") as f:
            f.write(f"path: {json.dumps(base)}\n")
            for split in splits:
                f.write(f"{split}: images/{split}\n")
            f.write("names:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(CLASSES)))
    if index:
        with open(os.path.join(out, "index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0]))
            w.writeheader()
            w.writerows(index)
    atomic_json(manifest, dict(status="complete", contract=contract, counts=counts))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands", default="rgb,lwir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(export(a.data, a.out, [b for b in a.bands.split(",") if b], seed=a.seed))


if __name__ == "__main__":
    main()
