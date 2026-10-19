"""M1 results note, explicit uncertainty, and reproducible qualitative/failure galleries."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from data.dataset import SceneDataset
from eval import metrics as M

MODELS = ("rgb_only", "lwir_only")


def interval(value, ci):
    if value is None or not math.isfinite(value):
        return "not estimable"
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi):
        return f"{100 * value:.1f}% [interval unavailable]"
    return f"{100 * value:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def gallery(data, output, cfg, records, ids, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ds = SceneDataset(str(data), "test", ["rgb", "lwir"], cfg, train=False,
                      seed=cfg["train"]["train"]["seed"])
    indexes = {m["scene_id"]: i for i, m in enumerate(ds.metas)}
    by_model = {k: {r["scene_id"]: r for r in v} for k, v in records.items()}
    if not ids:
        return
    fig, axes = plt.subplots(len(ids), 2, figsize=(10, 3 * len(ids)), squeeze=False)
    for row, sid in enumerate(ids):
        s = ds.load(indexes[sid])
        box = s["boxes"][0]
        cx, cy = (box[:2] + box[2:]) / 2
        half = max(16, 1.5 * max(box[2:] - box[:2]))
        for col, (band, model) in enumerate(zip(("rgb", "lwir"), MODELS)):
            ax = axes[row, col]
            arr = s["images"][band]
            ax.imshow(arr if band == "rgb" else arr[..., 0], cmap=None if band == "rgb" else "gray",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=False, ec="lime", lw=1))
            pred = by_model[model][sid]["preds"]
            for b, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                if label != 0 or score < cfg["train"]["eval"]["score_thresh"]:
                    continue
                ax.add_patch(Rectangle(b[:2], *(b[2:] - b[:2]), fill=False, ec="tomato", lw=1))
                ax.text(b[0], b[1], f"{score:.2f}", color="tomato", fontsize=7)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy + half, cy - half)
            meta = s["meta"]
            ax.set_title(f"{band} | {sid} | {meta['conditions']['lighting']} | "
                         f"{meta['target']['size_bin']} px | aspect {M.target_aspect_deg(meta):.0f} deg", fontsize=8)
            ax.axis("off")
    fig.suptitle(title + "\nTarget crops; green: ground truth, red: predictions at score >= 0.3", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=130)
    plt.close(fig)


def report(data, out, cfg, smoke=False):
    import torch
    data, out = Path(data), Path(out)
    rep = json.loads((out / "report.json").read_text())
    records = torch.load(out / "records.pt", map_location="cpu", weights_only=False)
    lines = ["# M1-lite results" + (" — SMOKE ONLY" if smoke else ""), "",
             ("Plumbing check on eight scenes and two training iterations per model. These are not convergence "
              "or M1 performance results." if smoke else
              "Synthetic MD-550 / L550E engine target. Evaluation uses the held-out test split."), "",
             "AP50 intervals use a scene percentile bootstrap. Recall intervals use Wilson target counts "
             "(one target per scene). Small slices and all-success/all-failure bootstrap intervals are unstable; "
             "a degenerate AP interval is not evidence of certainty. Recall uses centre matching for the "
             "2–4 and 4–8 px rows, and IoU 0.5 elsewhere. Values are percentages with 95% intervals.", "",
             "| Slice | n targets | Recall match | RGB AP50 | RGB recall | LWIR AP50 | LWIR recall |",
             "| --- | ---: | --- | --- | --- | --- | --- |"]
    if rep.get("backend") == "ultralytics":
        lines[2] += (" Ultralytics fallback: stride-8 architecture, 8-bit images, fixed per-scene training noise, "
                     "Ultralytics augmentation/letterboxing and IoU NMS. This is a different baseline.")
    slices = [("all", "overall", None), ("day", "conditions", "day"), ("night", "conditions", "night")]
    slices += [(b + " px", "bins", b) for b in cfg["render"]["size_bins"]["names"]]
    slices += [(c, "conditions", c) for c in ("aspect_head", "aspect_tail", "day_head", "day_tail", "night_head", "night_tail")]
    for title, section, key in slices:
        blocks = [rep["models"][m][section] if key is None else rep["models"][m][section].get(key) for m in MODELS]
        op = "op_centre" if key in ("2-4", "4-8") else "op_iou50"
        n = blocks[0][op]["n_gt"] if blocks[0] else 0
        cells = []
        for b in blocks:
            cells.extend([interval(b["ap50"], b["ap50_ci95"]), interval(b[op]["recall"], b[op]["recall_ci95"])]
                         if b else ["not estimable"] * 2)
        lines.append(f"| {title} | {n} | {'centre' if op == 'op_centre' else 'IoU 0.5'} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired model comparison", "",
              "LWIR minus RGB on identical test scenes; paired bootstrap intervals describe the difference. "
              "The plan's conservative headline rule also requires non-overlapping individual intervals. "
              "One seed does not measure training-run uncertainty.", ""]
    for metric, row in rep["comparisons"].get("lwir_only_vs_rgb_only", {}).items():
        vals = [row[k] for k in ("diff", "lo", "hi")]
        if any(v is None for v in vals):
            lines.append(f"- {metric}: not estimable.")
        else:
            lines.append(f"- {metric}: {vals[0] * 100:+.1f} points [{vals[1] * 100:+.1f}, {vals[2] * 100:+.1f}].")
    lines += ["", "## Size threshold and sensitivity", "",
              "[px50.csv](px50.csv) includes status and aspect counts. Missing crossings, absent aspect "
              "groups and extrapolations are labelled; do not interpret them as measured thresholds. "
              "[Detection versus size](pd_vs_size_centre.png) pools test targets. "
              "[Range conversion](pd_vs_range.png) is geometric only.", ""]
    sensitivity = out / "sensitivity/sensitivity.csv"
    with sensitivity.open(newline="") as f:
        rows = list(csv.DictReader(f))
    swings = {(r["model"], r["size_group"], r["input"]): (float(r["swing"]), r["ci_separated"])
              for r in rows if r["model"] == "lwir_only"}
    if swings:
        lines += ["Largest exploratory LWIR associations within size groups (not causal effects):", ""]
        for (_, size, inp), (swing, separated) in sorted(swings.items(), key=lambda kv: -kv[1][0])[:3]:
            lines.append(f"- {inp}, {size} px: {swing * 100:.1f}-point recall swing; "
                         f"quantile intervals {'separate' if separated == 'True' else 'overlap'}.")
    else:
        lines.append("Too few test targets for sensitivity quantiles; CSV files contain headers only.")
    hit = {m: {d["scene"]: d["detected"] for d in M.detections_per_target(records[m], cfg["train"]["eval"], "centre", 0)}
           for m in MODELS}
    ids = list(hit[MODELS[0]])
    failures = [sid for sid in ids if not all(hit[m][sid] for m in MODELS)]
    failures.sort(key=lambda sid: (hit[MODELS[0]][sid] == hit[MODELS[1]][sid], sid))
    gallery(data, out / "failure_gallery.png", cfg, records, failures[:8], "Exploratory failure/disagreement gallery")
    chosen = []
    missing = []
    for lighting in ("day", "night"):
        for aspect in ("head", "tail"):
            candidates = [r for r in records[MODELS[0]] if M.CONDITIONS[f"{lighting}_{aspect}"](r)]
            if candidates:
                r = max(candidates, key=lambda r: r["gts"][0]["size_px"])
                chosen.append(r["scene_id"])
            else:
                missing.append(f"{lighting} × {aspect}")
    gallery(data, out / "day_night_aspects.png", cfg, records, chosen, "Largest available test target per lighting/aspect slice")
    lines += ["", "## Qualitative checks", "",
              "Gallery selections are for inspection, not additional independent evidence.", ""]
    if failures:
        lines.append("[Failure gallery](failure_gallery.png): centre-match misses, disagreements first, then scene ID.")
    else:
        lines.append("No centre-match misses at the operating point in this evaluated set.")
    if chosen:
        lines.append("[Day/night and head/tail examples](day_night_aspects.png): largest target in each available slice.")
    if missing:
        lines.append("Unavailable slices: " + ", ".join(missing) + ".")
    lines += ["", "## Limits", "",
              "Synthetic data only; open sky only; clear air with water-vapour absorption, no haze or fog; "
              "one camera and one target per frame. Engine and exhaust surface temperatures and spectral "
              "properties are modelling assumptions, not measurements. Aspect slices "
              "are exploratory. No real-aircraft detection range, operational false-alarm rate, SWIR, fusion "
              "or distillation claim follows from M1.", "",
              "## Reproduction", "",
              "The machine-readable report records checkpoint hashes, evaluation settings, noise seed and "
              "test metadata hash. Use the bundled source and saved weights; run `python tools/m1.py evaluate "
              "--device cpu` from the project root (or the smoke command for this smoke report).", ""]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "results.md"
