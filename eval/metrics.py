"""Detection metrics (spec 8): precision, recall, AP50, AP50:95, centre-distance and NWD matching
for the tiny bins, false positives per image broken down by hard-negative class, all overall, per
pixel bin and per condition, plus detection probability vs pixel size with bootstrap CIs.

Per-bin protocol (COCO area ranges, with the bin in place of the area range): ground truths
outside the bin are ignored; a detection matched to an ignored ground truth is ignored; an
unmatched detection counts as a false positive only if its own size falls in the bin.
AP is COCO's 101-point interpolated average precision.
"""
from __future__ import annotations

import numpy as np

from eval.bins import NAMES, bin_of
from eval.nwd import centre_distance_matrix, centre_ok_matrix, iou_matrix, nwd_matrix

TARGET = 0


# ============================================================================ matching
def _ok_quality(P, G, kind, thr, cfg):
    if len(P) == 0 or len(G) == 0:
        z = np.zeros((len(P), len(G)))
        return z.astype(bool), z
    if kind == "iou":
        q = iou_matrix(P, G)
        return q >= thr, q
    if kind == "centre":
        c = cfg["centre_match"]
        return centre_ok_matrix(P, G, c["abs_px"], c["rel"]), -centre_distance_matrix(P, G)
    if kind == "nwd":
        q = nwd_matrix(P, G, cfg["nwd_match"]["C"])
        return q >= cfg["nwd_match"]["thresh"], q
    raise ValueError(kind)


def match_image(P, S, G, active, kind, thr, cfg):
    """Greedy by score; active ground truths are preferred, ignored ones only absorb detections."""
    n = len(P)
    tp, ign = np.zeros(n, bool), np.zeros(n, bool)
    gt_of = np.full(n, -1)
    if n == 0 or len(G) == 0:
        return tp, ign, gt_of
    ok, q = _ok_quality(P, G, kind, thr, cfg)
    used = np.zeros(len(G), bool)
    for i in np.argsort(-S, kind="stable"):
        for want_active in (True, False):
            cand = ok[i] & ~used & (active == want_active)
            if cand.any():
                j = int(np.argmax(np.where(cand, q[i], -np.inf)))
                used[j] = True
                gt_of[i] = j
                tp[i], ign[i] = want_active, not want_active
                break
    return tp, ign, gt_of


def coco_ap(scores, tp, n_gt) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(scores), kind="stable")
    t = np.asarray(tp, float)[o]
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def _preds(r, cls):
    p = r["preds"]
    m = np.asarray(p["labels"]) == cls
    return np.asarray(p["boxes"], float).reshape(-1, 4)[m], np.asarray(p["scores"], float)[m]


def _gts(r, cls, bin_name=None):
    g = [o for o in r["gts"] if o["label"] == cls]
    G = np.array([o["box"] for o in g], float).reshape(-1, 4)
    active = np.array([bin_name is None or o["size_bin"] == bin_name for o in g], bool)
    return g, G, active


def ap(records, kind="iou", thr=0.5, bin_name=None, cls=TARGET, cfg=None) -> float:
    scores, tps, n_gt = [], [], 0
    for r in records:
        P, S = _preds(r, cls)
        _, G, active = _gts(r, cls, bin_name)
        n_gt += int(active.sum())
        tp, ign, _ = match_image(P, S, G, active, kind, thr, cfg)
        if bin_name is not None and len(P):
            size = np.maximum(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
            outside = np.array([bin_of(s) != bin_name for s in size])
            ign |= ~tp & ~ign & outside
        scores += list(S[~ign])
        tps += list(tp[~ign])
    return coco_ap(scores, tps, n_gt)


def ap_50_95(records, bin_name=None, cls=TARGET, cfg=None) -> float:
    vals = [ap(records, "iou", t, bin_name, cls, cfg) for t in cfg["iou_thresholds"]]
    return float(np.nanmean(vals)) if not all(np.isnan(vals)) else float("nan")


# ============================================================================ operating point
def operating_point(records, cfg, kind="iou", thr=0.5, cls=TARGET, bin_name=None) -> dict:
    """Precision, recall (= Pd) and false positives per image at score >= score_thresh; false
    target detections are charged to the hard-negative class they land on (IoU >= negative_match_iou
    or centre inside the negative's box), otherwise to 'background'."""
    st = cfg["score_thresh"]
    tp_n = fp_n = n_gt = 0
    fp_by = {}
    for r in records:
        P, S = _preds(r, cls)
        keep = S >= st
        P, S = P[keep], S[keep]
        g, G, active = _gts(r, cls, bin_name)
        n_gt += int(active.sum())
        tp, ign, _ = match_image(P, S, G, active, kind, thr, cfg)
        if bin_name is not None and len(P):
            size = np.maximum(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
            ign |= ~tp & ~ign & np.array([bin_of(s) != bin_name for s in size])
        tp_n += int(tp.sum())
        fp = ~tp & ~ign
        fp_n += int(fp.sum())
        negs = [o for o in r["gts"] if o["label"] != cls]
        if fp.any():
            NB = np.array([o["box"] for o in negs], float).reshape(-1, 4)
            iou = iou_matrix(P[fp], NB) if len(negs) else np.zeros((int(fp.sum()), 0))
            ctr = (P[fp, :2] + P[fp, 2:]) / 2
            for i in range(int(fp.sum())):
                hit = None
                for j, o in enumerate(negs):
                    b = o["box"]
                    inside = b[0] - 1 <= ctr[i, 0] <= b[2] + 1 and b[1] - 1 <= ctr[i, 1] <= b[3] + 1
                    if iou[i, j] >= cfg["negative_match_iou"] or inside:
                        hit = o["cls_name"]
                        break
                fp_by[hit or "background"] = fp_by.get(hit or "background", 0) + 1
    n_img = max(1, len(records))
    from eval.sensitivity import wilson
    lo, hi = wilson(tp_n, n_gt)
    return dict(precision=tp_n / max(1, tp_n + fp_n), recall=tp_n / n_gt if n_gt else float("nan"),
                recall_ci95=dict(lo=lo, hi=hi, method="Wilson (target counts)"), n_gt=n_gt, tp=tp_n, fp=fp_n,
                fppi=fp_n / n_img, fppi_by_class={k: v / n_img for k, v in sorted(fp_by.items())})


# ============================================================================ Pd curves
def detections_per_target(records, cfg, kind="centre", thr=0.5):
    """For every target ground truth: (size_px, range_m, hfov, detected at the operating point)."""
    out = []
    st = cfg["score_thresh"]
    for r in records:
        P, S = _preds(r, TARGET)
        keep = S >= st
        g, G, active = _gts(r, TARGET)
        tp, _, gt_of = match_image(P[keep], S[keep], G, np.ones(len(g), bool), kind, thr, cfg)
        hit = set(gt_of[tp].tolist())
        for j, o in enumerate(g):
            out.append(dict(scene=r["scene_id"], size_px=o["size_px"], range_m=o["range_m"], hfov=r["camera"]["hfov_deg"],
                            detected=j in hit))
    return out


def pd_curve(dets, edges, n_boot=1000, seed=0) -> dict:
    """Detection probability per size bin with scene-level bootstrap 95% CIs."""
    size = np.array([d["size_px"] for d in dets])
    det = np.array([d["detected"] for d in dets], float)
    scenes = np.array([d["scene"] for d in dets])
    uniq = np.unique(scenes)
    idx_of = {s: np.nonzero(scenes == s)[0] for s in uniq}
    b = np.digitize(size, edges) - 1
    nb = len(edges) - 1
    if not len(dets):
        return dict(edges=list(map(float, edges)), pd=[float("nan")] * nb,
                    lo=[float("nan")] * nb, hi=[float("nan")] * nb, n=[0] * nb)
    if n_boot <= 0:
        raise ValueError("bootstrap count must be positive")

    def per_bin(ix):
        v = np.full(nb, np.nan)
        for k in range(nb):
            m = b[ix] == k
            if m.any():
                v[k] = det[ix][m].mean()
        return v
    base = per_bin(np.arange(len(det)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        boots.append(per_bin(np.concatenate([idx_of[s] for s in pick])))
    boots = np.array(boots)
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)          # empty size bins stay NaN
        lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    counts = np.array([(b == k).sum() for k in range(nb)])
    return dict(edges=list(map(float, edges)), pd=base.tolist(), lo=lo.tolist(), hi=hi.tolist(), n=counts.tolist())


def bootstrap_diff(rec_a, rec_b, fn, n_boot=1000, seed=0) -> dict:
    """CI of fn(rec_a) - fn(rec_b) on the same scenes (paired bootstrap over scenes)."""
    ids = [r["scene_id"] for r in rec_a]
    by_b = {r["scene_id"]: r for r in rec_b}
    pairs = [(r, by_b[s]) for r, s in zip(rec_a, ids) if s in by_b]
    if len(by_b) != len(rec_b) or len(set(ids)) != len(ids) or set(ids) != set(by_b):
        raise ValueError("paired comparison requires identical, unique scene IDs")
    if not pairs:
        return dict(diff=float("nan"), lo=float("nan"), hi=float("nan"), significant=False, n_scenes=0)
    rng = np.random.default_rng(seed)
    base = fn([p[0] for p in pairs]) - fn([p[1] for p in pairs])
    diffs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(pairs), len(pairs))
        a, b = [pairs[i][0] for i in k], [pairs[i][1] for i in k]
        diffs.append(fn(a) - fn(b))
    diffs = np.array(diffs, float)
    finite = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (float("nan"), float("nan"))
    return dict(diff=float(base), lo=float(lo), hi=float(hi), significant=bool(lo > 0 or hi < 0), n_scenes=len(pairs))


def ap_interval(records, cfg, kind="iou", bin_name=None, seed=0):
    """Scene bootstrap, caching matching so each resample only recomputes the AP ranking."""
    chunks = []
    for r in records:
        P, S = _preds(r, TARGET)
        _, G, active = _gts(r, TARGET, bin_name)
        tp, ign, _ = match_image(P, S, G, active, kind, 0.5 if kind == "iou" else 0.0, cfg)
        if bin_name is not None and len(P):
            outside = np.array([bin_of(s) != bin_name for s in np.max(P[:, 2:] - P[:, :2], axis=1)])
            ign |= ~tp & ~ign & outside
        chunks.append((S[~ign], tp[~ign], int(active.sum())))
    vals = []
    rng = np.random.default_rng(seed)
    n_boot = cfg.get("bootstrap", 1000)
    if n_boot <= 0:
        raise ValueError("bootstrap count must be positive")
    if chunks and sum(c[2] for c in chunks):
        for _ in range(n_boot):
            chosen = [chunks[i] for i in rng.integers(len(chunks), size=len(chunks))]
            vals.append(coco_ap(np.concatenate([c[0] for c in chosen]),
                                np.concatenate([c[1] for c in chosen]), sum(c[2] for c in chosen)))
    finite = np.asarray(vals)[np.isfinite(vals)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (float("nan"), float("nan"))
    return dict(lo=float(lo), hi=float(hi), method="scene percentile bootstrap", resamples=n_boot,
                finite_resamples=len(finite))


# ============================================================================ full report
CONDITIONS = {
    "day": lambda r: r["conditions"]["lighting"] == "day",
    "dusk": lambda r: r["conditions"]["lighting"] == "dusk",
    "night": lambda r: r["conditions"]["lighting"] == "night",
    "clear": lambda r: r["conditions"]["weather"] == "clear",
    "haze": lambda r: r["conditions"]["weather"] == "haze",
    "fog": lambda r: r["conditions"]["weather"] == "fog",
    "thermal_crossover": lambda r: r["conditions"]["thermal_state"] == "crossover",
    "warm_clutter": lambda r: r["conditions"].get("warm_clutter", False),
    "open_sky": lambda r: r["conditions"]["background"] == "open_sky",
    "horizon_clutter": lambda r: r["conditions"]["background"] == "horizon_clutter",
    "hfov_wide": lambda r: r["camera"]["hfov_deg"] >= 20,
    "hfov_narrow": lambda r: r["camera"]["hfov_deg"] < 20,
    # spec 8 aspect breakdown; aspect 0 = nose towards the camera (head-on), 180 = tail-on
    "aspect_head": lambda r: 0 <= r.get("target_aspect_deg", -1) <= 30,
    "aspect_beam": lambda r: 30 < r.get("target_aspect_deg", -1) < 150,
    "aspect_tail": lambda r: r.get("target_aspect_deg", -1) >= 150,
}
for _l in ("day", "night"):                       # lighting x aspect, the M1 question
    for _a in ("head", "tail"):
        CONDITIONS[f"{_l}_{_a}"] = (lambda r, l=_l, a=_a: CONDITIONS[l](r) and CONDITIONS[f"aspect_{a}"](r))


def target_aspect_deg(meta: dict) -> float:
    """Aspect of the labelled target (render/placement.py Placement.aspect), -1 if unknown."""
    for o in meta.get("extra", {}).get("objects", []):
        if o.get("cls") == "target" and "aspect_deg" in o:
            return float(o["aspect_deg"])
    return -1.0


def summary(records, cfg) -> dict:
    def block(rs, bin_name=None):
        d = dict(n_images=len(rs), ap50=ap(rs, "iou", 0.5, bin_name, cfg=cfg), ap50_95=ap_50_95(rs, bin_name, cfg=cfg),
                 ap50_ci95=ap_interval(rs, cfg, "iou", bin_name),
                 op_iou50=operating_point(rs, cfg, "iou", 0.5, bin_name=bin_name))
        if bin_name is None or bin_name in cfg["tiny_bins"]:
            d.update(ap_centre=ap(rs, "centre", 0.0, bin_name, cfg=cfg), ap_nwd=ap(rs, "nwd", 0.0, bin_name, cfg=cfg),
                     ap_centre_ci95=ap_interval(rs, cfg, "centre", bin_name),
                     op_centre=operating_point(rs, cfg, "centre", 0.0, bin_name=bin_name))
        return d
    out = dict(overall=block(records), bins={b: block(records, b) for b in NAMES}, conditions={})
    for name, f in CONDITIONS.items():
        rs = [r for r in records if f(r)]
        if rs:
            out["conditions"][name] = block(rs)
    edges = cfg["pd_size_edges_px"]
    for kind in ("centre", "iou"):
        out[f"pd_vs_size_{kind}"] = pd_curve(detections_per_target(records, cfg, kind, 0.5 if kind == "iou" else 0.0),
                                             edges, cfg.get("bootstrap", 1000))
    return out


# ============================================================================ running a model
def predict_records(model, ds, bands, cfg, device, batch: int = 4, drop_bands=()):
    """Run a detector over a SceneDataset. Bands in drop_bands are passed as missing (zeroed at the
    fusion), e.g. to test the student on LWIR alone."""
    import torch
    from data.dataset import CLASSES
    recs = []
    model.eval()
    for i0 in range(0, len(ds), batch):
        samples = [ds.load(i) for i in range(i0, min(len(ds), i0 + batch))]
        imgs = {b: (None if b in drop_bands else torch.from_numpy(np.stack([s["images"][b].transpose(2, 0, 1) for s in samples])).to(device))
                for b in bands}
        with torch.no_grad():
            preds = model.predict(imgs, cfg["conf"], cfg["nms_iou"], cfg["max_det"], cfg["nms_centre_px"])
        for s, p in zip(samples, preds):
            m = s["meta"]
            recs.append(dict(scene_id=s["scene_id"], preds={k: v.cpu().numpy() for k, v in p.items()},
                             gts=[dict(o, cls_name=CLASSES[o["label"]]) for o in s["objects"]],
                             conditions=m["conditions"], camera=m["camera"], visibility_km=s["visibility_km"],
                             target_aspect_deg=target_aspect_deg(m)))
    return recs


def quick_ap50(model, ds, bands, cfg, device) -> float:
    return ap(predict_records(model, ds, bands, cfg, device), "iou", 0.5, cfg=cfg)
