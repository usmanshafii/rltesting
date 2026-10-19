"""PyTorch dataset over the rendered scenes (spec 4, 6, 7).

Per sample and band: load the layers -> composite with haze (nominal visibility, or a fresh draw in
training) -> the rest of the PSF (sigma drawn in the spec range; base sigma0 was applied at render
time) -> registration jitter for LWIR / SWIR against RGB (training) -> sensor noise model -> [0, 1].
Scenes rendered with the camera rig (render/cameras.py, metadata extra.cameras) emulate each band's camera on the
output grid instead (data/camera_emulation.py): its PSF in its own pixels, its pixel size and coverage, and its own
noise draws (NETD, read noise, fixed pattern) from the scene's rig.
Then one spatial augmentation shared by all bands (training): horizontal flip and a random scale
whose lower end is capped so the smallest object never drops below `min_obj_px` (spec 7), with
scale-up allowed. Evaluation is deterministic per scene: the noise generator is seeded from the
scene index, so every model sees identical test images.

Boxes are COCO-style in metadata and returned as xyxy in output pixels.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

from data import camera_emulation as ce
from data import composite as cp
from noise import lwir_fpn, rgb_noise, swir_noise
from render import cameras, psf
from thermal import atmosphere
from thermal import atmosphere_bands as atb
from thermal.planck_lut import band_lut

CLASSES = ["target", "bird", "kite", "light_aircraft", "helicopter", "small_uav", "warm_clutter"]
BAND_CHANNELS = {"rgb": 3, "lwir": 1, "swir": 1, "lwir_cooled": 1, "mwir": 1}
BAND_CAMERA = {"rgb": "rgb", "lwir": "lwir_uncooled", "lwir_cooled": "lwir_cooled", "mwir": "mwir"}  # SWIR: deferred
BAND_SALT = {"rgb": 1, "lwir": 2, "swir": 3, "lwir_cooled": 4, "mwir": 5}


def scene_camera(meta: dict, band: str) -> dict | None:
    """The band's camera from the scene's rig, or None (legacy single camera, or no camera for the band)."""
    rig = (meta.get("extra") or {}).get("cameras")
    return (rig or {}).get("cameras", {}).get(BAND_CAMERA.get(band, ""))


def apply_teacher_zoom(meta: dict, sensors: dict) -> dict:
    """configs/sensors.yaml teacher_zoom (loader-only): the privileged zoom cameras set narrower than the student's
    field (render/cameras.zoom_in), so they keep their pixels on a distant target and see only the centre. The factor
    is drawn log-uniformly per scene and camera from the scene's own index (the same in every run). Returns a copy;
    applying it twice changes nothing. Scenes are rendered on the student's grid, so on the grid a zoomed camera gains
    by binning its finer pixels (temporal noise / z), not by finer detail (docs/privileged_teacher_case.md)."""
    tz = (sensors or {}).get("teacher_zoom") or {}
    rig = (meta.get("extra") or {}).get("cameras")
    if not tz.get("enabled") or not rig or rig.get("teacher_zoom_applied") or "scene_id" not in meta:
        return meta
    lo, hi = (float(v) for v in tz["factor"])
    cams = dict(rig["cameras"])
    for k, name in enumerate(tz.get("cameras", ["lwir_cooled", "mwir"])):
        if name not in cams or not cams[name].get("zoom"):
            continue
        g = np.random.default_rng([int(tz.get("seed", 0)), int(meta["scene_id"][1:]), k])
        z = float(np.exp(g.uniform(np.log(lo), np.log(hi)))) if hi > lo else lo
        cams[name] = cameras.zoom_in(cams[name], z, sensors["cameras"][name]["zoom_focal_mm"], rig["grid"])
    extra = dict(meta["extra"], cameras=dict(rig, cameras=cams, teacher_zoom_applied=True))
    return dict(meta, extra=extra)


def load_metas(root: str, split: str | None = None) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(root, "meta", "*.json"))):
        with open(p) as f:
            m = json.load(f)
        if split is None or m["split"] == split:
            out.append(m)
    return out


CNR_BANDS = ("rgb", "lwir", "lwir_cooled", "mwir", "swir")      # models/distill.BAND_INDEX order


def object_cnr(meta: dict) -> np.ndarray:
    """[N_objects, 5] contrast-to-noise per band, in gt_objects order: the target's row from the writer's nominal
    per-band CNR (data/writer.contrast_to_noise: noise-free preview against the scene's nominal sensor), the hard
    negatives' rows NaN (not measured; their distillation weight stays 1)."""
    objs = meta["extra"]["objects"]
    out = np.full((len(objs), len(CNR_BANDS)), np.nan, np.float32)
    for j, b in enumerate(CNR_BANDS):
        c = (meta.get("modalities") or {}).get(b, {}).get("cnr")
        if c is not None and len(objs):
            out[0, j] = float(c)
    return out


def gt_objects(meta: dict, use_blur_box: bool = False) -> list[dict]:
    """Target first, then the hard negatives, with everything the evaluation bins by."""
    objs = meta["extra"]["objects"]
    out = []
    for o in objs:
        b = o["bbox_blur_px"] if use_blur_box else o["bbox_px"]
        out.append(dict(box=[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], label=CLASSES.index(o["cls"]),
                        size_px=o["size_px"], size_bin=o["size_bin"], range_m=o["range_m"], name=o["name"]))
    return out


def _affine(img: np.ndarray, M: np.ndarray, out_hw) -> np.ndarray:
    """out(p) = img(M^-1 p) with bilinear interpolation. M (2x3) maps input to output in continuous
    pixel coordinates (pixel i spans [i, i+1), the convention of the boxes); SciPy indexes pixel
    centres, so the translation is shifted by A h - h with h = (0.5, 0.5)."""
    from scipy.ndimage import affine_transform
    A = np.vstack([M, [0, 0, 1]]).astype(float)
    h = np.array([0.5, 0.5])
    A[:2, 2] = A[:2, 2] + A[:2, :2] @ h - h
    Ai = np.linalg.inv(A)
    # scipy works in (row, col) = (y, x)
    R = np.array([[Ai[1, 1], Ai[1, 0]], [Ai[0, 1], Ai[0, 0]]])
    off = np.array([Ai[1, 2], Ai[0, 2]])
    chans = [affine_transform(img[..., c], R, off, output_shape=out_hw, order=1, mode="nearest") for c in range(img.shape[-1])]
    return np.stack(chans, -1)


def thermal_camera_noise(lwir_noise: dict, band: str, cam: dict) -> dict:
    """noise/lwir_fpn config of one thermal camera of the rig: the shared model with the scene's own draws (NETD, row and
    column temporal noise, residual fixed pattern) fixed. Cooled FPAs (cooled LWIR, MWIR): their residual non-uniformity
    after NUC is the rig's pixel and column FPN (the RNU bound, render/cameras.residual_fpn_k), so the optional uncooled
    stripe terms (lwir_fpn.REALISM_KEYS, which tools/m1_det2.py adds for the uncooled camera) are not applied."""
    cfg = dict(lwir_noise, netd_mk=[cam["netd_mk"]] * 2)
    if band != "lwir":
        for key in lwir_fpn.REALISM_KEYS:
            cfg.pop(key, None)
    for key, name in (("row_noise_k", "row_temporal_k"), ("column_temporal_k", "column_temporal_k"),
                      ("column_fpn_k", "column_fpn_k"), ("pixel_fpn_k", "pixel_fpn_k")):
        if name in cam:
            cfg[key] = [cam[name]] * 2
    return cfg


class SceneDataset:
    def __init__(self, root: str, split: str | None, bands, cfgs: dict, train: bool = False, seed: int = 0,
                 aug: dict | None = None, use_blur_box: bool = False, metas: list | None = None, cache_layers: int = 0):
        self.root, self.bands, self.cfgs = root, list(bands), cfgs
        self.train, self.seed, self.epoch = train, seed, 0
        self.aug = aug or {}
        self.use_blur_box = use_blur_box
        self.metas = [apply_teacher_zoom(m, cfgs.get("sensors"))
                      for m in (metas if metas is not None else load_metas(root, split))]
        self.W, self.H = cfgs["sensors"]["image"]["width"], cfgs["sensors"]["image"]["height"]
        self.cache_max = int(cache_layers)          # decoded layer sets kept per worker (0 = off; ~5-10 MB each)
        self._cache = {}

    def _layers(self, path: str):
        if self.cache_max <= 0:
            return cp.load_layers(path)
        if path not in self._cache:
            if len(self._cache) >= self.cache_max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = cp.load_layers(path)
        return self._cache[path]

    def __len__(self):
        return len(self.metas)

    def set_epoch(self, e: int):
        self.epoch = e

    # ------------------------------------------------------------------ conditions
    def _weather(self, meta, rng):
        w = meta["extra"]["weather"]
        wc = self.cfgs["render"]["weather"]
        if self.train and (meta["extra"].get("atmosphere") or {}).get("dust_from_visibility"):
            # the scene's sky was rendered with its own haze: jitter visibility around it only (factor 0.7-1.4, C)
            if rng.uniform() < wc.get("resample_in_training", 0.0):
                f = float(math.exp(rng.uniform(math.log(0.7), math.log(1.4))))
                return w["visibility_km"] * f, w["band_ratio"]
            return w["visibility_km"], w["band_ratio"]
        if self.train and rng.uniform() < wc.get("resample_in_training", 0.0):
            keys = list(wc["probs"])
            p = np.array([wc["probs"][k] for k in keys], float)
            kind = keys[int(rng.choice(len(keys), p=p / p.sum()))]
            lo, hi = wc["visibility_km"][kind]
            vis = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            ratio = {b: (float(rng.uniform(*v)) if isinstance(v, list) else float(v)) for b, v in wc["band_ratio"][kind].items()}
            lw = wc["band_ratio"][kind]["lwir"]
            for b, v in ((wc.get("band_ratio_thermal") or {}).get(kind) or {}).items():   # same aerosol as LWIR
                u = 0.5 if not isinstance(lw, list) or lw[1] <= lw[0] else (ratio["lwir"] - lw[0]) / (lw[1] - lw[0])
                ratio[b] = float(v[0] + min(1.0, max(0.0, u)) * (v[1] - v[0]))
            return vis, ratio
        return w["visibility_km"], w["band_ratio"]

    def band_image(self, meta: dict, band: str, rng, vis_km: float, ratio: dict, reg: bool) -> np.ndarray:
        S = self.cfgs["sensors"]
        layer, objects = self._layers(os.path.join(self.root, meta["modalities"][band]["layers"]))
        k = cp.extinction(vis_km, ratio[band])
        gas = atb.gas_table(meta, band)                 # layered thermal atmosphere (new scenes)
        wv = (atmosphere.scene_params(meta, self.cfgs["render"]["weather"].get("lwir_water_vapour"))
              if band == "lwir" and gas is None else None)
        img = cp.composite(layer, objects, k, meta["extra"]["weather"].get("aerosol_scale_height_m"), wv=wv, gas=gas)
        s0 = layer["psf_sigma0"]
        sig = float(rng.uniform(*S["psf_sigma_px"][band])) if self.train else meta["extra"]["psf"][band]["nominal"]
        cam = scene_camera(meta, band)
        if cam is not None:                      # the camera's PSF is in its own pixels
            sig = ce.grid_sigma(sig, cam["pixel_scale"], s0)
        img = psf.gaussian_blur(img, psf.extra_sigma(sig, s0))
        if reg and band in S["registration"]:
            r = S["registration"][band]
            th = math.radians(rng.uniform(-r["rot_deg"], r["rot_deg"]))
            rad, ang = rng.uniform(0, r["shift_px"]), rng.uniform(0, 2 * math.pi)
            cx, cy = self.W / 2, self.H / 2
            c, s = math.cos(th), math.sin(th)
            M = np.array([[c, -s, cx - c * cx + s * cy + rad * math.cos(ang)],
                          [s, c, cy - s * cx - c * cy + rad * math.sin(ang)]])
            img = _affine(img, M, (self.H, self.W))
        exp_ms = meta["camera"]["exposure_ms"]
        if cam is not None:
            src = img if band == "rgb" else img[..., 0]
            out, _ = ce.emulate(src, cam["pixel_scale"], cam["coverage"], self._camera_sensor(band, cam, meta, rng))
            return (out if band == "rgb" else out[..., None]).astype(np.float32)
        if band == "rgb":
            out, _ = rgb_noise.apply(img, rng, S["rgb_noise"], exp_ms)
        elif band == "lwir":
            out, _ = lwir_fpn.apply(img[..., 0], rng, S["lwir_noise"], meta["conditions"]["air_temp_c"] + 273.15)
            out = out[..., None]
        elif band in ("lwir_cooled", "mwir"):          # no rig: the uncooled stripe model on the band's own table
            band_um = (meta["extra"].get("thermal") or {}).get("bands_um", {}).get(band, (7.7, 9.3))
            out, _ = lwir_fpn.apply(img[..., 0], rng, S["lwir_noise"], meta["conditions"]["air_temp_c"] + 273.15,
                                    lut=band_lut(tuple(band_um)))
            out = out[..., None]
        else:
            out, _ = swir_noise.apply(img[..., 0], rng, S["swir_noise"], exp_ms)
            out = out[..., None]
        return out.astype(np.float32)

    def _camera_sensor(self, band: str, cam: dict, meta: dict, rng):
        """The camera's sensor model with the scene's own draws (render/cameras.sample_rig) fixed in its config."""
        S = self.cfgs["sensors"]
        pipes = S.get("camera_pipelines") or {}             # the camera's own processing (noise/isp.py), when switched on
        if not pipes.get("enabled", True):
            pipes = {}
        if band == "rgb":
            cfg = dict(S["rgb_noise"], read_noise_e=[cam["read_noise_e"]] * 2)
            exp_ms = meta["camera"]["exposure_ms"]
            if "rgb" in pipes:
                return lambda x: rgb_noise.apply_camera(x, rng, cfg, pipes["rgb"], exp_ms)[0]
            return lambda x: rgb_noise.apply(x, rng, cfg, exp_ms)[0]
        cfg = thermal_camera_noise(S["lwir_noise"], band, cam)
        lut = band_lut(tuple(cam["band_um"]))
        t_air = meta["conditions"]["air_temp_c"] + 273.15
        name = BAND_CAMERA[band]
        if name in pipes:
            return lambda x: lwir_fpn.apply_camera(x, rng, cfg, pipes[name], t_air, lut=lut)[0]
        return lambda x: lwir_fpn.apply(x, rng, cfg, t_air, lut=lut)[0]

    # ------------------------------------------------------------------ sample
    def _rng(self, i, band_salt=0):
        idx = int(self.metas[i]["scene_id"][1:])
        if self.train:
            return np.random.default_rng([self.seed, self.epoch, idx, band_salt])
        return np.random.default_rng([self.seed + 7919, idx, band_salt])

    def load(self, i: int) -> dict:
        meta = self.metas[i]
        rng = self._rng(i)
        vis, ratio = self._weather(meta, rng)
        # registration jitter models LWIR / SWIR misalignment against RGB, the label reference; a model
        # without RGB (lwir_only) sees its own band in label coordinates, so it gets no jitter
        reg = self.train and "rgb" in self.bands
        # noise stream salted by band NAME, not list position: an LWIR image is identical whether a
        # model loads [lwir] or [rgb, lwir] (identical test scenes; consistent contrast buckets)
        imgs = {b: self.band_image(meta, b, self._rng(i, BAND_SALT[b]), vis, ratio, reg=reg and b != "rgb")
                for b in self.bands}
        objs = gt_objects(meta, self.use_blur_box)
        boxes = np.array([o["box"] for o in objs], np.float32).reshape(-1, 4)
        labels = np.array([o["label"] for o in objs], np.int64)
        cnr = object_cnr(meta)
        if self.train:
            imgs, boxes, labels, keep = self._augment(imgs, boxes, labels, rng)
            cnr = cnr[keep]
        return dict(images=imgs, boxes=boxes, labels=labels, scene_id=meta["scene_id"], meta=meta, objects=objs,
                    visibility_km=vis, cnr=cnr)

    def _augment(self, imgs, boxes, labels, rng):
        """Returns (images, boxes, labels, indices of the kept objects)."""
        a = self.aug
        keep_idx = np.arange(len(boxes))
        if rng.uniform() < a.get("hflip", 0.5):
            imgs = {b: v[:, ::-1].copy() for b, v in imgs.items()}
            if len(boxes):
                boxes = np.c_[self.W - boxes[:, 2], boxes[:, 1], self.W - boxes[:, 0], boxes[:, 3]].astype(np.float32)
        lo, hi = a.get("scale", [1.0, 1.0])
        if len(boxes):
            smallest = float(np.max(boxes[:, 2:] - boxes[:, :2], axis=1).min())
            lo = max(lo, a.get("min_obj_px", 1.5) / max(smallest, 1e-3))      # spec 7: never below ~1 px
        if hi > lo and rng.uniform() < a.get("scale_prob", 0.5):
            s = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            if len(boxes):                  # keep one object (random) in view after scaling
                j = int(rng.integers(len(boxes)))
                px, py = (boxes[j, 0] + boxes[j, 2]) / 2, (boxes[j, 1] + boxes[j, 3]) / 2
            else:
                px, py = self.W / 2, self.H / 2

            def offset(size, p, m=8.0):
                # out = s * in + t. Zoom in: the frame stays covered (t in [size - s size, 0]) and the
                # chosen point stays m px inside. Zoom out: the image stays inside the frame.
                if s >= 1:
                    a0, a1 = max(size - s * size, m - s * p), min(0.0, size - m - s * p)
                else:
                    a0, a1 = 0.0, size - s * size
                return float(rng.uniform(a0, a1)) if a1 > a0 else float(np.clip(size / 2 - s * p, min(a0, a1), max(a0, a1)))
            tx, ty = offset(self.W, px), offset(self.H, py)
            M = np.array([[s, 0, tx], [0, s, ty]])
            imgs = {b: _affine(v, M, (self.H, self.W)) for b, v in imgs.items()}
            if len(boxes):
                boxes = boxes * s + np.array([tx, ty, tx, ty], np.float32)
                cb = np.c_[np.clip(boxes[:, [0, 2]], 0, self.W), np.clip(boxes[:, [1, 3]], 0, self.H)][:, [0, 2, 1, 3]]
                keep = ((cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1])) > 0.5 * ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
                boxes, labels, keep_idx = cb[keep].astype(np.float32), labels[keep], keep_idx[keep]
        return imgs, boxes, labels, keep_idx


# ============================================================================ torch glue
class TorchSceneDataset:
    """Map-style torch dataset around SceneDataset. Module-level so DataLoader workers can pickle it
    (Windows and macOS start workers by spawning)."""

    def __init__(self, ds: SceneDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        import torch
        s = self.ds.load(i)
        imgs = {b: torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1))) for b, v in s["images"].items()}
        cnr = s.get("cnr")
        cnr = torch.from_numpy(np.asarray(cnr, np.float32)) if cnr is not None else torch.full((len(s["boxes"]), 5), float("nan"))
        return imgs, dict(boxes=torch.from_numpy(s["boxes"]), labels=torch.from_numpy(s["labels"]),
                          scene_id=s["scene_id"], index=i, cnr=cnr)


def torch_dataset(ds: SceneDataset) -> TorchSceneDataset:
    return TorchSceneDataset(ds)


def collate(batch):
    import torch
    bands = batch[0][0].keys()
    imgs = {b: torch.stack([x[0][b] for x in batch]) for b in bands}
    return imgs, [x[1] for x in batch]
