"""Portable experiment identities and atomic JSON writes (also usable inside Blender)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def config_identity(cfgs):
    """The effective configs of an experiment (training, evaluation, export): every value, except the execution-only
    thread count and the location of the overlay."""
    effective = {k: cfgs[k] for k in ("target", "render", "sensors", "signatures")}
    effective = json.loads(json.dumps(effective))
    effective["render"]["render"].pop("threads", None)
    return digest(effective)


# Sensor settings the data loader applies after rendering (noise models, camera processing, registration jitter): a
# change to them must not invalidate rendered scenes, though it changes any experiment trained on them (config_identity).
# Their `nominal` entries stay, because the writer stores the nominal sensor's contrast-to-noise in each scene's metadata.
LOADER_ONLY_SENSORS = ("rgb_noise", "lwir_noise", "swir_noise", "registration", "camera_pipelines",
                       "teacher_zoom")


def render_identity(cfgs):
    """The configs that decide what is rendered: config_identity without the loader-only sensor settings (except their
    nominal values)."""
    effective = {k: cfgs[k] for k in ("target", "render", "sensors", "signatures")}
    effective = json.loads(json.dumps(effective))
    effective["render"]["render"].pop("threads", None)
    for key in LOADER_ONLY_SENSORS:
        section = effective["sensors"].pop(key, None)
        if isinstance(section, dict) and "nominal" in section:
            effective["sensors"][key] = {"nominal": section["nominal"]}
    return digest(effective)


def render_contract(cfgs, n_scenes, bands):
    return dict(schema=1, config_sha256=render_identity(cfgs), n_scenes=int(n_scenes),
                bands=sorted(bands), engine_samples=cfgs["render"]["render"]["engine_samples"],
                signature_model=cfgs["signatures"]["model_revision"],
                overlay=cfgs.get("overlay"))


def check_scene(meta, root, contract):
    """Reject incompatible data. Missing layers are recoverable by rerendering the scene."""
    previous = meta.get("extra", {}).get("render_contract")
    keys = ("config_sha256", "n_scenes", "bands")
    if previous is None or any(previous.get(k) != contract[k] for k in keys):
        raise ValueError(f"{meta['scene_id']}: render configuration, bands or split population differs "
                         "(or provenance is missing). Use a fresh directory or explicitly regenerate.")
    if set(meta["modalities"]) != set(contract["bands"]):
        raise ValueError(f"{meta['scene_id']}: metadata bands differ from the render contract")
    return all((Path(root) / m["layers"]).is_file() for m in meta["modalities"].values())


def dataset_identity(metas):
    """Metadata digest binds training to scene content, split, signatures and layer paths."""
    return digest(sorted(metas, key=lambda m: m["scene_id"]))
