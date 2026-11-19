"""M2-02: deterministic, fail-closed join to the original non-test census."""
from __future__ import annotations

import json
from pathlib import Path

from render.sampling import load_configs, sample_scene
from rl import contracts as C

ROOT = C.RL_ROOT.parent


def join_census_with_specs(census_path: str, overlay: str = "configs/m1.yaml", n_scenes: int = 600) -> list[dict]:
    """Regenerate specs, proving identity by range, unblurred box size and split.

    The census uses gt_objects' unblurred xyxy box, not the motion-blurred box.
    Config loading is explicit so KAL_CONFIG_OVERLAY cannot contaminate this join.
    No pixels are rendered, and no sealed test scene is sampled.
    """
    census = json.loads(Path(census_path).read_text(encoding="utf-8"))
    cfg = load_configs(root=str(ROOT), overlay=overlay)
    rows = []
    for sid, original in sorted(census.items()):
        if original["split"] not in ("train", "val"):
            raise ValueError(f"census join refuses sealed split: {sid}")
        if not sid.startswith("s") or not sid[1:].isdigit():
            raise ValueError(f"invalid scene ID: {sid}")
        spec, _ = sample_scene(int(sid[1:]), cfg, n_scenes=n_scenes)
        obj = spec["objects"][0]
        size = max(obj["bbox_px"]["w"], obj["bbox_px"]["h"])
        errors = {"range_relative": abs(obj["range_m"] / original["range_m"] - 1),
                  "size_relative": abs(size / original["size_px"] - 1)}
        if max(errors.values()) >= .001 or spec["split"] != original["split"]:
            raise ValueError(f"census join mismatch {sid}: {errors}, split {spec['split']}")
        # Identity includes conditions that affect radiance, not just geometry.
        if (spec["lighting"] != original["lighting"] or
                abs(spec["weather"]["visibility_km"] - original["visibility_km"]) > 1e-6 or
                bool(spec["clouds"]) != bool(original["clouds"])):
            raise ValueError(f"census join condition mismatch: {sid}")
        lw = spec["signatures"]["target"]["lwir"]
        rows.append(dict(original, scene_id=sid, aspect_deg=obj["aspect_deg"],
                         aspect_elevation_deg=obj["aspect_elevation_deg"],
                         engine_setting=lw["engine_head"]["thermal_profile"],
                         regenerated_range_m=obj["range_m"], regenerated_size_px=size,
                         join_error=errors))
    return rows
