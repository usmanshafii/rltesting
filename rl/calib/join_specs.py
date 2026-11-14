"""M2-02: regenerate each census scene's spec with render/sampling.sample_scene (M1 overlay, 600 scenes) to add
aspect and engine setting to docs/reference/privileged/data/census.json, without rendering.

Done when: the regenerated range and target size agree with census.json for all 126 scenes within 0.1 %.
"""
from __future__ import annotations


def join_census_with_specs(census_path: str, overlay: str = "configs/m1.yaml", n_scenes: int = 600) -> list[dict]:
    """Rows of census.json with aspect, engine setting and the regenerated range and size added."""
    raise NotImplementedError("M2-02: join the census with regenerated scene specs")
