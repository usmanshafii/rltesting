"""Episode sampling (task M1-06): one seed -> one complete EpisodeSpec, from the distributions in frozen.yaml.

Conditions (frozen.yaml episode.conditions):
  train          training distribution
  heldout_speed  speeds 50-56 m/s, never seen in training
  heldout_cloud  always cloudy, with twice the cloud cells
  low_heat       the engine at air temperature (the sensor model lowers thermal d')
M4's frozen evaluation sets (rl/evaluation/episodes.py) are lists of specs drawn here from pre-registered seeds.
"""
from __future__ import annotations

import numpy as np

from rl import contracts as C

CONDITIONS = tuple(C.FROZEN["episode"]["conditions"])


def sample_episode_spec(seed: int, condition: str = "train") -> C.EpisodeSpec:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    E = C.FROZEN["episode"]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))   # stream 0 of the episode = the scenario
    speed = E["speed_mps_heldout"] if condition == "heldout_speed" else E["speed_mps_train"]
    cloudy = True if condition == "heldout_cloud" else bool(rng.random() < E["cloudy_share"])
    n_lo, n_hi = E["cloud_cells"]
    if condition == "heldout_cloud":
        n_lo, n_hi = 2 * n_lo, 2 * n_hi
    n_cells = int(rng.integers(n_lo, n_hi + 1)) if cloudy else 0
    flat = rng.choice(C.GRID.n_el * C.GRID.n_az, size=min(n_cells, C.GRID.n_el * C.GRID.n_az), replace=False)
    cloud_cells = tuple(sorted((int(i // C.GRID.n_az), int(i % C.GRID.n_az)) for i in flat))
    margin = 5.0                                        # entry azimuth kept 5 deg inside the sector's edges
    return C.EpisodeSpec(
        seed=int(seed),
        drone_present=bool(rng.random() >= E["drone_free_share"]),
        speed_mps=float(rng.uniform(*speed)),
        entry_range_m=float(rng.uniform(*E["entry_range_m"])),
        entry_az_deg=float(rng.uniform(C.GRID.az_min + margin, C.GRID.az_max - margin)),
        entry_el_deg=float(rng.uniform(*E["entry_elevation_deg"])),
        lateral_offset_m=float(rng.uniform(*E["lateral_offset_m"])),
        weave_deg=float(rng.uniform(*E["weave_deg"])),
        weave_s=float(rng.uniform(*E["weave_s"])),
        span_m=float(rng.uniform(*E["span_m"])),
        is_day=bool(rng.random() < E["day_share"]),
        cloudy=cloudy,
        cloud_cells=cloud_cells,
        engine="low_heat" if condition == "low_heat" else "hot",
        condition=condition,
    )
