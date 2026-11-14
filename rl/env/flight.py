"""Drone flight (task M1-04): the level S-turn approach of render/track_sequence.flight_path, ported.

render/track_sequence.py imports Blender (bpy) at module level, so it cannot be imported here; `flight_path` below is a
line-for-line port (same arguments, same integration), and tests/rl/test_env.py checks it against the original source
on a fixed input. `Track` samples it at the environment's step times and turns positions into the sector's azimuth,
elevation and range, seen from the post at the origin.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from rl import contracts as C
from rl.env import geometry as geo

G = 9.81


def direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """render/placement.direction: x east, y north, z up."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])


def flight_path(C0, az_c, a):
    """Port of render/track_sequence.flight_path. Level S-turn approach. Returns per-frame
    (t, position, heading, bank, velocity)."""
    n = int(round(a.duration * a.fps))
    u = direction(az_c, 0.0)                            # horizontal look direction of the cue
    w = np.array([u[1], -u[0], 0.0])                    # to its right
    p = C0 + a.start_m * u + a.offset_m * w + np.array([0, 0, a.alt_m])
    head0 = (az_c + 180.0) % 360.0                      # flying back towards the site
    amp, period = a.weave_deg, a.weave_s
    dt = 1.0 / a.fps
    out = []
    sub = 8
    for k in range(n):
        t = k * dt
        heading = head0 + amp * math.sin(2 * math.pi * t / period)
        rate = math.radians(amp) * 2 * math.pi / period * math.cos(2 * math.pi * t / period)
        bank = math.degrees(math.atan(a.speed * rate / G))
        v = a.speed * direction(heading, 0.0)
        out.append(dict(t=t, position=p.copy(), heading=heading, bank=bank, velocity=v))
        for s in range(sub):                            # integrate the heading change within the frame
            ts = t + (s + 0.5) * dt / sub
            hs = head0 + amp * math.sin(2 * math.pi * ts / period)
            p = p + a.speed * direction(hs, 0.0) * dt / sub
    return out


def flight_positions(C0, az_c, a) -> np.ndarray:
    """The positions of flight_path, vectorised (same sub-step integration, about 100x faster). [n, 3]."""
    n = int(round(a.duration * a.fps))
    sub = 8
    dt = 1.0 / a.fps
    u = direction(az_c, 0.0)
    w = np.array([u[1], -u[0], 0.0])
    p0 = np.asarray(C0, float) + a.start_m * u + a.offset_m * w + np.array([0, 0, a.alt_m])
    head0 = (az_c + 180.0) % 360.0
    ts = np.arange(n)[:, None] * dt + (np.arange(sub)[None, :] + 0.5) * dt / sub
    hs = np.radians(head0 + a.weave_deg * np.sin(2 * np.pi * ts / a.weave_s))
    inc = (a.speed * dt / sub) * np.stack([np.sin(hs), np.cos(hs), np.zeros_like(hs)], axis=-1).reshape(-1, 3)
    pos = np.empty((n, 3))
    pos[0] = p0
    if n > 1:
        pos[1:] = p0 + np.cumsum(inc, axis=0)[np.arange(1, n) * sub - 1]
    return pos


def step_seconds() -> float:
    cam = C.FROZEN["cameras"]
    return cam["frames_per_step"] / cam["frame_hz"]


class Track:
    """The drone's direction and range at every environment step of one episode, from its EpisodeSpec."""

    def __init__(self, spec: C.EpisodeSpec, n_steps: int):
        dt = step_seconds()
        alt = spec.entry_range_m * math.tan(math.radians(spec.entry_el_deg))
        a = SimpleNamespace(duration=(n_steps + 1) * dt, fps=1.0 / dt, start_m=spec.entry_range_m,
                            offset_m=spec.lateral_offset_m, alt_m=alt, weave_deg=spec.weave_deg,
                            weave_s=spec.weave_s, speed=spec.speed_mps)
        pos = flight_positions(np.zeros(3), spec.entry_az_deg, a)
        self.range_m = np.linalg.norm(pos, axis=1)
        self.az_deg = np.degrees(np.arctan2(pos[:, 0], pos[:, 1]))          # signed, sector centre = 0
        self.el_deg = np.degrees(np.arcsin(np.clip(pos[:, 2] / self.range_m, -1.0, 1.0)))
        inside = np.array([geo.cell_of(az, el) is not None for az, el in zip(self.az_deg, self.el_deg)])
        out = np.flatnonzero(~inside)
        self.exit_step = int(out[0]) if len(out) else len(inside)          # first step outside the sector

    def at(self, k: int) -> tuple[float, float, float]:
        k = min(k, len(self.range_m) - 1)
        return float(self.az_deg[k]), float(self.el_deg[k]), float(self.range_m[k])
