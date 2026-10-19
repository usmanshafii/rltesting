"""Rotation blur of the propeller, as a time-averaged coverage texture. Pure NumPy.

The two-blade pusher turns at the shaft speed (target.yaml prop_rpm). Cycles blurs the aircraft's
translation from keyframes, but the propeller is rendered static, so its rotation over each band's
integration time is represented here instead:

    coverage(y, z) = fraction of the integration time during which a blade covers that point

computed from the blade planform, the angle at mid-exposure (shared by all bands: one instant) and the
sweep angle = 360 deg x rpm / 60 x t_int, with t_int the scene exposure for RGB and SWIR and the
microbolometer's effective integration time for LWIR. Short daylight exposures leave the blades partly
frozen; longer ones smear them into the uniform average the old disc assumed. The engine's drive flange
and spinner also turn, but they are bodies of revolution, so their rotation changes nothing in any band.

The disc is a thin closed solid, so a camera sees through two caps. Each cap therefore gets
alpha = 1 - sqrt(1 - coverage), which composes to the intended coverage; a single-cap alpha equal to
the coverage made the old disc about twice as opaque as intended.
"""
from __future__ import annotations

import math

import numpy as np


def sweep_deg(rpm: float, t_ms: float) -> float:
    return 360.0 * rpm / 60.0 * t_ms / 1000.0


def per_cap_alpha(coverage):
    """Alpha for each of two stacked caps so that 1 - (1 - a)^2 equals the wanted coverage."""
    c = np.clip(np.asarray(coverage, np.float64), 0.0, 1.0)
    return 1.0 - np.sqrt(1.0 - c)


def _grid(n: int, half_extent: float):
    t = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    Y, Z = np.meshgrid(t * half_extent, t * half_extent)      # rows = z (v), columns = y (u)
    return np.hypot(Y, Z), np.arctan2(Z, Y)


def _swept(r, phi, half_width_rad, n_items: int, phase_deg: float, sweep: float, radial_mask):
    """Exact time average of 'some item covers (r, phi)' while n_items equally spaced items (period
    P = 2 pi / n) turn through `sweep` degrees, centred on phase_deg. With w = 2 x half width, the
    covered time is a difference of G(y) = floor(y / P) w + min(y mod P, w), the integral of the
    periodic box train, so there is no time-sampling aliasing for thin blades or long sweeps."""
    P = 2.0 * math.pi / n_items
    hw = np.minimum(np.broadcast_to(np.asarray(half_width_rad, float), r.shape), P / 2.0)
    w = 2.0 * hw
    base = phi - math.radians(phase_deg) + hw
    D = math.radians(sweep)
    if D < 1e-9:
        cov = (np.mod(base, P) < w).astype(np.float64)
    else:
        def G(y):
            return np.floor(y / P) * w + np.minimum(np.mod(y, P), w)
        cov = (G(base + D / 2.0) - G(base - D / 2.0)) / D
    return np.where(radial_mask, np.clip(cov, 0.0, 1.0), 0.0)


def blade_coverage(n: int, p: dict, sweep: float) -> np.ndarray:
    """Coverage of the propeller disc over [-R, R]^2 in the disc plane (object Y right, Z up).
    p: radius_m, root_radius_m, chord_root_m, chord_tip_m, pitch_deg, blades, chord_scale, phase_deg."""
    R, r0 = p["radius_m"], p["root_radius_m"]
    r, phi = _grid(n, R)
    frac = np.clip((r - r0) / max(R - r0, 1e-9), 0.0, 1.0)
    chord = (p["chord_root_m"] + (p["chord_tip_m"] - p["chord_root_m"]) * frac) * p.get("chord_scale", 1.0)
    proj = chord * math.cos(math.radians(p["pitch_deg"]))          # chord seen along the shaft
    hw = np.arctan2(proj / 2.0, np.maximum(r, 1e-9))
    return _swept(r, phi, hw, int(p["blades"]), p["phase_deg"], sweep, (r >= r0) & (r <= R))


def mean_blade_coverage(p: dict, n: int = 256) -> float:
    """Blade area over disc area (the time-averaged coverage of a long exposure)."""
    cov = blade_coverage(n, dict(p, phase_deg=0.0), 0.0)
    return float(cov.mean() * 4.0 / math.pi)                        # square grid -> disc area


def scene_rotation(rng, g: dict, rpm_range, alpha: float, integration_ms: dict, cfg: dict) -> dict:
    """Per-scene rotation record (JSON-serialisable). g: derived target parameters (span units + span_m).
    alpha: sampled mean blade coverage; the blade chords are scaled so the long-exposure mean matches it."""
    s = g["span_m"]
    rpm = float(rng.uniform(*rpm_range))
    prop = dict(radius_m=g["prop_dia"] * s / 2, root_radius_m=g["spinner_dia"] * s / 2,
                chord_root_m=g["blade_root_chord"] * s, chord_tip_m=g["blade_tip_chord"] * s,
                pitch_deg=float(g["blade_pitch_deg"]), blades=2, phase_deg=float(rng.uniform(0, 360)))
    geo = mean_blade_coverage(dict(prop, chord_scale=1.0), 128)
    prop["chord_scale"] = float(alpha / geo) if geo > 0 else 1.0
    prop["mean_coverage"] = float(alpha)
    return dict(rpm=rpm, propeller=prop,
                integration_ms={b: float(v) for b, v in integration_ms.items()},
                sweep_deg={b: sweep_deg(rpm, v) for b, v in integration_ms.items()},
                texture_px=int(cfg.get("texture_px", 256)))


def band_alpha_images(rot: dict, band: str) -> dict:
    """Per-cap alpha image for one band: {'propeller_disc': (array, half_extent_m)}."""
    n, sw = rot["texture_px"], rot["sweep_deg"][band]
    prop = blade_coverage(n, rot["propeller"], sw)
    return {"propeller_disc": (per_cap_alpha(prop).astype(np.float32), rot["propeller"]["radius_m"])}
