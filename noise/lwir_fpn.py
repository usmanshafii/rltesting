"""LWIR sensor model, applied in the data loader (spec 3.2, Phase 3 brief). Pure NumPy.

Radiance -> 14-bit counts with a random responsivity (counts per kelvin near ambient) and offset
-> column fixed-pattern noise + residual pixel non-uniformity + Gaussian temporal noise at an
NETD of 30-80 mK (He et al., Applied Optics 2018 style stripe model) -> clipping and rounding
-> automatic gain control: a linear stretch between robust percentiles, like the 8-bit video an
uncooled core outputs.

Noise levels are kelvin-equivalent at the datasheet reference, a 30 C scene (NETD is quoted there).
A microbolometer's noise in counts does not follow the scene, but dL/dT does: against a cold sky or in
cold air the same counts are worth more kelvin (about 1.35x at 0 C, 1.75x at -20 C), so the noise is
set in radiance at T_REF_K rather than at the scene's air temperature.

The same model serves every thermal camera of the rig (render/cameras.py): pass the camera band's table (`lut`,
thermal/planck_lut.band_lut) for the cooled LWIR and MWIR cameras; the default is the 8-14 um table.
"""
from __future__ import annotations

import numpy as np

from thermal.planck_lut import lwir_lut


# Optional stripe terms (absent from configs/sensors.yaml, so earlier datasets and pilots reproduce
# exactly; tools/m1_det2.py adds them). config key -> parameter name.
REALISM_KEYS = {"column_gain_frac": "col_gain", "column_drift_k": "col_drift_k",
                "row_noise_k": "row_k", "column_temporal_k": "col_t_k"}


def sample_params(rng, cfg: dict) -> dict:
    u = lambda k: float(rng.uniform(*cfg[k]))  # noqa: E731
    p = dict(netd_k=u("netd_mk") / 1000.0, counts_per_k=u("counts_per_k"), offset=u("offset_counts"),
             col_k=u("column_fpn_k"), pix_k=u("pixel_fpn_k"), agc_lo=u("agc_low_pct"), agc_hi=u("agc_high_pct"))
    for key, name in REALISM_KEYS.items():          # drawn after the v1 parameters: v1 streams unchanged
        if key in cfg:
            p[name] = u(key)
    return p


def realistic_stripes(signal_counts: np.ndarray, rng, p: dict, cpk: float) -> np.ndarray:
    """Stripe components that are NOT constant down a column (counts). Each is optional:

    * column gain non-uniformity left after calibration (`col_gain`, a fraction): a column's stripe
      is g_c x (scene - calibration reference), so it changes wherever the scene changes along the
      column, e.g. a sky that warms towards the horizon or the target itself;
    * column offset drift along the readout (`col_drift_k`): 1/f noise in the column amplifiers gives
      each column a smooth random profile from top to bottom (5 knots, linear in between);
    * row noise (`row_k`) and temporal column noise (`col_t_k`): offsets redrawn every frame.

    Uncooled cameras keep such residual row and column noise after shutter correction, and it grows
    between corrections (1/f noise); amplitudes here are [C] assumptions (tools/m1_det2.py)."""
    h, w = signal_counts.shape
    out = np.zeros((h, w), np.float32)
    if p.get("col_gain", 0.0) > 0:
        out += signal_counts * (rng.standard_normal(w) * p["col_gain"]).astype(np.float32)[None, :]
    if p.get("col_drift_k", 0.0) > 0:
        knots = (rng.standard_normal((5, w)) * p["col_drift_k"] * cpk).astype(np.float32)
        y = np.linspace(0.0, 4.0, h)
        i0 = np.minimum(np.floor(y).astype(int), 3)
        t = (y - i0).astype(np.float32)[:, None]
        out += knots[i0] * (1 - t) + knots[i0 + 1] * t
    if p.get("row_k", 0.0) > 0:
        out += (rng.standard_normal(h) * p["row_k"] * cpk).astype(np.float32)[:, None]
    if p.get("col_t_k", 0.0) > 0:
        out += (rng.standard_normal(w) * p["col_t_k"] * cpk).astype(np.float32)[None, :]
    return out


def to_counts(L: np.ndarray, T_air_k: float, p: dict, lut=None) -> np.ndarray:
    lut = lut or lwir_lut()
    gain = p["counts_per_k"] / float(lut.dL_dT(T_air_k))           # counts per W m^-2 sr^-1
    return p["offset"] + gain * (np.asarray(L, np.float32) - float(lut.radiance(T_air_k)))


T_REF_K = 303.15


def _ref_scale(T_air_k: float, lut=None) -> float:
    """Counts per reference kelvin / counts per kelvin at ambient: dL/dT(30 C) / dL/dT(T_air)."""
    lut = lut or lwir_lut()
    return float(lut.dL_dT(T_REF_K) / lut.dL_dT(T_air_k))


def apply(L: np.ndarray, rng, cfg: dict, T_air_k: float, params: dict | None = None, lut=None) -> tuple[np.ndarray, dict]:
    """L: HxW radiance [W m^-2 sr^-1] in the band of `lut` (default 8-14 um). Returns (AGC image in [0, 1], params)."""
    p = params or sample_params(rng, cfg)
    c0 = to_counts(L, T_air_k, p, lut)
    h, w = c0.shape
    cpk = p["counts_per_k"] * _ref_scale(T_air_k, lut)      # counts per kelvin at the 30 C reference
    c = c0 + (rng.standard_normal(w) * p["col_k"] * cpk)[None, :]
    c = c + rng.standard_normal((h, w)) * p["pix_k"] * cpk
    c = c + rng.standard_normal((h, w)) * p["netd_k"] * cpk
    if any(p.get(n, 0.0) > 0 for n in REALISM_KEYS.values()):
        c = c + realistic_stripes(c0 - p["offset"], rng, p, cpk)    # calibration reference = ambient air
    top = 2 ** int(cfg["bits"]) - 1
    c = np.clip(np.round(c), 0, top).astype(np.float32)
    lo, hi = np.percentile(c, [p["agc_lo"], p["agc_hi"]])
    out = np.clip((c - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return out.astype(np.float32), dict(p, agc_range=[float(lo), float(hi)])


# ============================================================================ the whole camera (2026-10-17)
def sample_pipeline(rng, pipe: dict) -> dict:
    """One camera's corrections and output settings for a frame, from configs/sensors.yaml camera_pipelines.<camera>."""
    from noise import isp
    u = lambda r: float(rng.uniform(*r))  # noqa: E731
    ri = lambda r: int(rng.integers(int(r[0]), int(r[1]) + 1))  # noqa: E731
    d, rts, inu = pipe["defects"], pipe["rts"], pipe["inu"]
    return dict(defect_frac=u(d["frac"]), bad_lines=ri(d["bad_lines"]), residual_defect_frac=u(d["residual_frac"]),
                rts_frac=u(rts["frac"]), rts_amp_netd=u(rts["amplitude_netd"]), column_poly=u(pipe["column_poly_frac"]),
                shading_k=u(pipe["shading_k"]), inu_small=ri(inu["small"]), inu_large=ri(inu["large"]),
                inu_k=u(inu["amplitude_k"]), filter_attenuation=u(pipe["filter_attenuation"]),
                agc=isp.sample_agc(rng, pipe["agc"]))


def apply_camera(L: np.ndarray, rng, cfg: dict, pipe: dict, T_air_k: float, lut=None,
                 params: dict | None = None) -> tuple[np.ndarray, dict]:
    """A thermal camera's video frame: the sensor of `apply` plus the camera's corrections and display mapping
    (noise/isp.py; parameters in configs/sensors.yaml camera_pipelines; docs/sensor_noise.md).

    In counts: scene + temporal noise (NETD, temporal row and column noise as the camera's datasheet gives them) +
    residual fixed pattern (pixel and column offsets, the optional uncooled stripe terms, a per-column cubic in the
    signal after He et al. 2018, low-frequency shading, localised non-uniformities), the fixed-pattern part reduced by the
    camera's filters (`filter_attenuation`); blinking (RTS) pixels; 14-bit quantisation; defective pixels stuck at a
    rail and replaced from their neighbours, except a few missing from the defect map; then the AGC (histogram-based
    with detail enhancement, or linear) to [0, 1]."""
    from noise import isp
    p = params or sample_params(rng, cfg)
    q = sample_pipeline(rng, pipe)
    c0 = to_counts(L, T_air_k, p, lut)
    h, w = c0.shape
    cpk = p["counts_per_k"] * _ref_scale(T_air_k, lut)
    keep = 1.0 - q["filter_attenuation"]
    fixed = (rng.standard_normal(w) * p["col_k"] * cpk)[None, :] + rng.standard_normal((h, w)) * p["pix_k"] * cpk
    pf = dict(p, row_k=0.0, col_t_k=0.0)                                  # fixed-pattern stripe terms only
    if any(pf.get(n, 0.0) > 0 for n in REALISM_KEYS.values()):
        fixed = fixed + realistic_stripes(c0 - p["offset"], rng, pf, cpk)
    if q["column_poly"] > 0:                                              # column amplifier non-linearity (He 2018)
        lo, hi = float(c0.min()), float(c0.max())
        v = (c0 - lo) / max(hi - lo, 1.0)
        a2, a3 = (rng.uniform(-1.0, 1.0, (2, w)) * q["column_poly"] * (hi - lo)).astype(np.float32)
        fixed = fixed + a2[None, :] * v ** 2 + a3[None, :] * v ** 3
    if q["shading_k"] > 0:
        fixed = fixed + q["shading_k"] * cpk * isp.shading_field(rng, h, w)
    if q["inu_k"] > 0 and (q["inu_small"] or q["inu_large"]):
        fixed = fixed + q["inu_k"] * cpk * isp.inu_blobs(rng, h, w, q["inu_small"], q["inu_large"])
    c = c0 + keep * fixed + rng.standard_normal((h, w)) * p["netd_k"] * cpk
    c = c + (rng.standard_normal(h) * p.get("row_k", 0.0) * cpk)[:, None] + (rng.standard_normal(w) * p.get("col_t_k", 0.0) * cpk)[None, :]
    n_rts = int(q["rts_frac"] * h * w)
    if n_rts:                                                             # pixels caught in their high state this frame
        idx = rng.choice(h * w, n_rts, replace=False)
        on = rng.uniform(size=n_rts) < 0.5
        c.flat[idx[on]] += np.sign(rng.uniform(-1, 1, int(on.sum()))) * q["rts_amp_netd"] * p["netd_k"] * cpk
    top = 2 ** int(cfg["bits"]) - 1
    c = np.clip(np.round(c), 0, top).astype(np.float32)
    m = isp.defect_mask(rng, h, w, q["defect_frac"], q["bad_lines"])
    c[m] = rng.choice([0.0, float(top)], int(m.sum()))
    c = isp.replace_bad_pixels(c, m)                                      # the defect map is replaced from neighbours
    rm = isp.defect_mask(rng, h, w, q["residual_defect_frac"], cluster_prob=0.0)
    n_rm = int(rm.sum())                                                  # defects the map missed: hot / cold pixels
    c[rm] = np.clip(c[rm] + np.sign(rng.uniform(-1, 1, n_rm)) * rng.uniform(*pipe["defects"]["residual_offset_netd"], n_rm)
                    * p["netd_k"] * cpk, 0, top)
    out = isp.apply_agc(c, q["agc"])
    return out, dict(p, pipeline=q)


def nominal_sigma(T_air_k: float, cfg: dict, lut=None) -> float:
    """Temporal + column noise sigma in radiance units (for the CNR)."""
    n = cfg["nominal"]
    k = float(np.hypot(n["netd_mk"] / 1000.0, n["column_fpn_k"]))
    return k * float((lut or lwir_lut()).dL_dT(T_REF_K))       # noise radiance does not depend on the scene
