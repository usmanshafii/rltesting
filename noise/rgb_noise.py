"""RGB sensor model, applied in the data loader (spec 3.1). Pure NumPy.

Linear render radiance -> photo-electrons (radiance x exposure x conversion) -> heteroscedastic
Gaussian shot + read noise -> auto-exposure gain -> clipping and quantisation -> gamma.

This is the Brooks et al. (CVPR 2019) noise model: y ~ N(x, lambda_read + lambda_shot x) on linear
raw values x in [0, 1], with lambda_shot = gain / full_well and lambda_read = (gain sigma_read /
full_well)^2. Deriving both from a light level instead of drawing them independently makes night
and dusk frames noisier in a physically consistent way. Mosaicking / demosaicking is not modelled.
"""
from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])


def luminance(img):
    img = np.asarray(img, np.float32)
    return img @ LUMA.astype(np.float32) if img.ndim == 3 and img.shape[-1] == 3 else img


def sample_params(rng, cfg: dict) -> dict:
    u = lambda k: float(rng.uniform(*cfg[k]))  # noqa: E731
    return dict(e_per_unit_ms=u("electrons_per_unit_ms"), read_e=u("read_noise_e"), full_well=u("full_well_e"),
                ae_target=u("ae_target"), wb=rng.uniform(0.96, 1.04, 3).tolist())


def auto_exposure(median_e: float, full_well: float, ae_target: float, max_gain: float) -> tuple[float, float]:
    """The exposure time is fixed by the scene (it also sets the rendered motion blur), so the camera
    meets its target level with the aperture / ND (light reduction, never more light than wide
    open) and, when that is not enough, with gain. Returns (light factor <= 1, gain >= 1)."""
    target = ae_target * full_well
    if median_e > target:
        return target / median_e, 1.0
    return 1.0, float(min(target / max(median_e, 1e-6), max_gain))


def apply(L: np.ndarray, rng, cfg: dict, exposure_ms: float, params: dict | None = None) -> tuple[np.ndarray, dict]:
    """L: HxWx3 linear radiance (render units). Returns display-referred HxWx3 in [0, 1]."""
    p = params or sample_params(rng, cfg)
    e_open = np.clip(np.asarray(L, np.float32), 0, None) * (p["e_per_unit_ms"] * exposure_ms)
    e_open = e_open * np.asarray(p["wb"], np.float32)
    light, gain = auto_exposure(float(np.median(e_open.mean(-1))), p["full_well"], p["ae_target"], cfg["max_gain"])
    e_mean = e_open * light
    noisy = e_mean + np.sqrt(e_mean + p["read_e"] ** 2) * rng.standard_normal(e_mean.shape).astype(np.float32)
    noisy = np.minimum(noisy, p["full_well"])                           # pixel saturation
    raw = np.clip(noisy * gain / p["full_well"], 0.0, 1.0)
    q = 2 ** int(cfg["bits"]) - 1
    raw = np.round(raw * q) / q
    out = raw ** (1.0 / cfg["gamma"])
    return out.astype(np.float32), dict(p, gain=gain, light=light, lambda_shot=gain / p["full_well"],
                                        lambda_read=(gain * p["read_e"] / p["full_well"]) ** 2)


# ============================================================================ the whole camera (2026-10-17)
def sample_pipeline(rng, pipe: dict) -> dict:
    """One RGB camera's raw-noise and processing settings for a frame (configs/sensors.yaml camera_pipelines.rgb)."""
    u = lambda r: float(rng.uniform(*r))  # noqa: E731
    high = rng.uniform() < pipe["high_sensitivity_prob"]
    level = int(rng.integers(pipe["nr_level"][0], pipe["nr_level"][1] + 1))
    aperture = int(rng.integers(pipe["aperture_level"][0], pipe["aperture_level"][1] + 1))
    return dict(bits=int(rng.choice(pipe["adc_bits"])), max_gain=10 ** (pipe["max_gain_db"] / 20) * (4.0 if high else 1.0),
                high_sensitivity=bool(high), lam=u(pipe["read_tukey_lambda"]), row_frac=u(pipe["row_noise_frac"]),
                nr_level=level, aperture=aperture, gamma="standard" if rng.uniform() < pipe["gamma_standard_prob"] else "straight",
                ca_r=u(pipe["chromatic_aberration"]) * np.sign(rng.uniform(-1, 1)),
                ca_b=u(pipe["chromatic_aberration"]) * np.sign(rng.uniform(-1, 1)))


def apply_camera(L: np.ndarray, rng, cfg: dict, pipe: dict, exposure_ms: float,
                 params: dict | None = None) -> tuple[np.ndarray, dict]:
    """An RGB zoom camera's video frame (noise/isp.py; docs/sensor_noise.md).

    Sensor (Wei et al., CVPR 2020: their noise formation model with every term, which matched training on real paired
    data): radiance -> photo-electrons with the auto-exposure of `apply` (aperture / ND, then gain up to the camera's
    limit) -> Bayer RGGB sampling (IMX327 colour mosaic) -> Poisson photon shot noise -> Tukey-lambda read noise -> row
    (banding) noise -> gain, saturation and ADC quantisation (10 or 12 bits). Camera processing: bilinear demosaicking,
    white-balance error, lateral chromatic aberration, noise reduction (FCB-EV7520 levels Off and 1-5, acting only
    as the gain rises), edge enhancement (16 steps), display gamma. Returns display-referred HxWx3 in [0, 1]."""
    from noise import isp
    p = params or sample_params(rng, cfg)
    q = sample_pipeline(rng, pipe)
    e_open = np.clip(np.asarray(L, np.float32), 0, None) * (p["e_per_unit_ms"] * exposure_ms)
    light, gain = auto_exposure(float(np.median(e_open.mean(-1))), p["full_well"], p["ae_target"], q["max_gain"])
    raw = isp.mosaic_rggb(e_open * light)
    h, w = raw.shape
    raw = rng.poisson(np.clip(raw, 0, None)).astype(np.float32)                 # photon shot noise
    raw = raw + isp.tukey_lambda(rng, q["lam"], p["read_e"], (h, w))            # read noise (long-tailed)
    raw = raw + (rng.standard_normal(h) * q["row_frac"] * p["read_e"]).astype(np.float32)[:, None]   # row noise
    top = 2 ** q["bits"] - 1
    dn = np.round(np.clip(raw * gain / p["full_well"], 0.0, 1.0) * top) / top   # gain, saturation, quantisation
    rgb = isp.demosaic_bilinear(dn.astype(np.float32)) * np.asarray(p["wb"], np.float32)
    rgb = isp.chromatic_aberration(rgb, q["ca_r"], q["ca_b"])
    gain_db = 20 * np.log10(max(gain, 1.0))
    lo_db, hi_db = pipe["nr_gain_db"]
    nr = pipe["nr_sigma_px"][q["nr_level"]] * float(np.clip((gain_db - lo_db) / (hi_db - lo_db), 0.0, 1.0))
    rgb = isp.noise_reduction(rgb, nr)
    rgb = isp.sharpen(rgb, pipe["sharpen_amount_max"] * q["aperture"] / 15.0)
    out = isp.gamma_encode(rgb, q["gamma"])
    return out, dict(p, gain=gain, light=light, nr_sigma_px=nr, pipeline=q)


def nominal_sigma(L_level: float, exposure_ms: float, cfg: dict) -> float:
    """Noise sigma in radiance units at radiance L_level with the nominal sensor (for the CNR),
    with the same aperture / ND logic as apply() evaluated at that level."""
    n = cfg["nominal"]
    k = n["electrons_per_unit_ms"] * exposure_ms
    light, _ = auto_exposure(max(L_level, 0.0) * k, n["full_well_e"], n["ae_target"], cfg["max_gain"])
    k *= light
    return float(np.sqrt(max(L_level, 0.0) * k + n["read_noise_e"] ** 2) / k)
