"""Generic camera classes and the co-boresighted camera rig (CLAUDE.md sensor set rule; configs/sensors.yaml cameras).

Each scene draws one camera per sensor: RGB, LWIR uncooled, LWIR cooled and MWIR (SWIR is deferred). A camera is a
detector class with a format (pixels and pitch), a lens, a waveband and noise figures, each sampled over a range that
covers the manufacturer documents saved in docs/sources/cameras/ (readings.json). The field of view always follows
from the format and the focal length, HFOV = 2 atan(N p / 2 f) (pinhole; real lenses add a little distortion), so the
sampler cannot produce a combination of resolution, pitch and field of view that no camera has.

The rig: the student's uncooled LWIR camera (a fixed lens) sets the scene's output grid, the 640 x 512 network frame
spanning its field of view (configs/sensors.yaml image). The other cameras sit on the same mount and look along the
same axis. A zoom camera (RGB, cooled LWIR, MWIR) zooms to the narrowest setting that still covers the whole grid,
within its zoom range; if even its widest setting is narrower than the grid, it covers the centre only (coverage < 1).
One camera pixel then spans r = IFOV_camera / IFOV_grid output pixels: r < 1 for a finer camera (binned onto the
grid), r > 1 for a coarser one. data/camera_emulation.py reproduces each camera on the grid from r, its coverage and
its noise model. Pure NumPy.
"""
from __future__ import annotations

import math

import numpy as np

H_PLANCK, C_LIGHT, K_BOLTZ = 6.62607015e-34, 2.99792458e8, 1.380649e-23   # SI 2019 exact values


# ============================================================================ geometry
def hfov_deg(n_px: int, pitch_um: float, focal_mm: float) -> float:
    """Pinhole field of view across n_px pixels of pitch_um behind a focal_mm lens: 2 atan(N p / 2 f)."""
    return math.degrees(2.0 * math.atan(n_px * pitch_um * 1e-3 / (2.0 * focal_mm)))


def focal_for_hfov_mm(n_px: int, pitch_um: float, hfov: float) -> float:
    return n_px * pitch_um * 1e-3 / (2.0 * math.tan(math.radians(hfov) / 2.0))


def other_axis_deg(fov_deg: float, n_from: int, n_to: int) -> float:
    """Field of view across n_to pixels of a camera whose n_from pixels span fov_deg (square pixels)."""
    return math.degrees(2.0 * math.atan(math.tan(math.radians(fov_deg) / 2.0) * n_to / n_from))


# ============================================================================ sampling helpers
def _weighted(rng, items: list) -> dict:
    w = np.array([float(it.get("weight", 1.0)) for it in items])
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def _uniform(rng, r) -> float:
    return float(r[0]) if not isinstance(r, (list, tuple)) else float(rng.uniform(r[0], r[1]))


def _log_uniform(rng, r) -> float:
    return float(math.exp(rng.uniform(math.log(r[0]), math.log(r[1]))))


def _frame_rate(rng, cls: dict) -> float:
    fr = cls["frame_hz"]
    return float(_weighted(rng, [dict(v=v, weight=w) for v, w in zip(fr["options"], fr["weights"])])["v"])


# ============================================================================ physics: cooled integration time
def photon_radiance(band_um, t_k: float, n: int = 400) -> float:
    """In-band photon radiance of a black body, photons s-1 m-2 sr-1 (Planck, per wavelength, integrated)."""
    lam = np.linspace(band_um[0], band_um[1], n) * 1e-6
    lp = 2.0 * C_LIGHT / lam ** 4 / np.expm1(H_PLANCK * C_LIGHT / (lam * K_BOLTZ * t_k))
    return float(np.trapz(lp, lam))


def integration_ms(band_um, f_number: float, pitch_um: float, qe: float, capacity_e: float, well_fill: float,
                   t_scene_k: float = 300.0) -> float:
    """Integration time that fills the well to `well_fill` looking at a t_scene_k black body: on-axis photon
    irradiance at the focal plane pi L / (4 F^2) (paraxial, cold-shielded), x pixel area x quantum efficiency."""
    rate = math.pi * photon_radiance(band_um, t_scene_k) / (4.0 * f_number ** 2) * (pitch_um * 1e-6) ** 2 * qe
    return 1e3 * well_fill * capacity_e / rate


def residual_fpn_k(band_um, rnu_frac_of_dr: float, well_fill: float = 0.5, fill_range=(0.1, 0.9),
                   t_scene_k: float = 300.0) -> float:
    """Kelvin-equivalent of a residual non-uniformity quoted as a fraction of the dynamic range (STD/DR over a
    well-fill range): sigma_signal = rnu x (fill_hi - fill_lo) x capacity; responsivity at well_fill =
    well_fill x capacity x (dN/dT)/N, the relative slope of the in-band photon radiance at t_scene_k."""
    dn = (photon_radiance(band_um, t_scene_k + 0.5) - photon_radiance(band_um, t_scene_k - 0.5)) \
        / photon_radiance(band_um, t_scene_k)
    return rnu_frac_of_dr * (fill_range[1] - fill_range[0]) / (well_fill * dn)


# ============================================================================ cameras
def sample_fixed_lens(rng, cls: dict, narrow: bool = False) -> dict:
    """A fixed-lens camera: format, then a focal length log-uniform over the class's lens range, redrawn until the
    field of view lies in the class range. narrow: the longest lens the range allows (the narrowest view)."""
    fmt = _weighted(rng, cls["formats"])
    n, p = int(fmt["width"]), float(fmt["pitch_um"])
    f_lo, f_hi = cls["focal_mm"]
    h_lo, h_hi = cls["hfov_deg"]
    if narrow:
        f = min(f_hi, focal_for_hfov_mm(n, p, h_lo))
    else:
        for _ in range(200):
            f = _log_uniform(rng, (f_lo, f_hi))
            if h_lo <= hfov_deg(n, p, f) <= h_hi:
                break
        else:                                   # no lens of the range fits this format: the nearest edge
            f = min(max(focal_for_hfov_mm(n, p, 0.5 * (h_lo + h_hi)), f_lo), f_hi)
    return dict(width=n, height=int(fmt["height"]), pitch_um=p, focal_mm=f, hfov_deg=hfov_deg(n, p, f),
                format_source=fmt.get("source"), zoom=False)


def zoom_to_cover(fmt: dict, zoom_focal_mm, grid_hfov: float, grid_vfov: float) -> dict:
    """Narrowest zoom setting whose field covers the grid in both axes, clipped to the lens's focal range."""
    n, h, p = int(fmt["width"]), int(fmt["height"]), float(fmt["pitch_um"])
    t_need = max(math.tan(math.radians(grid_hfov) / 2.0), math.tan(math.radians(grid_vfov) / 2.0) * n / h)
    f = n * p * 1e-3 / (2.0 * t_need)
    f = min(max(f, zoom_focal_mm[0]), zoom_focal_mm[1])
    hf = hfov_deg(n, p, f)
    vf = other_axis_deg(hf, n, h)
    cov = [math.tan(math.radians(hf) / 2) / math.tan(math.radians(grid_hfov) / 2),
           math.tan(math.radians(vf) / 2) / math.tan(math.radians(grid_vfov) / 2)]
    cov = [1.0 if c >= 1.0 - 1e-9 else c for c in cov]          # an exact match is full coverage (rounding)
    return dict(width=n, height=h, pitch_um=p, focal_mm=f, hfov_deg=hf, vfov_deg=vf, coverage=list(cov),
                format_source=fmt.get("source"), zoom=True)


def zoom_in(cam: dict, z: float, zoom_focal_mm, grid: dict) -> dict:
    """The same zoom camera set z times narrower than its covering setting: focal length z f, clipped to the lens's
    longest. Fields of view, coverage of the grid, IFOV and pixel scale follow. A camera that gives up coverage for
    pixels on the target (docs/privileged_teacher_case.md, section 6). The f-number, NETD and noise draws are unchanged
    (a constant-aperture-ratio zoom, [C])."""
    n, h, p = int(cam["width"]), int(cam["height"]), float(cam["pitch_um"])
    f = min(float(cam["focal_mm"]) * float(z), float(zoom_focal_mm[1]))
    hf = hfov_deg(n, p, f)
    vf = other_axis_deg(hf, n, h)
    cov = [math.tan(math.radians(hf) / 2) / math.tan(math.radians(grid["hfov_deg"]) / 2),
           math.tan(math.radians(vf) / 2) / math.tan(math.radians(grid["vfov_deg"]) / 2)]
    out = dict(cam, focal_mm=f, hfov_deg=hf, vfov_deg=vf, coverage=[1.0 if c >= 1.0 - 1e-9 else c for c in cov],
               ifov_mrad=p / f, teacher_zoom=f / float(cam["focal_mm"]))
    out["pixel_scale"] = out["ifov_mrad"] / float(grid["ifov_mrad"])
    return out


def sample_rig(rng, cams: dict, grid_w: int, grid_h: int, narrow: bool = False) -> dict:
    """One camera per current sensor. Returns {"grid": ..., "cameras": {name: camera}}; every camera carries its
    format, focal length, fields of view, coverage of the grid, pixel scale r and its own noise draws."""
    student = cams["lwir_uncooled"]
    g = sample_fixed_lens(rng, student, narrow=narrow)
    grid_hfov = g["hfov_deg"]
    grid_vfov = other_axis_deg(grid_hfov, grid_w, grid_h)
    grid_ifov = 2.0 * math.tan(math.radians(grid_hfov) / 2.0) / grid_w          # rad per output pixel (centre)
    out = dict(grid=dict(width=grid_w, height=grid_h, hfov_deg=grid_hfov, vfov_deg=grid_vfov,
                         ifov_mrad=1e3 * grid_ifov, set_by="lwir_uncooled"), cameras={})
    for name, cls in cams.items():
        if name == "lwir_uncooled":
            cam = dict(g, vfov_deg=other_axis_deg(g["hfov_deg"], g["width"], g["height"]), coverage=[1.0, 1.0])
        else:
            cam = zoom_to_cover(_weighted(rng, cls["formats"]), cls["zoom_focal_mm"], grid_hfov, grid_vfov)
        cam["ifov_mrad"] = cam["pitch_um"] / cam["focal_mm"]                 # um / mm = mrad
        cam["pixel_scale"] = cam["ifov_mrad"] / out["grid"]["ifov_mrad"]      # r: grid pixels per camera pixel
        cam["detector"] = cls["detector"]
        cam["frame_hz"] = _frame_rate(rng, cls)
        if "detectors" in cls:                                               # MWIR: the detector sets the band
            det = _weighted(rng, cls["detectors"])
            cam.update(detector=det["name"], band_um=list(det["band_um"]), band_source=det.get("source"))
        else:
            cam["band_um"] = list(cls["band_um"])
        if "netd_mk" in cls:
            cam["netd_mk"] = _uniform(rng, cls["netd_mk"])
        if "read_noise_e" in cls:
            cam["read_noise_e"] = _uniform(rng, cls["read_noise_e"])
        if "f_number" in cls:
            cam["f_number"] = _uniform(rng, cls["f_number"])
        if "row_column_temporal_frac" in cls:                               # of the NETD, each axis
            cam["row_temporal_k"] = _uniform(rng, cls["row_column_temporal_frac"]) * cam["netd_mk"] / 1000.0
            cam["column_temporal_k"] = _uniform(rng, cls["row_column_temporal_frac"]) * cam["netd_mk"] / 1000.0
        if "residual_nonuniformity" in cls:                                 # cooled FPAs after NUC
            rn = cls["residual_nonuniformity"]
            bound = residual_fpn_k(cam["band_um"], rn["frac_of_dynamic_range"], rn["well_fill"], tuple(rn["fill_range"]))
            cam["pixel_fpn_k"] = float(rng.uniform(0.0, bound))
            cam["column_fpn_k"] = float(rng.uniform(0.0, rn["column_share"] * bound))
            cam["residual_fpn_bound_k"] = bound
        if "integration" in cls:                                            # cooled snapshot integration
            it = cls["integration"]
            qe, fill, cap = _uniform(rng, it["qe"]), _uniform(rng, it["well_fill"]), _uniform(rng, it["capacity_e"])
            t = integration_ms(cam["band_um"], cam["f_number"], cam["pitch_um"], qe, cap, fill)
            cam["integration_ms"] = min(t, 1000.0 / cam["frame_hz"])
            cam["integration_inputs"] = dict(qe=qe, well_fill=fill, capacity_e=cap)
        out["cameras"][name] = cam
    return out
