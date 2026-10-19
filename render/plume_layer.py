"""Exhaust plume in the thermal bands' layers (pure NumPy): one object crop per band with the four jets of
thermal/plume.py projected into the image, added next to the rendered target and negatives (data/writer.py). The
loader composites it like any object (data/composite.py): premultiplied emission P, absorbing coverage A, and its
own gas transmittance (the plume's spectrum is concentrated where the air absorbs, so the target's band-averaged
value would be wrong). Not a labelled object: the box stays the airframe's.
"""
from __future__ import annotations

import math

import numpy as np

from render import placement as pl
from render import psf
from thermal import atmosphere_bands as atb
from thermal import gas_bands as gb
from thermal import plume as pm

THERMAL = ("lwir", "lwir_cooled", "mwir")


def _camera(spec):
    c = spec["camera"]
    return pl.Camera(c["width"], c["height"], c["hfov_deg"], c["position"], c["azimuth_deg"], c["elevation_deg"],
                     c.get("roll_deg", 0.0))


def jets(spec: dict) -> list:
    """The scene's four exhaust jets in world coordinates (thermal/plume.Jet), or [] without a plume."""
    th = spec.get("thermal") or {}
    pdraw = th.get("plume")
    if not pdraw or not spec.get("engine"):
        return []
    o = spec["objects"][0]
    M = np.asarray(o["matrix_world"], float)
    R, t = M[:3, :3], M[:3, 3]
    lw = spec["signatures"]["target"]["lwir"]["exhaust_stub"]
    setting = lw.get("thermal_profile", "nominal")
    theta_out = pm.gas_exit_excess(float(lw["dT"]), setting)
    t_air = spec["air_temp_c"] + 273.15
    wind = float(o.get("wind_mps", 0.0)) * pl.direction(float(o.get("wind_dir_deg", 0.0)), 0.0)
    v_rel = np.asarray(o["velocity_mps"], float) - wind
    speed = float(np.linalg.norm(v_rel))
    alt = float(o.get("altitude_m", 0.0))
    ratio = pm.exit_velocity_ratio(setting, theta_out, t_air, max(speed, 1.0), max(alt, 0.0))
    from thermal import engine_heat as eh
    p_atm = eh.pressure_isa(max(alt, 0.0)) / 101325.0
    comp = pm.exhaust_composition(pdraw["equivalence_ratio"], pdraw["fuel_h_to_c"], pdraw["short_circuit"])
    eng = spec["engine"]
    cut = cut_n = None
    if eng.get("propeller_plane_x_m") is not None:           # the pusher propeller ends the hot plume
        cut = R @ np.array([eng["propeller_plane_x_m"], 0.0, 0.0]) + t
        cut_n = -(R @ np.array([1.0, 0.0, 0.0]))
    out = []
    for e in eng["exhaust_outlets"]:
        origin = R @ np.asarray(e["centre_m"], float) + t
        axis = ratio * (R @ np.asarray(e["axis"], float)) - v_rel / max(speed, 1e-6)
        out.append(pm.Jet(origin, axis, 2.0 * e["radius_m"] * 0.905, theta_out, t_air, p_atm, comp,
                          pdraw["k_mix"], pdraw["spread"], cut, cut_n))   # bore = 28.5 / 31.5 of the stub's outer d
    return out


def plume_object(spec: dict, band: str) -> dict | None:
    """The plume's object entry for one thermal band (crop, P, A, range_m, elevation_deg, t_gas), or None."""
    th = spec.get("thermal") or {}
    tables = (th.get("tables") or {}).get("bands", {})
    if band not in THERMAL or band not in tables or band not in th.get("bands_um", {}):
        return None
    js = jets(spec)
    if not js:
        return None
    cam = _camera(spec)
    band_um = tuple(th["bands_um"][band])
    w = atb.band_w(band_um)
    tab = tables[band]
    t_air = spec["air_temp_c"] + 273.15
    centre = np.mean([j.o for j in js], axis=0)
    view = centre - cam.C
    rng_m = float(np.linalg.norm(view))
    az, el = pl.azimuth_elevation(view)
    eps_bg = float(atb.sky_radiance_at(tab, max(el, 0.0)) / max(tab["L_air"], 1e-30))
    bg = {band: gb.planck_wn(t_air, w) * min(1.0, eps_bg)}
    n = (th["plume"].get("rays") or {})
    W, H = cam.W, cam.H
    pts, Pv, Av, spec_sum = [], [], [], np.zeros_like(w)
    for j in js:
        m = pm.jet_band_maps(j, view, {band: w}, bg, int(n.get("n_u", 12)), int(n.get("n_v", 7)))
        b = m["bands"][band]
        pts.append(m["anchors"])
        Pv.append(b["L_emit"] * m["dA"])
        Av.append(b["A_abs"] * m["dA"])
        spec_sum += b["spectrum_w_sr"]
    pts, Pv, Av = np.concatenate(pts), np.concatenate(Pv), np.concatenate(Av)
    u, v, z = cam.project(pts)
    ok = (z > 0) & np.isfinite(u) & np.isfinite(v)
    if not ok.any():
        return None
    s0 = float(spec["psf"][band]["sigma0"])
    pad = int(math.ceil(3 * s0)) + 2
    x0, y0 = max(0, int(math.floor(u[ok].min())) - pad), max(0, int(math.floor(v[ok].min())) - pad)
    x1, y1 = min(W, int(math.ceil(u[ok].max())) + pad), min(H, int(math.ceil(v[ok].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    foot = (z / cam.f) ** 2                                   # m2 seen by one pixel at each ray's depth
    P = np.zeros((y1 - y0, x1 - x0), np.float64)
    A = np.zeros_like(P)
    for uu, vv, pv, av, fo, k in zip(u, v, Pv, Av, foot, ok):
        if not k:
            continue
        fx, fy = uu - 0.5 - x0, vv - 0.5 - y0                  # pixel i spans [i, i + 1)
        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        ax, ay = fx - ix, fy - iy
        for dx, dy, wgt in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)), (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
            px, py = ix + dx, iy + dy
            if 0 <= px < P.shape[1] and 0 <= py < P.shape[0]:
                P[py, px] += wgt * pv / fo
                A[py, px] += wgt * av / fo
    P = psf.gaussian_blur(P[..., None].astype(np.float32), s0)[..., 0]
    A = np.clip(psf.gaussian_blur(A[..., None].astype(np.float32), s0)[..., 0], 0.0, 1.0)
    # gas transmittance of the path for the plume's own spectrum (no aerosol: the loader applies it, grey)
    prof = atb.profile_from(th["atmosphere"])
    win = atb.window_from(th["tables"]) if w.min() < atb.WINDOW_WN[1] else None
    t_gas = float(atb.path_transmittance(prof, max(el, 0.0), [rng_m], w, np.maximum(spec_sum, 1e-30),
                                         float(spec["camera"]["position"][2]), th["atmosphere"]["cont_scale"], win)[0])
    tgt = spec["objects"][0]
    in_front = rng_m < float(tgt["range_m"])
    return dict(name="plume", crop=[x0, y0, x1, y1], P=P[..., None].astype(np.float32), A=A.astype(np.float32),
                range_m=float(tgt["range_m"]) + (-0.05 if in_front else 0.05), elevation_deg=float(el), t_gas=t_gas,
                intensity_w_sr=float(np.sum(spec_sum) * atb.W_STEP), plume=True)
