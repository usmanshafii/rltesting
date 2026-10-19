"""8-14 um path absorption by water vapour, tied to the scene's humidity draw. Pure NumPy.

Two parts (docs/weak_spots_register.md, section 3.2):

* Water-vapour continuum, Roberts, Selby and Biberman (1976), Applied Optics 15:2085, as used in
  LOWTRAN: k(nu) = (4.18 + 5578 exp(-7.87e-3 nu)) exp(1800 (1/T - 1/296)) cm^2 g^-1 atm^-1, times the
  broadening pressure (e + 0.002 P) in atm, times the vapour density. Averaged uniformly over
  714-1250 cm^-1 (8-14 um). Gives ~0.04 /km at 10 C / 50 % RH, ~0.5 /km in a monsoon afternoon and
  ~1 /km at 35 C / 80 % RH. Self-broadening makes it scale with vapour density squared, so on slant
  paths it falls off with about half the ~2 km vapour scale height.
* Band-edge line absorption (CO2 near 13-14 um, water lines near 8 um), which saturates within the
  first ~100 m: tau_edge = 1 - a (1 - exp(-R / R_sat)), a = co2 + h2o * e / 45 hPa. With the
  continuum it reproduces FLIR's 80 % transmission for 100 m at 35 C and 80 % RH.

Applied as contrast transmittance in data/composite.py together with the aerosol term, in the loader
(no re-render). Grade C+/B-: the continuum is a fitted lab/field law; the edge term and the slant
scale height are engineering estimates. Aerosol extinction (haze, fog) stays in the weather model.
"""
from __future__ import annotations

import math

import numpy as np

P_ATM_HPA = 1013.25
R_V = 461.5                     # J kg^-1 K^-1, water vapour
NU = np.linspace(714.0, 1250.0, 64)      # cm^-1, 8-14 um


def continuum_sigma_per_km(t_air_c: float, e_hpa: float) -> float:
    """Band-average water-vapour continuum extinction at the surface, per km."""
    T = t_air_c + 273.15
    k = (4.18 + 5578.0 * np.exp(-7.87e-3 * NU)) * math.exp(1800.0 * (1.0 / T - 1.0 / 296.0))
    k = k * (e_hpa / P_ATM_HPA + 0.002)                     # cm^2 g^-1
    rho = e_hpa * 100.0 / (R_V * T) * 1e3                    # g m^-3
    return float(np.mean(k) * rho * 1e-6 * 1e5)             # g cm^-3 x cm^2 g^-1 = cm^-1 -> km^-1


def edge_fraction(e_hpa: float, cfg: dict) -> float:
    return float(cfg["edge_co2"] + cfg["edge_h2o_at_45hpa"] * min(e_hpa, 60.0) / 45.0)


def path_params(t_air_c: float, e_hpa: float | None, cfg: dict | None) -> dict | None:
    """Per-scene parameters, or None when disabled or the scene has no humidity (pre-v2 scenes)."""
    if not cfg or not cfg.get("enabled", False) or e_hpa is None:
        return None
    return dict(sigma_per_m=continuum_sigma_per_km(t_air_c, e_hpa) / 1000.0,
                scale_height_m=float(cfg["continuum_scale_height_m"]),
                edge=edge_fraction(e_hpa, cfg), edge_saturation_m=float(cfg["edge_saturation_m"]),
                vapour_pressure_hpa=float(e_hpa), air_temp_c=float(t_air_c))


def transmittance(dist_m, p: dict | None, elevation_deg: float | None = None):
    """Water-vapour transmittance over a path of length dist_m (array or scalar); 1 when p is None.
    Slant paths (elevation > 0.5 deg) use the continuum scale height; ground paths are horizontal."""
    d = np.asarray(dist_m, float)
    if p is None:
        return np.ones_like(d)
    s = p["sigma_per_m"]
    H = p["scale_height_m"]
    if elevation_deg is not None and elevation_deg > 0.5 and H:
        sn = math.sin(math.radians(elevation_deg))
        od = s * H / sn * (1.0 - np.exp(-d * sn / H))
    else:
        od = s * d
    edge = 1.0 - p["edge"] * (1.0 - np.exp(-d / p["edge_saturation_m"]))
    return np.exp(-od) * edge


def scene_params(meta_or_spec: dict, cfg: dict | None) -> dict | None:
    """From a scene's metadata (loader) or spec (renderer): air temperature + sampled vapour pressure."""
    ex = meta_or_spec.get("extra", meta_or_spec)
    sky = (((ex.get("signatures") or {}).get("environment") or {}).get("lwir") or {}).get("sky") or {}
    t_c = meta_or_spec.get("air_temp_c", (meta_or_spec.get("conditions") or {}).get("air_temp_c"))
    if t_c is None:
        return None
    return path_params(float(t_c), sky.get("vapour_pressure_hpa"), cfg)
