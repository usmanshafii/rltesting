"""Clear-sky 8-14 um radiance from surface air temperature and humidity. Pure NumPy.

Zenith emittance, Idso (1981), Water Resources Research 17(2):295-304, fitted on Arizona data:
    eps_z = 0.24 + 2.98e-8 * e0**2 * exp(3000 / T0)          e0 vapour pressure [hPa], T0 air [K]
Slant path: eps(theta) = 1 - (1 - eps_z) ** (1 / cos theta), theta the zenith angle (Granqvist and
Hjortsberg 1981). The slant law is monochromatic and is applied here to the band average, so the dry
sky at 10-30 deg elevation carries the largest model error; in humid air the low sky approaches air
temperature under either form. Sky radiance = eps(theta) * L_band(T0).

Evidence grade: B- for humid skies near the horizon, C+ for dry skies at 10-30 deg (unmeasured here;
the cheap check is a narrow-field 8-14 um IR thermometer at the zenith and at 25 deg elevation).
"""
from __future__ import annotations

import numpy as np

EPS_Z_MAX = 0.95


def saturation_vapour_pressure_hpa(t_c):
    """Magnus formula over water (6.112 hPa, 17.62, 243.12 C)."""
    t_c = np.asarray(t_c, float)
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def dew_point_c(e_hpa):
    x = np.log(np.asarray(e_hpa, float) / 6.112)
    return 243.12 * x / (17.62 - x)


def vapour_pressure_hpa(t_air_c: float, rh: float, dew_point_max_c: float | None = None) -> float:
    """Vapour pressure from relative humidity, capped at a climatological maximum dew point."""
    e = float(rh) * float(saturation_vapour_pressure_hpa(t_air_c))
    if dew_point_max_c is not None:
        e = min(e, float(saturation_vapour_pressure_hpa(dew_point_max_c)))
    return e


def zenith_emissivity(t_air_k: float, e_hpa: float) -> float:
    """Idso (1981) 8-14 um zenith emittance, capped at EPS_Z_MAX."""
    return float(min(EPS_Z_MAX, 0.24 + 2.98e-8 * e_hpa ** 2 * np.exp(3000.0 / t_air_k)))


def slant_emissivity(eps_z: float, elevation_deg):
    """Emittance along a ray at `elevation_deg` above the horizon; 1 at and below the horizon."""
    el = np.asarray(elevation_deg, float)
    s = np.sin(np.radians(np.clip(el, 1e-3, 90.0)))
    return np.where(el > 0, 1.0 - (1.0 - eps_z) ** (1.0 / s), 1.0)


def humidity_state(t_air_c: float, rh: float, dew_point_max_c: float | None = None) -> dict:
    """Everything the sky model needs (and the metadata records) for one scene."""
    e = vapour_pressure_hpa(t_air_c, rh, dew_point_max_c)
    return dict(vapour_pressure_hpa=e, dew_point_c=float(dew_point_c(e)),
                rh_effective=e / float(saturation_vapour_pressure_hpa(t_air_c)),
                emissivity_zenith=zenith_emissivity(t_air_c + 273.15, e))


def sky_dT_k(lut, t_air_k: float, eps_z: float, elevation_deg):
    """Sky brightness temperature minus air temperature at the given elevations."""
    L = slant_emissivity(eps_z, elevation_deg) * float(lut.radiance(t_air_k))
    return lut.temperature(L) - t_air_k
