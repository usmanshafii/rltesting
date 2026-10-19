"""Clear-air radiative transfer for the thermal cameras (CLAUDE.md sensor set rule): LWIR uncooled 8-14 um, LWIR cooled
7.7-9.3 um and the MWIR detector bands (3.6-4.9 um). Sky radiance by elevation, path transmittance and the direct
sun at the ground, from the scene's surface air temperature and humidity. Pure NumPy; docs/thermal_bands.md.

Atmosphere. Layers from the surface to 30 km, each uniform at its mid-height state:
  temperature  T(z) = T_sfc - lapse z up to the tropopause, isothermal above. Regional radiosondes (NOAA IGRA 2:
               Lahore City, Patiala and New Delhi, Oct 2023 - Sep 2026; tools/source_measurements.igra2_profiles)
               give the surface-500 hPa lapse rate, 10-90 % 4.5-7.7 K/km, median 5.7 K/km.
  pressure     hydrostatic for that temperature from the surface pressure.
  water vapour density rho_v(z) = rho_v(0) exp(-z / H_w), capped at saturation. H_w = precipitable water / surface
               vapour density from the same soundings: 10-90 % 0.96-2.8 km, from ~1.2 km in winter to ~2.7 km in the
               monsoon, and it grows with the surface vapour pressure (vapour_scale_height).
  CO2          well mixed, NOAA GML global mean 419-428 ppm over 2023-2026 (docs/sources/noaa_gml).
Gas absorption: the SP-3080 statistical narrow-band model of thermal/gas_bands.py (H2O, CO2; 5 cm-1 intervals,
Curtis-Godson along the slant path) plus the water-vapour continuum of Roberts, Selby and Biberman (1976) in 700-1300
cm-1 (thermal/atmosphere.py; its fit covers 8-14 um and is used to 7.7 um). Not modelled: the continuum in the MWIR
window, N2O (4.5 um), CO and CH4 in the air, ozone; tests/test_atmosphere_bands.py measures the MWIR window against the
ASTM G173-03 direct beam. Aerosol: grey per band, from the scene's visibility (data/composite.py), absorbing and
emitting a share 1 - omega0 of its extinction; in MWIR by day it also scatters sunlight into the sky (single
scattering, Henyey-Greenstein phase function).

Geometry: rays through spherical shells (Earth radius 6,371 km), so the horizon path is finite. Radiance along a ray
from the observer: L = sum_m B(T_m) (tau_{m-1} - tau_m) (SP-3080 eq. A-3), in W m-2 sr-1 per cm-1, integrated over
the camera's band (rectangular response).
"""
from __future__ import annotations

import functools
import math

import numpy as np

from thermal import gas_bands as gb

R_EARTH = 6.371e6
G0, R_D, R_V = 9.80665, 287.05, 461.5
W_STEP = 5.0
W_GRID = np.arange(700.0, 2800.0 + 0.1, W_STEP)          # cm-1: 3.57-14.3 um, every thermal camera band
Z_EDGES = np.array([0.0, 25.0, 50.0, 100.0, 200.0, 350.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0,
                    4000.0, 5000.0, 6500.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 20000.0, 30000.0])

# Regional radiosonde climatology (docs/sources/noaa_igra2/regional_profiles_2023-10_2026-09.json). The vapour scale
# height is tied to the surface vapour pressure by a log-log fit over the same soundings (vapour_scale_height).
LAPSE_K_PER_KM = (4.5, 7.7)          # 10-90 % of surface-500 hPa lapse rates
TROPOPAUSE_M = 16000.0               # C: subtropical tropopause; little matters above it in these bands
CO2_PPM = (419.0, 440.0)             # NOAA GML global mean 2023-2026 419-428 (S1); the top widened for the urban
                                     # boundary layer (C: no Lahore record)


def saturation_hpa(t_k):
    t_c = np.asarray(t_k, float) - 273.15
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))                  # Magnus over water (thermal/sky.py)


def vapour_scale_height(e_sfc_hpa, fit: dict | None = None, z_score: float = 0.0) -> float:
    """H_w [m] from the surface vapour pressure: ln H = a + b ln e, plus z_score x the residual spread (fit from
    tools/source_measurements.igra2_profiles; defaults are that fit)."""
    f = fit or VAPOUR_FIT
    return float(np.clip(math.exp(f["a"] + f["b"] * math.log(max(e_sfc_hpa, 0.5)) + z_score * f["sd"]),
                         f["h_min_m"], f["h_max_m"]))


# ln(H_w / m) = a + b ln(e_sfc / hPa) over 2,756 soundings (residual sd 0.37, a factor 1.45): ~1.1 km at 8 hPa, 1.7 km
# at 20 hPa, 2.3 km at 36 hPa; bounds = the 1-99 % range. Lahore's own 46 soundings (mostly October 2023) sit 0.38
# below the fit, inside the spread. tests/test_atmosphere_bands.py keeps these numbers equal to the JSON.
VAPOUR_FIT = dict(a=6.003, b=0.487, sd=0.369, h_min_m=628.0, h_max_m=3348.0)


class Profile:
    """Layered clear atmosphere above a station at the surface (z = 0 at the station)."""

    def __init__(self, t_sfc_k: float, e_sfc_hpa: float, p_sfc_hpa: float = 988.0, lapse_k_per_km: float = 5.7,
                 h_w_m: float = 1800.0, co2_ppm: float = 424.0, z_trop_m: float = TROPOPAUSE_M,
                 n2o_ppb: float = 338.0, ch4_ppb: float = 1990.0, co_ppb_surface: float = 300.0,
                 co_ppb_free: float = 100.0, boundary_layer_m: float = 1500.0):
        self.t_sfc, self.e_sfc, self.p_sfc = float(t_sfc_k), float(e_sfc_hpa), float(p_sfc_hpa)
        self.lapse, self.h_w, self.co2, self.z_trop = float(lapse_k_per_km) / 1000.0, float(h_w_m), float(co2_ppm), float(z_trop_m)
        self.n2o, self.ch4 = float(n2o_ppb), float(ch4_ppb)
        self.co_sfc, self.co_free, self.h_bl = float(co_ppb_surface), float(co_ppb_free), float(boundary_layer_m)
        z = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
        self.z_mid = z
        self.T = self.temperature(z)
        self.p_hpa = self.pressure(z)
        rho0 = self.e_sfc * 100.0 / (R_V * self.t_sfc)
        rho = rho0 * np.exp(-z / self.h_w)
        e = rho * R_V * self.T / 100.0
        self.e_hpa = np.minimum(e, saturation_hpa(self.T))

    def temperature(self, z):
        z = np.asarray(z, float)
        return self.t_sfc - self.lapse * np.minimum(z, self.z_trop)

    def pressure(self, z):
        z = np.asarray(z, float)
        T_t = self.t_sfc - self.lapse * self.z_trop
        low = self.p_sfc * (self.temperature(np.minimum(z, self.z_trop)) / self.t_sfc) ** (G0 / (R_D * self.lapse))
        return np.where(z <= self.z_trop, low, low * np.exp(-G0 * (z - self.z_trop) / (R_D * T_t)))

    def precipitable_water_mm(self) -> float:
        rho = self.e_hpa * 100.0 / (R_V * self.T)
        return float(np.sum(rho * np.diff(Z_EDGES)))                      # kg m-2 = mm

    def layer_state(self, i: int) -> dict:
        p = self.p_hpa[i] / 1013.25
        h2o = self.e_hpa[i] / 1013.25
        dry = p - h2o
        co = self.co_free + (self.co_sfc - self.co_free) * math.exp(-self.z_mid[i] / self.h_bl)     # ppb
        return dict(T=float(self.T[i]), p_total_atm=float(p), e_hpa=float(self.e_hpa[i]),
                    p_atm={"H2O": float(h2o), "CO2": float(dry * self.co2 * 1e-6), "N2": float(0.7808 * dry),
                           "O2": float(0.2095 * dry), "N2O": float(dry * self.n2o * 1e-9),
                           "CH4": float(dry * self.ch4 * 1e-9), "CO": float(dry * co * 1e-9)})


def ray_segments(prof: Profile, elevation_deg: float, z0_m: float = 0.0, max_dist_m: float | None = None) -> list:
    """Segments of a ray from height z0 at the given elevation through the layers (spherical shells), up to the top
    of the atmosphere or max_dist_m. Downward rays (elevation < 0) are followed until they reach the surface.
    Each segment: dict(layer index, length_m, start_m, end_m)."""
    e = math.radians(elevation_deg)
    r0 = R_EARTH + z0_m
    s_e = math.sin(e)
    out = []

    # every crossing of a layer boundary along the ray, in order; each interval between them lies in one layer
    events = []
    for zb in Z_EDGES:
        disc = (r0 * s_e) ** 2 - r0 ** 2 + (R_EARTH + zb) ** 2
        if disc < 0:
            continue
        root = math.sqrt(disc)
        for s in (-r0 * s_e - root, -r0 * s_e + root):
            if s > 1e-3:                 # the start shell's own root is 0 up to rounding (~1e-8 m at R = 6,371 km)
                events.append(s)
    events = sorted(set(round(s, 3) for s in events))
    s_prev = 0.0
    lim = max_dist_m if max_dist_m is not None else float("inf")
    for s in events + [float("inf")]:
        s_end = min(s, lim)
        if s_end <= s_prev:
            break
        mid = 0.5 * (s_prev + s_end) if math.isfinite(s_end) else s_prev + 1.0
        r_mid = math.sqrt(r0 ** 2 + mid ** 2 + 2 * r0 * mid * s_e)
        z_mid = r_mid - R_EARTH
        if z_mid < 0 or z_mid >= Z_EDGES[-1] or not math.isfinite(s_end):
            break
        i = int(np.clip(np.searchsorted(Z_EDGES, z_mid, side="right") - 1, 0, len(Z_EDGES) - 2))
        out.append(dict(layer=i, length_m=s_end - s_prev, start_m=s_prev, end_m=s_end))
        s_prev = s_end
        if s_end >= lim:
            break
    return out


@functools.lru_cache(maxsize=None)
def _mt_ckd():
    """MT_CKD 4.3 water-vapour continuum reference coefficients (docs/sources/aer_mt_ckd)."""
    import csv
    import os
    with open(os.path.join(gb.ROOT, "docs", "sources", "aer_mt_ckd", "mt_ckd_4.3_h2o_600-3000cm.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")][1:]
    a = np.array([[float(x) for x in r] for r in rows])
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


@functools.lru_cache(maxsize=None)
def _n2_cia():
    """N2 collision-induced fundamental (Lafferty et al. 1996 via LBLRTM; docs/sources/aer_lblrtm)."""
    import csv
    import os
    with open(os.path.join(gb.ROOT, "docs", "sources", "aer_lblrtm", "n2_cia_fundamental_lafferty1996.csv"),
              encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")][1:]
    a = np.array([[float(x) for x in r] for r in rows])
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


K_B = 1.380649e-23
LOSCHMIDT = 2.68675e19              # molecules cm-3 (LBLRTM's xlosmt)


def radiation_term(w, T):
    """MT_CKD / LBLRTM radiation term v tanh(hcv / 2kT) [cm-1]."""
    w = np.asarray(w, float)
    return w * np.tanh(gb.C2_WN * w / (2.0 * T))


def continuum_k_per_m(w, layer: dict, scale: float = 1.0):
    """Continuum absorption per metre at the layer's state: MT_CKD 4.3 water-vapour self and foreign continuum
    (mt_ckd_h2o_module.f90: C_self (296/T)^texp x_h2o rho_rat + C_for (1 - x_h2o) rho_rat, rho_rat = (p / 1013 mb)
    (296 K / T), times the radiation term and the vapour density), x `scale` (1 = MT_CKD 4.3; Roberts et al. 1976 is
    ~1.4 x stronger at 1000 cm-1), plus the N2 collision-induced fundamental near 4.3 um (LBLRTM n2_ver_1: N2-N2, -O2
    and -H2O partners, log interpolation in 1/T between 272 and 228 K)."""
    w = np.asarray(w, float)
    T = layer["T"]
    p_mb = layer["p_total_atm"] * 1013.25
    x_h2o = layer["e_hpa"] / p_mb
    wn, cs, cf, tx = _mt_ckd()
    rho_rat = (p_mb / 1013.0) * (296.0 / T)
    c = (np.interp(w, wn, cs) * (296.0 / T) ** np.interp(w, wn, tx) * x_h2o + np.interp(w, wn, cf) * (1.0 - x_h2o)) * rho_rat
    n_h2o = layer["e_hpa"] * 100.0 / (K_B * T) * 1e-6                    # molecules cm-3
    k = scale * c * radiation_term(w, T) * n_h2o * 100.0                  # cm-1 -> m-1
    # N2 CIA: tau = L rho_N2 rho_air (x_N2 k + x_O2 a_O2 k + x_H2O 9/7 a_H2O k) tanh(hcv / 2kT), densities in amagat
    v, k272, k228, a_h2o = _n2_cia()
    xt = (1.0 / T - 1.0 / 272.0) / (1.0 / 228.0 - 1.0 / 272.0)
    both = (k272 > 0) & (k228 > 0)
    kn = np.where(both, k272 * (np.where(both, k228, 1.0) / np.where(both, k272, 1.0)) ** xt,
                  k272 + (k228 - k272) * (T - 272.0) / (228.0 - 272.0))
    rho_air = p_mb * 100.0 / (K_B * T) * 1e-6 / LOSCHMIDT                # amagat
    x_n2, x_o2 = 0.7808 * (1 - x_h2o), 0.2095 * (1 - x_h2o)
    eff = x_n2 + x_o2 * (1.294 - 0.4545 * T / 296.0) + x_h2o * (9.0 / 7.0) * a_h2o
    kcia = np.interp(w, v, kn * eff, left=0.0, right=0.0) * rho_air * (x_n2 * rho_air) * np.tanh(gb.C2_WN * w / (2 * T))
    return k + kcia * 100.0                                               # cm-1 -> m-1


def _layer_segments(prof: Profile, segments: list) -> list:
    out = []
    for s in segments:
        L = prof.layer_state(s["layer"])
        out.append(dict(length_m=s["length_m"], T=L["T"], p_atm=L["p_atm"], _layer=L, _z=float(prof.z_mid[s["layer"]])))
    return out


def path_optics(prof: Profile, segments: list, w=W_GRID, aerosol: dict | None = None, cont_scale: float = 1.0,
                window: dict | None = None):
    """Cumulative optical depths at the end of each segment from the start of the ray, (n_seg, n_w):
    (gas incl. continua and the window absorber, aerosol extinction (grey, n_seg), segment temperatures).
    aerosol: {"k_per_m": surface extinction, "scale_height_m", "omega0"}; window: {"k_per_m": surface absorption,
    "scale_height_m", "w_lo", "w_hi"}, a grey absorber in [w_lo, w_hi] cm-1 (see sample_params)."""
    w = np.asarray(w, float)
    segs = _layer_segments(prof, segments)
    x = gb.path_depths(w, segs)
    x = x + np.cumsum([continuum_k_per_m(w, g["_layer"], cont_scale) * g["length_m"] for g in segs], axis=0)
    if window and window.get("k_per_m", 0.0) > 0:
        inside = (w >= window["w_lo"]) & (w <= window["w_hi"])
        H = window["scale_height_m"]
        x = x + np.cumsum([window["k_per_m"] * math.exp(-g["_z"] / H) * g["length_m"] for g in segs])[:, None] * inside
    x_aer = np.zeros(len(segs))
    if aerosol and aerosol.get("k_per_m", 0.0) > 0:
        H = aerosol.get("scale_height_m") or 1e9
        x_aer = np.cumsum([aerosol["k_per_m"] * math.exp(-g["_z"] / H) * g["length_m"] for g in segs])
    return x, x_aer, np.array([g["T"] for g in segs])


def _emission(x_gas, x_aer, T, w, omega0: float):
    """Radiance arriving at the start of a ray (SP-3080 eq. A-3) and the transmittance to its far end: each segment
    emits B(T) times its own absorptance (gas plus the absorbing share 1 - omega0 of the aerosol), attenuated by the
    total extinction in front of it."""
    tau = np.exp(-(x_gas + x_aer[:, None]))
    tau_prev = np.vstack([np.ones_like(w), tau[:-1]])
    x_abs = x_gas + (1.0 - omega0) * x_aer[:, None]
    seg_abs = 1.0 - np.exp(-np.diff(np.vstack([np.zeros_like(w), x_abs]), axis=0))
    return np.sum(gb.planck_wn(T[:, None], w[None, :]) * seg_abs * tau_prev, axis=0), tau[-1]


def ray_radiance(prof: Profile, elevation_deg: float, w=W_GRID, aerosol: dict | None = None, z0_m: float = 0.0,
                 max_dist_m: float | None = None, background=None, cont_scale: float = 1.0, window: dict | None = None):
    """Spectral radiance arriving at the observer along a ray [W m-2 sr-1 (cm-1)-1] and the spectral transmittance to
    the ray's far end; background: spectral radiance behind the far end (None: space)."""
    w = np.asarray(w, float)
    segs = ray_segments(prof, elevation_deg, z0_m, max_dist_m)
    if not segs:
        return (np.zeros_like(w) if background is None else np.asarray(background, float)), np.ones_like(w)
    x_gas, x_aer, T = path_optics(prof, segs, w, aerosol, cont_scale, window)
    L, tau = _emission(x_gas, x_aer, T, w, (aerosol or {}).get("omega0", 0.0))
    if background is not None:
        L = L + np.asarray(background, float) * tau
    return L, tau


def split_segments(segments: list, distances) -> tuple:
    """Subdivide ray segments at the given distances; returns (segments, index of the segment ending at each
    distance)."""
    cuts = sorted(float(d) for d in distances)
    out, ends = [], []
    for s in segments:
        a, b = s["start_m"], s["end_m"]
        pts = [a] + [c for c in cuts if a < c < b] + [b]
        for x0, x1 in zip(pts[:-1], pts[1:]):
            out.append(dict(s, start_m=x0, end_m=x1, length_m=x1 - x0))
    for d in cuts:
        ends.append(min(range(len(out)), key=lambda i: abs(out[i]["end_m"] - d)))
    return out, ends


def path_transmittance(prof: Profile, elevation_deg: float, distances, w, weight, z0_m: float = 0.0,
                       cont_scale: float = 1.0, window: dict | None = None):
    """Gas transmittance (no aerosol) from z0 along a ray to each distance, averaged over the band with the spectral
    weight (e.g. dB/dT at the air temperature for a near-ambient contrast, or a source spectrum)."""
    d = np.atleast_1d(np.asarray(distances, float))
    segs = ray_segments(prof, elevation_deg, z0_m, float(d.max()) * 1.000001)
    if not segs:
        return np.ones_like(d)
    segs, ends = split_segments(segs, d)
    x_gas, _, _ = path_optics(prof, segs, w, None, cont_scale, window)
    wt = np.asarray(weight, float)
    return np.array([float(np.sum(np.exp(-x_gas[i]) * wt) / np.sum(wt)) for i in ends])


def band_mask(band_um, w=W_GRID):
    lo, hi = 1e4 / band_um[1], 1e4 / band_um[0]
    return (np.asarray(w) >= lo - 1e-9) & (np.asarray(w) <= hi + 1e-9)


def band_w(band_um) -> np.ndarray:
    """The 5 cm-1 interval centres inside a band."""
    return W_GRID[band_mask(band_um)]


def band_integral(spec, band_um=None, w=None):
    """In-band integral of a spectral quantity per cm-1 (rectangular response, 5 cm-1 intervals); with band_um the
    spectrum is on W_GRID (or w) and is masked, without it the spectrum is already in-band."""
    a = np.asarray(spec, float)
    if band_um is not None:
        a = a[..., band_mask(band_um, W_GRID if w is None else w)]
    return np.sum(a, axis=-1) * W_STEP


def planck_band(T, band_um, w=None):
    ww = band_w(band_um) if w is None else np.asarray(w, float)
    return float(np.sum(gb.planck_wn(T, ww)) * W_STEP)


@functools.lru_cache(maxsize=None)
def _e490_grid():
    """ASTM E490 AM0 spectral irradiance on W_GRID [W m-2 (cm-1)-1] (docs/sources/astm, 1 au): the committed table
    on this grid (tools/source_measurements.e490_model_grid), else derived from the git-ignored workbook export."""
    import csv
    import os
    grid = os.path.join(gb.ROOT, "docs", "sources", "astm", "e490_am0_model_grid.csv")
    if os.path.exists(grid):
        with open(grid, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        w = np.array([float(r["wavenumber_cm1"]) for r in rows])
        if len(w) == len(W_GRID) and np.array_equal(w, W_GRID):
            return np.array([float(r["e490_w_m2_per_cm1"]) for r in rows])
    path = os.path.join(gb.ROOT, "docs", "sources", "astm", "ASTM_E490_AM0_from_NREL_workbook.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))[2:]
    a = np.array([[float(x) for x in r] for r in rows])
    lam_nm, e_nm = a[:, 0], a[:, 1]                                        # W m-2 nm-1
    lam = 1e7 / W_GRID                                                     # nm
    e = np.exp(np.interp(np.log(lam), np.log(lam_nm), np.log(np.maximum(e_nm, 1e-300))))
    return e * lam ** 2 / 1e7                                              # E_w = E_lambda lambda^2 / 1e7


def e490_wn(w=None):
    e = _e490_grid()
    return e if w is None else np.interp(np.asarray(w, float), W_GRID, e)


def solar_direct(prof: Profile, sun_elevation_deg: float, w=W_GRID, aerosol: dict | None = None, z0_m: float = 0.0,
                 cont_scale: float = 1.0, window: dict | None = None):
    """Direct solar spectral irradiance at height z0 on a surface facing the sun [W m-2 (cm-1)-1] and the spectral
    transmittance of the sun's path (ASTM E490 at 1 au; aerosol included when given)."""
    w = np.asarray(w, float)
    if sun_elevation_deg <= 0.0:
        return np.zeros_like(w), np.zeros_like(w)
    segs = ray_segments(prof, sun_elevation_deg, z0_m)
    x_gas, x_aer, _ = path_optics(prof, segs, w, aerosol, cont_scale, window)
    tau = np.exp(-(x_gas[-1] + x_aer[-1]))
    return e490_wn(w) * tau, tau


def hg_phase(cos_theta, g):
    """Henyey-Greenstein phase function, normalised to 4 pi over the sphere (P = 1 isotropic)."""
    return (1.0 - g * g) / (1.0 + g * g - 2.0 * g * np.asarray(cos_theta, float)) ** 1.5


def single_scatter_sky(e_sun_band: float, tau_aer: float, omega0: float, g: float, mu_view, mu_sun: float, cos_scatter):
    """Diffuse sky radiance at the ground from aerosol single scattering of the direct beam in a plane-parallel layer of
    optical depth tau_aer (Liou, An Introduction to Atmospheric Radiation, 2nd ed. 2002, section 6.4: the single-
    scattering diffuse transmission): omega0 F0 P / (4 pi) mu0 / (mu0 - mu) (exp(-tau / mu0) - exp(-tau / mu)), mu the
    cosine of the view zenith, mu0 that of the sun; F0 the beam irradiance above the aerosol. W m-2 sr-1."""
    mu = np.clip(np.asarray(mu_view, float), 1e-3, 1.0)
    mu0 = max(mu_sun, 1e-3)
    P = hg_phase(cos_scatter, g)
    near = np.abs(mu - mu0) < 1e-4
    a = np.where(near, tau_aer / mu0 * np.exp(-tau_aer / mu0),
                 mu0 / np.where(near, 1.0, mu0 - mu) * (np.exp(-tau_aer / mu0) - np.exp(-tau_aer / mu)))
    return omega0 * e_sun_band * P / (4.0 * math.pi) * a


# ============================================================================ per scene
GL_MU = np.array([0.0694318, 0.3300095, 0.6699905, 0.9305682])            # Gauss-Legendre on [0, 1]
GL_W = np.array([0.1739274, 0.3260726, 0.3260726, 0.1739274])
WINDOW_WN = (714.0, 1300.0)        # the grey window absorber spans both LWIR bands (8-14 and 7.7-9.3 um)


def hemispherical_emittance(prof: Profile, band_um, cont_scale=1.0, window=None, aerosol=None) -> float:
    """Clear-sky flux emittance in a band: 2 int eps(mu) mu dmu (4-point Gauss-Legendre in mu)."""
    w = band_w(band_um)
    eps = [float(np.sum(ray_radiance(prof, math.degrees(math.asin(m)), w, aerosol, cont_scale=cont_scale,
                                     window=window)[0]) * W_STEP) for m in GL_MU]
    return float(2.0 * np.sum(GL_W * GL_MU * np.array(eps)) / planck_band(prof.t_sfc, band_um))


def window_absorber_for(prof: Profile, target_eps: float, cont_scale: float, scale_height_m: float,
                        band_um=(8.0, 14.0)) -> float:
    """Surface absorption coefficient [m-1] of the grey window absorber that lifts the clear-sky 8-14 um flux
    emittance to target_eps (bisection on the zenith optical depth, 0-3)."""
    lo, hi = 0.0, 3.0
    if hemispherical_emittance(prof, band_um, cont_scale) >= target_eps:
        return 0.0
    for _ in range(10):                                  # zenith depth to 0.003
        mid = 0.5 * (lo + hi)
        win = dict(k_per_m=mid / scale_height_m, scale_height_m=scale_height_m, w_lo=WINDOW_WN[0], w_hi=WINDOW_WN[1])
        if hemispherical_emittance(prof, band_um, cont_scale, win) < target_eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / scale_height_m


def _logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_params(rng, cfg: dict, air_temp_c: float, e_hpa: float, p_sfc_hpa: float | None = None) -> dict:
    """Per-scene draws of the thermal atmosphere (render.yaml thermal_atmosphere); cheap: no radiative transfer."""
    z = float(np.clip(rng.normal(), -2.0, 2.0))
    return dict(
        model="layered_snb_v1", t_sfc_k=float(air_temp_c) + 273.15, e_sfc_hpa=float(e_hpa),
        p_sfc_hpa=float(p_sfc_hpa if p_sfc_hpa is not None else cfg["surface_pressure_hpa"]),
        lapse_k_per_km=float(rng.uniform(*cfg["lapse_k_per_km"])), h_w_z=z,
        h_w_m=vapour_scale_height(e_hpa, z_score=z), co2_ppm=float(rng.uniform(*cfg["co2_ppm"])),
        n2o_ppb=float(rng.uniform(*cfg["n2o_ppb"])), ch4_ppb=float(rng.uniform(*cfg["ch4_ppb"])),
        co_ppb_surface=_logu(rng, *cfg["co_ppb_surface"]), co_ppb_free=float(cfg["co_ppb_free"]),
        boundary_layer_m=float(cfg["boundary_layer_m"]), cont_scale=float(rng.uniform(*cfg["continuum_scale"])),
        window_u=float(rng.uniform(0.0, 1.0)), aerosol_u=float(rng.uniform(0.0, 1.0)),
        aerosol_g=float(rng.uniform(*cfg["aerosol"]["asymmetry_g"])))


def profile_from(params: dict) -> Profile:
    return Profile(params["t_sfc_k"], params["e_sfc_hpa"], params["p_sfc_hpa"], params["lapse_k_per_km"],
                   params["h_w_m"], params["co2_ppm"], n2o_ppb=params["n2o_ppb"], ch4_ppb=params["ch4_ppb"],
                   co_ppb_surface=params["co_ppb_surface"], co_ppb_free=params["co_ppb_free"],
                   boundary_layer_m=params["boundary_layer_m"])


SKY_EL_DEG = (0.25, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)
PATH_DIST_M = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0, 15000.0,
               20000.0)


def band_aerosol(params: dict, cfg: dict, band: str, visibility_km: float, band_ratio: float) -> dict:
    """Grey aerosol of a thermal band: extinction 3.912 / V x the scene's band ratio (data/composite.py), single-
    scattering albedo omega0 at the scene's shared aerosol quantile inside the band's range (render.yaml)."""
    lo, hi = cfg["aerosol"]["omega0"][band]
    return dict(k_per_m=3.912 / (visibility_km * 1000.0) * band_ratio, scale_height_m=float(cfg["aerosol"]["scale_height_m"]),
                omega0=float(lo + params["aerosol_u"] * (hi - lo)))


def scene_tables(params: dict, cfg: dict, bands: dict, air_temp_c: float, sun: dict | None, weather: dict,
                 objects: list, camera_height_m: float = 0.0) -> dict:
    """Everything a thermal render pass and the loader need, per band (bands: {name: (lo_um, hi_um)}):
    sky radiance by elevation (gas + aerosol emission), the air's in-band radiance, the direct sun at the ground,
    gas transmittance tables for horizontal paths and for each labelled object's slant path, the MWIR single-
    scattering inputs. Radiances in W m-2 sr-1, irradiances in W m-2."""
    prof = profile_from(params)
    out = dict(params=params, bands={})
    # the window absorber: drawn between the clear-air model and Idso (1981), as a share u of the gap (sample_params)
    from thermal import sky as thsky
    eps_model = hemispherical_emittance(prof, (8.0, 14.0), params["cont_scale"])
    eps_idso = float(thsky.zenith_emissivity(params["t_sfc_k"], params["e_sfc_hpa"]))
    target = eps_model + params["window_u"] * max(0.0, eps_idso - eps_model)
    H = float(cfg["window_absorber_scale_height_m"])
    k_win = window_absorber_for(prof, target, params["cont_scale"], H) if target > eps_model + 1e-4 else 0.0
    window = dict(k_per_m=k_win, scale_height_m=H, w_lo=WINDOW_WN[0], w_hi=WINDOW_WN[1])
    out["window"] = dict(eps_hemispherical_model=eps_model, eps_hemispherical_idso=eps_idso, target=target,
                         k_per_m=k_win, scale_height_m=H, zenith_tau=k_win * H)
    t_air = params["t_sfc_k"]
    for name, band_um in bands.items():
        w = band_w(band_um)
        aer = band_aerosol(params, cfg, name, weather["visibility_km"], weather["band_ratio"].get(name, 0.2))
        win = window if w.min() < WINDOW_WN[1] else None
        L_sky = [float(np.sum(ray_radiance(prof, el, w, aer, camera_height_m, cont_scale=params["cont_scale"],
                                           window=win)[0]) * W_STEP) for el in SKY_EL_DEG]
        dbdt = gb.planck_wn(t_air + 0.5, w) - gb.planck_wn(t_air - 0.5, w)
        horiz = path_transmittance(prof, 0.0, PATH_DIST_M, w, dbdt, camera_height_m, params["cont_scale"], win)
        objs = {}
        for o in objects:
            objs[o["name"]] = float(path_transmittance(prof, max(float(o["elevation_deg"]), 0.0), [o["range_m"]], w,
                                                       dbdt, camera_height_m, params["cont_scale"], win)[0])
        b = dict(band_um=list(band_um), sky_elevation_deg=list(SKY_EL_DEG), sky_radiance=L_sky,
                 L_air=planck_band(t_air, band_um), path_distance_m=list(PATH_DIST_M), tau_horizontal=horiz.tolist(),
                 tau_objects=objs, aerosol=aer, e_sun_toa=float(np.sum(e490_wn(w)) * W_STEP))
        if sun and sun.get("elevation_deg", 0.0) > 0.0:
            el = float(sun["elevation_deg"])
            e_dir, _ = solar_direct(prof, el, w, aer, cont_scale=params["cont_scale"], window=win)
            e_gas, _ = solar_direct(prof, el, w, None, cont_scale=params["cont_scale"], window=win)
            b["e_sun_ground"] = float(np.sum(e_dir) * W_STEP)                 # direct beam, gas + aerosol
            b["e_sun_above_aerosol"] = float(np.sum(e_gas) * W_STEP)          # gas only: the scattering source
            b["tau_aerosol_vertical"] = aer["k_per_m"] * aer["scale_height_m"]
        else:
            b["e_sun_ground"] = 0.0
        out["bands"][name] = b
    return out


def sky_radiance_at(table: dict, elevation_deg) -> np.ndarray:
    """Interpolate a band's sky table in elevation (log-elevation below 10 deg); below the horizon: None."""
    el = np.clip(np.asarray(elevation_deg, float), table["sky_elevation_deg"][0], 90.0)
    return np.interp(np.log(el), np.log(table["sky_elevation_deg"]), table["sky_radiance"])


def horizontal_transmittance(table: dict, dist_m) -> np.ndarray:
    d = np.asarray(dist_m, float)
    lg = np.log(np.clip(d, 1.0, None))
    xs = np.log(np.concatenate([[1.0], table["path_distance_m"]]))
    ys = np.concatenate([[1.0], table["tau_horizontal"]])
    return np.interp(lg, xs, ys)


def window_from(tables: dict) -> dict | None:
    """The scene's window absorber (scene_tables) in path_optics form, or None."""
    wt = (tables or {}).get("window") or {}
    if wt.get("k_per_m", 0.0) <= 0:
        return None
    return dict(k_per_m=wt["k_per_m"], scale_height_m=wt.get("scale_height_m", 1500.0), w_lo=WINDOW_WN[0],
                w_hi=WINDOW_WN[1])


def ensure_tables(spec: dict, render_cfg: dict) -> dict | None:
    """The scene's thermal tables (scene_tables), computed once and kept in spec["thermal"]["tables"]; None when the
    scene has no layered thermal atmosphere (older scenes, or thermal_atmosphere disabled)."""
    th = spec.get("thermal")
    if not th or not th.get("atmosphere"):
        return None
    if not th.get("tables"):
        th["tables"] = scene_tables(th["atmosphere"], render_cfg["thermal_atmosphere"],
                                    {b: tuple(v) for b, v in th["bands_um"].items()}, spec["air_temp_c"],
                                    spec["sun"] if spec.get("lighting") != "night" else None, spec["weather"],
                                    spec["objects"], float(spec["camera"]["position"][2]))
    return th["tables"]


def gas_table(meta_or_spec: dict, band: str) -> dict | None:
    """A band's thermal table from a scene's spec or metadata (extra.thermal), or None (legacy LWIR model)."""
    th = meta_or_spec.get("thermal") or (meta_or_spec.get("extra") or {}).get("thermal") or {}
    tab = ((th.get("tables") or {}).get("bands") or {}).get(band)
    if tab is None or (band == "lwir" and not th.get("layered_lwir")):
        return None
    return tab
