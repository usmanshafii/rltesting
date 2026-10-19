"""Infrared absorption and emission of water vapour and CO2: the statistical narrow-band model of NASA SP-3080
(Ludwig, Malkmus, Reardon and Thomson, 1973, "Handbook of infrared radiation from combustion gases") as implemented in
NIST's RADCAL (Grosshandler 1993; github.com/firemodels/radcal, Source/rcal.f90, commit dbc8d81 of 2025-04-24),
ported to NumPy. Used for the cooled-LWIR and MWIR bands: clear-sky radiance, path transmittance and the exhaust
plume (CLAUDE.md sensor set rule; docs/thermal_bands.md). Pure NumPy.

Data (docs/sources/nasa_sp3080, docs/sources/nist_radcal):
  H2O    mean absorption coefficient k(T, w) at STP, 50-9425 cm-1 every 25 cm-1, 300-2500 K: SP-3080 Table A2-35
         (RADCAL's copy, five entries corrected to the printed table: tools/source_measurements.radcal_tables);
         fine-structure parameter 1/d: RADCAL's fit to SP-3080 (section 5.2.2.3).
  CO2    RADCAL's band models: the 4.3 um fundamental (harmonic-oscillator / rigid-rotor model with the band strength
         2700 cm-2 atm-1 at STP, SP-3080 Table 5-13), scaled by 273/300 onto SP-3080's printed Table A2-29 (see
         SP3080_SCALE_43; checked at 300, 600 and 1200 K by tests/test_gas_bands.py),
         the 9.4 / 10.4 um bands, and the 15 um table (co2_15um_k.csv). The 2.7 and 2.0 um bands lie outside the
         project's thermal bands and are not ported (SWIR is deferred).
  CO     the 4.7 um fundamental from SP-3080's printed Tables A2-2 (k) and A2-3 (1/d) at 300, 600 and 1200 K
         (docs/sources/nasa_sp3080/co_tablesA2-2_A2-3.csv): log k and 1/d linear in wavenumber, both linear in
         temperature (the exhaust plume of a rich-running two-stroke carries a few per cent CO).
  Air    N2O, CH4 and the air's own CO (200-320 K): narrow-band tables built from the HITRAN2020 line list
         (docs/sources/hitran; tools/source_measurements.hitran_snb), with HITRAN's air-broadened half-widths. CO above
         320 K (the exhaust) comes from SP-3080 (blended over 300-340 K); HITRAN's CO agrees with SP-3080's printed
         300 K column within about 20 % across the band core and 7 % in band strength (tests/test_gas_bands.py).
  Widths collision half-width, SP-3080 eq. 5-34 with RADCAL's broadening coefficients (SP-3080 Table 5-19); Doppler
         half-width 5.94e-6 w sqrt(T / 273 M), eq. 5-35.

Model, per narrow interval: the Goody random band model of Lorentz lines, X_c = X* sqrt(a_c) / sqrt(a_c + X*/4), and of
Doppler lines, X_D = 1.7 a_D sqrt(ln(1 + (0.588 X* / a_D)^2)), combined by SP-3080 eqs. 5-25 and 5-26; along a
non-uniform path the Curtis-Godson averages of a_c and a_D weighted by X* (SP-3080 eqs. A-5 to A-7). X* = sum k u, the
absorber amount u = p L 273 / T (cm atm at STP). Transmittance tau = exp(-sum over species X). The water-vapour
continuum and aerosols are not part of this model: thermal/atmosphere.py adds them.
"""
from __future__ import annotations

import csv
import functools
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RADCAL = os.path.join(ROOT, "docs", "sources", "nist_radcal")
CACHE = os.path.join(HERE, "cache")

C1_WN = 1.191042972e-8       # 2 h c^2 [W m-2 sr-1 (cm-1)^-4], CODATA 2018 exact constants
C2_WN = 1.438776877          # h c / k [cm K]
Q2 = 1.4388                  # RADCAL's h c / k [cm K] inside the band models (kept as in the code)
TINY = 1e-300

# SP-3080 eq. 5-34 broadening coefficients (RADCAL GAMMA): absorber -> {broadener: gamma [cm-1 atm-1]};
# "self" is the resonant self-broadening term, which scales as 273/T instead of sqrt(273/T).
GAMMA = {"CO2": {"CO2": 0.09, "H2O": 0.07, "CO": 0.06, "O2": 0.055, "N2": 0.07, "self": 0.01},
         "H2O": {"CO2": 0.12, "H2O": 0.09, "CO": 0.10, "O2": 0.04, "N2": 0.09, "self": 0.44},
         "CO": {"CO2": 0.07, "H2O": 0.06, "CO": 0.06, "O2": 0.05, "N2": 0.06, "self": 0.0}}
MOLAR_MASS = {"CO2": 44.0, "H2O": 18.0, "CO": 28.0, "N2O": 44.0, "CH4": 16.0}
SP3080 = os.path.join(ROOT, "docs", "sources", "nasa_sp3080")


def planck_wn(T, w):
    """Black-body spectral radiance per wavenumber [W m-2 sr-1 (cm-1)-1]; w in cm-1."""
    T = np.asarray(T, float)
    w = np.asarray(w, float)
    x = np.clip(C2_WN * w / np.maximum(T, 1e-9), None, 700.0)
    return C1_WN * w ** 3 / np.expm1(x)


# ============================================================================ tables
@functools.lru_cache(maxsize=None)
def _table(fname):
    with open(os.path.join(RADCAL, fname), encoding="utf-8") as f:
        rows = list(csv.reader(f))
    temps = [float(h.split("_")[1].rstrip("K")) for h in rows[0][1:]]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    return np.array(temps), data[:, 0], data[:, 1:].T          # temperatures, wavenumbers, k[T, w]


def _bilinear(temps, wns, table, T, w):
    """Linear in temperature between the table rows, linear in wavenumber (RADCAL's interpolation)."""
    T = np.clip(np.asarray(T, float), temps[0], temps[-1] - 1e-6)
    i = np.clip(np.searchsorted(temps, T, side="right") - 1, 0, len(temps) - 2)
    tt = (T - temps[i]) / (temps[i + 1] - temps[i])
    dw = wns[1] - wns[0]
    jf = (np.asarray(w, float) - wns[0]) / dw
    j = np.clip(np.floor(jf).astype(int), 0, len(wns) - 2)
    ww = np.clip(jf - j, 0.0, 1.0)
    return (table[i, j] * (1 - tt) * (1 - ww) + table[i + 1, j] * tt * (1 - ww)
            + table[i, j + 1] * (1 - tt) * ww + table[i + 1, j + 1] * tt * ww)


def doppler_halfwidth(w, T, molar_mass):
    """SP-3080 eq. 5-35 [cm-1]."""
    return 5.94e-6 * np.asarray(w, float) * np.sqrt(np.asarray(T, float) / (273.0 * molar_mass))


def collision_halfwidth(absorber: str, T, p_atm: dict):
    """SP-3080 eq. 5-34 with RADCAL's coefficients: sum over broadeners gamma_j p_j sqrt(273/T) + gamma_self p_i 273/T."""
    g = GAMMA[absorber]
    a = 273.0 / np.asarray(T, float)
    out = sum(g[k] * p_atm.get(k, 0.0) for k in ("CO2", "H2O", "CO", "O2", "N2")) * np.sqrt(a)
    return out + g["self"] * p_atm.get(absorber, 0.0) * a


# ============================================================================ H2O
def h2o_params(w, T, gc):
    """(k [cm-1 atm-1 at STP], a_c = gamma_c / d, a_D = gamma_D / d) of water vapour, 50-9300 cm-1."""
    w = np.asarray(w, float)
    T = np.asarray(T, float) * np.ones_like(w)
    temps, wns, tab = _table("h2o_k_sp3080.csv")
    k = _bilinear(temps, wns, tab, T, w)
    tt = np.clip(T, 300.0, 2499.99)
    d = -2.294 + 0.3004e-2 * tt - 0.366e-6 * T ** 2                     # as in RADCAL (TTEMP, then TEMP)
    dinv = np.exp(0.7941 * np.sin(0.0036 * w - 8.043) + d)              # 1/d [cm], fit to SP-3080
    inside = (w >= 50.0) & (w < 9300.0)
    k = np.where(inside, k, 0.0)
    return k, np.where(inside, gc * dinv, 1.0), np.where(inside, doppler_halfwidth(w, T, 18.0) * dinv, 1.0)


# ============================================================================ CO2
BE_CO2 = 0.391635            # rotational constant [cm-1]
OM1, OM2, OM3 = 1354.91, 673.0, 2396.49
A_ROT = 0.0030875
X13, X23, X33 = -19.37, -12.53, -12.63


def _co2_43(w: float, T: float):
    """RADCAL CO2, 1975 < w <= 2474 cm-1: the 4.3 um band (dipole approximation, alpha = 2700) below 2395 cm-1,
    and above it the (000)-(021)/(010)-(031) and (000)-(101)/(010)-(111) transitions. Returns (S/d at T [cm-1 atm-1
    per unit density], 1/d [cm]) before the STP conversion; a direct line-by-line port of the Fortran loops."""
    T0 = 300.0
    B, A = BE_CO2, A_ROT
    xbar = 0.5 * (0.5 * X13 + X23)
    om12 = 0.5 * (0.5 * OM1 + OM2)
    sdweak = 0.0
    sdstrg = 0.0
    passes = [(2700.0, OM3, "43")] if w <= 2395.0 else [(28.5, 2.0 * OM2 + OM3, "27a"), (42.3, OM1 + OM3, "27b")]
    for alpha, omprim, kind in passes:
        aa = alpha * B * Q2 / (A * (1.0 - math.exp(-OM3 * Q2 / T0)) * (1.0 - math.exp(-om12 * Q2 / T0)) ** 3
                               * (1.0 + math.exp(-om12 * Q2 / T0)) * (1.0 - math.exp(-omprim * Q2 / T0)))
        bb = ((1.0 - math.exp(-Q2 * w / T)) * (1.0 - math.exp(-Q2 * OM3 / T)) * (1.0 - math.exp(-om12 * Q2 / T)) ** 3
              * (1.0 + math.exp(-om12 * Q2 / T)) * (1.0 - math.exp(-Q2 * omprim / T)))
        cc = aa * bb * w * T0 / T ** 2
        for j in range(1, 21):
            v = float(j - 1)
            g = 0.25 * (v + 1.0) * (v + 3.0) if j % 2 == 0 else 0.25 * (v + 2.0) * (v + 2.0)
            if kind != "43":
                vbar1 = -1.0 + (v + 5.0) / 6.0 if j % 2 == 0 else -1.0 + (v + 3.0) * (v + 4.0) / (6.0 * (v + 2.0))
            for k in range(1, 11):
                v3 = float(k - 1)
                if kind == "43":
                    qq = (v3 + 1.0) * g * math.exp(-(v3 * OM3 + v * om12) * Q2 / T)
                    omvv3 = OM3 + 0.5 * X13 + X23 + 2.0 * X33 + xbar * v + 2.0 * X33 * v3
                else:
                    qq = (v3 + 1.0) * g * math.exp(-(v3 * OM3 + v * om12) * Q2 / T) * (vbar1 + 1.0)
                    if kind == "27b":
                        omvv3 = 3715.0 - 47.0 * v3 if v <= 0.0 else 3728.0 - 5.0 * v - 47.0 * v3
                    else:
                        omvv3 = 3613.0 - 47.0 * v3 if v <= 0.0 else 3598.0 - 18.0 * v - 47.0 * v3
                gam = B - A * (v3 + 1.0)
                delta = A * (w - omvv3)
                if gam * gam <= delta:
                    break
                dd = 2.0 * math.sqrt(gam ** 2 - delta)
                omvbar = omvv3 * (1.0 - math.exp(-omvv3 * Q2 / T))
                f1, f2 = gam - 0.5 * dd, gam + 0.5 * dd
                ee = Q2 * gam / (A * A * T)
                if ee * delta * (1.0 + 0.5 * A / gam) <= -78.0 or ee * 2.0 * gam * f1 >= 78.0:
                    break
                ff = math.exp(ee * delta * (1.0 + 0.5 * A / gam))
                sminus = cc * qq / omvbar * abs(f1) * ff * math.exp(-ee * 2.0 * gam * f1)
                splus = 0.0 if ee * 2.0 * gam * f2 >= 78.0 else cc * qq / omvbar * abs(f2) * ff * math.exp(-ee * 2.0 * gam * f2)
                gg = sdweak
                sdweak = (sminus + splus) / dd + sdweak
                test = (sdweak - gg) / sdweak if sdweak > 0.0 else float("nan")
                if test < 1.0e-4:
                    break
                sdstrg = math.sqrt(0.5 * g) * (math.sqrt(sminus) + math.sqrt(splus)) / dd + sdstrg
    if sdweak <= 1e-300:
        return 0.0, None
    return sdweak, sdstrg * sdstrg / sdweak


# RADCAL's 4.3 um model runs high against SP-3080's own printed Table A2-29 by the same factor at every temperature
# (band integral 2200-2395 cm-1: x1.096 at 300 K, x1.095 at 600 K, x1.101 at 1200 K), which is 300/273 = 1.099 to
# within 0.3 %: most likely a 300 K instead of 273 K reference amount somewhere in the code (not traced). The printed
# table is the source (CLAUDE.md evidence rule 5), so the model is scaled by 273/300. It then reproduces the table's
# band integral within 0.4 % at 300, 600 and 1200 K, and point by point with an rms of 9, 4 and 4 % where k is at
# least a tenth of the peak (worst: the 2350 cm-1 gap between the branches at 300 K, +29 %; the band wings, where k
# is under a hundredth of the peak, differ by up to a factor 2). docs/sources/nasa_sp3080/co2_43um_k_tableA2-29.csv,
# tests/test_gas_bands.py.
SP3080_SCALE_43 = 273.0 / 300.0


def _co2_scalar(w: float, T: float):
    """(k at STP [cm-1 atm-1], 1/d [cm]) of CO2 at one wavenumber, RADCAL's branches below 2474 cm-1."""
    if 1975.0 < w <= 2474.0:
        s, dinv = _co2_43(w, T)
        return (s * T / 273.0 * SP3080_SCALE_43, dinv) if dinv is not None else (0.0, None)
    if 880.0 < w <= 1100.0:
        bcnt = (960.8, 1063.6)
        oma, omb = OM3, (OM1 + 2.0 * OM2) / 2.0
        s = 0.0
        for atot0, bc in zip((0.0219, 0.0532), bcnt):
            atot = (300.0 / T * atot0 * math.exp(Q2 * omb * (1.0 / 300.0 - 1.0 / T)) * (1.0 - math.exp(-Q2 * (oma - omb) / T))
                    / ((1.0 - math.exp(-Q2 * oma / T)) * (1.0 - math.exp(-omb * Q2 / T))))
            s += atot * Q2 / (4.0 * BE_CO2 * T) * abs(w - bc) * math.exp(-Q2 / (4.0 * BE_CO2 * T) * (w - bc) ** 2)
        return s * T / 273.0, 1.0 / (4.0 * BE_CO2)
    if 500.0 < w <= 880.0:
        temps, wns, tab = _table("co2_15um_k.csv")
        tt = min(max(T, 300.0), 2399.99)
        k = float(_bilinear(temps, wns, tab, tt, w))
        if k <= 1e-300:
            return 0.0, None
        t1, t2, t3, d1, d2, d3 = 300.0, 550.0, 830.0, 1.2, 8.0, 30.0       # RADCAL's quadratic for 1/d
        dinv = (d1 * (tt - t2) * (tt - t3) / ((t1 - t2) * (t1 - t3)) + d2 * (tt - t1) * (tt - t3) / ((t2 - t1) * (t2 - t3))
                + d3 * (tt - t1) * (tt - t2) / ((t3 - t1) * (t3 - t2)))
        return k, dinv
    return 0.0, None


T_GRID = np.arange(180.0, 1201.0, 5.0)        # K: atmosphere to exhaust plume
W_STEP = 5.0


@functools.lru_cache(maxsize=None)
def _co2_grid():
    """(k, 1/d) of CO2 on a 5 cm-1 x 5 K grid, 500-2475 cm-1 and 180-1200 K, computed once and cached."""
    path = os.path.join(CACHE, "co2_snb_radcal_sp3080scaled_500-2475cm_180-1200K.npz")
    w = np.arange(500.0, 2475.0 + 0.1, W_STEP)
    if os.path.exists(path):
        z = np.load(path)
        if z["k"].shape == (len(T_GRID), len(w)):
            return w, z["k"], z["dinv"]
    k = np.zeros((len(T_GRID), len(w)))
    dinv = np.ones_like(k)
    for i, T in enumerate(T_GRID):
        for j, ww in enumerate(w):
            kk, dd = _co2_scalar(float(ww), float(T))
            k[i, j] = kk
            dinv[i, j] = dd if dd is not None else 1.0
    try:
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, k=k, dinv=dinv)
    except OSError:
        pass
    return w, k, dinv


def co2_params(w, T, gc):
    """(k at STP [cm-1 atm-1], a_c, a_D) of CO2 at wavenumbers w and temperature T (interpolated on the cached grid)."""
    wg, kg, dg = _co2_grid()
    w = np.asarray(w, float)
    T = np.asarray(T, float) * np.ones_like(w)
    k = _bilinear(T_GRID, wg, kg, T, w)
    dinv = _bilinear(T_GRID, wg, dg, T, w)
    k = np.where((w > 500.0) & (w <= 2475.0), np.maximum(k, 0.0), 0.0)
    return k, gc * dinv, doppler_halfwidth(w, T, 44.0) * dinv


# ============================================================================ CO
@functools.lru_cache(maxsize=None)
def _co_tables():
    """SP-3080 Tables A2-2 / A2-3 on a common 25 cm-1 grid, 1400-2350 cm-1, per column (300, 600, 1200 K): log k and
    1/d interpolated linearly in wavenumber between the printed rows of each column; zero k outside them."""
    with open(os.path.join(SP3080, "co_tablesA2-2_A2-3.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    head, rows = rows[0], rows[1:]
    w_tab = np.array([float(r[0]) for r in rows])
    grid = np.arange(1400.0, 2350.0 + 0.1, 25.0)
    temps = np.array([300.0, 600.0, 1200.0])
    k = np.zeros((3, len(grid)))
    dinv = np.ones((3, len(grid)))
    for i, t in enumerate(("300K", "600K", "1200K")):
        kc = np.array([float(r[head.index(f"k_{t}")]) if r[head.index(f"k_{t}")] else np.nan for r in rows])
        dc = np.array([float(r[head.index(f"invd_{t}")]) if r[head.index(f"invd_{t}")] else np.nan for r in rows])
        ok = np.isfinite(kc) & (kc > 0)
        inside = (grid >= w_tab[ok].min()) & (grid <= w_tab[ok].max())
        k[i, inside] = np.exp(np.interp(grid[inside], w_tab[ok], np.log(kc[ok])))
        okd = np.isfinite(dc)
        dinv[i] = np.interp(grid, w_tab[okd], dc[okd])
    return temps, grid, k, dinv


def co_params(w, T, gc):
    """(k at STP [cm-1 atm-1], a_c, a_D) of CO, 1400-2350 cm-1; T clipped to the table's 300-1200 K."""
    temps, grid, tk, td = _co_tables()
    w = np.asarray(w, float)
    T = np.clip(np.asarray(T, float) * np.ones_like(w), 300.0, 1200.0 - 1e-6)
    k = _bilinear(temps, grid, tk, T, w)
    dinv = _bilinear(temps, grid, td, T, w)
    k = np.where((w >= 1400.0) & (w <= 2350.0), np.maximum(k, 0.0), 0.0)
    return k, gc * dinv, doppler_halfwidth(w, T, 28.0) * dinv


# ============================================================================ HITRAN-derived species (air)
HITRAN = os.path.join(ROOT, "docs", "sources", "hitran")


@functools.lru_cache(maxsize=None)
def _hitran_table(mol: str):
    with open(os.path.join(HITRAN, f"snb_{mol.lower()}_hitran2020.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    head, rows = rows[0], rows[1:]
    temps = np.array([float(h[2:-1]) for h in head if h.startswith("k_")])
    a = np.array([[float(x) for x in r] for r in rows])
    nt = len(temps)
    w = a[:, 0]
    grid = np.arange(w.min(), w.max() + 0.1, 5.0)                      # empty intervals: no absorption
    j = np.round((w - grid[0]) / 5.0).astype(int)
    k = np.zeros((nt, len(grid)))
    dinv = np.ones((nt, len(grid)))
    g = np.full(len(grid), 0.07)
    n = np.full(len(grid), 0.75)
    k[:, j], dinv[:, j] = a[:, 4:4 + nt].T, a[:, 4 + nt:4 + 2 * nt].T
    g[j], n[j] = a[:, 1], a[:, 2]
    return temps, grid, k, dinv, g, n


def hitran_params(mol: str, w, T, p_total_atm: float):
    """(k at STP, a_c, a_D) from the HITRAN narrow-band table of `mol` (200-320 K; T clipped), with the interval's
    air-broadened half-width gamma_air(296) p (296 / T)^n."""
    temps, grid, tk, td, g, n = _hitran_table(mol)
    w = np.asarray(w, float)
    T = np.clip(np.asarray(T, float) * np.ones_like(w), temps[0], temps[-1] - 1e-6)
    k = _bilinear(temps, grid, tk, T, w)
    dinv = _bilinear(temps, grid, td, T, w)
    gam = np.interp(w, grid, g) * p_total_atm * (296.0 / T) ** np.interp(w, grid, n)
    inside = (w >= grid[0] - 2.5) & (w <= grid[-1] + 2.5)
    return (np.where(inside, np.maximum(k, 0.0), 0.0), gam * dinv,
            doppler_halfwidth(w, T, MOLAR_MASS[mol]) * dinv)


def _co_any(w, T, p_atm):
    """CO: HITRAN table at air temperatures, SP-3080 Tables A2-2/A2-3 for the exhaust, blended over 300-340 K."""
    T = float(T)
    hot = co_params(w, max(T, 300.0), collision_halfwidth("CO", T, p_atm))
    if T >= 340.0:
        return hot
    cold = hitran_params("CO", w, min(T, 320.0), sum(p_atm.values()))
    f = 0.0 if T <= 300.0 else (T - 300.0) / 40.0
    return tuple((1.0 - f) * c + f * h for c, h in zip(cold, hot))


# ============================================================================ band model
def goody_lorentz(xstar, ac):
    return np.where(xstar < 1e-15, xstar, np.sqrt(ac) * xstar / np.sqrt(ac + 0.25 * xstar))


def goody_doppler(xstar, ad):
    return 1.7 * ad * np.sqrt(np.log1p((0.58824 * xstar / np.maximum(ad, TINY)) ** 2))


def combined_depth(xstar, ac, ad):
    """SP-3080 eqs. 5-25 / 5-26: optical depth X of Lorentz and Doppler lines together."""
    xs = np.maximum(xstar, TINY)
    yd = np.maximum(1.0 - (goody_doppler(xs, ad) / xs) ** 2, 2.2e-16)
    yc = np.maximum(1.0 - (goody_lorentz(xs, ac) / xs) ** 2, 2.2e-16)
    y = np.maximum(1.0 / yc ** 2 + 1.0 / yd ** 2 - 1.0, 1.0)
    x = xs * np.sqrt(1.0 - y ** -0.5)
    return np.where(xstar < 1e-20, xstar, x)


SPECIES = ("H2O", "CO2", "CO", "N2O", "CH4")


def species_params(s: str, w, T: float, p_atm: dict):
    """(k at STP, a_c, a_D) of species s in a gas of temperature T and partial pressures p_atm [atm]."""
    if s == "H2O":
        return h2o_params(w, T, collision_halfwidth("H2O", T, p_atm))
    if s == "CO2":
        return co2_params(w, T, collision_halfwidth("CO2", T, p_atm))
    if s == "CO":
        return _co_any(w, T, p_atm)
    return hitran_params(s, w, T, sum(p_atm.values()))


def curtis_godson(xs_inc, ac_inc, ad_inc) -> dict:
    """One absorber along a path of sections: the cumulative X* = sum k u, the X*-weighted fine-structure parameters
    (SP-3080 eqs. A-5 to A-7) and the optical depth X at the end of each section. Inputs (n_sections, ...)."""
    xs = np.cumsum(np.asarray(xs_inc, float), axis=0)
    wsum = np.where(xs > 1e-20, xs, 1.0)
    ac = np.where(xs > 1e-20, np.cumsum(np.asarray(ac_inc, float) * xs_inc, axis=0) / wsum, 1.0)
    ad = np.where(xs > 1e-20, np.cumsum(np.asarray(ad_inc, float) * xs_inc, axis=0) / wsum, 1.0)
    return dict(xstar=xs, ac=ac, ad=ad, x=combined_depth(xs, ac, ad))


_PARAM_MEMO: dict = {}


def _params_memo(s: str, w: np.ndarray, T: float, p: dict):
    """species_params for a gas state, memoised: a layered atmosphere repeats the same layer states in every ray."""
    key = (s, float(T), tuple(sorted(p.items())), w.shape[0], float(w[0]), float(w[-1]))
    v = _PARAM_MEMO.get(key)
    if v is None:
        if len(_PARAM_MEMO) > 20000:
            _PARAM_MEMO.clear()
        v = _PARAM_MEMO[key] = species_params(s, w, T, p)
    return v


def path_depths(w, segments) -> np.ndarray:
    """Cumulative optical depth X(w) at the end of each segment of a non-uniform path (Curtis-Godson).
    segments: sequence of dicts {length_m, T, p_atm: {"H2O": .., "CO2": .., "CO": .., "N2": .., "O2": ..}}.
    Returns (n_segments, n_w): X summed over species from the start of the path to the end of each segment."""
    w = np.asarray(w, float)
    out = np.zeros((len(segments), len(w)))
    for s in SPECIES:
        xs_inc, ac_inc, ad_inc = [], [], []
        for seg in segments:
            T, p = float(seg["T"]), seg["p_atm"]
            if p.get(s, 0.0) <= 0.0:
                xs_inc.append(np.zeros_like(w)); ac_inc.append(np.ones_like(w)); ad_inc.append(np.ones_like(w))
                continue
            u = 273.0 / T * seg["length_m"] * 100.0 * p[s]                      # cm atm at STP
            k, ac, ad = _params_memo(s, w, T, p)
            xs_inc.append(k * u); ac_inc.append(ac); ad_inc.append(ad)
        out += curtis_godson(np.array(xs_inc), np.array(ac_inc), np.array(ad_inc))["x"]
    return out


def homogeneous_emissivity(w, T, length_m, p_atm: dict) -> np.ndarray:
    """Spectral emissivity 1 - tau of a uniform slab (an exhaust jet, say)."""
    return 1.0 - np.exp(-path_depths(w, [dict(length_m=length_m, T=T, p_atm=p_atm)])[-1])
