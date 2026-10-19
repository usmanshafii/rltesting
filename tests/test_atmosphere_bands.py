"""thermal/atmosphere_bands.py: the layered clear atmosphere of the thermal cameras against its sources.

* NOAA IGRA 2 regional radiosondes: the vapour-height fit and the lapse range are the soundings' own numbers.
* ASTM G173-03: the MWIR window's direct-beam transmittance at AM1.5 under G173's own atmosphere.
* Idso (1981): the clear-air 8-14 um flux emittance sits below Idso's fit, and the scene's window absorber closes the
  share of the gap it draws.
* Geometry, limits and units: spherical shells, an opaque band radiates the air's Planck radiance, the single-
  scattering sky, the ASTM E490 in-band irradiance.
"""
import csv
import json
import math
import os

import numpy as np
import pytest

from thermal import atmosphere_bands as ab
from thermal import gas_bands as gb
from thermal import sky as thsky

SRC = os.path.join(gb.ROOT, "docs", "sources")
PROFILES = os.path.join(SRC, "noaa_igra2", "regional_profiles_2023-10_2026-09.json")
SOUNDINGS = os.path.join(SRC, "noaa_igra2", "regional_soundings_2023-10_2026-09.csv")
TA_CFG = dict(surface_pressure_hpa=988.0, lapse_k_per_km=[4.5, 7.7], co2_ppm=[419.0, 440.0], n2o_ppb=[336.0, 340.0],
              ch4_ppb=[1920.0, 2100.0], co_ppb_surface=[150.0, 3000.0], co_ppb_free=100.0, boundary_layer_m=1500.0,
              continuum_scale=[1.0, 1.44], window_absorber_scale_height_m=1500.0,
              aerosol=dict(scale_height_m=1500.0, asymmetry_g=[0.6, 0.8],
                           omega0=dict(lwir=[0.3, 0.7], lwir_cooled=[0.3, 0.7], mwir=[0.6, 0.95])))


def test_vapour_fit_and_lapse_are_the_soundings_numbers():
    from render.sampling import load_configs
    js = json.load(open(PROFILES))
    fit = js["vapour_scale_height_fit"]
    assert (ab.VAPOUR_FIT["a"], ab.VAPOUR_FIT["b"], ab.VAPOUR_FIT["sd"]) == (fit["a"], fit["b"], fit["sd"])
    assert ab.VAPOUR_FIT["h_min_m"] == pytest.approx(fit["h_1_99_percent_m"][0], abs=1.0)
    assert ab.VAPOUR_FIT["h_max_m"] == pytest.approx(fit["h_1_99_percent_m"][1], abs=1.0)
    lap = js["all_months_percentiles_10_25_50_75_90"]["lapse_sfc_500_k_per_km"]
    cfg = load_configs()["render"]["thermal_atmosphere"]
    assert cfg["lapse_k_per_km"] == pytest.approx([lap[0], lap[-1]], abs=0.05)
    assert ab.LAPSE_K_PER_KM == tuple(cfg["lapse_k_per_km"])
    # the fit returns the soundings' precipitable water: median ratio near 1, half of them within a factor 1.35
    rows = list(csv.DictReader(open(SOUNDINGS)))
    ratio = []
    for r in rows:
        e, t = float(r["e_sfc_hpa"]), float(r["t_sfc_c"]) + 273.15
        rho0 = e * 100.0 / (ab.R_V * t)
        ratio.append(rho0 * ab.vapour_scale_height(e) / float(r["pw_mm"]))
    ratio = np.array(ratio)
    assert np.median(ratio) == pytest.approx(1.0, abs=0.08)
    assert np.mean(np.abs(np.log(ratio)) < np.log(1.35)) > 0.5


def test_profile_hydrostatics_and_precipitable_water():
    p = ab.Profile(300.0, 20.0, 988.0, 6.5, 2000.0, 424.0)
    z = np.array([0.0, 5000.0, 11000.0, 16000.0, 20000.0])
    pr = p.pressure(z)
    assert pr[0] == pytest.approx(988.0)
    assert pr[1] / pr[0] == pytest.approx(((300.0 - 32.5) / 300.0) ** (9.80665 / (287.05 * 0.0065)), rel=1e-9)
    assert np.all(np.diff(pr) < 0)
    rho0 = 20.0 * 100.0 / (ab.R_V * 300.0)
    assert p.precipitable_water_mm() == pytest.approx(rho0 * 2000.0, rel=0.08)     # exponential column (layers)
    assert np.all(p.e_hpa <= ab.saturation_hpa(p.T) + 1e-9)


def test_ray_geometry_spherical_shells():
    p = ab.Profile(300.0, 15.0)
    up = ab.ray_segments(p, 90.0)
    assert sum(s["length_m"] for s in up) == pytest.approx(30000.0, rel=1e-6)
    flat = ab.ray_segments(p, 0.0)
    total = sum(s["length_m"] for s in flat)
    assert total == pytest.approx(math.sqrt((ab.R_EARTH + 30000.0) ** 2 - ab.R_EARTH ** 2), rel=1e-6)
    for el in (0.5, 5.0, 30.0):                          # every elevation reaches the top, layer by layer upward
        segs = ab.ray_segments(p, el)
        assert segs and [s["layer"] for s in segs] == sorted(s["layer"] for s in segs)
    part = ab.ray_segments(p, 3.0, max_dist_m=1234.0)
    assert sum(s["length_m"] for s in part) == pytest.approx(1234.0, rel=1e-9)


def test_g173_mwir_window_direct_beam():
    rows = list(csv.reader(open(os.path.join(SRC, "astm", "ASTM_G173-03_from_NREL_workbook.csv"))))[2:]
    a = np.array([[float(x) for x in r] for r in rows])
    lam, etr, direct = a[:, 0], a[:, 1], a[:, 3]
    # G173's atmosphere: 1976 US Standard Atmosphere, 1.42 cm precipitable water, 370 ppm CO2, AM1.5, rural aerosol
    # AOD 0.084 at 500 nm; the era's N2O and CH4
    T0, H = 288.15, 2000.0
    e0 = 14.2 / H * ab.R_V * T0 / 100.0
    p = ab.Profile(T0, e0, 1013.25, 6.5, H, 370.0, z_trop_m=11000.0, n2o_ppb=320.0, ch4_ppb=1700.0,
                   co_ppb_surface=100.0, co_ppb_free=100.0)
    w = ab.W_GRID
    _, tau = ab.solar_direct(p, 90.0 - 48.19, w)
    l_nm = 1e7 / w
    tau = tau * np.exp(-0.084 * (l_nm / 500.0) ** -1.37 * 1.5)
    g = np.interp(l_nm, lam, direct / np.maximum(etr, 1e-9))
    m = (l_nm >= 3600) & (l_nm <= 4000)
    wt = ab.e490_wn(w)[m]
    model, ref = np.sum(tau[m] * wt) / wt.sum(), np.sum(g[m] * wt) / wt.sum()
    assert model == pytest.approx(ref, rel=0.05)


def test_clear_air_below_idso_and_window_absorber_closes_the_gap():
    for tc, e in ((5.0, 6.0), (25.0, 12.0), (30.0, 20.0), (32.0, 30.0)):
        T = tc + 273.15
        p = ab.Profile(T, e, 988.0, 5.7, ab.vapour_scale_height(e), 424.0)
        eps = ab.hemispherical_emittance(p, (8.0, 14.0))
        gap = thsky.zenith_emissivity(T, e) - eps
        assert 0.0 < gap < 0.2
    T, e = 298.15, 12.0
    p = ab.Profile(T, e, 988.0, 5.7, ab.vapour_scale_height(e), 424.0)
    target = thsky.zenith_emissivity(T, e)
    k = ab.window_absorber_for(p, target, 1.0, 1500.0)
    win = dict(k_per_m=k, scale_height_m=1500.0, w_lo=ab.WINDOW_WN[0], w_hi=ab.WINDOW_WN[1])
    assert ab.hemispherical_emittance(p, (8.0, 14.0), 1.0, win) == pytest.approx(target, abs=0.005)


def test_opaque_band_radiates_air_planck_and_transmittance_falls():
    p = ab.Profile(300.0, 15.0, 988.0, 0.001, 2000.0, 424.0)           # nearly isothermal
    w = ab.band_w((4.2, 4.4))                                           # CO2 4.3 um band centre
    L, tau = ab.ray_radiance(p, 1.0, w)
    assert np.sum(L) * ab.W_STEP == pytest.approx(ab.planck_band(300.0, (4.2, 4.4)), rel=0.01)
    wl = ab.band_w((8.0, 14.0))
    dbdt = gb.planck_wn(300.5, wl) - gb.planck_wn(299.5, wl)
    t = ab.path_transmittance(p, 0.0, [10.0, 100.0, 1000.0, 5000.0], wl, dbdt)
    assert np.all(np.diff(t) < 0) and t[0] > 0.97 and 0.0 < t[-1] < t[0]


def test_single_scatter_sky_limits():
    F, om, g = 5.0, 0.9, 0.7
    # optically thin: omega0 F P tau / (4 pi mu)
    L = ab.single_scatter_sky(F, 1e-4, om, g, 0.5, 0.8, 0.3)
    assert float(L) == pytest.approx(om * F * float(ab.hg_phase(0.3, g)) * 1e-4 / (4 * math.pi * 0.5), rel=1e-3)
    # continuous through mu = mu0
    a = ab.single_scatter_sky(F, 0.3, om, g, 0.6 - 1e-3, 0.6, 0.5)
    b = ab.single_scatter_sky(F, 0.3, om, g, 0.6, 0.6, 0.5)
    assert float(a) == pytest.approx(float(b), rel=5e-3)
    # the phase function integrates to 4 pi
    c = np.linspace(-1, 1, 20001)
    assert np.trapz(ab.hg_phase(c, g), c) * 2 * math.pi == pytest.approx(4 * math.pi, rel=1e-3)


def test_e490_in_band_matches_register():
    rows = {r["band"]: r for r in csv.DictReader(open(os.path.join(SRC, "astm", "band_integrals.csv")))}
    e = float(np.sum(ab.e490_wn(ab.band_w((8.0, 14.0)))) * ab.W_STEP)
    assert e == pytest.approx(float(rows["lwir"]["e490_am0"]), rel=0.02)


def test_scene_tables_shapes_and_physics():
    rng = np.random.default_rng(3)
    prm = ab.sample_params(rng, TA_CFG, 30.0, 20.0)
    bands = {"lwir": (8.0, 14.0), "lwir_cooled": (7.7, 9.3), "mwir": (3.7, 4.8)}
    weather = dict(visibility_km=10.0, band_ratio=dict(lwir=0.2, lwir_cooled=0.2, mwir=0.3))
    tab = ab.scene_tables(prm, TA_CFG, bands, 30.0, dict(elevation_deg=45.0), weather,
                          [dict(name="target", range_m=2000.0, elevation_deg=5.0)])
    for b, t in tab["bands"].items():
        sky = np.array(t["sky_radiance"])
        assert np.all(np.diff(sky) < 0) and sky[0] < t["L_air"]          # brighter toward the horizon, below air
        assert 0.0 < t["tau_objects"]["target"] < 1.0
        assert 0.0 < t["e_sun_ground"] < t["e_sun_toa"]
    assert tab["bands"]["mwir"]["e_sun_toa"] == pytest.approx(
        float(np.sum(ab.e490_wn(ab.band_w((3.7, 4.8)))) * ab.W_STEP))
    assert tab["window"]["target"] >= tab["window"]["eps_hemispherical_model"] - 1e-9
