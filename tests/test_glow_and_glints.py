"""Night red glow of hot parts (RGB) and sun glints (LWIR): the physics behind render/signatures.rgb_self_emission
and sun_glint_params / glint_radiance, tied to configs/signatures.yaml (pure NumPy; the Blender wiring in
render/rgb.py and render/lwir.py evaluates the same expressions)."""
import csv
import json
import math
import os

import numpy as np
import pytest

from render import sampling, signatures as sg
from thermal.planck_lut import LWIR_BAND_UM, band_radiance_direct

CFG = sampling.load_configs(overlay="")
SIG = CFG["signatures"]
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "sources")
T_SUN, R_SUN, AU = 5772.0, 6.957e8, 1.495978707e11            # IAU 2015 B3 nominal values; IAU 2012 B2 au
OMEGA_SUN = math.pi * (R_SUN / AU) ** 2
# In-band irradiances of the ASTM G173-03 and E490 standard spectra (tools/source_measurements.py)
with open(os.path.join(SRC, "astm", "band_integrals.csv"), encoding="utf-8") as _f:
    ASTM = {r["band"]: r for r in csv.DictReader(_f)}


def _band(T, band, n=400):
    return float(band_radiance_direct(np.array([T]), tuple(band), n)[0])


# ---------------------------------------------------------------- night glow (RGB)
def test_glow_threshold_and_colour():
    he = SIG["rgb"]["hot_emission"]
    airglow_min = CFG["render"]["sky"]["night"]["airglow_rgb"][0]
    black = dict(albedo=0.0)
    t_min = he["min_temp_c"] + 273.15
    assert sg.rgb_self_emission(black, t_min - 1.0, SIG) == (0.0, 0.0, 0.0)
    r, g, b = sg.rgb_self_emission(black, t_min, SIG)
    assert r < 0.1 * airglow_min                                  # below the threshold it would be negligible ...
    assert sg.rgb_self_emission(black, t_min + 20.0, SIG)[0] > 0.1 * airglow_min   # ... and it is not far below
    r, g, b = sg.rgb_self_emission(dict(albedo=0.1), 773.15, SIG)
    assert r > 20 * g > 0 and g > 20 * b                          # a dull red glow (Wien tail)
    assert r > airglow_min                                        # a 500 C stub outshines the faintest night sky
    assert sg.rgb_self_emission(dict(albedo=0.1), 673.15, SIG)[0] < airglow_min    # a 400 C one does not


def test_glow_calibration_from_the_sun():
    """In-band direct sun at air mass 1.5: the ASTM G173-03 direct + circumsolar standard spectrum. Cross-check: a
    5772 K Sun (IAU 2015) through Rayleigh (Hansen and Travis 1974) and aerosol extinction (AOD 0.084 at 500 nm, the
    G173 condition) lands within 10 %."""
    he = SIG["rgb"]["hot_emission"]
    for c, (lo, hi) in he["bands_um"].items():
        assert he["inband_solar_irradiance"][c] == pytest.approx(float(ASTM["rgb_" + c]["g173_direct"]), abs=0.1), c
        lam = np.linspace(lo, hi, 201)
        toa = np.array([_band(T_SUN, (l - 2.5e-4, l + 2.5e-4), 5) / 5e-4 for l in lam]) * OMEGA_SUN
        tau_r = 0.008569 * lam ** -4 * (1 + 0.0113 * lam ** -2 + 0.00013 * lam ** -4)
        tau_a = 0.084 * (lam / 0.5) ** -1.3
        direct = float(np.trapz(toa * np.exp(-1.5 * (tau_r + tau_a)), lam))
        assert direct == pytest.approx(he["inband_solar_irradiance"][c], rel=0.10), c
    assert he["nishita_sun_units"] == SIG["swir"]["hot_emission"]["nishita_sun_units"]   # one render-unit scale
    assert SIG["swir"]["hot_emission"]["inband_solar_irradiance"] == pytest.approx(
        float(ASTM["swir"]["g173_direct"]), rel=0.01)


def test_solar_tables_are_the_standards():
    """The saved spectra reproduce the standards' own totals (G173-03: global tilt 1000.4, direct 900.1 W/m2; E490
    normalised to 1366.1 W/m2), and the code's Sun is the IAU 2015 nominal Sun read from the resolution."""
    assert float(ASTM["g173_total"]["g173_global_tilt"]) == pytest.approx(1000.4, abs=0.1)
    assert float(ASTM["g173_total"]["g173_direct"]) == pytest.approx(900.1, abs=0.1)
    assert float(ASTM["e490_total"]["e490_am0"]) == pytest.approx(1366.1, abs=0.1)
    with open(os.path.join(SRC, "iau", "constants.json"), encoding="utf-8") as f:
        iau = json.load(f)
    assert (iau["effective_temperature_k"], iau["solar_radius_m"], iau["astronomical_unit_m"]) == (T_SUN, R_SUN, AU * 1.0)
    assert sg.SUN_ANGULAR_RADIUS_RAD == pytest.approx(iau["solar_radius_m"] / iau["astronomical_unit_m"], rel=1e-12)


# ---------------------------------------------------------------- sun glints (LWIR)
def _glint(el=45.0, eps_z=0.3, lobe=2.0, fs=1.0):
    class R:                                                      # deterministic "rng" returning the given draws
        def __init__(self):
            self.v = [lobe, fs]

        def uniform(self, *a):
            return self.v.pop(0)
    return sg.sun_glint_params(R(), SIG, el, eps_z)


def test_glint_sun_irradiance_and_transmittance():
    g = SIG["lwir"]["sun_glint"]
    assert g["sun_toa_irradiance"] == pytest.approx(float(ASTM["lwir"]["e490_am0"]), rel=0.005)     # ASTM E490
    # cross-check: the 5772 K black body is brighter, because the Sun's 10 um brightness temperature is lower
    assert 1.1 < _band(T_SUN, LWIR_BAND_UM, 4000) * OMEGA_SUN / g["sun_toa_irradiance"] < 1.3
    p = _glint(el=30.0, eps_z=0.3)
    assert p["transmittance"] == pytest.approx(0.7 ** 2)          # (1 - eps_z)^(1 / sin 30 deg)
    assert p["irradiance_w_m2"] == pytest.approx(g["sun_toa_irradiance"] * 0.49)
    assert sg.sun_glint_params(np.random.default_rng(0), SIG, -10.0, 0.3) is None      # night: no sun
    assert sg.SUN_ANGULAR_RADIUS_RAD == pytest.approx(math.radians(0.2665), rel=0.01)


def test_glint_lobe_conserves_the_reflected_flux_and_mirrors_the_sun():
    p = _glint(lobe=3.0, fs=0.6)
    eps = 0.1
    th = np.linspace(0.0, 10 * math.sqrt(p["sigma2_rad2"]), 4000)       # angle from the mirror direction
    L = np.array([sg.glint_radiance(eps, p, math.cos(t)) for t in th])
    flux = float(np.trapz(L * 2 * math.pi * np.sin(th), th))             # integral over solid angle
    assert flux == pytest.approx((1 - eps) * 0.6 * p["irradiance_w_m2"], rel=0.01)
    m = _glint(lobe=1e-6, fs=1.0)                                         # a perfect mirror returns the sun itself
    sun_radiance = m["irradiance_w_m2"] / OMEGA_SUN
    assert sg.glint_radiance(0.0, m, 1.0) == pytest.approx(sun_radiance, rel=0.01)


def test_glints_are_rare_small_and_very_bright():
    """FLIR's note: glints read hundreds of degrees too hot. A 1 deg glint on bare cast aluminium makes a 300 K part
    read over 250 K too hot, a near-mirror one outshines the hottest exhaust stub; only within a few lobe widths of
    the mirror direction; black surfaces barely glint."""
    hottest_stub = 273.15 + 50 + max(p["exhaust_stub"][1] for p in SIG["lwir"]["target"]["engine_profiles"].values())
    p = _glint(el=50.0, eps_z=0.25, lobe=1.0, fs=0.8)
    t = np.arange(300.0, 900.0, 0.5)
    reads = float(np.interp(_band(300.0, LWIR_BAND_UM) + sg.glint_radiance(0.09, p, 1.0),
                            band_radiance_direct(t, LWIR_BAND_UM, 400), t))
    assert reads - 300.0 > 250.0                                                  # bare cast aluminium
    assert sg.glint_radiance(0.09, p, 1.0) > 10 * _band(300.0, LWIR_BAND_UM)       # vs a black 300 K skin
    mirror = _glint(el=50.0, eps_z=0.25, lobe=0.0, fs=1.0)
    assert sg.glint_radiance(0.09, mirror, 1.0) > 5 * _band(hottest_stub, LWIR_BAND_UM)
    off = math.cos(4 * math.sqrt(p["sigma2_rad2"]))
    assert sg.glint_radiance(0.09, p, off) < 1e-3 * sg.glint_radiance(0.09, p, 1.0)
    assert sg.glint_radiance(0.97, p, 1.0) < 0.05 * sg.glint_radiance(0.09, p, 1.0)


def test_scene_records_glow_and_glint_draws():
    spec, _ = sampling.sample_scene(3, CFG, n_scenes=720)
    rgb = spec["signatures"]["target"]["rgb"]
    assert all("glow_dT" in p for p in rgb.values())
    caps = SIG["lwir"]["target"]["surface_temperature_cap_c"]
    for slot, cap in caps.items():
        assert spec["air_temp_c"] + rgb[slot]["glow_dT"] <= cap + 1e-9
    g = spec["sun_glint"]
    if spec["sun"]["elevation_deg"] > 0:
        assert g and 0.0 < g["transmittance"] < 1.0 and g["peak_per_reflectance"] > 0
    else:
        assert g is None
