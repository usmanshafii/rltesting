"""Humidity-driven clear sky (thermal/sky.py), config overlays, skin split and noise reference."""
import os

import numpy as np
import pytest

from noise import lwir_fpn
from render import sampling, signatures
from thermal import sky
from thermal.planck_lut import lwir_lut

LUT = lwir_lut()


@pytest.mark.parametrize("dew_c, e_ref, eps_ref, dT_ref", [
    (-5.0, 4.2, 0.25, (-2.5, -24.6, -41.1)),      # dry
    (10.0, 12.3, 0.33, (-0.7, -15.4, -29.9)),
    (24.0, 29.8, 0.76, (0.0, -0.3, -2.2)),        # humid: the low sky sits at air temperature
])
def test_idso_reference_values(dew_c, e_ref, eps_ref, dT_ref):
    """Air 30 C; sky temperature below air at 5, 15 and 25 deg elevation."""
    e = float(sky.saturation_vapour_pressure_hpa(dew_c))
    assert e == pytest.approx(e_ref, abs=0.1)
    eps = sky.zenith_emissivity(303.15, e)
    assert eps == pytest.approx(eps_ref, abs=0.01)
    dT = sky.sky_dT_k(LUT, 303.15, eps, np.array([5.0, 15.0, 25.0]))
    assert dT == pytest.approx(dT_ref, abs=0.6)


def test_slant_law_limits_and_dew_point_cap():
    el = np.array([-10.0, 0.0, 1.0, 10.0, 45.0, 90.0])
    e = sky.slant_emissivity(0.3, el)
    assert e[0] == e[1] == 1.0 and np.all(np.diff(e[1:]) < 0) and e[-1] == pytest.approx(0.3)
    hot = sky.humidity_state(45.0, 0.9, dew_point_max_c=28.0)          # desert heat cannot be monsoon-humid
    assert hot["dew_point_c"] == pytest.approx(28.0, abs=0.05) and hot["rh_effective"] < 0.4
    assert sky.humidity_state(10.0, 0.5)["dew_point_c"] == pytest.approx(0.0, abs=0.3)


def test_scene_records_humidity_sky_and_skin_split():
    cfgs = sampling.load_configs(overlay="")
    spec, _ = sampling.sample_scene(3, cfgs, 720)
    s = spec["signatures"]["environment"]["lwir"]["sky"]
    assert s["model"] == "idso1981" and 0.24 <= s["emissivity_zenith"] <= sky.EPS_Z_MAX
    assert s["dT_at_camera_k"] <= 0.0
    assert spec["config_overlay"] is None
    skin = cfgs["signatures"]["lwir"]["target"]["groups"]["skin"]
    rng = np.random.default_rng(4)
    for state in ("day", "crossover", "night"):
        for _ in range(30):
            t = signatures.sample_target(rng, cfgs["signatures"], state)
            p = t["lwir"]["skin"]
            assert skin["dT"][state][0] <= p["dT"] <= skin["dT"][state][1]
            assert skin["dT_upper"][state][0] <= p["dT_upper"] <= skin["dT_upper"][state][1]
            assert t["lwir"]["fuselage"]["dT_upper"] == p["dT_upper"]           # one paint draw
            lo, hi = cfgs["signatures"]["rgb"]["target"]["propeller_disc_alpha"]
            assert lo <= t["rgb"]["propeller_disc"]["alpha"] <= hi <= 0.13             # blade geometry, not 0.15-0.35


def test_overlay_merges_and_records_provenance(tmp_path):
    ov = tmp_path / "ov.yaml"
    ov.write_text("name: t\nrender:\n  dataset:\n    families: {open_sky: 1.0, horizon: 0.0}\n"
                  "sensors:\n  hfov_deg: [8.7834]\n  hfov_probs: [1.0]\n")
    base = sampling.load_configs(overlay="")
    cfgs = sampling.load_configs(overlay=str(ov))
    assert cfgs["render"]["dataset"]["families"] == {"open_sky": 1.0, "horizon": 0.0}
    assert cfgs["render"]["dataset"]["n_scenes"] == base["render"]["dataset"]["n_scenes"]      # merged, not replaced
    assert cfgs["render"]["target"] == base["render"]["target"]                                # flight settings untouched
    assert cfgs["sensors"]["hfov_deg"] == [8.7834] and cfgs["overlay"]["name"] == "t"
    bad = tmp_path / "bad.yaml"
    bad.write_text("target:\n  wingspan: 3.0\n")
    with pytest.raises(ValueError):
        sampling.load_configs(overlay=str(bad))


def test_m1_overlay_scene():
    cfgs = sampling.load_configs(overlay=os.path.join("configs", "m1.yaml"))
    spec, _ = sampling.sample_scene(7, cfgs, cfgs["render"]["dataset"]["n_scenes"])
    assert spec["family"] == "open_sky" and spec["lighting"] in ("day", "night")
    assert spec["camera"]["hfov_deg"] == pytest.approx(spec["cameras"]["grid"]["hfov_deg"]) and len(spec["objects"]) == 1
    assert 5.0 - 1e-6 <= spec["camera"]["hfov_deg"] <= 46.0 + 1e-6          # the uncooled camera's lenses set the grid
    assert spec["config_overlay"]["name"] == "m1-four-sensor"
    assert cfgs["render"]["target"]["airspeed_mps"] == {"default": 44.0, "range": [39.0, 56.0]}


def test_lwir_noise_is_referenced_to_a_30C_scene():
    """Same sensor, colder air: the noise in counts is unchanged, so it is worth more kelvin of scene."""
    S = sampling.load_configs(overlay="")["sensors"]["lwir_noise"]
    p = dict(netd_k=0.05, counts_per_k=50.0, offset=6000.0, col_k=0.0, pix_k=0.0, agc_lo=0.0, agc_hi=100.0)
    k_equiv = {}
    for Ta in (253.15, 303.15):
        img = np.full((512, 512), float(LUT.radiance(Ta)), np.float32)
        out, q = lwir_fpn.apply(img, np.random.default_rng(0), S, Ta, params=p)
        lo, hi = q["agc_range"]
        counts = out * (hi - lo) + lo
        k_equiv[Ta] = float(counts.std()) / p["counts_per_k"]       # kelvin at this ambient (gain is per ambient K)
    ratio = float(LUT.dL_dT(303.15) / LUT.dL_dT(253.15))
    assert k_equiv[303.15] == pytest.approx(0.05, rel=0.03)
    assert k_equiv[253.15] / k_equiv[303.15] == pytest.approx(ratio, rel=0.03) and ratio > 1.6
    assert lwir_fpn.nominal_sigma(253.15, S) == pytest.approx(lwir_fpn.nominal_sigma(303.15, S))
