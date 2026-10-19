"""LWIR water-vapour path loss, surface temperature caps, SWIR paint families, camera pointing, px50 fits."""
import math

import numpy as np
import pytest

from data import composite as cp
from eval import pd_fit
from render import sampling, signatures
from thermal import atmosphere as atm
from thermal import sky

CFGS = sampling.load_configs(overlay="")
WV = CFGS["render"]["weather"]["lwir_water_vapour"]


def test_continuum_magnitudes_and_flir_anchor():
    assert atm.continuum_sigma_per_km(10.0, 0.5 * float(sky.saturation_vapour_pressure_hpa(10.0))) == pytest.approx(0.04, abs=0.02)
    e_hot = 0.8 * float(sky.saturation_vapour_pressure_hpa(35.0))
    assert atm.continuum_sigma_per_km(35.0, e_hot) == pytest.approx(1.0, abs=0.15)
    p = atm.path_params(35.0, e_hot, WV)
    assert float(atm.transmittance(100.0, p)) == pytest.approx(0.80, abs=0.03)       # FLIR 2016: 80 % over 100 m
    monsoon = atm.path_params(31.4, 30.2, WV)
    assert 0.2 < float(atm.transmittance(2400.0, monsoon)) < 0.35
    dry = atm.path_params(13.0, 9.9, WV)
    assert float(atm.transmittance(2400.0, dry)) > 0.7


def test_transmittance_limits_and_slant_paths():
    p = atm.path_params(30.0, 25.0, WV)
    d = np.array([0.0, 500.0, 2000.0, 5000.0])
    t = atm.transmittance(d, p)
    assert t[0] == pytest.approx(1.0) and np.all(np.diff(t) < 0)
    assert float(atm.transmittance(5000.0, p, 20.0)) > float(atm.transmittance(5000.0, p, 0.0))   # vapour thins with height
    assert np.all(atm.transmittance(d, None) == 1.0)
    assert atm.path_params(30.0, 25.0, dict(WV, enabled=False)) is None
    assert atm.path_params(30.0, None, WV) is None                                        # pre-v2 scene: unaffected


def test_composite_attenuates_object_contrast_not_sky():
    sky_level, obj_level = 30.0, 40.0
    layer = {"sky": np.full((20, 20), sky_level, np.float32)}
    A = np.ones((4, 4), np.float32)
    obj = dict(name="target", crop=(8, 8, 12, 12), P=A * obj_level, A=A, range_m=2400.0, elevation_deg=10.0)
    p = atm.path_params(31.4, 30.2, WV)
    clean = cp.composite(layer, [obj], 0.0)
    wet = cp.composite(layer, [obj], 0.0, wv=p)
    t = float(atm.transmittance(2400.0, p, 10.0))
    assert wet[0, 0, 0] == pytest.approx(sky_level)                                       # sky untouched
    assert clean[10, 10, 0] - sky_level == pytest.approx(obj_level - sky_level)
    assert wet[10, 10, 0] - sky_level == pytest.approx(t * (obj_level - sky_level), rel=1e-4)


def test_scene_records_path_loss_and_caps():
    spec, _ = sampling.sample_scene(11, CFGS, 720)
    w = spec["weather"]["lwir_water_vapour"]
    assert w is not None and w["vapour_pressure_hpa"] == pytest.approx(
        spec["signatures"]["environment"]["lwir"]["sky"]["vapour_pressure_hpa"])
    t = spec["objects"][0]["lwir_wv_transmission"]
    assert 0.0 < t <= 1.0
    for slot, cap in (CFGS["signatures"]["lwir"]["target"].get("surface_temperature_cap_c") or {}).items():
        p = spec["signatures"]["target"]["lwir"][slot]
        assert spec["air_temp_c"] + p["dT"] <= cap + 1e-9 and p["dT_cap_k"] == pytest.approx(cap - spec["air_temp_c"])


def test_surface_caps_clip_and_keep_the_draw():
    slots = {"exhaust_stub": {"dT": 420.0}, "engine_head": {"dT": 150.0}, "skin": {"dT": 3.0}}
    signatures.apply_surface_caps(slots, 45.0, {"exhaust_stub": 400.0, "engine_head": 250.0})
    w, h = slots["exhaust_stub"], slots["engine_head"]
    assert w["capped"] and w["dT"] == pytest.approx(355.0) and w["dT_uncapped"] == 420.0
    assert not h["capped"] and h["dT"] == 150.0 and "dT_uncapped" not in h
    assert "capped" not in slots["skin"]


def test_swir_paint_families():
    rng = np.random.default_rng(3)
    mix = CFGS["signatures"]["swir"]["target"]["groups"]["paint"]["reflectance"]["mixture"]
    seen = {}
    for _ in range(600):
        p = signatures.sample_target(rng, CFGS["signatures"], "day")["swir"]["skin"]
        fam = p["reflectance_family"]
        lo, hi = mix[fam]["range"]
        assert lo <= p["reflectance"] <= hi
        seen[fam] = seen.get(fam, 0) + 1
    assert set(seen) == set(mix) and max(p for f in mix for p in [mix[f]["range"][1]]) == pytest.approx(0.60)
    assert seen["mixed"] < seen["carbon_black"] and seen["mixed"] < seen["ir_reflective"]


def test_open_sky_camera_is_weighted_low():
    cc = CFGS["render"]["camera"]
    lo, hi = cc["look_elevation_deg"]["open_sky"]
    rng = np.random.default_rng(5)
    els = np.array([sampling.look_elevation(rng, cc, "open_sky", lo, hi) for _ in range(4000)])
    assert els.min() >= lo and els.max() <= hi
    assert np.median(els) == pytest.approx(math.sqrt(lo * hi), rel=0.06)                 # log-uniform: ~17 deg, not 32.5
    hz = np.array([sampling.look_elevation(rng, cc, "horizon", -1.0, 5.0) for _ in range(500)])
    assert hz.min() >= -1.0 and np.median(hz) == pytest.approx(2.0, abs=0.4)             # horizon family stays uniform


def test_px50_fit_recovers_known_curve():
    rng = np.random.default_rng(0)
    dets = []
    for i in range(1500):
        g = ("beam", "head", "tail")[i % 3]
        size = 2 ** rng.uniform(1, 7)
        true50 = {"beam": 8.0, "head": 12.0, "tail": 6.0}[g]
        p = 1 / (1 + math.exp(-2.5 * (math.log2(size) - math.log2(true50))))
        dets.append(dict(scene=f"s{i}", size_px=size, detected=bool(rng.uniform() < p), group=g))
    res = pd_fit.summarise(dets, n_boot=60)
    assert res["px50_beam"]["value"] == pytest.approx(8.0, rel=0.12)
    assert res["ratio_head_beam"]["value"] == pytest.approx(1.5, rel=0.15)
    assert res["px50_beam"]["lo"] < 8.0 < res["px50_beam"]["hi"] or abs(res["px50_beam"]["value"] - 8.0) < 0.5
    flat = [dict(d, detected=False) for d in dets[:30]] + [dict(dets[30], detected=True)]
    assert pd_fit.px50(np.array([-3.0, 0.0, 0.0, 0.0]), "beam") == float("inf")
    unavailable = pd_fit.summarise(flat[:5])
    assert unavailable["px50_beam"]["status"] == "insufficient_targets"
    assert np.isnan(unavailable["px50_beam"]["value"])


def test_sensitivity_categories_and_inputs():
    from eval import sensitivity as SE
    spec, _ = sampling.sample_scene(4, CFGS, 720)
    meta = dict(extra=dict(signatures=spec["signatures"], objects=spec["objects"]),
                conditions=dict(lighting=spec["lighting"]))
    c = SE.scene_categories(meta)
    assert c["thermal_profile"] in CFGS["signatures"]["lwir"]["target"]["engine_profiles"]
    assert c["swir_paint_family"] in ("carbon_black", "mixed", "ir_reflective")
    assert c["engine_cap_acted"] in ("yes", "no")      # CHT-limit caps on the head and fins
    v = SE.scene_inputs(meta)
    for k in ("engine_head_dT_k", "exhaust_stub_dT_k", "engine_visible_cm2", "engine_hot_visible_cm2"):
        assert k in v, k
    assert v["engine_hot_visible_cm2"] <= v["engine_visible_cm2"]
    dets = [dict(scene=f"s{i}", size_px=12.0, detected=i % 2 == 0) for i in range(40)]
    cats = {f"s{i}": dict(thermal_profile="low_power" if i % 2 else "high_power") for i in range(40)}
    rows = SE.category_table(dets, cats, "lwir_only")
    assert {r["value"] for r in rows} == {"low_power", "high_power"} and rows[0]["swing"] == pytest.approx(1.0)
