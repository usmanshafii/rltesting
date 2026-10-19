"""MD-550 / L550E regression checks: engine surface priors, independent bands and SWIR hot emission."""
import numpy as np
import pytest

import engine_l550e as el
import kal_geometry as kg
from render import sampling, signatures

CFG = sampling.load_configs()
SIG = CFG["signatures"]


@pytest.mark.parametrize("state", ["day", "crossover", "night"])
def test_engine_surface_profiles_and_slot_coverage(state):
    lw = SIG["lwir"]["target"]
    seen = set()
    independent_profiles = False
    rng = np.random.default_rng(13)
    for _ in range(100):
        sampled = signatures.sample_target(rng, SIG, state)
        for band in ("rgb", "lwir", "swir"):
            assert set(sampled[band]) == set(kg.SLOTS)
        for slot in (*kg.ENGINE_SLOTS, "tail_cone"):
            p, sw = sampled["lwir"][slot], sampled["swir"][slot]
            seen.add(p["thermal_profile"])
            lo, hi = lw["engine_profiles"][p["thermal_profile"]][slot]
            assert lo <= p["dT"] <= hi
            lo, hi = lw["engine_profiles"][sw["emit_thermal_profile"]][slot]
            assert lo <= sw["emit_dT"] <= hi
            assert p["dT"] != sw["emit_dT"]
            independent_profiles |= p["thermal_profile"] != sw["emit_thermal_profile"]
        # one profile per scene and band: every engine surface shares it
        assert len({sampled["lwir"][s]["thermal_profile"] for s in kg.ENGINE_SLOTS}) == 1
        assert all(p["hot_emission_enabled"] for p in sampled["swir"].values())
        lo, hi = lw["groups"]["skin"]["dT"][state]
        assert lo <= sampled["lwir"]["skin"]["dT"] <= hi
    assert seen == set(lw["engine_profiles"])
    assert independent_profiles


def test_engine_profiles_order_by_physics():
    """Within each power setting the exhaust stubs are hottest and the carburettors coolest; higher power
    settings are hotter (tests/test_engine_heat.py checks that the ranges equal the heat model)."""
    order = ("exhaust_stub", "engine_head", "engine_barrel", "engine_case", "engine_flange", "engine_intake")
    prof = SIG["lwir"]["target"]["engine_profiles"]
    for name, p in prof.items():
        mids = [sum(p[s]) / 2 for s in order]
        assert mids == sorted(mids, reverse=True), name
    for slot in ("exhaust_stub", "engine_head", "engine_barrel", "engine_case"):
        assert sum(prof["low_power"][slot]) < sum(prof["nominal"][slot]) < sum(prof["high_power"][slot]), slot
    assert sum(p["weight"] for p in prof.values()) == pytest.approx(1.0)


def test_swir_reflectance_not_copied_from_rgb():
    rng = np.random.default_rng(42)
    samples = [signatures.sample_target(rng, SIG, "day") for _ in range(300)]
    for slot in kg.ENGINE_SLOTS:
        a = [s["rgb"][slot]["albedo"] for s in samples]
        b = [s["swir"][slot]["reflectance"] for s in samples]
        assert abs(np.corrcoef(a, b)[0, 1]) < 0.2


def test_exhaust_and_heads_reach_swir_emission_temperatures():
    """Every power setting's exhaust and hottest heads pass hot_emission.min_temp_c at ordinary air temperatures,
    so SWIR sees the engine glow at night; the carburettors do not on a temperate day."""
    he = SIG["swir"]["hot_emission"]
    for p in SIG["lwir"]["target"]["engine_profiles"].values():
        assert 15.0 + p["exhaust_stub"][1] > he["min_temp_c"] and 15.0 + p["engine_head"][1] > he["min_temp_c"]
        assert 15.0 + p["engine_intake"][1] < he["min_temp_c"]


def test_scene_records_engine_model_and_visible_surfaces():
    spec, _ = sampling.sample_scene(5, CFG, 720)
    assert spec["signature_model"] == "md550-engine-v3"
    assert "MD-550" in spec["target_variant"]["params"]["engine_model"]
    areas = spec["objects"][0]["hot_visible_cm2"]
    assert set(areas) == {*kg.ENGINE_SLOTS, "engine_total", "hot_total"}
    assert areas["engine_total"] == pytest.approx(sum(areas[k] for k in kg.ENGINE_SLOTS))
    assert areas["hot_total"] == pytest.approx(sum(areas[k] for k in kg.HOT_SLOTS))
    eng = spec["engine"]
    assert eng["envelope_m"] == [el.LENGTH_M, el.WIDTH_M, el.HEIGHT_M]
    assert len(eng["exhaust_outlets"]) == 4
    assert eng["mounting_face_x_m"] > eng["flange_face_x_m"]          # +X forward: the flange faces the propeller
    for key in ("mass_kg", "power_kw", "rpm_max", "displacement_cc"):  # appearance model: no performance data
        assert key not in eng


def test_exhaust_outlets_sit_below_the_engine_in_the_output_frame():
    g = kg.derive(kg.defaults(CFG["target"]))
    eng = kg.engine_metadata(g)
    z = [o["centre_m"][2] for o in eng["exhaust_outlets"]]
    y = sorted(o["centre_m"][1] for o in eng["exhaust_outlets"])
    assert max(z) < 0.0 and np.allclose(z, z[0])                       # all four point down, below the axis
    assert y[0] < 0 < y[-1] and y[0] == pytest.approx(-y[-1])          # two per bank, mirrored
    for o in eng["exhaust_outlets"]:
        assert eng["flange_face_x_m"] < o["centre_m"][0] < eng["mounting_face_x_m"]
