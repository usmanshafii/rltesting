"""thermal/plume.py and render/plume_layer.py: exhaust chemistry, the jet field, and the plume layer of a scene."""
import numpy as np
import pytest

from render import sampling
from thermal import plume as pm

CFGS = sampling.load_configs()


@pytest.mark.parametrize("phi", [0.9, 1.0, 1.1, 1.2])
def test_exhaust_composition_balances(phi):
    y = 1.87
    c = pm.exhaust_composition(phi, y, 0.0)
    assert sum(c.values()) == pytest.approx(1.0)
    carbon = c["CO2"] + c["CO"]
    hydrogen = 2 * c["H2O"] + 2 * c["H2"]
    oxygen = 2 * c["CO2"] + c["CO"] + c["H2O"] + 2 * c["O2"]
    nitrogen = 2 * c["N2"]
    assert hydrogen / carbon == pytest.approx(y)
    o2_per_c = (1 + y / 4) / phi
    assert oxygen / carbon == pytest.approx(2 * o2_per_c)
    assert nitrogen / oxygen == pytest.approx(2 * 3.773 / 2)
    if phi > 1:
        # Heywood 1988, 4.9.1: K = x_CO x_H2O / (x_CO2 x_H2) = 3.5; CO about three times H2
        assert c["CO"] * c["H2O"] / (c["CO2"] * c["H2"]) == pytest.approx(3.5, rel=1e-6)
        assert 2.0 < c["CO"] / c["H2"] < 4.5 and c["O2"] == 0.0
    else:
        assert c["CO"] == 0.0 and c["H2"] == 0.0
    if phi == 1.1:
        assert 0.02 < c["CO"] < 0.04 and 0.10 < c["CO2"] < 0.13


def test_short_circuit_dilutes_with_air():
    a, b = pm.exhaust_composition(1.1, 1.87, 0.0), pm.exhaust_composition(1.1, 1.87, 0.5)
    assert b["CO2"] < a["CO2"] and b["O2"] > 0.0
    assert b["CO2"] / b["H2O"] == pytest.approx(a["CO2"] / a["H2O"])


def test_exit_gas_within_the_settings_band():
    from thermal import engine_heat as eh
    for s, v in eh.POWER_SETTINGS.items():
        lo = pm.gas_exit_excess(0.0, s)
        hi = pm.gas_exit_excess(1e4, s)
        assert lo < hi <= v["egt_excess_k"][1] and lo >= 0.9 * v["egt_excess_k"][0]
        assert 0.3 < pm.exit_velocity_ratio(s, 0.5 * (lo + hi), 300.0, 47.0, 1000.0) < 2.5


def test_jet_field_similarity():
    jet = pm.Jet([0, 0, 0], [1, 0, 0], 0.03, 500.0, 300.0, 1.0, pm.exhaust_composition(1.0, 1.87), 10.0, 0.094)
    xc = jet.x_c
    assert jet.mixture_fraction(np.array([[0.5 * xc, 0, 0]]))[0] == pytest.approx(1.0)
    x = 4 * xc
    assert jet.mixture_fraction(np.array([[x, 0, 0]]))[0] == pytest.approx(0.25)
    b = 0.094 * x
    assert jet.mixture_fraction(np.array([[x, b, 0]]))[0] == pytest.approx(0.125, rel=1e-6)   # half-width
    st = jet.state(1.0)
    assert st["T"] == pytest.approx(800.0) and st["p_atm"]["CO2"] == pytest.approx(jet.comp["CO2"])


def test_plume_layer_of_a_scene():
    from render.plume_layer import plume_object
    from thermal import atmosphere_bands as atb
    spec = None
    for idx in range(40):
        s, _ = sampling.sample_scene(idx, CFGS, 40)
        if s.get("thermal") and s["thermal"].get("plume") and s["objects"][0]["range_m"] < 600:
            spec = s
            break
    assert spec is not None
    atb.ensure_tables(spec, CFGS["render"])
    for band in ("lwir", "lwir_cooled", "mwir"):
        o = plume_object(spec, band)
        assert o is not None and o["name"] == "plume" and o["plume"]
        assert o["P"].min() >= 0.0 and o["P"].max() > 0.0
        assert 0.0 <= o["A"].min() and o["A"].max() <= 1.0
        assert 0.0 < o["t_gas"] <= 1.0 and o["intensity_w_sr"] > 0.0
        x0, y0, x1, y1 = o["crop"]
        assert o["P"].shape[:2] == (y1 - y0, x1 - x0)
