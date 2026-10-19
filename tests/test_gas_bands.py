"""thermal/gas_bands.py against the printed tables of NASA SP-3080 (Ludwig et al. 1973; docs/sources/nasa_sp3080).

* Table A1-1, the handbook's own worked example (pure CO2 at 1 atm, 60 cm in 10 sections, 3400-3600 cm-1): the
  band-model arithmetic (eq. 5-34 and 5-35 widths, Curtis-Godson X* and a_c, the Goody curve of growth, the combined
  Lorentz-Doppler depth and the emission sum) is re-derived from the printed inputs.
* Table A2-29 / A2-33: the CO2 4.3 um k and 1/d of the port of NIST RADCAL, scaled onto the printed k.
* Tables A2-2 / A2-3: CO, read straight from the printed table.
* Table A2-35: the five H2O entries where RADCAL's copy differs from the print carry the printed value.
"""
import csv
import os

import numpy as np
import pytest

from thermal import gas_bands as gb

SP = os.path.join(gb.ROOT, "docs", "sources", "nasa_sp3080")


def _csv(path):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def _num(v):
    return float(v) if v != "" else np.nan


# ---------------------------------------------------------------------------- Table A1-1
A11 = _csv(os.path.join(SP, "table_A1-1_example.csv"))
# Printed values that contradict the rest of their own row (each re-derived from the neighbouring columns):
K_MISPRINTS = {(3500, 2): 0.01769,      # printed 0.1769: X* rises 0.03128 -> 0.07998 over du = 2.753
               (3400, 5): 0.01862}      # printed 0.1862: X* rises 0.03531 -> 0.06227 over du = 1.448
TAU_MISPRINTS = {(3600, 1),             # printed 0.2765 next to X = 1.018 (exp(-1.018) = 0.361); the printed sum
                 (3500, 3)}             # carries it. Printed 0.8075 next to X = 0.2183 (exp(-0.2183) = 0.8039)
# Section 10 is inconsistent in the print (its X* step at 3500 and 3600 cm-1 is not k du), so the cumulative columns are
# checked over sections 1-9; the printed a_D column does not follow from gamma_D / d except at 3400 cm-1 and is unused
# (the text, document p. 382: Doppler broadening is negligible in this example).


def _example(w):
    S = sorted((r for r in A11 if int(r["w"]) == w), key=lambda r: int(r["section"]))
    col = {k: np.array([_num(r[k]) for r in S]) for k in S[0] if k not in ("section", "w")}
    for (ww, sec), v in K_MISPRINTS.items():
        if ww == w:
            col["k"][sec - 1] = v
    return col


@pytest.mark.parametrize("w", [3400, 3450, 3500, 3600])
def test_table_a1_1_band_model_arithmetic(w):
    c = _example(w)
    T = c["T"]
    assert np.allclose(c["du"], 6.0 * 273.0 / T, atol=6e-4)                         # 6 cm sections at STP
    gc = gb.collision_halfwidth("CO2", T, {"CO2": 1.0})                              # pure CO2, 1 atm (p. 381)
    assert np.allclose(gc, c["gc"], atol=6e-5)
    # eq. 5-35; the example rounds 5.94e-6 / sqrt(44) to 0.895e-6 (p. 381), 0.05 % below, which moves the last digit
    assert np.allclose(gb.doppler_halfwidth(w, T, 44.0), c["gD"], atol=1.1e-5)
    cg = gb.curtis_godson(c["k"] * c["du"], gc * c["inv_d"], gb.doppler_halfwidth(w, T, 44.0) * c["inv_d"])
    s = slice(0, 9)
    assert np.allclose(cg["xstar"][s], c["xstar"][s], rtol=1.5e-3)
    assert np.allclose(cg["ac"][s], c["ac"][s], rtol=1.5e-3)
    assert np.allclose(gb.goody_lorentz(cg["xstar"], cg["ac"])[s], c["xc"][s], rtol=1.5e-3)
    assert np.allclose(cg["x"][s], c["x"][s], rtol=3e-3)
    tau = np.exp(-cg["x"])
    ok = np.array([(w, i + 1) not in TAU_MISPRINTS for i in range(10)])
    assert np.allclose(tau[s][ok[s]], c["tau"][s][ok[s]], atol=1.5e-3)


def test_table_a1_1_planck_and_emission_sum():
    rel = []
    for w in (3400, 3450, 3500, 3600):
        c = _example(w)
        n0 = gb.planck_wn(c["T"], w) * 100.0                     # W m-2 -> 1e-6 W cm-2 (sr cm-1)^-1
        ok = np.isfinite(c["N0x1e6"])
        rel += list(np.abs(n0[ok] - c["N0x1e6"][ok]) / np.maximum(1e-3 * c["N0x1e6"][ok], 0.005))   # 2 decimals
        # emerging radiance, SP-3080 eq. A-3: sum of N0(T_m) (tau_{m-1} - tau_m) from the observer's side
        cg = gb.curtis_godson(c["k"] * c["du"], gb.collision_halfwidth("CO2", c["T"], {"CO2": 1.0}) * c["inv_d"],
                              gb.doppler_halfwidth(w, c["T"], 44.0) * c["inv_d"])
        tau = np.concatenate([[1.0], np.exp(-cg["x"])])
        total = float(np.sum(n0 * (tau[:-1] - tau[1:])))
        # 3600 cm-1: the printed sum was built on the misprinted tau of section 1 (+1.7 % here)
        assert total == pytest.approx(c["sumN0dtau"][-1], rel=0.02 if w == 3600 else 0.005)
    rel = np.array(rel)
    # Planck: within 0.1 % or the printed 2 decimals (older constants: +0.04 %), except five printed entries that are
    # 1.5-4 % off the Planck function of their own section temperature (3450 cm-1 sections 1, 9, 10; 3600 cm-1 6, 7)
    assert np.sum(rel > 1.0) <= 5 and np.max(rel) < 50.0


# ---------------------------------------------------------------------------- CO2 4.3 um
A229 = _csv(os.path.join(SP, "co2_43um_k_tableA2-29.csv"))
# Table A2-33 (SLG 1/d, 4.3 um band), PDF page 468, read off pages/sp3080_pdfpage468.png, 2250-2390 cm-1 every 10
A233 = {300: [2.553, 2.553, 2.553, 2.468, 2.383, 2.293, 1.910, 1.528, 1.203, 1.165, 2.849, 1.313, 1.090, 1.099, 1.460],
        600: [13.73, 13.27, 12.39, 11.06, 8.965, 7.826, 6.521, 5.739, 5.739, 7.304, 8.005, 4.618, 2.413, 1.763, 1.473],
        1200: [104.8, 92.21, 80.68, 69.15, 58.78, 84.47, 64.90, 46.35, 37.08, 35.02, 23.47, 16.13, 7.701, 3.394, 1.469]}


@pytest.mark.parametrize("T", [300, 600, 1200])
def test_co2_43um_k_against_table_a2_29(T):
    w = np.array([float(r["wavenumber_cm-1"]) for r in A229])
    v = np.array([_num(r[f"k_{T}K_per_cm_atm_STP"]) for r in A229])
    m = np.array([gb._co2_scalar(float(x), float(T))[0] for x in w])
    band = np.isfinite(v) & (w >= 2200)
    assert np.trapz(m[band], w[band]) == pytest.approx(np.trapz(v[band], w[band]), rel=0.006)
    core = np.isfinite(v) & (v >= 0.1 * np.nanmax(v))
    r = m[core] / v[core]
    assert np.sqrt(np.mean((r - 1.0) ** 2)) < 0.10 and 0.8 < r.min() and r.max() < 1.3
    # the unscaled RADCAL model sits 300/273 above the print at every temperature
    assert np.trapz(m[band], w[band]) / gb.SP3080_SCALE_43 / np.trapz(v[band], w[band]) == pytest.approx(300 / 273, rel=0.006)
    # interpolated grid used by the renderer = the direct evaluation
    k_grid = gb.co2_params(w[band], float(T), 0.07)[0]
    assert np.allclose(k_grid, m[band], rtol=0.02, atol=1e-3 * np.nanmax(v))


@pytest.mark.parametrize("T", [300, 600, 1200])
def test_co2_43um_line_spacing_against_table_a2_33(T):
    w = np.arange(2250.0, 2391.0, 10.0)
    m = np.array([gb._co2_scalar(float(x), float(T))[1] for x in w])
    r = m / np.array(A233[T])
    core = w >= 2310.0              # where k is large; in the 300 K wings the print holds a constant 2.553
    assert np.all(np.abs(np.log(r[core])) < np.log(1.6))
    assert np.exp(np.mean(np.abs(np.log(r[core])))) < 1.2


# ---------------------------------------------------------------------------- CO and H2O
def test_co_reproduces_tables_a2_2_a2_3():
    rows = _csv(os.path.join(SP, "co_tablesA2-2_A2-3.csv"))
    for T in (300, 600, 1200):
        for r in rows:
            k, d = _num(r[f"k_{T}K"]), _num(r[f"invd_{T}K"])
            w = float(r["wavenumber_cm-1"])
            if np.isfinite(k) and k > 1e-6:
                kk, ac, ad = gb.co_params(np.array([w]), float(T), 1.0)
                assert kk[0] == pytest.approx(k, rel=1e-6)
                if np.isfinite(d):
                    assert ac[0] == pytest.approx(d, rel=1e-6)                      # a_c = gc / d with gc = 1
    k6 = gb.co_params(np.array([2175.0]), 450.0, 1.0)[0][0]                          # halfway between columns
    assert k6 == pytest.approx(0.5 * (3.581 + 2.423), rel=1e-6)


def test_h2o_table_carries_printed_values():
    from tools.source_measurements import SP3080_H2O_CORRECTIONS
    temps, wns, tab = gb._table("h2o_k_sp3080.csv")
    assert tab.shape == (6, 376) and wns[0] == 50 and wns[-1] == 9425
    for (w, T), (_radcal, printed, _page) in SP3080_H2O_CORRECTIONS.items():
        assert tab[list(temps).index(T), list(wns).index(w)] == pytest.approx(printed)


def test_path_model_limits():
    w = np.arange(2000.0, 2500.0, 5.0)
    air = {"H2O": 0.01, "CO2": 420e-6, "N2": 0.78, "O2": 0.21}
    eps_1km = gb.homogeneous_emissivity(w, 290.0, 1000.0, air)
    assert eps_1km[(w >= 2300) & (w <= 2370)].min() > 0.999                          # CO2 4.3 um: opaque in 1 km
    assert eps_1km[(w >= 2450) & (w <= 2495)].max() < 0.2                            # 4.0-4.1 um window
    # weak-line limit: X -> X* for a vanishing amount; strong limit: X ~ sqrt(a_c X*) (square-root law)
    assert gb.combined_depth(np.array([1e-6]), np.array([0.5]), np.array([0.01]))[0] == pytest.approx(1e-6, rel=1e-3)
    big = gb.goody_lorentz(np.array([1e4]), np.array([0.1]))[0]
    assert big == pytest.approx(np.sqrt(0.1 * 4 * 1e4), rel=1e-3)
    # splitting a uniform path in sections changes nothing (Curtis-Godson is exact for a uniform path)
    seg = dict(length_m=100.0, T=290.0, p_atm=air)
    one = gb.path_depths(w, [dict(seg, length_m=300.0)])[-1]
    three = gb.path_depths(w, [seg, seg, seg])[-1]
    assert np.allclose(one, three, rtol=1e-9)
