"""Physics behind the engine signature: properties, correlations, consistency checks and the YAML link."""
import math

import numpy as np
import pytest

import engine_l550e as el
from render import sampling, signatures
from thermal import engine_heat as eh
from thermal.planck_lut import SWIR_BAND_UM, band_radiance_direct, lwir_lut

CFG = sampling.load_configs(overlay="")
SIG = CFG["signatures"]


# ---------------------------------------------------------------- air properties and correlations vs references
def test_air_properties_match_tables():
    """Incropera tab. A.4 (1 atm): nu(300 K) 15.89e-6 m2/s, k(300 K) 0.0263, k(800 K) 0.0573, mu(800 K) 369.8e-7."""
    assert eh.viscosity(300.0) / eh.density(300.0) == pytest.approx(15.89e-6, rel=0.02)
    assert eh.conductivity(300.0) == pytest.approx(0.0263, rel=0.03)
    assert eh.conductivity(800.0) == pytest.approx(0.0573, rel=0.03)
    assert eh.viscosity(800.0) == pytest.approx(369.8e-7, rel=0.03)
    assert eh.density(255.65, 5000.0) == pytest.approx(0.7364, rel=0.01)       # ISA at 5,000 m


def test_correlations_against_independent_forms():
    re = 1.0e4                                                                   # Hilpert: C 0.193, m 0.618
    assert eh.nu_churchill_bernstein(re) == pytest.approx(0.193 * re ** 0.618 * 0.71 ** (1 / 3), rel=0.10)
    re = 5.0e4                                                                   # Dittus-Boelter (cooling)
    assert eh.nu_gnielinski(re) == pytest.approx(0.023 * re ** 0.8 * 0.71 ** 0.3, rel=0.15)
    assert eh.nu_flat_plate_laminar(1.0e5) == pytest.approx(0.664 * 1.0e5 ** 0.5 * 0.71 ** (1 / 3))
    assert eh.fin_efficiency(0.0) == 1.0 and eh.fin_efficiency(1.0) == pytest.approx(math.tanh(1.0))
    assert eh.cavity_emissivity(1.0, 0.2) == 1.0 and eh.cavity_emissivity(0.3, 1.0) == pytest.approx(0.3)
    assert eh.cavity_emissivity(0.3, 0.1) > 0.3                                  # a groove looks darker than its wall


def test_fin_mean_excess_is_efficiency_times_base():
    """Exact fin theory: the area-mean excess of a straight fin equals eta x the root excess, and the tip excess
    is root / cosh(m L_c)."""
    st = eh.fin_state(150.0, 44.0, 0.0, 288.15, 0.7, 140.0)
    assert st["theta_mean"] == pytest.approx(150.0 * st["eta"])
    m_lc = math.acosh(150.0 / st["theta_tip"])
    assert st["eta"] == pytest.approx(math.tanh(m_lc) / m_lc, rel=1e-9)


# ---------------------------------------------------------------- engine data and propeller
def test_published_engine_data_are_self_consistent():
    assert el.displacement_m3() * 1e6 == pytest.approx(548.0, rel=0.002)       # pi/4 B^2 S x 4
    wall = el.FIN_PACK["core_r"] - el.BORE_M / 2
    assert 0.004 <= wall <= 0.010                                                # cast barrel around a 66 mm bore
    assert el.rpm_at_power_fraction(1.0) == el.RATED_RPM
    lo, hi = CFG["target"]["propeller"]["prop_rpm"]
    assert el.rpm_at_power_fraction(0.15) == pytest.approx(lo, rel=0.01) and hi == el.RATED_RPM


def test_engine_geometry_matches_the_limbach_drawings():
    """Dimensions measured on the L550E / L550EF datasheet drawings (scale from their own dimension lines, +-1 mm)."""
    fp, hd, hs, st, pl = el.FIN_PACK, el.HEAD, el.HEATSINK, el.STUB, el.PLUG
    assert (fp["y1"] - fp["y0"]) / fp["fins"] * 1e3 == pytest.approx(6.67, abs=0.1)       # fin pitch
    assert fp["dx"] * 1e3 == pytest.approx(108, abs=2) and fp["dz"] * 1e3 == pytest.approx(114, abs=2)
    assert 2 * hd["y1"] * 1e3 == pytest.approx(314, abs=4)                                 # "without heat sinks"
    assert 2 * hs["y1"] * 1e3 == pytest.approx(364, abs=2) and hs["y0"] == hd["y1"]        # "with optional heat sinks"
    assert 2 * pl["y1"] == pytest.approx(el.WIDTH_M)                                         # 410 plug cap to cap
    assert 2 * st["r"] * 1e3 == pytest.approx(31.5, abs=1) and 2 * st["y"] * 1e3 == pytest.approx(178, abs=2)
    assert -st["z_flange"] * 1e3 == pytest.approx(67, abs=2) and -st["z_end"] * 1e3 == pytest.approx(162.5, abs=2)
    (_, _), (_, _), (z0, z1) = el.envelope()
    assert (z1 - z0) * 1e3 == pytest.approx(301.7, abs=2)                    # carburettor top to stack end
    x = sorted(xc for _, side, xc in el.CYLINDERS if side == +1)
    assert (x[1] - x[0]) * 1e3 == pytest.approx(110, abs=2)                  # cylinder pitch within a bank (side view)


def test_model_equals_the_saved_source_measurements():
    """docs/sources holds what tools/source_measurements.py read off the manufacturer's drawings and chart; the
    engine module must carry the same numbers (within the drawings' pixel resolution)."""
    import csv, json, os
    src = os.path.join(os.path.dirname(__file__), "..", "docs", "sources", "limbach")
    rows = list(csv.DictReader(open(os.path.join(src, "sfc_chart_digitised.csv"), encoding="utf-8")))
    chart = {int(r["rpm"]): (340.0 if "gridline" in r["sfc_g_per_hph"] else float(r["sfc_g_per_hph"])) for r in rows}
    assert chart == el.SFC_CHART_G_PER_HPH
    m = json.load(open(os.path.join(src, "measurements.json"), encoding="utf-8"))
    fp, st = el.FIN_PACK, el.STUB
    assert (fp["y1"] - fp["y0"]) / fp["fins"] * 1e3 == pytest.approx(m["ef_front"]["fin_pitch_mm"], abs=0.3)
    assert fp["fin_t"] * 1e3 == pytest.approx(m["ef_front"]["fin_face_spacing_mm"], abs=0.3)
    assert fp["dx"] * 1e3 == pytest.approx(m["ef_side"]["cylinder_end_width_mm"], abs=2)
    w = m["e_front"]["width_lines_mm"]                                       # 410 / heat sinks / none / stacks
    assert 2 * el.PLUG["y1"] * 1e3 == pytest.approx(w[0]) and 2 * el.HEATSINK["y1"] * 1e3 == pytest.approx(w[1], abs=1)
    assert 2 * el.HEAD["y1"] * 1e3 == pytest.approx(w[2], abs=1) and 2 * st["y"] * 1e3 == pytest.approx(w[3], abs=1)
    assert 2 * st["r"] * 1e3 == pytest.approx(m["e_front"]["stack_od_mm"], abs=0.5)
    dims = [tuple(d) for d in m["e_front"]["vertical_dims_mm"]]
    assert (0.0, 162.5) in dims and -st["z_end"] * 1e3 == pytest.approx(162.5)  # crank axis to stack end
    assert (-54.2, 59.8) in dims and fp["dz"] * 1e3 == pytest.approx(114.0, abs=1)   # fin plate height


# ---------------------------------------------------------------- sourced anchors -> model inputs
def test_fuel_chemistry_and_sourced_consumption():
    assert eh.afr_stoich(1.87) == pytest.approx(14.6, abs=0.05)                # petrol CH1.87
    assert eh.afr_stoich(18 / 8) == pytest.approx(15.13, abs=0.03)             # iso-octane C8H18
    lo, hi = eh.bsfc_l550e_chart()                                             # L550E chart: 335-434 g/HPh
    assert lo == pytest.approx(334.3 / 0.7457, rel=1e-6) and hi == pytest.approx(437.4 / 0.7355, rel=1e-6)
    r = eh.bsfc_rotax_503()                                                    # 25 / 15 l/h, 37 kW / 75 %
    assert r["cont75"][0] == pytest.approx(15 * 0.720 * 1000 / (0.75 * 37), rel=1e-9)
    assert eh.INPUTS["bsfc_g_kwh"] == (pytest.approx(r["cont75"][0]), pytest.approx(hi))
    a_lo, a_hi = eh.INPUTS["afr"]
    assert a_lo == pytest.approx(eh.afr_stoich(1.87) / 1.2) and a_hi == pytest.approx(eh.afr_stoich(2.25) / 1.0)
    # cooling load / shaft power = (10-11 % of the fuel energy) / brake efficiency 1 / (BSFC LHV)
    c_lo, c_hi = eh.INPUTS["cooling_over_shaft"]
    assert c_lo == pytest.approx(0.10 * eh.INPUTS["bsfc_g_kwh"][0] * eh.LHV_FUEL / 3.6e9)
    assert 0.4 < c_lo < c_hi < 0.85


def test_temperature_bands_come_from_the_rotax_limits():
    """Excess over a 15 C day of the Rotax 447/503/582 operator's manual limits (spark-plug-seat CHT, EGT)."""
    ps = eh.POWER_SETTINGS
    assert ps["nominal"]["cht_excess_k"] == (180 - 15, 230 - 15)               # 503 / 447 normal ranges
    assert ps["nominal"]["egt_excess_k"] == (460 - 15, 620 - 15)               # 503 / 582 normal ranges
    assert ps["high_power"]["cht_excess_k"] == (180 - 15, 260 - 15)            # 503 normal min -> 447 maximum
    # 503 normal max -> maximum; the manual prints that normal max as 580 C and as 1,000 F (538 C): both covered
    assert ps["high_power"]["egt_excess_k"] == (pytest.approx((1000 - 32) / 1.8 - 15), 650 - 15)
    assert ps["low_power"]["cht_excess_k"][1] == 230 - 15                      # reduced throttle can run as hot
    assert ps["low_power"]["cht_excess_k"][0] == pytest.approx((180 - 15) * 0.30 / 0.50)
    assert eh.CHT_LIMIT_C == eh.ROTAX["503"]["cht_max_c"] == 250.0              # = Limbach L 1700 TCDS limit


def test_emissivities_come_from_the_flir_table():
    e, r = eh.EMISSIVITY_SOURCES, eh.emissivity_ranges()
    assert eh.INPUTS["eps_stub"] == (e["stainless_untreated"], e["soot"])
    assert r["engine_head"] == (e["al_as_received"], e["al_anodised_dull"][1])
    assert r["engine_case"] == r["tail_cone"] == e["black_lacquer"]
    assert SIG["lwir"]["target"]["groups"]["skin"]["emissivity"] == list(e["black_lacquer"])


def test_propeller_inflow_is_momentum_theory():
    """Actuator disc: no power, no inflow; at the disc v_i = (sqrt(V^2 + 2T/(rho A)) - V) / 2 with T = P/V; ahead of
    it the vortex-cylinder decay 1 - x / sqrt(x^2 + R^2); never negative and below the disc value upstream."""
    assert eh.prop_inflow_ratio(0.0, 45.0, 0.0, 288.15, 0.58, 0.1) == 0.0
    p, v, d = 30_000.0, 45.0, 0.58
    rho, area = eh.density(288.15), math.pi * d ** 2 / 4
    v_i = 0.5 * (math.sqrt(v ** 2 + 2 * (p / v) / (rho * area)) - v)
    assert eh.prop_inflow_ratio(p, v, 0.0, 288.15, d, 0.0) == pytest.approx(v_i / v)
    up = eh.prop_inflow_ratio(p, v, 0.0, 288.15, d, 0.2)
    assert up == pytest.approx(v_i / v * (1 - 0.2 / math.hypot(0.2, d / 2))) and 0 < up < v_i / v


def test_propeller_diameter_range_keeps_tip_mach_subsonic():
    """Upper end of prop_dia_m: helical tip Mach at rated speed and the fastest airspeed stays <= 0.76 at ISA
    15 C and <= 0.81 on a -20 C day (props are kept below ~0.8-0.85)."""
    d_max = CFG["target"]["propeller"]["prop_dia_m"]["range"][1]
    v_max = CFG["render"]["target"]["airspeed_mps"]["range"][1]
    assert el.tip_mach(d_max, el.RATED_RPM, v_max, 288.15) <= 0.76
    assert el.tip_mach(d_max, el.RATED_RPM, v_max, 253.15) <= 0.81


# ---------------------------------------------------------------- consistency checks of the heat model
def test_flight_envelope_matches_render_config():
    assert tuple(eh.ENV["airspeed_mps"]) == tuple(CFG["render"]["target"]["airspeed_mps"]["range"])
    assert tuple(t - 273.15 for t in eh.ENV["t_air_k"]) == pytest.approx(tuple(CFG["render"]["air_temp_c"]["default"]))



@pytest.mark.parametrize("setting", list(eh.POWER_SETTINGS))
def test_exhaust_model_inputs_are_physical(setting):
    s = eh.POWER_SETTINGS[setting]
    for f in s["f"]:
        for bsfc in eh.INPUTS["bsfc_g_kwh"]:
            for afr in eh.INPUTS["afr"]:
                op = eh.operating_state(f, bsfc, afr)
                # published power <-> published displacement: crankcase-scavenged two-strokes run DR ~0.3 (1/4 throttle)
                # to 1.1-1.2 (full throttle) (Ausserer 2016, pp. 270 and 306)
                assert 0.25 <= op["delivery_ratio"] <= 1.25
                for egt in s["egt_excess_k"]:
                    assert eh.exhaust_share(egt, afr) <= 0.30         # energy left after shaft, walls, short-circuit
                    for v in eh.ENV["airspeed_mps"]:
                        st = eh.stub_state(egt, op["exhaust_kg_s_per_cyl"], v, 0.0, 288.15, 0.75, 1.6, 0.85)
                        assert st["re_i"] > 3000                      # Gnielinski range
                        assert 0.2 < st["phi"] < 0.85 and st["ntu"] < 0.2


def test_naca488_free_air_cooling_anchor():
    """NACA Report 488 (Biermann and Pinkel 1934): finned cylinders across a free air stream. Reference point read
    from its fig. 9 (q = 0.0892 B.t.u./sq.in./F/hr at 0.175 in fin space, 76 mph, 4.66 in cylinder, sea-level air),
    q ~ V^0.796 (fig. 10), q ~ s^0.322 (fig. 9), similitude in diameter (eq. 2-4) and density (eq. 6)."""
    c = eh.NACA488
    assert 1055.056 / (3600 * 0.0254 ** 2 * 5 / 9) == pytest.approx(817.67, rel=1e-4)   # B.t.u./sq.in./F/hr -> W/m2K
    assert eh.h_naca488(c["v_ref_mps"], c["rho_ref"], c["s_ref_m"], c["d_ref_m"]) == pytest.approx(0.0892 * 817.67)
    h1 = eh.h_naca488(20.0, 1.2, 0.004, 0.08)
    assert eh.h_naca488(40.0, 1.2, 0.004, 0.08) == pytest.approx(h1 * 2 ** 0.796)
    assert eh.h_naca488(20.0, 0.6, 0.004, 0.08) == pytest.approx(eh.h_naca488(10.0, 1.2, 0.004, 0.08))   # mass flow
    # the digitised figure reproduces the report's own velocity exponent and its quoted q at 76 mph
    import csv, os
    rows = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), "..", "docs", "sources", "naca",
                                                 "fig9_digitised.csv"), encoding="utf-8")))
    q175 = {float(r["air_speed_mph"]): float(r["q_at_0.175in"]) for r in rows}
    slope = np.polyfit(np.log(list(q175)), np.log(list(q175.values())), 1)[0]
    assert slope == pytest.approx(0.796, abs=0.02) and 0.086 <= q175[76.0] <= 0.092


@pytest.mark.parametrize("setting", list(eh.POWER_SETTINGS))
def test_free_air_cooling_model_in_its_validity_range(setting):
    s = eh.POWER_SETTINGS[setting]
    g = eh.fin_geometry()
    inch = 0.0254
    assert 0.10 * inch <= g["pitch_m"] <= 0.60 * inch and 0.04 * inch <= g["t_m"] <= 0.27 * inch     # NACA 488's
    assert 0.37 * inch <= g["length_m"] <= 1.47 * inch                                                # tested fins
    lo, hi = eh.NACA488["v_range_mps"]
    for f in s["f"]:
        for v in eh.ENV["airspeed_mps"]:
            for alt in eh.ENV["alt_m"]:
                for t in eh.ENV["t_air_k"]:
                    for kf in eh.INPUTS["k_fin"]:
                        for inflow in (0.0, 1.0):
                            k = kf + inflow * eh.inflow_bound(f, v, alt, t)
                            st = eh.fin_state(s["cht_excess_k"][1], v, alt, t, k, 120.0)
                            # equivalent sea-level speeds stay within -30 % / +10 % of the tested 30-150 mph
                            assert 0.7 * lo <= st["v_naca_eq"] <= 1.1 * hi
                            assert 0.8 < st["eta"] <= 1.0
    # independent physics agrees with the measurement at the envelope centre (laminar channel model vs NACA 488)
    mid = lambda r: 0.5 * (r[0] + r[1])  # noqa: E731
    t_f = mid(eh.ENV["t_air_k"]) + 0.5 * mid(s["cht_excess_k"])
    ch = eh.fin_channel_h(0.7 * mid(eh.ENV["airspeed_mps"]), g["flow_len_m"], g["gap_m"], t_f, mid(eh.ENV["alt_m"]), 1.0)
    st = eh.fin_state(mid(s["cht_excess_k"]), mid(eh.ENV["airspeed_mps"]), mid(eh.ENV["alt_m"]), mid(eh.ENV["t_air_k"]),
                      0.7, 140.0)
    assert ch["h"] == pytest.approx(st["h"], rel=0.30)


@pytest.mark.parametrize("setting", list(eh.POWER_SETTINGS))
def test_free_air_cooling_carries_the_sourced_load(setting):
    """Barrel fins (NACA 488) + heat sinks reject the sourced cooling load (AFIT share / Limbach SFC) inside the
    Rotax CHT band at the best-cooled corner of the flight envelope; at its centre the head runs towards the top of
    the band (more so at high power, as AFIT notes for pusher engines). The heat sinks and the propeller's inflow
    only help."""
    hb, hbi = eh.fin_heat_balance(setting), eh.fin_heat_balance(setting, with_inflow=True)
    hb0, best = eh.fin_heat_balance(setting, with_heatsink=False), eh.fin_heat_balance(setting, at="best")
    assert best["ratio"] <= 1.0
    assert 1.0 <= hbi["ratio"] <= hb["ratio"] < hb0["ratio"] <= 3.0
    assert 0.05 < hb["q_heatsink_w"] / (hb["q_barrel_w"] + hb["q_heatsink_w"]) < 0.30


def test_heatsinks_are_a_cooling_component():
    """Limbach's optional heat sinks (fitted: the 410 mm plug-cap width needs them) are their own surface class
    with a temperature from fin theory at the head and the black finish of the datasheet photos."""
    import kal_geometry as kg
    assert "engine_heatsink" in kg.ENGINE_SLOTS and "engine_heatsink" in kg.HOT_SLOTS
    assert sum(p["slot"] == "engine_heatsink" for p in el.parts()) == el.N_CYL
    for setting, r in eh.envelope_cache().items():
        assert r["engine_heatsink"][1] <= r["engine_head"][1] + 1e-9            # never hotter than the head
        assert r["engine_heatsink"][0] >= 0.75 * r["engine_head"][0]             # fin efficiency >= 0.75 at worst
    lo, hi = eh.emissivity_ranges()["engine_heatsink"]
    assert lo > 0.9 and hi <= 1.0                                                # black, slightly raised by the grooves
    caps = SIG["lwir"]["target"]["surface_temperature_cap_c"]
    assert caps["engine_heatsink"] == eh.CHT_LIMIT_C


def test_carburettor_evaporative_cooling():
    cool = [eh.carb_evaporative_cooling_k(a, h) for a in eh.INPUTS["afr"] for h in eh.H_FG_FUEL]
    assert 19.0 < min(cool) and max(cool) < 30.0


@pytest.mark.parametrize("setting", list(eh.POWER_SETTINGS))
def test_plume_lwir_contribution_is_small(setting):
    """8-14 um is the atmospheric window because H2O and CO2 hardly absorb there; by Kirchhoff the plume hardly
    emits there either. Even with a x3 margin and a generous plume area it stays below ~20 % of the stubs'
    contrast, inside the stubs' own emissivity uncertainty."""
    s = eh.POWER_SETTINGS[setting]
    egt = 0.5 * sum(s["egt_excess_k"])
    op = eh.operating_state(0.5 * sum(s["f"]), 450.0, 13.0)
    st = eh.stub_state(egt, op["exhaust_kg_s_per_cyl"], 47.5, 0.0, 288.15, 0.75, 1.6, 0.85)
    assert eh.plume_lwir_emissivity(st["theta_gas_out"]) < 0.02
    assert eh.plume_to_stub_lwir_ratio(st["theta_gas_out"], st["theta_wall"], 0.75) < 0.2


# ---------------------------------------------------------------- the YAML is the model's output
def test_signatures_yaml_equals_the_heat_model():
    prof = SIG["lwir"]["target"]["engine_profiles"]
    assert set(prof) == set(eh.POWER_SETTINGS)
    for name, ranges in eh.yaml_ranges().items():
        assert prof[name]["weight"] == pytest.approx(eh.POWER_SETTINGS[name]["weight"])
        for slot, (lo, hi) in ranges.items():
            assert prof[name][slot] == [pytest.approx(lo), pytest.approx(hi)], (name, slot)
    groups = SIG["lwir"]["target"]["groups"]
    for group, (lo, hi) in eh.emissivity_ranges().items():
        assert groups[group]["emissivity"] == [pytest.approx(lo), pytest.approx(hi)], group
    caps = SIG["lwir"]["target"]["surface_temperature_cap_c"]
    assert caps["engine_head"] == caps["engine_heatsink"] == caps["engine_barrel"] == eh.CHT_LIMIT_C


def test_every_drawn_temperature_fits_the_planck_tables():
    lut = lwir_lut()
    lo_air, hi_air = (t + 273.15 for t in CFG["render"]["air_temp_c"]["default"])
    prof = SIG["lwir"]["target"]["engine_profiles"]
    hottest = max(p["exhaust_stub"][1] for p in prof.values())
    coldest = min(p["engine_intake"][0] for p in prof.values())
    assert hi_air + hottest <= lut.t_max and lo_air + coldest >= lut.t_min


# ---------------------------------------------------------------- SWIR self-emission
def test_swir_threshold_and_target_emission_is_physics():
    he = SIG["swir"]["hot_emission"]
    units = he["nishita_sun_units"] / he["inband_solar_irradiance"]
    glow_min, glow_max = SIG["swir"]["night_airglow"]
    L = lambda tc: float(band_radiance_direct(np.array([tc + 273.15]), SWIR_BAND_UM, 2000)[0])  # noqa: E731
    assert L(he["min_temp_c"]) * units < 0.1 * glow_min                          # below the threshold: negligible
    assert L(he["min_temp_c"] + 10.0) * units > 0.1 * glow_min                    # ... and not far below it
    assert 0.5 * L(200.0) * units > 10 * glow_max                                 # a 200 C head glows at night
    hot = dict(reflectance=0.3, hot_emission_enabled=True)
    day = dict(swir_hot_emission=False)                                           # scene switch off (daylight)
    assert signatures.swir_self_emission(hot, 773.15, day, SIG) > 0               # target: emits anyway
    assert signatures.swir_self_emission(dict(hot, hot_emission_enabled=False), 773.15, day, SIG) == 0
    neg = dict(reflectance=0.3)                                                   # hard negative: follows the switch
    assert signatures.swir_self_emission(neg, 773.15, day, SIG) == 0
    assert signatures.swir_self_emission(neg, 773.15, dict(swir_hot_emission=True), SIG) > 0
    assert signatures.swir_self_emission(hot, he["min_temp_c"] + 273.15 - 1.0, day, SIG) == 0


def test_caps_apply_to_both_bands_and_keep_the_draw():
    lw = {"engine_head": {"dT": 230.0}}
    sw = {"engine_head": {"emit_dT": 240.0, "reflectance": 0.3}}
    signatures.apply_surface_caps(lw, 45.0, {"engine_head": 250.0})
    signatures.apply_surface_caps(sw, 45.0, {"engine_head": 250.0}, key="emit_dT")
    assert lw["engine_head"]["dT"] == pytest.approx(205.0) and lw["engine_head"]["dT_uncapped"] == 230.0
    assert sw["engine_head"]["emit_dT"] == pytest.approx(205.0) and sw["engine_head"]["emit_dT_capped"]
    assert sw["engine_head"]["emit_dT_uncapped"] == 240.0


def test_mwir_emissivities_come_from_the_flir_sw_entries():
    """signatures.yaml mwir.target.emissivity = engine_heat.emissivity_ranges_mwir() (FLIR Table 21.1 SW and T rows),
    and the cooled LWIR ranges contain the uncooled LWIR ones (same LW rows, painted lower bounds widened)."""
    sw = eh.EMISSIVITY_SOURCES_SW
    r = eh.emissivity_ranges_mwir()
    em = SIG["mwir"]["target"]["emissivity"]
    for group, (lo, hi) in r.items():
        assert em[group] == [pytest.approx(lo), pytest.approx(hi)], group
    assert r["skin"] == sw["paints"] and r["exhaust"][0] == sw["stainless_untreated"]
    assert r["engine_flange"][1] == sw["al_anodised_dull"][1] < eh.EMISSIVITY_SOURCES["al_anodised_dull"][1]
    lw, cool = SIG["lwir"]["target"]["groups"], SIG["lwir_cooled"]["target"]["emissivity"]
    for group, rng in cool.items():
        lo, hi = lw[group]["emissivity"]
        assert rng[0] <= lo + 1e-9 and rng[1] >= hi - 1e-9, group
