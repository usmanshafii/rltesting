"""Every derived engine number, saved as data: the heat-transfer derivation (thermal/engine_heat.py), the engine's
visible area on the mesh and its LWIR contrast against the airframe, the plume check, the night glow and the sun
glints. docs/engine_signature_physics.md quotes these files.

    python tools/engine_report.py              -> docs/reference/engine/data/
    python tools/engine_report.py --quick      (fewer Monte Carlo draws; for a check, not for the saved files)

Files (all committed, all re-made by this script):
  heat_report.txt       thermal/engine_heat.report(): sourced inputs, checks, heat balance, envelopes, YAML lines
  engine_profiles.csv   per power setting and surface: the model's envelope and the signatures.yaml range (K)
  heat_balance.csv      per power setting: cooling load against the free-air capacity of the barrel fins (NACA 488)
                        and the heat sinks, at the envelope centre, with the propeller inflow bound, without the
                        heat sinks and at the best-cooled corner
  emissivity.csv        LWIR emissivity range of each surface (FLIR table, groove cavities)
  cooling.csv           NACA 488 h and fin efficiencies of the barrel fins and heat sinks over the envelope, the
                        equivalent test speeds, and the laminar channel model against NACA 488 at the centre
  visible_area.csv      visible projected area (cm2) of each slot by aspect, ray-cast on the canonical mesh
  lwir_contrast.csv     engine vs airframe LWIR contrast intensity by aspect, power setting and thermal state
  plume.csv             exhaust plume / stub LWIR contrast at the band centres (best estimate and x3 margin); in
                        SWIR, Planck's gas / wall ratio and the plume emissivity at which the plume equals the stub
  night_glow.csv        RGB glow of the stubs: threshold temperatures and the share of scenes where it shows
  sun_glint.csv         glint draws by percentile and how much too hot a glint reads on each finish
  summary.json          the headline numbers

Contrast intensity of a set of surfaces: I = sum(A_visible x eps x (L(T_air + dT) - L(T_air))) in W/sr, with L the
8-14 um band radiance (Planck), at the centres of the signatures.yaml ranges and a 15 C day. Painted skin is blended
from its underside dT to its upper-surface dT_upper by t = (n_z + 1) / 2, as the LWIR shader does (render/lwir.py).
The reflected surroundings are left out (an emission contrast), the same for engine and airframe. The airframe is the
painted skin plus the tail cone, which is warmed by the engine (its own column; engine_and_tail_cone_over_skin counts
it with the engine). engine_over_airframe_previous_definition repeats the 2026-10-10 calculation (all other surfaces
at one underside skin temperature) so the effect of the heat sinks can be compared with the numbers quoted then.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "target")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import kal_geometry as kg  # noqa: E402
import kal_measure as km  # noqa: E402
from render import sampling  # noqa: E402
from render import signatures as sg  # noqa: E402
from thermal import engine_heat as eh  # noqa: E402
from thermal import sky as thsky  # noqa: E402
from thermal.planck_lut import LWIR_BAND_UM, band_radiance_direct  # noqa: E402

OUT = os.path.join(ROOT, "docs", "reference", "engine", "data")
T_AIR_K = 288.15
STATES = ("day", "crossover", "night")
# az 0 = camera dead ahead, 90 = port, 180 = behind; el + = camera above (kal_measure.direction)
ASPECTS = [(az, el) for el in (30, 0, -20, -30, -60) for az in (0, 45, 90, 135, 180)] + [(0, -90)]
NOT_COUNTED = ("propeller", "propeller_disc")          # neither engine nor airframe skin (blades near ambient)


def mid(r):
    return 0.5 * (float(r[0]) + float(r[1]))


def L(t_k):
    return float(band_radiance_direct(np.array([t_k]), LWIR_BAND_UM, 400)[0])


def _write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _r(x, n=3):
    return round(float(x), n)


# ============================================================================ heat model
def heat_tables():
    profiles, balance = [], []
    yr = eh.yaml_ranges()
    for name, env in eh.envelope_cache().items():
        for surf, (lo, hi) in env.items():
            profiles.append(dict(setting=name, surface=surf, model_min_k=_r(lo, 1), model_max_k=_r(hi, 1),
                                 yaml_min_k=yr[name][surf][0], yaml_max_k=yr[name][surf][1]))
        for case, kw in (("centre", {}), ("centre_with_inflow_bound", dict(with_inflow=True)),
                         ("centre_without_heatsinks", dict(with_heatsink=False)), ("best_corner", dict(at="best"))):
            hb = eh.fin_heat_balance(name, **kw)
            balance.append(dict(setting=name, case=case, load_per_cyl_kw=_r(hb["q_per_cyl_w"] / 1e3),
                                barrel_fins_kw=_r(hb["q_barrel_w"] / 1e3), heatsink_kw=_r(hb["q_heatsink_w"] / 1e3),
                                h_barrel_w_m2k=_r(hb["h_barrel"], 1), load_over_capacity=_r(hb["ratio"])))
    emis = [dict(surface=k, eps_min=lo, eps_max=hi) for k, (lo, hi) in eh.emissivity_ranges().items()]
    return profiles, balance, emis


def cooling_table():
    """Free-air cooling over the flight envelope and the installation factors, per power setting: the NACA 488 h on
    the barrel fins and the heat sinks, their fin efficiencies, the sea-level speed of the same mass flow (NACA 488
    tested 30-150 mph = 13-67 m/s), and the independent laminar channel model against NACA 488 at the centre."""
    rows = []
    for name, s in eh.POWER_SETTINGS.items():
        fb, fh = [], []
        for c in eh._corners(dict(v=eh.ENV["airspeed_mps"], alt=eh.ENV["alt_m"], t=eh.ENV["t_air_k"],
                                  kf=eh.INPUTS["k_fin"], f=s["f"], inflow=(0.0, 1.0), kal=eh.INPUTS["k_al"],
                                  hsf=eh.INPUTS["hs_h_factor"])):
            k = c["kf"] + c["inflow"] * eh.inflow_bound(c["f"], c["v"], c["alt"], c["t"])
            fb.append(eh.fin_state(s["cht_excess_k"][1], c["v"], c["alt"], c["t"], k, c["kal"]))
            fh.append(eh.heatsink_state(s["cht_excess_k"][1], c["v"], c["alt"], c["t"], k, c["hsf"], c["kal"]))
        g = eh.fin_geometry()
        t_f = mid(eh.ENV["t_air_k"]) + 0.5 * mid(s["cht_excess_k"])
        ch = eh.fin_channel_h(0.7 * mid(eh.ENV["airspeed_mps"]), g["flow_len_m"], g["gap_m"], t_f, mid(eh.ENV["alt_m"]),
                              1.0)
        st = eh.fin_state(mid(s["cht_excess_k"]), mid(eh.ENV["airspeed_mps"]), mid(eh.ENV["alt_m"]),
                          mid(eh.ENV["t_air_k"]), 0.7, 140.0)
        rows.append(dict(setting=name,
                         barrel_h_min=_r(min(x["h"] for x in fb), 1), barrel_h_max=_r(max(x["h"] for x in fb), 1),
                         barrel_eta_min=_r(min(x["eta"] for x in fb)), barrel_eta_max=_r(max(x["eta"] for x in fb)),
                         v_naca_eq_min_mps=_r(min(x["v_naca_eq"] for x in fb), 1),
                         v_naca_eq_max_mps=_r(max(x["v_naca_eq"] for x in fb), 1),
                         heatsink_h_min=_r(min(x["h"] for x in fh), 1), heatsink_h_max=_r(max(x["h"] for x in fh), 1),
                         heatsink_eta_min=_r(min(x["eta"] for x in fh)), heatsink_eta_max=_r(max(x["eta"] for x in fh)),
                         centre_h_naca488=_r(st["h"], 1), centre_h_channel_model=_r(ch["h"], 1),
                         channel_over_naca=_r(ch["h"] / st["h"])))
    return rows


def geometry_summary():
    g, h, s = eh.fin_geometry(), eh.heatsink_geometry(), eh.stub_geometry()
    mm = lambda x: _r(x * 1e3, 2)  # noqa: E731
    return dict(barrel_fins=dict(fins=g["n"], area_m2_per_cyl=_r(g["area_m2"], 4), fin_height_mm=mm(g["length_m"]),
                                 thickness_mm=mm(g["t_m"]), pitch_mm=mm(g["pitch_m"]), gap_mm=mm(g["gap_m"]),
                                 flow_length_mm=mm(g["flow_len_m"])),
                heatsink=dict(fins=int(eh.el.HEATSINK["fins"]), area_m2_per_cyl=_r(h["area_m2"], 4),
                              fin_depth_mm=mm(h["depth_m"]), thickness_mm=mm(h["t_m"]), pitch_mm=mm(h["pitch_m"]),
                              gap_mm=mm(h["gap_m"])),
                exhaust_stack=dict(od_mm=mm(s["d_o"]), id_mm=mm(s["d_i"]), length_mm=mm(s["length"])))


def plume_table():
    rows = []
    for name, s in eh.POWER_SETTINGS.items():
        op = eh.operating_state(mid(s["f"]), mid(eh.INPUTS["bsfc_g_kwh"]), mid(eh.INPUTS["afr"]))
        for eps in (eh.INPUTS["eps_stub"][0], mid(eh.INPUTS["eps_stub"]), eh.INPUTS["eps_stub"][1]):
            st = eh.stub_state(mid(s["egt_excess_k"]), op["exhaust_kg_s_per_cyl"], mid(eh.ENV["airspeed_mps"]), 0.0,
                               T_AIR_K, eps, mid(eh.INPUTS["f_pulse"]), mid(eh.INPUTS["k_stub"]))
            row = dict(setting=name, eps_stub=_r(eps), stub_wall_excess_k=_r(st["theta_wall"], 1),
                       gas_at_outlet_excess_k=_r(st["theta_gas_out"], 1))
            for safety in (1.0, 3.0):
                row[f"plume_eps_x{safety:.0f}"] = _r(eh.plume_lwir_emissivity(st["theta_gas_out"], T_AIR_K,
                                                                               safety=safety), 4)
                row[f"plume_over_stub_x{safety:.0f}"] = _r(eh.plume_to_stub_lwir_ratio(
                    st["theta_gas_out"], st["theta_wall"], eps, T_AIR_K, safety=safety))
            # SWIR (not calculated for want of a sourced hot-water band model): Planck alone gives the gas / wall
            # radiance ratio in 0.9-1.7 um and the band-average plume emissivity at which the plume would equal the stub
            swir = (band_radiance_direct(np.array([T_AIR_K + st["theta_gas_out"]]), (0.9, 1.7), 2000)[0]
                    / band_radiance_direct(np.array([T_AIR_K + st["theta_wall"]]), (0.9, 1.7), 2000)[0])
            row["swir_gas_over_wall_planck"] = _r(swir, 1)
            row["swir_break_even_plume_eps_pct"] = _r(100.0 * eps / (3.2 * swir), 3)
            rows.append(row)
    return rows


# ============================================================================ visibility and LWIR contrast
def contrast_tables(sig):
    cfg = kg.load_config()
    mesh = kg.build_mesh(kg.defaults(cfg))
    dirs = np.array([km.direction(az, el) for az, el in ASPECTS])
    vis = km.visible_triangle_area(mesh, dirs)
    area, slot_idx, nz = vis["area"], vis["slot"], vis["normal"][:, 2]
    slots = [kg.SLOTS[i] for i in slot_idx]
    lw = sig["lwir"]["target"]
    engine = set(kg.ENGINE_SLOTS)
    la = L(T_AIR_K)
    vis_rows = []
    for k, (az, el) in enumerate(ASPECTS):
        row = dict(azimuth_deg=az, elevation_deg=el)
        for s in kg.SLOTS:
            row[s] = _r(area[k][[x == s for x in slots]].sum() * 1e4, 1)
        vis_rows.append(row)
    rows = []
    for setting, prof in lw["engine_profiles"].items():
        for state in STATES:
            # per-triangle emission contrast: eps x (L(T_air + dT) - L(T_air)), skin blended by the normal
            per_tri = np.zeros(len(slots))
            for i, s in enumerate(slots):
                if s in NOT_COUNTED:
                    continue
                g = lw["groups"][lw["slots"][s]]
                eps = mid(g["emissivity"])
                if "dT_profile" in g:
                    per_tri[i] = eps * (L(T_AIR_K + mid(prof[g["dT_profile"]])) - la)
                    continue
                lo = mid(g["dT"][state] if isinstance(g["dT"], dict) else g["dT"])
                up = mid(g["dT_upper"][state]) if "dT_upper" in g else lo
                t = 0.5 * (nz[i] + 1.0)
                per_tri[i] = eps * ((1 - t) * (L(T_AIR_K + lo) - la) + t * (L(T_AIR_K + up) - la))
            # previous definition (2026-10-10, before the heat sinks): every non-engine slot but the disc, decals
            # and window at the underside day centre (+2.75 K) and eps 0.9; kept so the two can be compared
            prev = np.where([s not in engine and s not in ("propeller_disc", "decal", "nose_window") for s in slots],
                            0.9 * (L(T_AIR_K + 2.75) - la), 0.0)
            is_eng = np.array([s in engine for s in slots])
            is_tail = np.array([s == "tail_cone" for s in slots])
            is_skin = np.array([s not in engine and s not in NOT_COUNTED and s != "tail_cone" for s in slots])
            is_hs = np.array([s == "engine_heatsink" for s in slots])
            is_barrel = np.array([s == "engine_barrel" for s in slots])
            for k, (az, el) in enumerate(ASPECTS):
                ie, it, isk = (float(area[k][m] @ per_tri[m]) for m in (is_eng, is_tail, is_skin))
                ihs, ib = float(area[k][is_hs] @ per_tri[is_hs]), float(area[k][is_barrel] @ per_tri[is_barrel])
                ip = float(area[k] @ prev)
                rows.append(dict(setting=setting, state=state, azimuth_deg=az, elevation_deg=el,
                                 engine_area_cm2=_r(area[k][is_eng].sum() * 1e4, 1),
                                 tail_cone_area_cm2=_r(area[k][is_tail].sum() * 1e4, 1),
                                 skin_area_cm2=_r(area[k][is_skin].sum() * 1e4, 1),
                                 engine_w_sr=_r(ie, 4), tail_cone_w_sr=_r(it, 4), skin_w_sr=_r(isk, 4),
                                 engine_over_airframe=_r(ie / (it + isk), 2) if it + isk > 0 else "",
                                 engine_and_tail_cone_over_skin=_r((ie + it) / isk, 2) if isk > 0 else "",
                                 engine_over_airframe_previous_definition=_r(ie / ip, 2) if ip > 0 else "",
                                 heatsink_share_of_engine=_r(ihs / ie) if ie > 0 else "",
                                 barrel_share_of_engine=_r(ib / ie) if ie > 0 else ""))
    return vis_rows, rows


# ============================================================================ night glow (RGB)
def glow_table(cfg, n):
    """Monte Carlo of the scene sampler's own draws (render/sampling.sample_scene): lighting, air temperature, the
    RGB pass's engine-profile and temperature draw (signatures.add_rgb_glow_draw) with the caps, the stub albedo, and
    the scene's night-sky floor (render.yaml airglow_rgb). The glow shows where the stub's red-band radiance passes
    that floor."""
    sig, R = cfg["signatures"], cfg["render"]
    rng = np.random.default_rng(2026)
    caps = sig["lwir"]["target"]["surface_temperature_cap_c"]
    he = sig["rgb"]["hot_emission"]
    exh = sig["rgb"]["target"]["groups"]["exhaust"]
    at = R["air_temp_c"]
    floor = R["sky"]["night"]["airglow_rgb"]
    counts = dict(scenes=0, night=0, stub_above_threshold=0, glow_over_scene_sky=0, glow_over_faintest_sky=0)
    hottest = []
    for _ in range(n):
        lighting = sampling._choice(rng, R["dataset"]["lighting"])
        state = sig["lwir"]["states"][lighting]
        air_c = float(rng.uniform(*(at["desert_day"] if lighting == "day" and rng.uniform() < at["desert_day_prob"]
                                    else at["default"])))
        target = {"rgb": {s: {} for s in sig["lwir"]["target"]["slots"]}}
        sg.add_rgb_glow_draw(rng, target, sig, state)
        sg.apply_surface_caps(target["rgb"], air_c, caps, key="glow_dT")
        t_stub = air_c + 273.15 + target["rgb"]["exhaust_stub"]["glow_dT"]
        p = dict(albedo=float(rng.uniform(*exh["albedo"])))
        sky = sampling._logu(rng, *floor)
        counts["scenes"] += 1
        counts["night"] += int(lighting == "night")
        hottest.append(t_stub - 273.15)
        if t_stub < he["min_temp_c"] + 273.15:
            continue
        counts["stub_above_threshold"] += 1
        if lighting != "night":
            continue
        r = sg.rgb_self_emission(p, t_stub, sig)[0]
        counts["glow_over_scene_sky"] += int(r > sky)
        counts["glow_over_faintest_sky"] += int(r > floor[0])
    # temperatures at which a black stub's red glow passes the faintest and the brightest night-sky floor
    def t_for(level):
        lo, hi = he["min_temp_c"] + 273.15, 1200.0
        for _ in range(60):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if sg.rgb_self_emission(dict(albedo=0.0), m, sig)[0] < level else (lo, m)
        return 0.5 * (lo + hi) - 273.15
    rows = [dict(quantity=k, value=v) for k, v in counts.items()]
    rows += [dict(quantity="share_of_all_scenes_glow_over_scene_sky", value=_r(counts["glow_over_scene_sky"] / n, 4)),
             dict(quantity="share_of_all_scenes_glow_over_faintest_sky",
                  value=_r(counts["glow_over_faintest_sky"] / n, 4)),
             dict(quantity="share_of_night_scenes_glow_over_scene_sky",
                  value=_r(counts["glow_over_scene_sky"] / max(counts["night"], 1), 4)),
             dict(quantity="threshold_min_temp_c", value=he["min_temp_c"]),
             dict(quantity="black_stub_c_for_faintest_sky", value=_r(t_for(floor[0]), 1)),
             dict(quantity="black_stub_c_for_brightest_sky_floor", value=_r(t_for(floor[1]), 1)),
             dict(quantity="hottest_stub_draw_c", value=_r(max(hottest), 1)),
             dict(quantity="stub_draw_c_p99", value=_r(np.percentile(hottest, 99), 1))]
    return rows


# ============================================================================ sun glints (LWIR)
def _inverse_band(levels):
    t = np.arange(200.0, 3000.0, 0.5)
    lt = band_radiance_direct(t, LWIR_BAND_UM, 400)
    return np.interp(levels, lt, t)


def glint_table(cfg, n):
    """Monte Carlo of the scene's glint draw: lighting, sun elevation, air temperature and humidity -> the Idso
    clear-sky zenith emittance -> in-band sun at the ground (signatures.sun_glint_params). Then how hot a glint at
    the mirror angle makes a 300 K surface read, for each finish's emissivity (centre of its range)."""
    sig, R = cfg["signatures"], cfg["render"]
    rng = np.random.default_rng(41)
    at = R["air_temp_c"]
    sky = sig["lwir"]["environment"]["sky"]
    draws = []
    for _ in range(n):
        lighting = sampling._choice(rng, R["dataset"]["lighting"])
        sun_el = float(rng.uniform(*R["sky"]["sun_elevation_deg"][lighting]))
        if sun_el <= 0:
            continue
        state = sig["lwir"]["states"][lighting]
        air_c = float(rng.uniform(*(at["desert_day"] if lighting == "day" and rng.uniform() < at["desert_day_prob"]
                                    else at["default"])))
        hs = thsky.humidity_state(air_c, float(rng.uniform(*sky["rh"][state])), sky.get("dew_point_max_c"))
        g = sg.sun_glint_params(rng, sig, sun_el, hs["emissivity_zenith"])
        draws.append((g["peak_per_reflectance"], g["transmittance"], g["lobe_deg"]))
    d = np.array(draws)
    finishes = dict(cast_al_as_received=0.09, stainless_untreated=0.28, cast_al_blast_cleaned=0.46,
                    carburettor_mid=mid(sig["lwir"]["target"]["groups"]["engine_intake"]["emissivity"]),
                    stub_mid=mid(sig["lwir"]["target"]["groups"]["exhaust"]["emissivity"]),
                    skin_paint_mid=mid(sig["lwir"]["target"]["groups"]["skin"]["emissivity"]),
                    heatsink_mid=mid(sig["lwir"]["target"]["groups"]["engine_heatsink"]["emissivity"]))
    l300 = L(300.0)
    rows = [dict(quantity="scenes_with_sun_above_horizon_share", percentile="", value=_r(len(d) / n, 4))]
    for pc in (10, 50, 90, 99, 100):
        peak, tau, lobe = (float(np.percentile(d[:, j], pc)) for j in range(3))
        rows.append(dict(quantity="peak_per_reflectance_w_m2_sr", percentile=pc, value=_r(peak, 2)))
        rows.append(dict(quantity="transmittance", percentile=pc, value=_r(tau, 3)))
        rows.append(dict(quantity="lobe_deg", percentile=pc, value=_r(lobe, 2)))
        for f, eps in finishes.items():
            t_app = float(_inverse_band(l300 + (1.0 - eps) * peak))
            rows.append(dict(quantity=f"reads_too_hot_k_{f}_eps{eps:.2f}", percentile=pc, value=_r(t_app - 300.0, 1)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n = 4000 if a.quick else 100000
    out = OUT if not a.quick else os.path.join(ROOT, "out", "engine_report_quick")
    globals()["OUT"] = out
    os.makedirs(out, exist_ok=True)
    cfg = sampling.load_configs(overlay="")
    sig = cfg["signatures"]
    with open(os.path.join(out, "heat_report.txt"), "w", encoding="utf-8") as f:
        f.write(eh.report() + "\n")
    profiles, balance, emis = heat_tables()
    _write_csv("engine_profiles.csv", profiles)
    _write_csv("heat_balance.csv", balance)
    _write_csv("emissivity.csv", emis)
    cooling = cooling_table()
    _write_csv("cooling.csv", cooling)
    plume = plume_table()
    _write_csv("plume.csv", plume)
    vis_rows, contrast = contrast_tables(sig)
    _write_csv("visible_area.csv", vis_rows)
    _write_csv("lwir_contrast.csv", contrast)
    glow = glow_table(cfg, n)
    _write_csv("night_glow.csv", glow)
    glint = glint_table(cfg, n)
    _write_csv("sun_glint.csv", glint)

    def pick(setting, state, az, el):
        return next(r for r in contrast if (r["setting"], r["state"], r["azimuth_deg"], r["elevation_deg"])
                    == (setting, state, az, el))
    key_aspects = [(0, 0), (90, -30), (180, -20), (0, -90)]
    summary = dict(
        model_revision=sig["model_revision"],
        geometry=geometry_summary(),
        heat_balance_ratio={r["setting"] + "/" + r["case"]: r["load_over_capacity"] for r in balance},
        engine_over_airframe_day={f"{s} az{az} el{el}": pick(s, "day", az, el)["engine_over_airframe"]
                                  for s in sig["lwir"]["target"]["engine_profiles"] for az, el in key_aspects},
        engine_and_tail_cone_over_skin_day={f"{s} az{az} el{el}": pick(s, "day", az, el)["engine_and_tail_cone_over_skin"]
                                            for s in sig["lwir"]["target"]["engine_profiles"] for az, el in key_aspects},
        engine_over_airframe_previous_definition_day={
            f"{s} az{az} el{el}": pick(s, "day", az, el)["engine_over_airframe_previous_definition"]
            for s in sig["lwir"]["target"]["engine_profiles"] for az, el in key_aspects},
        heatsink_share_nominal_day={f"az{az} el{el}": pick("nominal", "day", az, el)["heatsink_share_of_engine"]
                                    for az, el in key_aspects},
        plume_over_stub={f"{r['setting']} eps{r['eps_stub']}": [r["plume_over_stub_x1"], r["plume_over_stub_x3"]]
                         for r in plume},
        night_glow={r["quantity"]: r["value"] for r in glow},
        sun_glint_reads_too_hot_k={f"{r['quantity'][len('reads_too_hot_k_'):]} p{r['percentile']}": r["value"]
                                   for r in glint if r["percentile"] in (50, 99) and r["quantity"].startswith("reads")},
        monte_carlo_draws=n)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("written to", out)


if __name__ == "__main__":
    main()
