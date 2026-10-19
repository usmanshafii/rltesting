"""Engine surface temperatures from heat transfer: the physics behind configs/signatures.yaml (pure NumPy).

LWIR records emissivity x Planck radiance of each surface (plus the reflected surroundings); SWIR adds the
self-emission of parts above ~250 C. Both depend on the surface temperatures of the MD-550 / L550E
installation, so instead of guessing them this module derives them for each power setting from the
engine's air flow and heat load (target/engine_l550e.py ratings and geometry) and from standard convection
correlations. Every range in configs/signatures.yaml lwir.target.engine_profiles comes from here:

    python thermal/engine_heat.py        derivation table, consistency checks and the YAML lines

tests/test_engine_heat.py fails when the YAML and this model disagree. The renderer samples the YAML ranges
(fast, readable, no solver in the render loop). Temperatures are excesses over ambient air, theta = T - T_air:
every heat source and sink below scales with that difference, so excesses hold across ambient temperatures
(the ambient only enters through the air properties, swept over -20..+50 C and 0..5,000 m).
docs/engine_signature_physics.md walks through the same equations with numbers and sources.

 Sources (tiers S1-S4 as in CLAUDE.md; full list and page references in docs/engine_signature_physics.md 2b):
   [L550E-DS]  Limbach L550E datasheet (manufacturer, S1 for the L550E): ratings, SFC chart, drawing.
   [ROTAX-OM]  BRP-Rotax Operator's Manual 447 UL / 503 UL / 582 UL (manufacturer, S2: closest analogue with
               published limits; the 503 UL is an air-cooled carburetted two-stroke of 37 kW take-off power).
   [EASA-E082] EASA type-certificate data sheet E.082, Limbach L 1700 (regulator, S2: Limbach's own engine).
   [AUSSERER]  Ausserer (2016), AFIT PhD dissertation AFIT-ENY-DS-16-S-055, measured energy balances of 28-85 cm3
               air-cooled two-strokes (peer reviewed as J. Eng. Gas Turbines Power 140, 112803, 2018; S2).
   [FLIR-TAB]  FLIR ETS3xx user's manual T810252 (2017) Table 21.1, referenced emissivities (S2/S3).
   [NACA-488]  Biermann and Pinkel (1934), NACA Report 488: wind-tunnel heat transfer of finned cylinders across a
               free air stream, no baffles (S2, high relevance: how the L550E is cooled).

 S1 Operating state. Shaft power P = f x P_rated, f drawn per power setting. Fuel flow m_f = P x BSFC,
    exhaust flow per cylinder m_ex = m_f (1 + AFR) / 4, shaft speed n = n_rated f^(1/3) (fixed-pitch
    propeller, P ~ n^3). BSFC: the [L550E-DS] full-throttle chart and the [ROTAX-OM] consumptions. AFR: the
    stoichiometric ratio from fuel chemistry over the equivalence ratio of carburetted small two-strokes
    [AUSSERER]. Check: delivery ratio DR = AFR m_f / (rho_a V_d n) must be 0.25-0.95, the range of a
    crankcase-scavenged two-stroke; it ties the published displacement to the published power.
 S2 Exhaust stubs (thin steel tubes, hot gas inside, cross-flow outside). Wall excess theta_w = phi theta_g,
    phi = h_i' / (h_i' + h_o + h_r) with h_i' = h_i D_i / D_o per outer area. h_i: Gnielinski pipe flow x
    entrance factor (1 + (D/L)^0.7) x a pulsation factor (blow-down pulses); h_o: Churchill-Bernstein
    cylinder in cross-flow; h_r = eps sigma (T_w^2 + T_a^2)(T_w + T_a). The gas cools along the stub,
    theta_g(x) = theta_g0 exp(-NTU x / L), NTU = U pi D_o L / (m_ex c_p), so the visible mean is
    phi theta_g0 (1 - e^-NTU) / NTU. Gas temperature at the port theta_g0: the EGT band of the setting, from
    the [ROTAX-OM] limits; check: implied sensible exhaust share x_ex = theta_g0 (1 + AFR) c_p / LHV must stay
    below the share left after shaft work, cooling and the unburnt (short-circuited and incompletely burnt)
    charge [AUSSERER] (<= 0.30).
 S3 Cylinder fins and heads. Base excess = cylinder-head temperature excess (CHT band: the range the engine
    is operated in; [ROTAX-OM] normal and maximum CHT at the spark-plug seat, which is where the [L550E-DS]
    14 mm CHT thermocouple sits). Fins are straight fins of corrected length L_c = L + t/2,
    efficiency eta = tanh(m L_c) / (m L_c), m = sqrt(2 h / (k t)); a fin's area-weighted mean excess is
    exactly eta x its base excess. h on the fins: the free-air cooling of the L550E (no fan or shroud), measured by
    [NACA-488] for finned cylinders across a free air stream, h ~ V^0.796 gap^0.322 with similitude in diameter and
    density (h_naca488); the laminar channel model (fin_channel_h) is kept as an independent cross-check. Local
    velocity = flight speed x (wake factor + the pusher propeller's inflow, bounded by momentum theory). Heat sinks
    (Limbach's option, fitted): fins across the stream at the head temperature, h = hs_h_factor x the NACA 488
    value. Check: the barrel fins plus the heat sinks must reject the cooling load Q = (cooling share [AUSSERER] /
    brake efficiency) x shaft power inside the CHT band (fin_heat_balance: load / capacity reported at the
    envelope centre, with the inflow bound, without the heat sinks and at the best-cooled corner). Fin and heat-sink
    geometry: measured on the [L550E-DS] drawings (engine_l550e.FIN_PACK, HEATSINK).
    Apparent emissivity of the fin pack: an isothermal-groove cavity, eps_a = eps / (eps + (1 - eps) F),
    F = gap / (gap + 2 L), area-weighted with the fin edges.
 S4 Case, drive flange, tail cone: conduction-fed, cooled by the charge and the air stream; fractions of the
    CHT / case excess (C: no measurement of these parts on a comparable engine was found).
 S5 Carburettors. Fuel evaporation cools the charge by up to dT_evap = h_fg / (AFR c_p) (~20-29 K); the
    body sits between charge and ambient, so it runs below ambient (the physics of carburettor icing).
 S6 Exhaust plume: a quick 8-14 um bound here (water-vapour continuum of thermal/atmosphere.py, Roberts et al.
    1976, plus the CO2 band edge) gives the plume-to-stub LWIR contrast ratio. The rendered plume is
    thermal/plume.py: the exhaust's CO2, H2O and CO through SP-3080 / HITRAN narrow bands, cut at the pusher
    propeller's plane, in all thermal bands. It confirms the bound in LWIR (<= 2 % of the stubs) and makes the
    plume a major MWIR source (CO2 4.3 um: 10-70 % of the stubs at the source, before the air's own CO2 absorbs
    the band centre).
"""
from __future__ import annotations

import functools
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, os.path.join(ROOT, "target")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import engine_l550e as el  # noqa: E402

SIGMA = 5.670374419e-8          # Stefan-Boltzmann constant, W m-2 K-4
R_AIR = 287.05                  # J kg-1 K-1
CP_AIR = 1006.0                 # J kg-1 K-1, air near ambient
CP_EXHAUST = 1150.0             # J kg-1 K-1, combustion products at 600-900 K (Heywood 1988, ch. 4)
PR = 0.71                       # Prandtl number of air and of the exhaust gas
LHV_FUEL = 43.0e6               # J kg-1, gasoline (Heywood 1988, app. D); the 2-4 % two-stroke oil changes it < 1 %
H_FG_FUEL = (300e3, 350e3)      # J kg-1, gasoline heat of vaporisation (Heywood 1988, app. D)

# Stub wall: 1.5 mm steel tube (C, typical exhaust tubing); the stub geometry is engine_l550e.STUB.
STUB_WALL_M = 0.0015

# ---------------------------------------------------------------- environment envelope (render.yaml)
ENV = dict(airspeed_mps=(39.0, 56.0),            # render.yaml airspeed range
           alt_m=(0.0, 5000.0),                  # sea level to the reported operating ceiling (render.yaml note)
           t_air_k=(253.15, 323.15))             # render.yaml air_temp_c -20..+50 C

# ================================================================ sourced anchors (docs/engine_signature_physics.md 2b)
T_REF_C = 15.0      # ISA sea-level day: the manuals' absolute temperatures become excesses over this air temperature
# [ROTAX-OM] BRP-Rotax, Operator's Manual for engine types 447 UL SCDI, 503 UL DCDI, 582 UL DCDI (initial issue
# 1999-05-01), sec. 10 items 4-5, p. 10-1: cylinder-head temperature read at the spark-plug seat, normal and maximum;
# exhaust gas temperature normal and maximum. Sec. 10.3.7, p. 10-15: CHT and EGT can be HIGHER at reduced throttle
# openings (more hot residual gas). The 582 is liquid cooled, so only its EGT is used. The manual's own table prints
# the 503's normal EGT as 460-580 C next to 860-1000 F, and 1,000 F is 538 C: the two readings are both kept (the
# evidence rule widens a range to cover a conflict it cannot resolve).
ROTAX = {"447": dict(cht_normal_c=(190.0, 230.0), cht_max_c=260.0),
         "503": dict(cht_normal_c=(180.0, 220.0), cht_max_c=250.0, egt_normal_c=(460.0, 580.0),
                     egt_normal_f=(860.0, 1000.0), egt_max_c=650.0),
         "582": dict(egt_normal_c=(500.0, 620.0), egt_max_c=650.0)}


def f_to_c(f: float) -> float:
    return (f - 32.0) / 1.8
# [ROTAX-OM] sec. 8.3, p. 8-1, 503 UL DCDI: 25 l/h at take-off (37 kW, sec. 10 item 2), 15 l/h at 75 % continuous
# power, specific consumption 550 g/kWh. Petrol density 720-775 kg/m3 at 15 C (EN 228 limits) turns l/h into g/kWh.
ROTAX_503_FUEL = dict(takeoff_l_h=25.0, cont75_l_h=15.0, sfc_g_kwh=550.0, takeoff_kw=37.0)
FUEL_DENSITY_KG_L = (0.720, 0.775)
HP_KW = (0.73550, 0.74570)          # "HP" of the [L550E-DS] chart read as metric PS or as mechanical hp
# [AUSSERER] p. v: at peak power the heat lost from the cylinder (the cooling load, friction included) is 10-11 % of
# the fuel energy in 28-85 cm3 air-cooled two-strokes, and thermal losses do not grow until displacement falls
# below ~10 cm3, so the share carries to the L550E's 137 cm3 cylinders. Pp. 307-311: at part throttle the share
# rises while short-circuiting falls (the low end of cooling_over_shaft is therefore a bound).
COOLING_SHARE = (0.10, 0.11)
# [AUSSERER] p. 316: small engines are air carburetted in their stock form and run at equivalence ratios 1.1-1.2;
# pp. 284-285: run rich for cooling, tuned to 1.1, a simple carburettor swinging 0.8-1.2 over the speed range;
# peak power near 1.05. Aero practice also forbids lean running ([ROTAX-OM] 10.3.7). Range used: 1.0-1.2.
PHI = (1.0, 1.2)
FUEL_H_TO_C = (1.87, 2.25)          # petrol ~CH1.87; iso-octane C8H18 (AVGAS 100LL is mostly iso-paraffins)


def afr_stoich(h_to_c: float) -> float:
    """Stoichiometric air/fuel mass ratio of CH_y from the oxygen balance: (1 + y/4) mol O2 per mol C, air 20.95 %
    O2 by volume and 28.96 g/mol; carbon 12.011, hydrogen 1.008 g/mol. CH1.87 -> 14.6, C8H18 -> 15.1."""
    return (1.0 + h_to_c / 4.0) / 0.2095 * 28.96 / (12.011 + 1.008 * h_to_c)


def bsfc_l550e_chart() -> tuple:
    """[L550E-DS] full-throttle SFC chart, 4,000-7,500 rpm, in g/kWh (both readings of "HP")."""
    vals = [g / k for g in el.SFC_CHART_G_PER_HPH.values() for k in HP_KW]
    return min(vals), max(vals)


def bsfc_rotax_503() -> dict:
    """[ROTAX-OM] 503 UL consumptions in g/kWh at take-off and at 75 % power (EN 228 density limits)."""
    f = ROTAX_503_FUEL
    to = tuple(f["takeoff_l_h"] * d * 1000.0 / f["takeoff_kw"] for d in FUEL_DENSITY_KG_L)
    c75 = tuple(f["cont75_l_h"] * d * 1000.0 / (0.75 * f["takeoff_kw"]) for d in FUEL_DENSITY_KG_L)
    return dict(takeoff=to, cont75=c75, stated=f["sfc_g_kwh"])


def bsfc_band() -> tuple:
    """Lowest: the 503 UL at 75 % (part throttle traps more charge, [AUSSERER] pp. 307-311); highest: the L550E
    chart at 4,000 rpm full throttle."""
    r = bsfc_rotax_503()
    lo = min(r["cont75"][0], bsfc_l550e_chart()[0])
    hi = max(bsfc_l550e_chart()[1], r["takeoff"][1], r["stated"])
    return lo, hi


def afr_band() -> tuple:
    return afr_stoich(FUEL_H_TO_C[0]) / PHI[1], afr_stoich(FUEL_H_TO_C[1]) / PHI[0]


def cooling_over_shaft_band(bsfc=None) -> tuple:
    """Cooling load / shaft power = cooling share of the fuel energy / brake efficiency, eta_b = 1 / (BSFC LHV)."""
    lo_b, hi_b = bsfc or bsfc_band()
    eta = lambda b: 3.6e9 / (b * LHV_FUEL)  # noqa: E731  (g/kWh -> fraction)
    return COOLING_SHARE[0] / eta(lo_b), COOLING_SHARE[1] / eta(hi_b)


# ---------------------------------------------------------------- uncertain inputs, each with its grade
INPUTS = dict(
    bsfc_g_kwh=bsfc_band(),          # B  [L550E-DS] chart 449-590 g/kWh at full throttle; [ROTAX-OM] 389-419 at 75 %
    afr=afr_band(),                  # B- stoichiometry / equivalence ratio 1.0-1.2 [AUSSERER] -> 12.2-15.1
    k_stub=(0.7, 1.0),               # C  local air speed at the stubs / flight speed, before the propeller's inflow
                                     #    (behind the body, near free stream; + prop_inflow_ratio, momentum theory)
    f_pulse=(1.2, 2.0),              # C+ exhaust-pulse enhancement of internal heat transfer over steady pipe flow
    eps_stub=(0.28, 0.95),           # B  [FLIR-TAB] stainless untreated 0.28 (LW), rolled 0.45, oxidised 0.85, soot 0.95
    beta_fin=(0.80, 0.95),           # C+ mean fin-root excess / CHT excess (the barrel runs cooler than the plug seat)
    k_fin=(0.5, 0.9),                # C  air speed at the cylinders / flight speed (fuselage wake, ~45 mm boundary
                                     #    layer at the neck), before the propeller's inflow
    hs_h_factor=(0.25, 1.0),         # C  heat-sink h / NACA 488 h of aligned fins: its fins stand across the ram air
    k_al=(120.0, 160.0),             # B  cast aluminium alloy conductivity, W m-1 K-1
    head_over_cht=(0.85, 1.00),      # C+ head block and plug: at the CHT sensing point (plug seat, [L550E-DS] 14 mm
                                     #    thermocouple; [ROTAX-OM]) or slightly cooler towards the heat sink
    case_over_cht=(0.25, 0.45),      # C  crankcase: cooled from inside by the scavenge charge, fed from the cylinders
    flange_over_case=(0.30, 0.80),   # C  rotating drive disc: conduction along the crank vs disc convection
    tail_over_case=(0.10, 0.40),     # C  airframe neck at the mounting face: conduction + radiation from the engine
    carb_evap_share=(0.30, 0.60),    # C  share of the fuel evaporated in the carburettor body
    carb_charge_coupling=(0.5, 0.8),  # C  carb body temperature position between ambient and the cooled charge
    carb_soak_over_case=(0.0, 0.15),  # C  heat soak from the warm intake spacer
    cooling_over_shaft=cooling_over_shaft_band(),  # B- [AUSSERER] 10-11 % of the fuel energy / eta_b -> 0.47-0.78
)

# ---------------------------------------------------------------- power settings drawn by the sampler
# f: shaft power fraction (sampled; the fractions and weights are coverage choices, C). Temperature bands are
# excesses over a 15 C day, from the [ROTAX-OM] limits of the closest analogue:
#   nominal     CHT: the 447/503 normal ranges 180-230 C.  EGT: the 503/582 normal ranges 460-620 C.
#   high_power  CHT: 503 normal minimum to the 447 maximum, 180-260 C (capped at CHT_LIMIT_C in the scene).
#               EGT: 503 normal maximum (538 C as printed in F, 580 C as printed in C) to the maximum: 538-650 C.
#   low_power   upper ends: the normal maxima, because the manual warns that CHT and EGT can be higher at reduced
#               throttle (p. 10-15). CHT lower end: physics bound theta ~ Q_cool at a given cooling flow, with the
#               cooling load falling no faster than the power (its share of the fuel energy rises at part throttle,
#               [AUSSERER] pp. 307-311): nominal minimum x 0.30 / 0.50. EGT lower end 400 C: C+ estimate (no source).
def _excess(c):
    return float(c) - T_REF_C


F_SETTINGS = {"low_power": (0.30, 0.50), "nominal": (0.50, 0.75), "high_power": (0.75, 1.00)}
_CHT_NOM = (_excess(min(ROTAX["503"]["cht_normal_c"][0], ROTAX["447"]["cht_normal_c"][0])),
            _excess(max(ROTAX["503"]["cht_normal_c"][1], ROTAX["447"]["cht_normal_c"][1])))
_EGT_NOM = (_excess(min(ROTAX["503"]["egt_normal_c"][0], ROTAX["582"]["egt_normal_c"][0])),
            _excess(max(ROTAX["503"]["egt_normal_c"][1], ROTAX["582"]["egt_normal_c"][1])))
POWER_SETTINGS = {
    "low_power":  dict(weight=0.25, f=F_SETTINGS["low_power"],
                       cht_excess_k=(_CHT_NOM[0] * F_SETTINGS["low_power"][0] / F_SETTINGS["nominal"][0], _CHT_NOM[1]),
                       egt_excess_k=(_excess(400.0), _excess(ROTAX["503"]["egt_normal_c"][1]))),
    "nominal":    dict(weight=0.50, f=F_SETTINGS["nominal"], cht_excess_k=_CHT_NOM, egt_excess_k=_EGT_NOM),
    "high_power": dict(weight=0.25, f=F_SETTINGS["high_power"],
                       cht_excess_k=(_excess(ROTAX["503"]["cht_normal_c"][0]), _excess(ROTAX["447"]["cht_max_c"])),
                       egt_excess_k=(_excess(min(ROTAX["503"]["egt_normal_c"][1], f_to_c(ROTAX["503"]["egt_normal_f"][1]))),
                                     _excess(ROTAX["503"]["egt_max_c"]))),
}
# Absolute CHT cap: the 503 UL maximum [ROTAX-OM] and Limbach's certified L 1700 limit ([EASA-E082] IV.1), both
# 250 C (the smaller 447 allows 260 C). On a hot day the engine cannot hold a setting whose head would pass it; the
# scene clips the head and fins there (signatures.yaml surface_temperature_cap_c).
CHT_LIMIT_C = 250.0

# ---------------------------------------------------------------- pusher propeller inflow (momentum theory)
PROP = dict(dia_m=(0.50, 0.64), hub_aft_flange_m=0.035)     # configs/target.yaml prop_dia_m range, prop_hub_aft_flange_m


def prop_inflow_ratio(p_shaft, airspeed, alt_m, t_air_k, dia_m, x_up_m) -> float:
    """Upper bound of the axial velocity the pusher propeller adds at a point x_up ahead of its disc, over the flight
    speed. Actuator-disc momentum theory with an ideal propeller (thrust T <= P / V): induced velocity at the disc
    v_i = (sqrt(V^2 + 2 T / (rho A)) - V) / 2; on the axis ahead of the disc the semi-infinite vortex cylinder gives
    v_i (1 - x / sqrt(x^2 + R^2)). The real propeller (efficiency < 1) adds less, so 0 .. this bound is sampled."""
    rho = density(t_air_k, alt_m)
    area = math.pi * dia_m ** 2 / 4.0
    thrust = p_shaft / airspeed
    v_i = 0.5 * (math.sqrt(airspeed ** 2 + 2.0 * thrust / (rho * area)) - airspeed)
    r = dia_m / 2.0
    return v_i * (1.0 - x_up_m / math.sqrt(x_up_m ** 2 + r ** 2)) / airspeed


def engine_x_up_min() -> float:
    """Shortest distance from the propeller disc to a cylinder centre (m): hub behind the flange + cylinder x."""
    return PROP["hub_aft_flange_m"] + min(xc for _, _, xc in el.CYLINDERS)


def inflow_bound(f, airspeed, alt_m, t_air_k) -> float:
    """Largest inflow ratio over the propeller diameters (smaller disc -> faster inflow) at the nearest cylinder."""
    return max(prop_inflow_ratio(f * el.RATED_POWER_W, airspeed, alt_m, t_air_k, d, engine_x_up_min())
               for d in PROP["dia_m"])


# ============================================================================ air properties
def viscosity(t_k):
    """Dynamic viscosity of air (Sutherland): 1.716e-5 Pa s at 273.15 K, S = 110.4 K. Within 2 % of tables
    to 1,000 K; used for the exhaust gas too (combustion products differ by < 10 %)."""
    return 1.716e-5 * (t_k / 273.15) ** 1.5 * (273.15 + 110.4) / (t_k + 110.4)


def conductivity(t_k):
    """Thermal conductivity of air, 0.0241 (T / 273.15)^0.81 W m-1 K-1: within 3 % of tables, 250-900 K."""
    return 0.0241 * (t_k / 273.15) ** 0.81


def pressure_isa(alt_m):
    """ISA troposphere static pressure (Pa)."""
    return 101325.0 * (1.0 - 2.25577e-5 * alt_m) ** 5.25588


def density(t_k, alt_m=0.0):
    """Ideal gas at the ISA pressure of the altitude and the given temperature (kg m-3)."""
    return pressure_isa(alt_m) / (R_AIR * t_k)


# ============================================================================ correlations
def nu_churchill_bernstein(re, pr=PR):
    """Mean Nusselt number of a circular cylinder in cross-flow, Churchill & Bernstein (1977); Re Pr > 0.2."""
    return 0.3 + 0.62 * re ** 0.5 * pr ** (1 / 3) / (1 + (0.4 / pr) ** (2 / 3)) ** 0.25 \
        * (1 + (re / 282000.0) ** 0.625) ** 0.8


def nu_gnielinski(re, pr=PR):
    """Fully developed turbulent pipe flow, Gnielinski (1976), 3e3 < Re < 5e6; Petukhov friction factor."""
    f = (0.790 * math.log(re) - 1.64) ** -2
    return (f / 8) * (re - 1000.0) * pr / (1 + 12.7 * math.sqrt(f / 8) * (pr ** (2 / 3) - 1))


def nu_flat_plate_laminar(re, pr=PR):
    """Mean Nusselt number of a laminar flat plate, Nu_L = 0.664 Re_L^0.5 Pr^(1/3) (Re_L < 5e5)."""
    return 0.664 * re ** 0.5 * pr ** (1 / 3)


def fin_efficiency(m_lc):
    """Straight fin with an adiabatic tip at the corrected length: eta = tanh(m L_c) / (m L_c)."""
    return 1.0 if m_lc < 1e-9 else math.tanh(m_lc) / m_lc


def cavity_emissivity(eps, f_open):
    """Apparent emissivity of an isothermal diffuse groove whose opening is the fraction f_open of its internal
    surface: eps / (eps + (1 - eps) f_open) (Modest, Radiative Heat Transfer, ch. 6)."""
    return eps / (eps + (1.0 - eps) * f_open)


# ============================================================================ geometry from engine_l550e
def fin_geometry() -> dict:
    """Fin pack per cylinder: rounded-rectangle fins around the barrel (engine_l550e.FIN_PACK)."""
    fp = el.FIN_PACK
    a_rect = fp["dx"] * fp["dz"] - (4 - math.pi) * fp["corner"] ** 2
    a_face = a_rect - math.pi * fp["core_r"] ** 2
    perim = 2 * (fp["dx"] + fp["dz"]) - (8 - 2 * math.pi) * fp["corner"]
    n, t = int(fp["fins"]), fp["fin_t"]
    pitch = (fp["y1"] - fp["y0"]) / n
    length = math.sqrt(a_rect / math.pi) - fp["core_r"]          # area-equivalent annulus height
    return dict(area_m2=n * (2 * a_face + perim * t), length_m=length, lc_m=length + t / 2, t_m=t,
                gap_m=pitch - t, pitch_m=pitch, flow_len_m=fp["dx"], n=n)


def stub_geometry() -> dict:
    """Exhaust stack of the [L550E-DS] drawing: 31.5 mm OD, flange to end 95.5 mm (engine_l550e.STUB)."""
    st = el.STUB
    d_o = 2 * st["r"]
    return dict(d_o=d_o, d_i=d_o - 2 * STUB_WALL_M, length=st["z_flange"] - st["z_end"],
                outlet_d=2 * st["outlet_r"], outlet_len=st["outlet_len"])


# ============================================================================ S1 operating state
def operating_state(f, bsfc_g_kwh, afr, alt_m=0.0, t_air_k=288.15) -> dict:
    p = f * el.RATED_POWER_W
    m_f = p * bsfc_g_kwh / 3.6e9                            # g/kWh -> kg/J
    rpm = el.rpm_at_power_fraction(f)
    m_air = afr * m_f
    dr = m_air / (density(t_air_k, alt_m) * el.displacement_m3() * rpm / 60.0)   # two-stroke: one charge per rev
    return dict(shaft_w=p, fuel_kg_s=m_f, fuel_power_w=m_f * LHV_FUEL, rpm=rpm, delivery_ratio=dr,
                exhaust_kg_s_per_cyl=(m_air + m_f) / el.N_CYL)


# ============================================================================ S2 exhaust stubs
def stub_state(theta_g0, m_ex, airspeed, alt_m, t_air_k, eps, f_pulse, k_stub) -> dict:
    """Mean visible wall excess of one stub and the gas excess leaving it."""
    g = stub_geometry()
    t_g = t_air_k + theta_g0
    re_i = 4.0 * m_ex / (math.pi * g["d_i"] * viscosity(t_g))
    nu_i = nu_gnielinski(re_i) * (1.0 + (g["d_i"] / g["length"]) ** 0.7) * f_pulse
    h_i = nu_i * conductivity(t_g) / g["d_i"]
    h_io = h_i * g["d_i"] / g["d_o"]                        # per unit outer area
    v = k_stub * airspeed
    t_w = t_air_k + 0.5 * theta_g0
    for _ in range(50):                                     # film properties and radiation depend on T_w
        t_f = 0.5 * (t_w + t_air_k)
        re_o = v * g["d_o"] * density(t_f, alt_m) / viscosity(t_f)
        h_o = nu_churchill_bernstein(re_o) * conductivity(t_f) / g["d_o"]
        h_r = eps * SIGMA * (t_w ** 2 + t_air_k ** 2) * (t_w + t_air_k)
        phi = h_io / (h_io + h_o + h_r)
        t_new = t_air_k + phi * theta_g0
        if abs(t_new - t_w) < 1e-6:
            break
        t_w = t_new
    u = 1.0 / (1.0 / h_io + 1.0 / (h_o + h_r))
    ntu = u * math.pi * g["d_o"] * g["length"] / (m_ex * CP_EXHAUST)
    mean = (1.0 - math.exp(-ntu)) / ntu
    return dict(theta_wall=phi * theta_g0 * mean, phi=phi, ntu=ntu, h_i=h_i, h_o=h_o, h_r=h_r, re_i=re_i,
                re_o=re_o, theta_gas_out=theta_g0 * math.exp(-ntu))


def exhaust_share(theta_g0, afr) -> float:
    """Sensible exhaust enthalpy / fuel energy implied by a port gas excess (energy balance)."""
    return theta_g0 * (1.0 + afr) * CP_EXHAUST / LHV_FUEL


# ============================================================================ S3 fins and heads
NU_PARALLEL_PLATES_FD = 7.54   # fully developed laminar flow between isothermal parallel plates, on D_h = 2 x gap
                               # (Incropera et al., Fundamentals of Heat and Mass Transfer, Table 8.1)


def fin_channel_h(u, flow_len, gap, t_f, alt_m, f_tu) -> dict:
    """Mean heat-transfer coefficient on the faces of a fin channel (gap between two fins, length along the flow).
    Developing laminar boundary layers (flat plate, delta = 5 x / sqrt(Re_x)) until they meet in mid-gap at
    x_m = (gap / 10)^2 u / nu; fully developed parallel-plate flow (Nu_Dh = 7.54) beyond; the mean of the two over
    the length, x the turbulence factor. Then the channel air's own heating: with isothermal walls the outlet
    excess falls as exp(-NTU), NTU = h 2L / (rho u gap c_p), so the effective h (on the inlet excess) is
    h (1 - e^-NTU) / NTU (Incropera eq. 8.41b)."""
    rho, mu, k = density(t_f, alt_m), viscosity(t_f), conductivity(t_f)
    nu = mu / rho
    x_m = min(flow_len, (gap / 10.0) ** 2 * u / nu)
    h_dev = nu_flat_plate_laminar(u * x_m / nu) * k / x_m
    h_fd = NU_PARALLEL_PLATES_FD * k / (2.0 * gap)
    h = (h_dev * x_m + h_fd * (flow_len - x_m)) / flow_len * f_tu
    ntu = h * 2.0 * flow_len / (rho * u * gap * CP_AIR)
    h_eff = h * (1.0 - math.exp(-ntu)) / ntu
    return dict(h=h_eff, h_wall=h, x_merge=x_m, ntu_air=ntu, re=u * flow_len / nu,
                bl_m=5.0 * flow_len / math.sqrt(u * flow_len / nu))


# [NACA-488] Biermann, A. E. and Pinkel, B. (1934). Heat transfer from finned metal cylinders in an air stream. NACA
# Report 488 (wind tunnel; 18 finned cylinders, root diameter 4.66 in, axis across a free air stream, no baffles;
# fin pitch 0.10-0.60 in, thickness 0.04-0.27 in, width 0.37-1.47 in; 30-150 mph). The average surface heat-transfer
# coefficient q depends mainly on air speed and fin space: q ~ V^0.796 (p. 11, fig. 10) and q = L s^0.322 (p. 16,
# fig. 9). Reference point read from fig. 9 (docs/sources/naca/fig9_digitised.csv; it reproduces the report's 0.796
# and its quoted q = 0.087-0.091 at 76 mph): q = 0.0892 B.t.u. / sq.in. / F / hr at s = 0.175 in, 76 mph, sea-level
# air of 0.0734 lb/cu.ft (29.92 in Hg, 80 F). Other diameters by similitude (eq. 2-4): q_x = q_T(J V, s / J) / J with
# J = D_x / D_T; other densities by equal mass flow (eq. 6).
NACA488 = dict(q_ref_w_m2k=0.0892 * 817.67, s_ref_m=0.175 * 0.0254, v_ref_mps=76 * 0.44704, rho_ref=1.1757,
               d_ref_m=4.66 * 0.0254, v_exp=0.796, s_exp=0.322, v_range_mps=(30 * 0.44704, 150 * 0.44704))


def h_naca488(u_local, rho, gap_m, root_d_m) -> float:
    """Average fin-surface h (W m-2 K-1) of a finned cylinder across an air stream [NACA-488], at local speed
    u_local and air density rho, fin space gap_m, root diameter root_d_m."""
    c = NACA488
    j = root_d_m / c["d_ref_m"]
    v_eq = u_local * rho / c["rho_ref"]                     # sea-level speed of the same mass flow (eq. 6)
    return (c["q_ref_w_m2k"] * (v_eq / c["v_ref_mps"]) ** c["v_exp"] * (gap_m / c["s_ref_m"]) ** c["s_exp"]
            * j ** (c["v_exp"] - c["s_exp"] - 1.0))


def fin_state(theta_base, airspeed, alt_m, t_air_k, k_fin, k_al) -> dict:
    """Mean fin-surface excess for a mean fin-root excess theta_base: NACA 488 h on the barrel fins at the local
    speed k_fin x airspeed, straight-fin efficiency at the measured fin height and thickness."""
    g = fin_geometry()
    rho = density(t_air_k, alt_m)
    u = k_fin * airspeed
    h = h_naca488(u, rho, g["gap_m"], 2.0 * el.FIN_PACK["core_r"])
    m = math.sqrt(2.0 * h / (k_al * g["t_m"]))
    eta = fin_efficiency(m * g["lc_m"])
    return dict(theta_mean=theta_base * eta, theta_tip=theta_base / math.cosh(m * g["lc_m"]), eta=eta, h=h,
                v_naca_eq=u * rho / NACA488["rho_ref"])


def heatsink_geometry() -> dict:
    """Limbach heat sink (engine_l550e.HEATSINK): fins across the crank axis, each a plate in a y-z plane."""
    hs = el.HEATSINK
    depth = hs["y1"] - hs["y0"] - hs["base_t"]
    pitch = hs["dx"] / hs["fins"]
    plug_share = hs["plug_well_w"] / hs["dz"]                # part of the fin height cut by the plug well
    area = hs["fins"] * 2.0 * depth * hs["dz"] * (1.0 - plug_share)
    return dict(area_m2=area, depth_m=depth, lc_m=depth + hs["fin_t"] / 2, t_m=hs["fin_t"], pitch_m=pitch,
                gap_m=pitch - hs["fin_t"])


def heatsink_state(theta_root, airspeed, alt_m, t_air_k, k_fin, hs_h_factor, k_al) -> dict:
    """Mean surface excess of the heat-sink fins for a root (head) excess theta_root. The fins stand across the
    ram air, so their channels open across the stream rather than along it: h = hs_h_factor x the NACA 488 value
    for aligned fins at the same local speed (C), then the straight-fin efficiency."""
    g = heatsink_geometry()
    rho = density(t_air_k, alt_m)
    h = hs_h_factor * h_naca488(k_fin * airspeed, rho, g["gap_m"], 2.0 * el.FIN_PACK["core_r"])
    m = math.sqrt(2.0 * h / (k_al * g["t_m"]))
    eta = fin_efficiency(m * g["lc_m"])
    return dict(theta_mean=theta_root * eta, eta=eta, h=h)


def fin_apparent_emissivity(eps_surface, geometry=None) -> float:
    """Fin pack seen across the fin openings: grooves (gap x fin height) plus the fin edges."""
    g = geometry or fin_geometry()
    f_open = g["gap_m"] / (g["gap_m"] + 2.0 * g.get("length_m", g.get("depth_m")))
    edge = g["t_m"] / g["pitch_m"]
    return edge * eps_surface + (1.0 - edge) * cavity_emissivity(eps_surface, f_open)


def fin_heat_balance(setting: str, with_inflow: bool = False, with_heatsink: bool = True, at: str = "centre") -> dict:
    """Heat the free-air cooling can reject at the band centres, against the cooling load (cooling_over_shaft x
    shaft power): barrel fins at the NACA 488 h, plus the heat sink at its own h (fins across the stream), each
    as h A eta theta. at="centre": the centre of the flight envelope and of the installation factors (plus the
    propeller's inflow bound if with_inflow); at="best": the fastest, densest corner with the inflow bound, the
    highest wake factor and heat-sink h. ratio = cooling load / that capacity: > 1 at the centre means the engine
    needs faster or denser air than the centre to hold the CHT band (or runs towards the top of it)."""
    s = POWER_SETTINGS[setting]
    mid = lambda r: 0.5 * (r[0] + r[1])  # noqa: E731
    pick = (lambda r: r[1]) if at == "best" else mid  # noqa: E731
    g, gh = fin_geometry(), heatsink_geometry()
    q_cyl = mid(INPUTS["cooling_over_shaft"]) * mid(s["f"]) * el.RATED_POWER_W / el.N_CYL
    theta_c = mid(s["cht_excess_k"])
    theta_b = theta_c * mid(INPUTS["beta_fin"])
    env = {k: mid(v) for k, v in ENV.items()}
    if at == "best":
        env = dict(airspeed_mps=ENV["airspeed_mps"][1], alt_m=ENV["alt_m"][0], t_air_k=ENV["t_air_k"][0])
        with_inflow = True
    k = pick(INPUTS["k_fin"])
    if with_inflow:
        k += inflow_bound(mid(s["f"]), env["airspeed_mps"], env["alt_m"], env["t_air_k"])
    st = fin_state(theta_b, env["airspeed_mps"], env["alt_m"], env["t_air_k"], k, pick(INPUTS["k_al"]))
    q_barrel = st["h"] * g["area_m2"] * st["eta"] * theta_b
    q_hs = 0.0
    if with_heatsink:
        hs = heatsink_state(theta_c * mid(INPUTS["head_over_cht"]), env["airspeed_mps"], env["alt_m"], env["t_air_k"],
                            k, pick(INPUTS["hs_h_factor"]), pick(INPUTS["k_al"]))
        q_hs = hs["h"] * gh["area_m2"] * hs["eta"] * theta_c * mid(INPUTS["head_over_cht"])
    return dict(q_per_cyl_w=q_cyl, q_barrel_w=q_barrel, q_heatsink_w=q_hs, h_barrel=st["h"],
                ratio=q_cyl / (q_barrel + q_hs))


# ============================================================================ S5 carburettor
def carb_evaporative_cooling_k(afr, h_fg) -> float:
    """Charge temperature drop if all fuel evaporates into it: h_fg / (AFR c_p)."""
    return h_fg / (afr * CP_AIR)


# ============================================================================ S6 exhaust plume
def plume_lwir_emissivity(theta_gas, t_air_k=288.15, path_m=None, x_h2o=0.12, x_co2=0.10, safety=3.0) -> float:
    """8-14 um emissivity of the exhaust jet at the outlet: water-vapour continuum of thermal/atmosphere.py
    (Roberts et al. 1976) at the gas temperature and vapour pressure, plus the CO2 13-14 um band edge whose
    ambient share (render.yaml edge_co2 = 3 %) saturates within ~50 m at 420 ppm, scaled to the plume's CO2
    (optically thin: depth ~ concentration x path). x a safety factor for hot bands absent from both."""
    from thermal.atmosphere import continuum_sigma_per_km
    g = stub_geometry()
    path = g["outlet_d"] if path_m is None else path_m
    t_g = t_air_k + theta_gas
    e_hpa = x_h2o * 1013.25
    tau_cont = continuum_sigma_per_km(t_g - 273.15, e_hpa) / 1000.0 * path
    sat_len = 50.0 * 420e-6 / x_co2                          # m: same optical depth at plume CO2
    edge = 0.03 * (1.0 - math.exp(-path / sat_len))
    return float(min(1.0, safety * (1.0 - math.exp(-tau_cont) + edge)))


def plume_to_stub_lwir_ratio(theta_gas, theta_stub, eps_stub, t_air_k=288.15, area_ratio=3.2, safety=3.0) -> float:
    """LWIR contrast (above ambient) of the plume region hotter than half the outlet excess over that of a
    stub. area_ratio: that region (about 5 x 1.5 outlet diameters for a jet bent over by an equal cross-flow)
    over the stub's projected area; safety: the hot-band margin of plume_lwir_emissivity (1 = best estimate)."""
    from thermal.planck_lut import LWIR_BAND_UM, band_radiance_direct
    def rad(t):
        return float(band_radiance_direct(np.array([t]), LWIR_BAND_UM)[0])
    la = rad(t_air_k)
    eg = plume_lwir_emissivity(theta_gas, t_air_k, safety=safety)
    dg = eg * (rad(t_air_k + theta_gas) - la)
    ds = eps_stub * (rad(t_air_k + theta_stub) - la)
    return area_ratio * dg / ds


# ============================================================================ envelope per power setting
def _corners(ranges: dict):
    keys = list(ranges)
    for combo in itertools.product(*[ranges[k] for k in keys]):
        yield dict(zip(keys, combo))


def envelope(setting: str) -> dict:
    """(min, max) excess (K) of every engine surface class over all corners of the inputs and the flight
    envelope, for one power setting. Monotonic in each input, so corners bound the range."""
    s = POWER_SETTINGS[setting]
    out = {}
    # exhaust stubs
    vals = []
    for c in _corners(dict(f=s["f"], bsfc=INPUTS["bsfc_g_kwh"], afr=INPUTS["afr"], egt=s["egt_excess_k"],
                           v=ENV["airspeed_mps"], alt=ENV["alt_m"], t=ENV["t_air_k"], eps=INPUTS["eps_stub"],
                           fp=INPUTS["f_pulse"], ks=INPUTS["k_stub"], inflow=(0.0, 1.0))):
        op = operating_state(c["f"], c["bsfc"], c["afr"])
        k = c["ks"] + c["inflow"] * inflow_bound(c["f"], c["v"], c["alt"], c["t"])
        vals.append(stub_state(c["egt"], op["exhaust_kg_s_per_cyl"], c["v"], c["alt"], c["t"], c["eps"], c["fp"],
                               k)["theta_wall"])
    out["exhaust_stub"] = (min(vals), max(vals))
    # barrel fins (engine_barrel), heat sinks (engine_heatsink) and the head and plug (engine_head)
    fins, sinks = [], []
    for c in _corners(dict(cht=s["cht_excess_k"], beta=INPUTS["beta_fin"], head=INPUTS["head_over_cht"],
                           v=ENV["airspeed_mps"], alt=ENV["alt_m"], t=ENV["t_air_k"], kf=INPUTS["k_fin"],
                           hsf=INPUTS["hs_h_factor"], kal=INPUTS["k_al"], f=s["f"], inflow=(0.0, 1.0))):
        k = c["kf"] + c["inflow"] * inflow_bound(c["f"], c["v"], c["alt"], c["t"])
        fins.append(fin_state(c["cht"] * c["beta"], c["v"], c["alt"], c["t"], k, c["kal"])["theta_mean"])
        sinks.append(heatsink_state(c["cht"] * c["head"], c["v"], c["alt"], c["t"], k, c["hsf"], c["kal"])["theta_mean"])
    out["engine_barrel"] = (min(fins), max(fins))
    out["engine_heatsink"] = (min(sinks), max(sinks))
    lo, hi = s["cht_excess_k"]
    out["engine_head"] = (lo * INPUTS["head_over_cht"][0], hi * INPUTS["head_over_cht"][1])
    case = (lo * INPUTS["case_over_cht"][0], hi * INPUTS["case_over_cht"][1])
    out["engine_case"] = case
    out["engine_flange"] = (case[0] * INPUTS["flange_over_case"][0], case[1] * INPUTS["flange_over_case"][1])
    out["tail_cone"] = (case[0] * INPUTS["tail_over_case"][0], case[1] * INPUTS["tail_over_case"][1])
    # carburettor: coolest when much fuel evaporates in it and its body follows the charge; warmest with heat soak
    cool = max(carb_evaporative_cooling_k(a, h) for a in INPUTS["afr"] for h in H_FG_FUEL)
    out["engine_intake"] = (-cool * INPUTS["carb_evap_share"][1] * INPUTS["carb_charge_coupling"][1],
                            case[1] * INPUTS["carb_soak_over_case"][1])
    return out


@functools.lru_cache(maxsize=None)
def _envelope_cached(setting: str) -> tuple:
    return tuple(envelope(setting).items())


def envelope_cache() -> dict:
    """envelope() of every power setting, computed once per process."""
    return {name: dict(_envelope_cached(name)) for name in POWER_SETTINGS}


def yaml_ranges(step: float = 5.0) -> dict:
    """Envelopes rounded outward to `step` K: what configs/signatures.yaml engine_profiles must contain."""
    res = {}
    for name, env in envelope_cache().items():
        res[name] = {k: (math.floor(lo / step) * step, math.ceil(hi / step) * step) for k, (lo, hi) in env.items()}
    return res


# ============================================================================ S7 electric-like variant (2026-10-19)
# The electric-like signature arm (docs/privileged_teacher_case.md 6c, configs/m1_electric_like.yaml): this engine's
# geometry and its own free-air cooling surfaces, carrying an electric drive's loss heat at the same shaft power
# instead of the combustion engine's cooling load. A variant of this engine's heat, not the REB 30 of ../fyp
# (CLAUDE.md). The drive rejects (1 - eta) / eta x P through the cylinders' fins and heat sinks; there is no exhaust
# gas and no fuel. Every fin root of a cylinder sits at one excess theta = q / sum(h A eta) over its barrel fins and
# heat sink, with h and eta from fin_state / heatsink_state at the same corners as envelope(). The controller's losses
# (2 %, [HEPPERLE-2012]) and the battery's stay inside the fuselage and are not modelled, so the arm errs cold.
# [EU-2019/1781] Commission Regulation (EU) 2019/1781, Annex I (OJ L 272, 25.10.2019, pp. 84-85): minimum efficiencies
#   of 50 Hz induction motors at rated output, 2-4 poles: IE3 15 kW 91.9-92.1 %, 37 kW 93.7-93.9 %; IE4 15 kW 93.3-93.9 %,
#   37 kW 94.8-95.2 % (S1 for those motors, S2 here: a motor class, not an aero drive).
# [HEPPERLE-2012] Hepperle (2012), Electric Flight - Potential and Limitations, NATO STO-MP-AVT-209 paper 9 (DLR),
#   p. 9-9 Fig. 7: electric motor 95 %, controller 98 % in the on-board chain of a ~50 kW electric aircraft (assumed
#   typical values, p. 9-8; S2). Both: docs/sources/electric_drive/readings.json.
MOTOR_EFFICIENCY_SOURCES = dict(ie3_15kw=(0.919, 0.921), ie3_37kw=(0.937, 0.939), ie4_15kw=(0.933, 0.939),
                                ie4_37kw=(0.948, 0.952), hepperle_motor=0.95, hepperle_controller=0.98)
MOTOR_EFFICIENCY = (0.90, 0.96)      # B-: the sourced 0.919-0.952, widened ~2 points below (part load, light air-cooled
                                     #     aero motors) and ~1 above (high-efficiency permanent-magnet drives)
CASE_OVER_ROOT_ELECTRIC = (0.40, 1.00)   # C: end bells and mount, fed from the stator by conduction; no scavenge
                                         #    charge cools them from inside as it does the crankcase


ELECTRIC_DRAWS, ELECTRIC_QUANTILES = 4000, (5.0, 95.0)


def electric_like_envelope(setting: str, n: int = ELECTRIC_DRAWS, seed: int = 0) -> dict:
    """Central 90 % range (K) of every engine surface class when the same fins and heat sinks reject an electric drive's
    loss heat (1 - eta) / eta x shaft power, for one power setting. The inputs of envelope() are drawn uniformly and
    independently (flight envelope, installation factors, motor efficiency). envelope()'s corners would stack every
    C-graded input at its extreme at once (7-185 K here), and no operating band caps this arm as the CHT band caps the
    combustion engine; the central interval of the propagated inputs is the least-delta range (CLAUDE.md, evidence rule
    4). Deterministic (seeded)."""
    s = POWER_SETTINGS[setting]
    g, gh = fin_geometry(), heatsink_geometry()
    rng = np.random.default_rng([seed, list(POWER_SETTINGS).index(setting)])
    u = lambda r: float(rng.uniform(r[0], r[1]))  # noqa: E731
    rows = []
    for _ in range(n):
        f, eta, v, alt, t = u(s["f"]), u(MOTOR_EFFICIENCY), u(ENV["airspeed_mps"]), u(ENV["alt_m"]), u(ENV["t_air_k"])
        k = u(INPUTS["k_fin"]) + float(rng.uniform()) * inflow_bound(f, v, alt, t)
        fs = fin_state(1.0, v, alt, t, k, u(INPUTS["k_al"]))                        # per K of root excess
        hs = heatsink_state(1.0, v, alt, t, k, u(INPUTS["hs_h_factor"]), u(INPUTS["k_al"]))
        q = (1.0 - eta) / eta * f * el.RATED_POWER_W / el.N_CYL                     # loss heat through one cylinder
        root = q / (fs["h"] * g["area_m2"] * fs["eta"] + hs["h"] * gh["area_m2"] * hs["eta"])
        case = root * u(CASE_OVER_ROOT_ELECTRIC)
        rows.append(dict(engine_head=root, engine_barrel=root * fs["eta"], engine_heatsink=root * hs["eta"],
                         engine_case=case, engine_flange=case * u(INPUTS["flange_over_case"]),
                         tail_cone=case * u(INPUTS["tail_over_case"])))
    lo, hi = ELECTRIC_QUANTILES
    out = {k: (float(np.percentile([r[k] for r in rows], lo)), float(np.percentile([r[k] for r in rows], hi)))
           for k in rows[0]}
    out["engine_intake"] = (0.0, 0.0)           # no fuel: no evaporative cooling of the carburettor bodies
    out["exhaust_stub"] = (0.0, 0.0)            # no exhaust gas
    return out


def electric_like_yaml_ranges(step: float = 1.0) -> dict:
    """electric_like_envelope rounded outward to `step` K: what configs/m1_electric_like.yaml engine_profiles hold."""
    return {name: {k: (math.floor(lo / step) * step, math.ceil(hi / step) * step)
                   for k, (lo, hi) in electric_like_envelope(name).items()} for name in POWER_SETTINGS}


# [FLIR-TAB] FLIR Systems, User's manual FLIR ETS3xx series, T810252 (2017), ch. 21 Table 21.1: a compilation from
# Bramson (Infrared Radiation, Plenum), Wolfe & Zissis (The Infrared Handbook, ONR) and FLIR/AGEMA E-Box measurements
# (ref. 9), each with its temperature, waveband (LW = 8-14 um, T = total) and reference; "recommendations only".
# Entries used (material, condition, temperature, band: emissivity):
EMISSIVITY_SOURCES = {
    "al_as_received": 0.09,          # aluminium, as received plate/sheet, 100 C, T (refs 2, 4)
    "al_cast_blasted": 0.46,         # aluminium, cast, blast cleaned, 70 C, LW (ref 9)
    "al_polished": (0.04, 0.06),     # aluminium, polished, 50-100 C, T (ref 1)
    "al_oxidised_strongly": (0.2, 0.3),  # 50-500 C, T (ref 1)
    "al_anodised_dull": (0.95, 0.97),    # anodised black / light grey, dull, 70 C, LW (ref 9)
    "black_lacquer": (0.87, 0.98),   # black shiny sprayed 0.87 (20 C, T); black dull 0.96-0.98 (40-100 C, T) (ref 1)
    "paints_lw": (0.92, 0.94),       # 8 colours and qualities, 70 C, LW (ref 9)
    "stainless_untreated": 0.28,     # sheet, untreated, somewhat scratched, 70 C, LW (ref 9)
    "stainless_oxidised": 0.85,      # type 18-8, oxidised at 800 C, 60 C, T (ref 2)
    "soot": 0.95,                    # candle soot, 20 C, T (ref 2); lampblack 0.95-0.97 (ref 1)
}


# The same table's SW (2-5 um) entries, for the MWIR camera (CLAUDE.md sensor set rule); T-band ("total") entries
# above serve every band. (material, condition, temperature, band: emissivity, FLIR reference number)
EMISSIVITY_SOURCES_SW = {
    "al_anodised_dull": (0.61, 0.67),    # anodised light grey / black, dull, 70 C, SW (ref 9); LW 0.95-0.97
    "al_cast_blasted": 0.47,             # cast, blast cleaned, 70 C, SW (ref 9)
    "paints": (0.84, 0.97),              # 8 colours and qualities 0.88-0.96 (70 C, ref 9); oil black flat 0.94, black
                                         # gloss 0.92, grey flat 0.97, plastic white 0.84, plastic black 0.95 (20 C, ref 6)
    "paints_black": (0.92, 0.95),        # oil black gloss 0.92, flat 0.94, plastic black 0.95 (20 C, SW, ref 6)
    "lacquer_on_al": (0.50, 0.53),       # 3 colours sprayed on aluminium, 70 C, SW (ref 9): thin coats are part clear
    "stainless_untreated": 0.30,         # sheet, untreated, somewhat scratched, 70 C, SW (ref 9); polished 0.18
}


def emissivity_ranges_mwir() -> dict:
    """MWIR (2-5 um) emissivity ranges of the engine and airframe surfaces from the SW and T entries of [FLIR-TAB], with
    the same finish logic as emissivity_ranges(): bare metal to finished (anodised or painted, whichever spans more),
    grooves raise the fin packs."""
    e, sw = EMISSIVITY_SOURCES, EMISSIVITY_SOURCES_SW
    hs = heatsink_geometry()
    bare = e["al_as_received"]                                       # T band
    finished = sw["paints_black"][1]                                 # black paint 0.95 > anodised 0.67 (both covered)
    return dict(engine_head=(bare, round(sw["paints_black"][0] + 0.02, 2)),
                engine_heatsink=(round(fin_apparent_emissivity(sw["paints_black"][0], hs), 2),
                                 round(fin_apparent_emissivity(sw["paints_black"][1], hs), 2)),
                engine_barrel=(round(fin_apparent_emissivity(bare), 2), round(fin_apparent_emissivity(finished), 2)),
                engine_case=(sw["lacquer_on_al"][0], sw["paints_black"][1]),   # thin lacquer to black paint
                engine_intake=(e["al_polished"][0], e["al_oxidised_strongly"][1]),   # T band
                engine_flange=(e["al_polished"][1], sw["al_anodised_dull"][1]),
                exhaust=(sw["stainless_untreated"], e["soot"]),                 # SW untreated -> T oxidised / soot
                tail_cone=sw["paints"], skin=sw["paints"])


def emissivity_ranges() -> dict:
    """LWIR emissivity ranges of the engine surfaces (signatures.yaml groups), from [FLIR-TAB] and geometry.
    Cast aluminium runs from as-received (0.09) through blast-cleaned castings (0.46) to dull anodised or
    painted (0.97); the head block's finish is open (bare or finished); the heat sinks are black on both photos."""
    e = EMISSIVITY_SOURCES
    bare, finished = e["al_as_received"], e["al_anodised_dull"][1]
    hs = heatsink_geometry()
    return dict(engine_head=(bare, finished),
                engine_heatsink=(round(fin_apparent_emissivity(e["black_lacquer"][0], hs), 2),   # black finish
                                 round(fin_apparent_emissivity(e["black_lacquer"][1], hs), 2)),  # (both photos)
                engine_barrel=(round(fin_apparent_emissivity(bare), 2), round(fin_apparent_emissivity(finished), 2)),
                engine_case=e["black_lacquer"],                         # black painted case and covers
                engine_intake=(e["al_polished"][0], e["al_oxidised_strongly"][1]),  # bare die-cast carb bodies
                engine_flange=(e["al_polished"][1], e["al_anodised_dull"][0]),      # plated or gold-anodised disc
                exhaust=INPUTS["eps_stub"],                             # stainless, heat-tinted -> oxidised / sooted
                tail_cone=e["black_lacquer"])                           # painted airframe at the mounting face


# ============================================================================ report
def report() -> str:
    g, sg = fin_geometry(), stub_geometry()
    mid = lambda r: 0.5 * (r[0] + r[1])  # noqa: E731
    lines = ["MD-550 / L550E surface temperature derivation (excess over ambient, K)", ""]
    lines.append(f"displacement {el.displacement_m3() * 1e6:.1f} cc ([L550E-DS] 548); fin pack per cylinder "
                 f"{g['n']} fins, {g['area_m2']:.3f} m2, fin height {g['length_m'] * 1e3:.1f} mm, gap "
                 f"{g['gap_m'] * 1e3:.2f} mm; stub D {sg['d_o'] * 1e3:.1f}/{sg['d_i'] * 1e3:.1f} mm, "
                 f"L {sg['length'] * 1e3:.1f} mm")
    r503 = bsfc_rotax_503()
    lines.append(f"sourced inputs: BSFC {INPUTS['bsfc_g_kwh'][0]:.0f}-{INPUTS['bsfc_g_kwh'][1]:.0f} g/kWh "
                 f"([L550E-DS] chart {bsfc_l550e_chart()[0]:.0f}-{bsfc_l550e_chart()[1]:.0f}; [ROTAX-OM] 503 take-off "
                 f"{r503['takeoff'][0]:.0f}-{r503['takeoff'][1]:.0f}, 75 % {r503['cont75'][0]:.0f}-"
                 f"{r503['cont75'][1]:.0f}, stated {r503['stated']:.0f}); AFR {INPUTS['afr'][0]:.1f}-{INPUTS['afr'][1]:.1f} "
                 f"(stoichiometric {afr_stoich(FUEL_H_TO_C[0]):.2f} / {afr_stoich(FUEL_H_TO_C[1]):.2f}, phi "
                 f"{PHI[0]}-{PHI[1]}); cooling / shaft {INPUTS['cooling_over_shaft'][0]:.2f}-"
                 f"{INPUTS['cooling_over_shaft'][1]:.2f}")
    lines.append("")
    for name, s in POWER_SETTINGS.items():
        f_mid = mid(s["f"])
        op = operating_state(f_mid, mid(INPUTS["bsfc_g_kwh"]), mid(INPUTS["afr"]))
        st = stub_state(mid(s["egt_excess_k"]), op["exhaust_kg_s_per_cyl"], 47.5, 0.0, 288.15, mid(INPUTS["eps_stub"]),
                        mid(INPUTS["f_pulse"]), mid(INPUTS["k_stub"]))
        dr_lo = min(operating_state(f, b, a)["delivery_ratio"] for f in s["f"] for b in INPUTS["bsfc_g_kwh"]
                    for a in INPUTS["afr"])
        dr_hi = max(operating_state(f, b, a)["delivery_ratio"] for f in s["f"] for b in INPUTS["bsfc_g_kwh"]
                    for a in INPUTS["afr"])
        xe = (exhaust_share(s["egt_excess_k"][0], INPUTS["afr"][0]), exhaust_share(s["egt_excess_k"][1], INPUTS["afr"][1]))
        hb, hbi = fin_heat_balance(name), fin_heat_balance(name, with_inflow=True)
        hb0, hbb = fin_heat_balance(name, with_heatsink=False), fin_heat_balance(name, at="best")
        inflow = [inflow_bound(f, v, 0.0, 288.15) for f in s["f"] for v in ENV["airspeed_mps"]]
        lines.append(f"[{name}] f {s['f'][0]:.2f}-{s['f'][1]:.2f}: shaft {op['shaft_w'] / 1e3:.1f} kW, "
                     f"{op['rpm']:.0f} rpm, exhaust {op['exhaust_kg_s_per_cyl'] * 1e3:.1f} g/s per cylinder; CHT band "
                     f"{s['cht_excess_k'][0]:.0f}-{s['cht_excess_k'][1]:.0f} K, EGT band {s['egt_excess_k'][0]:.0f}-"
                     f"{s['egt_excess_k'][1]:.0f} K (over {T_REF_C:.0f} C)")
        lines.append(f"   checks: delivery ratio {dr_lo:.2f}-{dr_hi:.2f} (0.25-1.25 OK); sensible exhaust share "
                     f"{xe[0]:.2f}-{xe[1]:.2f} (<= 0.30 OK)")
        lines.append(f"   heat balance per cylinder: load {hb['q_per_cyl_w'] / 1e3:.2f} kW vs free-air cooling "
                     f"{(hb['q_barrel_w'] + hb['q_heatsink_w']) / 1e3:.2f} kW (barrel fins {hb['q_barrel_w'] / 1e3:.2f} at "
                     f"NACA 488 h {hb['h_barrel']:.0f} W/m2K, heat sink {hb['q_heatsink_w'] / 1e3:.2f}): ratio "
                     f"{hb['ratio']:.2f}; {hbi['ratio']:.2f} with the propeller inflow bound; {hb0['ratio']:.2f} without the "
                     f"heat sinks; {hbb['ratio']:.2f} at the best-cooled corner of the envelope")
        lines.append(f"   propeller inflow at the nearest cylinder, sea level: up to {min(inflow):.2f}-{max(inflow):.2f} "
                     f"x flight speed (momentum theory, ideal propeller)")
        lines.append(f"   stub at centre: Re_i {st['re_i']:.0f}, h_i {st['h_i']:.0f}, h_o {st['h_o']:.0f}, "
                     f"h_r {st['h_r']:.0f} W/m2K, phi {st['phi']:.2f}, NTU {st['ntu']:.3f}")
        lines.append(f"   plume/stub LWIR contrast at centre: "
                     f"{plume_to_stub_lwir_ratio(st['theta_gas_out'], st['theta_wall'], mid(INPUTS['eps_stub'])):.3f}")
        for k, (lo, hi) in envelope(name).items():
            lines.append(f"   {k:14s} {lo:7.1f} .. {hi:7.1f}")
        lines.append("")
    lines.append("signatures.yaml lwir.target.engine_profiles (rounded outward to 5 K):")
    for name, r in yaml_ranges().items():
        body = ", ".join(f"{k}: [{lo:.1f}, {hi:.1f}]" for k, (lo, hi) in r.items())
        lines.append(f"  {name}: {{weight: {POWER_SETTINGS[name]['weight']:.2f}, {body}}}")
    lines.append("")
    lines.append("LWIR emissivity ranges: " + ", ".join(f"{k} {lo:.2f}-{hi:.2f}" for k, (lo, hi) in emissivity_ranges().items()))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
