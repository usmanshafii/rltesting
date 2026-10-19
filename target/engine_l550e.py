"""MD-550 / Limbach L550E pusher installation: one geometry for the dataset mesh and the presentation model.

The KAL's engine is not published (evidence level D). By the user's decision of 2026-10-08 the model uses
the engine of the most similar drone (target.yaml G1): the MD-550, a copy of the Limbach L550E
(4-cylinder two-stroke boxer, air cooled, one carburettor per cylinder). What is modelled is what a camera
can see and what sets its signature: the external envelope and visible layout (this file), and the
surface temperatures of each part (thermal/engine_heat.py derives them from heat transfer). The ratings
below are used ONLY by that derivation (engine air flow and heat load) and by the propeller rotation
blur; nothing else reads them. Grades A-D as in the configs/signatures.yaml header; source tiers S1-S4 as in
CLAUDE.md. Source [L550E-DS]: Limbach Flugmotoren, "LIMBACH L550E Technical Specification" (2 pp., limflug.de
downloads, listed 2025-08-01; checked 2026-10-10), and its sister sheet [L550EF-DS] (same cylinders, heads and
case; fuel injected):

    rated 37 kW (50 HP) at 7,500 rpm; bore 66 x stroke 40 mm, 548 cm3;      B  [L550E-DS] p. 1 (S1 for the L550E;
        16 kg; cast Al cylinders, Nikasil bores; 4 diaphragm carburettors        an analogue for the KAL)
    envelope 300 x 410 x 301 mm                                              B  [L550E-DS] p. 2 drawing: 410 plug
                                                                                 cap to plug cap, 301.7 carburettor to
                                                                                 exhaust-stack end (measured, 1.588
                                                                                 px/mm); 300 [L550EF-DS] p. 2 label
    4 finned cylinders as 2 opposed banks, a drive flange disc and ignition
        units at the propeller end, exhaust stacks on one side of the crank
        axis and carburettors on the other                                   B  [L550E-DS] p. 2 drawing and photos
    installed carburettors up / stubs down                                   B  KAL photo P4 consistent; analogue G1
    fin pack, head block, stub size and position                             B- measured on the [L550E-DS] and
                                                                                 [L550EF-DS] drawings (scale from their
                                                                                 own dimension lines, +-1 mm); see
                                                                                 docs/engine_signature_physics.md 2b
    bank offset, case and carburettor boxes, ignition units                  C  inside the envelope

Engine frame (metres): x forward from the drive-flange face (the propeller side, i.e. aft on the aircraft)
towards the mounting face, y to starboard, z up, origin on the crankshaft axis. kal_geometry maps it to the
airframe. Every part returned here is a closed primitive; the dataset mesh uses them as they are, the
presentation model (kal_hero) adds individual cooling fins.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- published data (level B, [L550E-DS])
LENGTH_M, WIDTH_M, HEIGHT_M = 0.300, 0.410, 0.301
BORE_M, STROKE_M = 0.066, 0.040
N_CYL = 4
RATED_POWER_W, RATED_RPM = 37_000.0, 7_500.0     # inputs of the heat balance and the blur only
# Full-throttle specific fuel consumption from the [L550E-DS] p. 2 chart, digitised by tools/source_measurements.py
# (docs/sources/limbach/sfc_chart_digitised.csv; +-3 g/HPh; the 6,625 rpm bin runs along the 340 gridline):
# ~435 g/HPh at 4,000-5,000 rpm falling to ~335 g/HPh at 7,000-7,500 rpm. "HP" is converted both as PS
# (0.7355 kW) and as hp (0.7457 kW), since 37 kW = 50 HP fits either (thermal/engine_heat.bsfc_band).
SFC_CHART_G_PER_HPH = {4125: 436.1, 4375: 434.1, 4625: 434.1, 4875: 437.4, 5125: 430.8, 5375: 410.8, 5625: 393.5,
                       5875: 384.9, 6125: 372.2, 6375: 354.9, 6625: 340.0, 6875: 337.6, 7125: 335.6, 7375: 334.3}


def displacement_m3(per_cylinder: bool = False) -> float:
    """Swept volume pi/4 B^2 S (x 4): 136.8 cc per cylinder, 547 cc in all (published 548 cc)."""
    v = math.pi / 4.0 * BORE_M ** 2 * STROKE_M
    return v if per_cylinder else N_CYL * v

# ---------------------------------------------------------------- layout
# Measured on the manufacturer drawings (B-): [L550EF-DS] front view 1.707 px/mm (its 392 and 128 + 235 mm dimension
# lines agree within 0.3 %), side view 1.575 px/mm (300 + 204 mm); [L550E-DS] front view 1.588 px/mm (its 410 mm line,
# cross-checked by the 364 mm across the heat sinks and the 114 mm fin height measured on the EF sheet).
FLANGE = dict(x0=0.000, x1=0.022, r=0.075, rim_r=0.084)      # gold knurled drive disc and black rim (photo; C)
FRONT_COVER = dict(x0=0.022, x1=0.050, w=0.112, h=0.112)     # C
CASE = dict(x0=0.050, x1=0.286, w=0.115, h=0.125)            # black crankcase (C)
MOUNT = dict(x0=0.286, x1=0.300, r=0.072)                    # rear cover / mounting face (C)
# Barrel fin stack per cylinder: rounded-rectangle plates 108 (along the crank) x 114 mm (side view outline; front
# views -54..+60 mm), 12 fins at a 6.67 mm pitch, ~2.5 mm thick (spacing of the drawn fin faces), from 66 mm out to
# 146 mm from the crank axis (front views). Corner radius and barrel root radius: C.
FIN_PACK = dict(dx=0.108, dz=0.114, y0=0.066, y1=0.146, corner=0.022, core_r=0.041,
                fins=12, fin_t=0.0025)
# Head end: the cylinder head, 146 -> 157 mm out ([L550E-DS]: ~314 mm across "without heat sinks"), then Limbach's
# optional heat sink, 157 -> 182 mm ([L550E-DS]: 364 mm across "with optional heat sinks"; fitted, as the 410 mm
# plug-cap width needs it). Outline ~100 x 93 mm (EF front view height, side view inner outline).
HEAD = dict(y0=0.146, y1=0.157, dx=0.100, dz=0.093, corner=0.020)
# Heat sink (photos on both datasheets): a black finned block on each head, its fins thin plates across the crank
# axis (stacked along x, each in a y-z plane), closed on the propeller side, a central well for the spark plug.
# ~12 fins (+-2, counted on the photos and the side-view drawing), ~2 mm thick (C), 25 mm deep (157 -> 182 mm),
# 93 mm tall; base plate 4 mm (C). Its cooling and temperature: thermal/engine_heat.heatsink_state.
HEATSINK = dict(y0=0.157, y1=0.182, dx=0.100, dz=0.093, corner=0.020, fins=12, fin_t=0.002, base_t=0.004,
                plug_well_w=0.026)
PLUG = dict(y0=0.182, y1=0.205, r=0.0095)                    # spark plug and cap: 410 mm plug cap to plug cap
# cylinder centres along x: 110 mm apart within a bank (side view); the port bank offset by the crank-pin spacing (C)
CYLINDERS = (("stbd_front", +1, 0.090), ("stbd_rear", +1, 0.200),
             ("port_front", -1, 0.120), ("port_rear", -1, 0.230))
# carburettor stack scaled so the installed height (carburettor top to stub end) stays 301 mm (C)
CARBS = dict(w=0.040, d=0.044, h=0.042, spacer_h=0.016, horn_r=0.016, horn_h=0.018, y=0.034)
# Exhaust stacks ([L550E-DS] front view): 31.5 mm OD, centres 88.5 mm either side (178 mm apart), flange 67 mm and
# end 162.5 mm from the crank axis. The mesh tube starts at the fin pack (57 mm) so it stays attached; the flange-to-
# end length (95.5 mm) is what the heat model uses. Wall 1.5 mm: C (thermal/engine_heat.STUB_WALL_M).
STUB = dict(r=0.01575, outlet_r=0.01575, y=0.0885, z_top=-0.057, z_flange=-0.067, z_end=-0.1625, outlet_len=0.016)
IGNITION = (dict(x0=0.026, x1=0.060, y=0.0, z0=0.0625, z1=0.104, w=0.060),
            dict(x0=0.026, x1=0.060, y=0.0, z0=-0.112, z1=-0.0625, w=0.060))


# ---------------------------------------------------------------- propeller physics used by the renders
GAMMA_AIR, R_AIR = 1.4, 287.05


def rpm_at_power_fraction(fraction: float, rated_rpm: float = RATED_RPM) -> float:
    """Shaft speed of a fixed-pitch propeller at a fraction of rated power. Propeller power
    P = C_P rho n^3 D^5; at a near-constant power coefficient C_P, n / n_rated = (P / P_rated)^(1/3).
    15 % power -> 0.53 x 7,500 = 3,990 rpm: the lower end of target.yaml prop_rpm."""
    return rated_rpm * float(fraction) ** (1.0 / 3.0)


def tip_mach(dia_m: float, rpm: float, airspeed_mps: float, air_temp_k: float) -> float:
    """Helical tip Mach number sqrt((pi n D)^2 + V^2) / a, a = sqrt(gamma R T). Propellers are kept below
    about 0.8-0.85: above it compressibility losses and noise rise steeply. Bounds target.yaml prop_dia_m."""
    v_tip = math.hypot(math.pi * rpm / 60.0 * dia_m, airspeed_mps)
    return v_tip / math.sqrt(GAMMA_AIR * R_AIR * air_temp_k)


def cylinder_axes():
    """[(name, side, x centre)] of the four cylinders (engine frame)."""
    return list(CYLINDERS)


def parts() -> list[dict]:
    """Closed primitives of the installation (engine frame, metres).

    kind "box": centre c, size (sx, sy, sz); kind "cyl": axis from p0 to p1, radius r0 -> r1;
    kind "rrect": rounded-rectangle prism along y from y0 to y1 at (x, z), size (dx, dz), corner radius.
    `slot` is the thermal / optical surface class used by configs/signatures.yaml.
    """
    out = []
    out.append(dict(name="drive_flange", slot="engine_flange", kind="cyl",
                    p0=(FLANGE["x0"], 0, 0), p1=(FLANGE["x1"], 0, 0), r0=FLANGE["r"], r1=FLANGE["r"]))
    out.append(dict(name="flange_rim", slot="engine_case", kind="cyl",
                    p0=(FLANGE["x0"] + 0.004, 0, 0), p1=(FLANGE["x1"] + 0.004, 0, 0),
                    r0=FLANGE["rim_r"], r1=FLANGE["rim_r"]))
    out.append(dict(name="front_cover", slot="engine_case", kind="box",
                    c=((FRONT_COVER["x0"] + FRONT_COVER["x1"]) / 2, 0, 0),
                    size=(FRONT_COVER["x1"] - FRONT_COVER["x0"], FRONT_COVER["w"], FRONT_COVER["h"])))
    out.append(dict(name="crankcase", slot="engine_case", kind="box",
                    c=((CASE["x0"] + CASE["x1"]) / 2, 0, 0), size=(CASE["x1"] - CASE["x0"], CASE["w"], CASE["h"])))
    out.append(dict(name="mount_face", slot="engine_case", kind="cyl",
                    p0=(MOUNT["x0"], 0, 0), p1=(MOUNT["x1"], 0, 0), r0=MOUNT["r"], r1=MOUNT["r"]))
    for name, s, xc in CYLINDERS:
        fp = FIN_PACK
        out.append(dict(name=f"fins_{name}", slot="engine_barrel", kind="rrect", x=xc, z=0.0, side=s,
                        y0=fp["y0"], y1=fp["y1"], dx=fp["dx"], dz=fp["dz"], corner=fp["corner"]))
        hd, hs = HEAD, HEATSINK
        out.append(dict(name=f"head_{name}", slot="engine_head", kind="rrect", x=xc, z=0.0, side=s,
                        y0=hd["y0"], y1=hd["y1"], dx=hd["dx"], dz=hd["dz"], corner=hd["corner"]))
        out.append(dict(name=f"heatsink_{name}", slot="engine_heatsink", kind="rrect", x=xc, z=0.0, side=s,
                        y0=hs["y0"], y1=hs["y1"], dx=hs["dx"], dz=hs["dz"], corner=hs["corner"]))
        out.append(dict(name=f"plug_{name}", slot="engine_head", kind="cyl",
                        p0=(xc, s * PLUG["y0"], 0), p1=(xc, s * PLUG["y1"], 0), r0=PLUG["r"], r1=PLUG["r"]))
        # exhaust: port on the underside of the barrel, a stub straight down and a short wider outlet
        st = STUB
        out.append(dict(name=f"stub_{name}", slot="exhaust_stub", kind="cyl",
                        p0=(xc, s * st["y"], st["z_top"]), p1=(xc, s * st["y"], st["z_end"] + st["outlet_len"]),
                        r0=st["r"], r1=st["r"]))
        out.append(dict(name=f"outlet_{name}", slot="exhaust_stub", kind="cyl",
                        p0=(xc, s * st["y"], st["z_end"] + st["outlet_len"]), p1=(xc, s * st["y"], st["z_end"]),
                        r0=st["outlet_r"], r1=st["outlet_r"]))
        # carburettor on its intake spacer, side by side on top of the case (photo: 2 x 2 layout)
        cb = CARBS
        y = s * cb["y"]
        z0 = CASE["h"] / 2
        out.append(dict(name=f"spacer_{name}", slot="engine_case", kind="box",
                        c=(xc, y, z0 + cb["spacer_h"] / 2), size=(cb["w"] * 0.8, cb["d"] * 0.8, cb["spacer_h"])))
        out.append(dict(name=f"carb_{name}", slot="engine_intake", kind="box",
                        c=(xc, y, z0 + cb["spacer_h"] + cb["h"] / 2), size=(cb["w"], cb["d"], cb["h"])))
        out.append(dict(name=f"horn_{name}", slot="engine_intake", kind="cyl",
                        p0=(xc, y, z0 + cb["spacer_h"] + cb["h"]),
                        p1=(xc, y, z0 + cb["spacer_h"] + cb["h"] + cb["horn_h"]), r0=cb["horn_r"], r1=cb["horn_r"]))
    for k, ig in enumerate(IGNITION):
        out.append(dict(name=f"ignition_{k}", slot="engine_case", kind="box",
                        c=((ig["x0"] + ig["x1"]) / 2, ig["y"], (ig["z0"] + ig["z1"]) / 2),
                        size=(ig["x1"] - ig["x0"], ig["w"], ig["z1"] - ig["z0"])))
    return out


def outlets() -> list[dict]:
    """Exhaust outlet centres and the outward axis (engine frame): where each plume starts."""
    return [dict(name=name, centre=(xc, s * STUB["y"], STUB["z_end"]), axis=(0.0, 0.0, -1.0),
                 radius=STUB["outlet_r"]) for name, s, xc in CYLINDERS]


def envelope(ps: list[dict] | None = None) -> tuple[tuple[float, float], ...]:
    """(x, y, z) extents of the primitives, for the published-envelope check."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for p in ps or parts():
        if p["kind"] == "box":
            for i in range(3):
                lo[i] = min(lo[i], p["c"][i] - p["size"][i] / 2)
                hi[i] = max(hi[i], p["c"][i] + p["size"][i] / 2)
        elif p["kind"] == "cyl":
            r = max(p["r0"], p["r1"])
            for q in (p["p0"], p["p1"]):
                ax = [abs(p["p1"][i] - p["p0"][i]) > 1e-12 for i in range(3)]
                for i in range(3):
                    pad = 0.0 if ax[i] else r
                    lo[i] = min(lo[i], q[i] - pad)
                    hi[i] = max(hi[i], q[i] + pad)
        elif p["kind"] == "rrect":
            ys = sorted((p["side"] * p["y0"], p["side"] * p["y1"]))
            for i, (a, b) in enumerate(((p["x"] - p["dx"] / 2, p["x"] + p["dx"] / 2), ys,
                                        (p["z"] - p["dz"] / 2, p["z"] + p["dz"] / 2))):
                lo[i] = min(lo[i], a)
                hi[i] = max(hi[i], b)
    return tuple((lo[i], hi[i]) for i in range(3))


def check_envelope(tol: float = 0.003) -> list[str]:
    """Published L550E envelope: 300 x 410 x 301 mm. Returns problems (empty = consistent)."""
    (x0, x1), (y0, y1), (z0, z1) = envelope()
    bad = []
    for name, got, want in (("length", x1 - x0, LENGTH_M), ("width", y1 - y0, WIDTH_M), ("height", z1 - z0, HEIGHT_M)):
        if abs(got - want) > tol:
            bad.append(f"{name} {got * 1000:.1f} mm vs published {want * 1000:.0f} mm")
    return bad


if __name__ == "__main__":
    (x0, x1), (y0, y1), (z0, z1) = envelope()
    print(f"envelope L {1000 * (x1 - x0):.1f} x W {1000 * (y1 - y0):.1f} x H {1000 * (z1 - z0):.1f} mm "
          f"(published {LENGTH_M * 1000:.0f} x {WIDTH_M * 1000:.0f} x {HEIGHT_M * 1000:.0f})")
    print("problems:", check_envelope() or "none")
    print(len(parts()), "parts")
