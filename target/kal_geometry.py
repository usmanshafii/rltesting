"""KAL target geometry as a pure-NumPy mesh (no Blender dependency).

build_target.py turns this mesh into a Blender object. tools/draw_reference.py and
tools/photo_check.py use the same mesh, so the reference drawing, the photo overlay
and the rendered model cannot diverge.

Engine: the Geran-2's MD-550 (Limbach L550E copy), user decision of 2026-10-08. Its parts come from
target/engine_l550e.py in fixed metres; the airframe scales with the sampled wingspan, the engine does not.

Internal frame (span units): x aft from the wing apex, y to starboard, z up.
Output frame (metres, Blender): +X forward, +Y to port, +Z up, origin on the
fuselage axis midway between the nose tip and the propeller hub.
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import engine_l550e as el  # noqa: E402

SLOTS = ["skin", "control_surface", "fin", "fuselage", "markings", "decal",
         "engine_head", "engine_barrel", "engine_case", "engine_intake", "engine_flange", "exhaust_stub",
         "tail_cone", "propeller", "propeller_disc", "antenna", "nose_window", "engine_heatsink"]
# Engine surface classes (configs/signatures.yaml gives each its own temperature and emissivity)
ENGINE_SLOTS = ("engine_head", "engine_heatsink", "engine_barrel", "engine_case", "engine_intake", "engine_flange",
                "exhaust_stub")
HOT_SLOTS = ("engine_head", "engine_heatsink", "engine_barrel", "exhaust_stub")
SMOOTH_SLOTS = {"skin", "control_surface", "fuselage", "markings", "tail_cone"}
DECAL_PARTS = {"wing_text_stbd", "wing_text_port", "fin_text_stbd", "fin_text_port", "flag"}
# Coarse part groups, stored per face in Blender as the INT attribute `kal_part`
PART_NAMES = ["wing", "fin", "fuselage", "engine", "exhaust", "spinner", "propeller_disc",
              "blade", "probe", "antenna", "decal"]


def part_group(part: str) -> str:
    if part in DECAL_PARTS or part == "nose_window":
        return "decal"
    for prefix, group in (("fin_", "fin"), ("eng_stub_", "exhaust"), ("eng_outlet_", "exhaust"),
                          ("eng_", "engine"), ("blade_", "blade"), ("antenna_", "antenna")):
        if part.startswith(prefix):
            return group
    return {"prop_disc": "propeller_disc"}.get(part, part)


DEFAULT_YAML = os.path.join(HERE, "..", "configs", "target.yaml")


# ============================================================================ parameters
def load_config(path: str = DEFAULT_YAML) -> dict:
    """PyYAML if available, else the bundled strict subset parser (Blender has no PyYAML)."""
    try:
        import yaml
    except ImportError:
        import _yamlite
        return _yamlite.load(path)
    with open(path) as f:
        return yaml.safe_load(f)


def _walk(d, fn):
    for k, v in d.items():
        if isinstance(v, dict) and ("default" in v or "range" in v):
            fn(k, v)
        elif isinstance(v, dict):
            _walk(v, fn)
        else:
            fn(k, v)


def defaults(cfg: dict) -> dict:
    """Flatten the YAML into {name: value}, taking `default` for ranged entries."""
    out = {}
    _walk(cfg, lambda k, v: out.__setitem__(k, v["default"] if isinstance(v, dict) else v))
    return out


def ranges(cfg: dict) -> dict:
    out = {}
    _walk(cfg, lambda k, v: out.__setitem__(k, tuple(v["range"])) if isinstance(v, dict) and "range" in v else None)
    return out


def derive(p: dict) -> dict:
    """All dependent quantities. Sweep and length are derived, never entered."""
    g = dict(p)
    s = p["span_m"]
    cr = p["root_chord"]
    ct = cr * p["tip_over_root"]
    tan_l = (cr - ct) / 0.5
    Rf = p["forebody_dia"] / 2
    eng0 = p["engine_start_over_root"] * cr                 # mounting face, end of the fuselage
    eng1 = eng0 + el.LENGTH_M / s                           # drive-flange face (propeller side)
    g.update(
        tip_chord=ct, te_x=cr, tip_le_x=cr - ct, tan_sweep=tan_l,
        sweep_deg=math.degrees(math.atan(tan_l)),
        nose_x=-p["nose_over_root"] * cr,
        R_fore=Rf, R_tube=Rf * p["tube_over_forebody"], R_pod=Rf * p["pod_over_forebody"],
        R_neck=Rf * p["neck_over_forebody"],
        dome_len=p["dome_len_over_forebody"] * p["forebody_dia"],
        collar_x=Rf * tan_l,                    # leading edge meets the forebody side here
        pod0=p["pod_start_over_root"] * cr, pod1=p["pod_max_over_root"] * cr,
        neck_x=p["neck_over_root"] * cr,
        eng0=eng0, eng1=eng1,
        hub_x=eng1 + p["prop_hub_aft_flange_m"] / s,
        elevon_c=p["elevon_chord_over_root"] * cr,
        fin_root_chord=p["fin_root_over_tip_chord"] * ct,
        prop_dia=p["prop_dia_m"] / s, spinner_len=p["spinner_len_m"] / s, spinner_dia=p["spinner_dia_m"] / s,
        blade_root_chord=p["blade_root_chord_m"] / s, blade_tip_chord=p["blade_tip_chord_m"] / s,
    )
    g["band1"] = g["collar_x"] - 0.010
    g["band0"] = g["band1"] - p["band_width"]
    g["length"] = g["hub_x"] - g["nose_x"]
    g["length_with_probe"] = g["length"] + p["probe_len"]
    g["x_mid"] = 0.5 * (g["nose_x"] + g["hub_x"])
    g["fin_total_height"] = p["fin_up_height"] + p["fin_down_height"]
    rk = math.tan(math.radians(p["fin_te_rake_deg"]))
    g["fin_te_top_x"] = g["te_x"] - p["fin_up_height"] * rk      # + rake tilts the top forward
    g["fin_te_bot_x"] = g["te_x"] + p["fin_down_height"] * rk
    # engine frame -> internal: x_int = eng1 - x_e / s (x_e runs forward from the flange), y, z scaled
    fins_fwd = max(xc + el.FIN_PACK["dx"] / 2 for _, _, xc in el.CYLINDERS)
    g["engine_fins_front_x"] = eng1 - fins_fwd / s                # most forward fin-pack face (internal)
    g["engine_mount_radius"] = el.MOUNT["r"] / s
    return g


def check(g: dict) -> list[str]:
    """Consistency rules. An empty list means a valid variant."""
    bad = []
    lo, hi = g["le_sweep_limits_deg"]
    if not lo <= g["sweep_deg"] <= hi:
        bad.append(f"sweep {g['sweep_deg']:.1f} outside {lo}-{hi}")
    if not (g["nose_x"] + g["dome_len"] < g["band0"] < g["collar_x"] < g["collar_x"] + g["collar_step_len"]
            < g["pod0"] < g["pod1"] < g["neck_x"] < g["eng0"] + 1e-9):
        bad.append("fuselage stations out of order")
    if g["fin_root_chord"] > g["tip_chord"] + 1e-9:
        bad.append("fin root chord longer than the wing tip chord")
    if g["fin_tip_chord"] >= g["fin_root_chord"]:
        bad.append("fin tip chord not shorter than its root chord")
    if g["elevon_c"] >= 0.9 * g["tip_chord"]:
        bad.append("elevon chord too large for the tip chord")
    if g["engine_fins_front_x"] < g["te_x"] - 1e-9:
        bad.append("engine cylinders cut the wing trailing edge")
    if g["R_neck"] < g["engine_mount_radius"] * 0.98:
        bad.append("fuselage neck narrower than the engine mounting face")
    if g["eng1"] >= g["hub_x"] - g["spinner_len"] / 2 + 1e-9:
        bad.append("engine overlaps the propeller hub")
    if g["R_tube"] <= 0.5 * g["wing_tc"] * g["te_x"]:
        bad.append("wing root thicker than the fuselage tube")
    ax_max = max(g["antenna_x_from_nose"]) + g["antenna_chord"]
    if g["nose_x"] + ax_max >= g["band0"]:
        bad.append("antennas overlap the band")
    return bad


def sample_variant(cfg: dict, index: int, seed: int | None = None, max_tries: int = 500) -> dict:
    """Variant `index`: uniform draw inside every range, resampled until the derived
    geometry is valid. Each index has its own stream seeded by (base seed, index), so a
    variant is identical whether 5 or 5,000 are generated."""
    base_seed = int(seed if seed is not None else cfg.get("variants", {}).get("seed", 0))
    rng = np.random.default_rng([base_seed, int(index)])
    base, rg = defaults(cfg), ranges(cfg)
    for _ in range(max_tries):
        p = dict(base)
        for k, (a, b) in rg.items():
            p[k] = float(rng.uniform(a, b))
        if not check(derive(p)):
            p["variant_index"], p["variant_seed"] = int(index), base_seed
            return p
    raise RuntimeError("no valid variant found; ranges are inconsistent")


# ============================================================================ mesh builder
class MeshBuilder:
    def __init__(self):
        self.v: list = []
        self.f: list = []
        self.slot: list = []
        self.part: list = []
        self.uv: dict = {}           # face index -> [(u, v)] per corner (decals only)

    def add_verts(self, pts) -> int:
        i0 = len(self.v)
        self.v.extend([tuple(float(c) for c in q) for q in pts])
        return i0

    def add_face(self, idx, slot, part, uv=None):
        self.f.append(tuple(int(i) for i in idx))
        self.slot.append(SLOTS.index(slot))
        self.part.append(part)
        if uv is not None:
            self.uv[len(self.f) - 1] = list(uv)

    def orient_part(self, part):
        """Flip every face of a closed part if its signed volume is negative (outward normals)."""
        V = np.asarray(self.v)
        ids = [i for i, p in enumerate(self.part) if p == part]
        vol = 0.0
        for i in ids:
            f = self.f[i]
            for k in range(1, len(f) - 1):
                vol += np.dot(V[f[0]], np.cross(V[f[k]], V[f[k + 1]])) / 6.0
        if vol < 0:
            for i in ids:
                self.f[i] = self.f[i][::-1]
                if i in self.uv:
                    self.uv[i] = self.uv[i][::-1]

    # ---- primitives (each produces one closed, consistently wound part)
    def ring_loft(self, rings, slot_fn, part, cap0=None, cap1=None, tip0=None):
        M = len(rings[0])
        starts = [self.add_verts(r) for r in rings]
        if tip0 is not None:
            t = self.add_verts([tip0])
            for i in range(M):
                self.add_face((t, starts[0] + (i + 1) % M, starts[0] + i), slot_fn(-1, i), part)
        for j in range(len(rings) - 1):
            a, b = starts[j], starts[j + 1]
            for i in range(M):
                i2 = (i + 1) % M
                self.add_face((a + i, a + i2, b + i2, b + i), slot_fn(j, i), part)
        if cap0:
            self.add_face([starts[0] + i for i in range(M)][::-1], cap0, part)
        if cap1:
            self.add_face([starts[-1] + i for i in range(M)], cap1, part)
        self.orient_part(part)

    def cylinder(self, p0, p1, r0, r1, n, slot, part):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        ax = p1 - p0
        ax /= np.linalg.norm(ax)
        t = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(ax, t); e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        ring = lambda c, r: np.array([c + r * (math.cos(a) * e1 + math.sin(a) * e2) for a in ang])
        if r1 <= 1e-9:   # cone: base cap + fan to the apex
            i0 = self.add_verts(ring(p0, r0))
            tip = self.add_verts([p1])
            self.add_face([i0 + i for i in range(n)][::-1], slot, part)
            for i in range(n):
                self.add_face((i0 + i, i0 + (i + 1) % n, tip), slot, part)
            self.orient_part(part)
            return
        self.ring_loft([ring(p0, r0), ring(p1, r1)], lambda j, i: slot, part, cap0=slot, cap1=slot)

    def box(self, c, size, slot, part):
        """Axis-aligned closed box, centre c and full size (internal units)."""
        hx, hy, hz = (v / 2 for v in size)
        V = [(c[0] + sx * hx, c[1] + sy * hy, c[2] + sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        i0 = self.add_verts(V)
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.add_face([i0 + k for k in q], slot, part)
        self.orient_part(part)

    def rrect_prism(self, x, z, y0, y1, dx, dz, corner, n_corner, slot, part):
        """Rounded-rectangle outline in the x-z plane extruded along y from y0 to y1 (closed)."""
        r = min(corner, dx / 2, dz / 2)
        pts = []
        for cx, cz, a0 in ((x + dx / 2 - r, z + dz / 2 - r, 0), (x - dx / 2 + r, z + dz / 2 - r, 90),
                           (x - dx / 2 + r, z - dz / 2 + r, 180), (x + dx / 2 - r, z - dz / 2 + r, 270)):
            for k in range(n_corner + 1):
                a = math.radians(a0 + 90 * k / n_corner)
                pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
        rings = [np.array([(px, y, pz) for px, pz in pts]) for y in (y0, y1)]
        self.ring_loft(rings, lambda j, i: slot, part, cap0=slot, cap1=slot)

    def plate(self, poly_xz, y_center, thickness, slot, part):
        """Extrude a planar polygon in the x-z plane into a thin plate along y."""
        poly = np.asarray(poly_xz, float)
        n = len(poly)
        a = self.add_verts([(x, y_center - thickness / 2, z) for x, z in poly])
        b = self.add_verts([(x, y_center + thickness / 2, z) for x, z in poly])
        self.add_face([a + i for i in range(n)][::-1], slot, part)
        self.add_face([b + i for i in range(n)], slot, part)
        for i in range(n):
            i2 = (i + 1) % n
            self.add_face((a + i, a + i2, b + i2, b + i), slot, part)
        self.orient_part(part)

    def decal(self, corners, part, uv=((0, 0), (1, 0), (1, 1), (0, 1)), normal=None):
        """Single-sided quad. `normal` fixes the visible side."""
        c = np.asarray(corners, float)
        if normal is not None:
            n = np.cross(c[1] - c[0], c[2] - c[0])
            if np.dot(n, normal) < 0:
                c = c[::-1]
                uv = tuple(uv)[::-1]
        i0 = self.add_verts(c)
        self.add_face((i0, i0 + 1, i0 + 2, i0 + 3), "decal", part, uv=uv)


def _naca_half(s, tc):
    s = np.asarray(s, float)
    return 5 * tc * (0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * s ** 2 + 0.2843 * s ** 3 - 0.1036 * s ** 4)


def radius_at(g, x) -> float:
    """Fuselage radius at station x: dome, forebody, smooth collar taper, tube, pod, neck (constant to the
    engine mounting face)."""
    nx, dl = g["nose_x"], g["dome_len"]
    Rf, Rt, Rp, Rn = g["R_fore"], g["R_tube"], g["R_pod"], g["R_neck"]
    c0, c1 = g["collar_x"], g["collar_x"] + g["collar_step_len"]
    if x <= nx:
        return 0.0
    if x < nx + dl:
        t = (nx + dl - x) / dl
        return Rf * math.sqrt(max(0.0, 1 - t * t))
    if x < c0:
        return Rf
    if x < c1:
        t = (x - c0) / (c1 - c0)
        return Rf + (Rt - Rf) * (0.5 - 0.5 * math.cos(math.pi * t))     # smooth (cosine) taper, no kink
    if x < g["pod0"]:
        return Rt
    if x < g["pod1"]:
        t = (x - g["pod0"]) / (g["pod1"] - g["pod0"])
        return Rt + (Rp - Rt) * math.sin(t * math.pi / 2)
    if x < g["neck_x"]:
        t = (x - g["pod1"]) / (g["neck_x"] - g["pod1"])
        return Rn + (Rp - Rn) * math.cos(t * math.pi / 2)
    return Rn


def wing_upper_z(g, x, y) -> float:
    """Upper-surface height of the wing at (x, y); used to seat decals."""
    le = abs(y) * g["tan_sweep"]
    c = g["te_x"] - le
    s = min(max((x - le) / c, 0.0), 1.0)
    return g["wing_z"] + float(_naca_half(s, g["wing_tc"])) * c


def fin_outline(g):
    """Fin plate outline in the x-z plane: straight TE through the wing TE, swept LEs, root = fin root chord."""
    te, zw = g["te_x"], g["wing_z"]
    hu, hd = g["fin_up_height"], g["fin_down_height"]
    frc, ftc = g["fin_root_chord"], g["fin_tip_chord"]
    tt, tb = g["fin_te_top_x"], g["fin_te_bot_x"]
    return [(te - frc, zw), (tt - ftc, zw + hu), (tt, zw + hu), (te, zw),
            (tb, zw - hd), (tb - ftc, zw - hd)]


def wing_text_frame(g, side: int):
    """Centre, reading direction and letter-up direction (internal x-y) of the KAL lettering on one wing.

    Baseline parallel to that wing's leading edge, letter tops toward the leading edge, reading left to
    right for a viewer behind the aircraft (P8, P3): root -> tip on the starboard wing, tip -> root on port."""
    t = g["tan_sweep"]
    n = math.hypot(t, 1.0)
    c = np.array([g["wing_text_center_over_root"] * g["te_x"], side * g["wing_text_y"]])
    if side > 0:
        read, up = np.array([t, 1.0]) / n, np.array([-1.0, t]) / n
    else:
        read, up = np.array([-t, 1.0]) / n, np.array([-1.0, -t]) / n
    return c, read, up


def engine_point(g, q) -> tuple:
    """Engine frame (metres, x forward from the flange) -> internal frame (span units)."""
    s = g["span_m"]
    return (g["eng1"] - q[0] / s, q[1] / s, q[2] / s)


# ============================================================================ mesh
def _engine(mb: MeshBuilder, g: dict):
    """L550E / MD-550 primitives from engine_l550e.parts(), each a closed part named eng_*."""
    s = g["span_m"]
    n = int(g.get("engine_segments", 16))
    for prt in el.parts():
        name = "eng_" + prt["name"]          # eng_stub_* / eng_outlet_* -> part group "exhaust"
        if prt["kind"] == "cyl":
            mb.cylinder(engine_point(g, prt["p0"]), engine_point(g, prt["p1"]), prt["r0"] / s, prt["r1"] / s,
                        n, prt["slot"], name)
        elif prt["kind"] == "box":
            cx, cy, cz = engine_point(g, prt["c"])
            mb.box((cx, cy, cz), tuple(v / s for v in prt["size"]), prt["slot"], name)
        elif prt["kind"] == "rrect":
            xi, _, zi = engine_point(g, (prt["x"], 0.0, prt["z"]))
            y0, y1 = sorted((prt["side"] * prt["y0"] / s, prt["side"] * prt["y1"] / s))
            mb.rrect_prism(xi, zi, y0, y1, prt["dx"] / s, prt["dz"] / s, prt["corner"] / s, 3, prt["slot"], name)
        else:
            raise ValueError(f"unknown engine primitive {prt['kind']}")


def build_mesh(p: dict) -> dict:
    g = derive(p)
    problems = check(g)
    if problems:
        raise ValueError("; ".join(problems))
    mb = MeshBuilder()
    zw = g["wing_z"]

    # ---------------- fuselage: body of revolution, dome -> forebody -> collar taper -> tube -> pod -> neck
    nseg = int(g["fuselage_segments"])
    nx, dl = g["nose_x"], g["dome_len"]
    xs = list(nx + dl * (1 - np.cos(np.linspace(0.08, 1, 8) * math.pi / 2)))
    xs += [g["band0"], g["band1"], g["collar_x"]]
    xs += list(np.linspace(g["collar_x"], g["collar_x"] + g["collar_step_len"], 5)[1:])
    xs += list(np.linspace(g["collar_x"] + g["collar_step_len"], g["pod0"], 4)[1:])
    xs += list(np.linspace(g["pod0"], g["pod1"], 5)[1:])
    xs += list(np.linspace(g["pod1"], g["neck_x"], 5)[1:])
    xs += [g["eng0"]]
    xs = sorted(set(round(x, 6) for x in xs if x > nx + 1e-6))
    ang = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    rings = [np.array([(x, radius_at(g, x) * math.cos(a), radius_at(g, x) * math.sin(a)) for a in ang]) for x in xs]

    def fus_slot(j, i):
        if j < 0:
            return "fuselage"
        xm = 0.5 * (xs[j] + xs[j + 1])
        if g["band0"] - 1e-9 <= xm <= g["band1"] + 1e-9:
            return "markings"
        if xm > g["neck_x"] - 0.04:
            return "tail_cone"
        return "fuselage"
    mb.ring_loft(rings, fus_slot, "fuselage", cap1="tail_cone", tip0=(nx, 0, 0))

    # ---------------- wing: closed symmetric sections, straight TE, hinge line inserted
    npts = int(g["airfoil_points"])
    ys_half = sorted(set([0.0, 0.035, *g["elevon_breaks"], 0.185, 0.39, 0.46, 0.5]))
    ys = [-y for y in reversed(ys_half[1:])] + ys_half

    def section(y):
        le = abs(y) * g["tan_sweep"]
        c = g["te_x"] - le
        hinge = 1 - g["elevon_c"] / c
        # same index layout in every section: npts points LE->hinge (clustered at the LE),
        # 3 points hinge->TE, so faces never twist and the hinge line is one index everywhere
        s_fwd = hinge * (1 - np.cos(np.linspace(0, math.pi / 2, npts + 1)))
        s = np.r_[s_fwd, np.linspace(hinge, 1, 4)[1:]]
        zt = _naca_half(s, g["wing_tc"]) * c
        zt[-1] = 0.0
        up = [(le + si * c, y, zw + zi) for si, zi in zip(s, zt)]
        lo = [(le + si * c, y, zw - zi) for si, zi in zip(s[1:-1][::-1], zt[1:-1][::-1])]
        return np.array(up + lo), s, npts

    secs = [section(y) for y in ys]
    M = len(secs[0][0])
    starts = [mb.add_verts(sec[0]) for sec in secs]
    b0, _, b2 = g["elevon_breaks"]
    for j in range(len(ys) - 1):
        ymid = 0.5 * (abs(ys[j]) + abs(ys[j + 1]))
        s_arr = secs[j][1]
        K = len(s_arr) - 1
        hinge_idx = secs[j][2]
        for i in range(M):
            i2 = (i + 1) % M
            ci = i if i <= K else M - i
            ci2 = i2 if i2 <= K else M - i2
            aft = min(ci, ci2) >= hinge_idx
            slot = "control_surface" if (aft and b0 - 1e-9 <= ymid <= b2 + 1e-9) else "skin"
            a, b = starts[j], starts[j + 1]
            mb.add_face((a + i, a + i2, b + i2, b + i), slot, "wing")
    mb.add_face([starts[0] + i for i in range(M)][::-1], "skin", "wing")
    mb.add_face([starts[-1] + i for i in range(M)], "skin", "wing")
    mb.orient_part("wing")

    # ---------------- tip fins
    poly = fin_outline(g)
    for s, name in ((1, "fin_stbd"), (-1, "fin_port")):
        mb.plate(poly, s * 0.5, g["fin_thickness"], "fin", name)

    # ---------------- decals: wing text (parallel to each leading edge), fin text, flag
    L, H = g["wing_text_size"]
    for s, name in ((1, "wing_text_stbd"), (-1, "wing_text_port")):
        c, rd, up = wing_text_frame(g, s)
        quad = [c - rd * L / 2 - up * H / 2, c + rd * L / 2 - up * H / 2,
                c + rd * L / 2 + up * H / 2, c - rd * L / 2 + up * H / 2]
        z = max(wing_upper_z(g, q[0], q[1]) for q in quad) + 0.0015
        corners = [(q[0], q[1], z) for q in quad]       # uv (0,0),(1,0),(1,1),(0,1): u along the reading direction
        mb.decal(corners, name, normal=(0, 0, 1))
    if g.get("fin_text", True):
        # largest band inside the fin: z from -0.45 hd to +0.2 hu, x from the LE at that height to near the TE
        z0, z1 = zw - 0.45 * g["fin_down_height"], zw + 0.2 * g["fin_up_height"]

        def le_at(z):
            (x0, zz0), (x1, zz1) = (poly[0], poly[1]) if z >= zw else (poly[0], poly[5])
            t = (z - zz0) / (zz1 - zz0)
            return x0 + t * (x1 - x0)
        xa = max(le_at(z0), le_at(z1)) + 0.006
        xb = min(g["fin_te_top_x"], g["fin_te_bot_x"], g["te_x"]) - 0.008
        for s, name in ((1, "fin_text_stbd"), (-1, "fin_text_port")):
            yo = s * (0.5 + g["fin_thickness"] / 2 + 0.0008)
            # text runs along the fin height, reading downward, readable from outboard
            corners = [(xb, yo, z1), (xb, yo, z0), (xa, yo, z0), (xa, yo, z1)]
            mb.decal(corners, name, normal=(0, s, 0))
    fx = g["flag_x_over_root"] * g["te_x"]
    fl, fw = g["flag_size"]
    zf = radius_at(g, fx) + 0.0025
    mb.decal([(fx - fl / 2, -fw / 2, zf), (fx + fl / 2, -fw / 2, zf), (fx + fl / 2, fw / 2, zf), (fx - fl / 2, fw / 2, zf)],
             "flag", normal=(0, 0, 1))

    # ---------------- engine: MD-550 / L550E (fixed metres), shaft to the propeller hub
    _engine(mb, g)
    hub = g["hub_x"]
    mb.cylinder((g["eng1"] - 0.001, 0, 0), (hub - g["spinner_len"] / 2, 0, 0), 0.022 / g["span_m"], 0.022 / g["span_m"],
                12, "engine_flange", "eng_prop_hub")

    # ---------------- propeller: spinner, disc and/or blades
    mb.cylinder((hub - g["spinner_len"] / 2, 0, 0), (hub + g["spinner_len"] / 2, 0, 0),
                g["spinner_dia"] / 2, 0.0, 14, "propeller", "spinner")
    R = g["prop_dia"] / 2
    mode = g.get("prop_mode", "disc")
    if mode in ("blades", "both"):
        pitch = math.radians(g["blade_pitch_deg"])
        th = 0.006
        for s, tag in ((1, "a"), (-1, "b")):
            pts = []
            for r, c in ((g["spinner_dia"] / 2, g["blade_root_chord"]), (R, g["blade_tip_chord"])):
                for dx, dn in ((-c / 2, -th / 2), (c / 2, -th / 2), (c / 2, th / 2), (-c / 2, th / 2)):
                    pts.append((hub + dx * math.sin(pitch) + dn * math.cos(pitch),
                                dx * math.cos(pitch) - dn * math.sin(pitch), s * r))
            i0 = mb.add_verts(pts)
            for q in ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
                mb.add_face([i0 + k for k in q], "propeller", f"blade_{tag}")
            mb.orient_part(f"blade_{tag}")
    if mode in ("disc", "both"):
        mb.cylinder((hub + 0.004, 0, 0), (hub + 0.008, 0, 0), R, R, int(g["prop_disc_segments"]),
                    "propeller_disc", "prop_disc")

    # ---------------- nose probe and two blade antennas on top of the nose
    mb.cylinder((nx - g["probe_len"], 0, 0), (nx + 0.01, 0, 0), g["probe_dia"] / 2, g["probe_dia"] / 2, 6, "antenna", "probe")
    for k, axn in enumerate(g["antenna_x_from_nose"]):
        x = nx + axn
        c, h = g["antenna_chord"], g["antenna_height"]
        zb = min(radius_at(g, x), radius_at(g, x + c)) - 0.004
        rk = h * math.tan(math.radians(g["antenna_rake_deg"]))
        mb.plate([(x, zb), (x + c, zb), (x + c + rk * 0.6, zb + h), (x + rk, zb + h)], 0.0, 0.003, "antenna", f"antenna_{k}")

    if g.get("nose_window"):
        x0 = nx + 0.6 * dl
        r = radius_at(g, x0)
        mb.decal([(x0, -0.015, -r - 0.001), (x0 + 0.03, -0.015, -r - 0.001), (x0 + 0.03, 0.015, -r - 0.001), (x0, 0.015, -r - 0.001)],
                 "nose_window", normal=(0, 0, -1))
        mb.slot[-1] = SLOTS.index("nose_window")

    return dict(vertices_span=np.array(mb.v), faces=mb.f, face_slots=np.array(mb.slot),
                face_parts=mb.part, decal_uv=mb.uv, derived=g)


# ============================================================================ frames and export
def internal_to_output(g: dict, pts) -> np.ndarray:
    """Span-unit internal frame -> metres, +X forward, +Y port, +Z up, origin mid-length on the axis."""
    P = np.atleast_2d(np.asarray(pts, float))
    return np.c_[-(P[:, 0] - g["x_mid"]), -P[:, 1], P[:, 2]] * g["span_m"]


def to_output_frame(mesh: dict) -> np.ndarray:
    return internal_to_output(mesh["derived"], mesh["vertices_span"])


def engine_metadata(g: dict) -> dict:
    """Engine identity, evidence and placement for scene metadata (output frame, metres). Appearance only:
    the envelope and where the hot exhaust outlets sit, no performance data."""
    outs = []
    for o in el.outlets():
        c = internal_to_output(g, [engine_point(g, o["centre"])])[0]
        outs.append(dict(name=o["name"], centre_m=c.tolist(), axis=[0.0, 0.0, -1.0], radius_m=o["radius"]))
    return dict(model=g.get("engine_model", "MD-550 / L550E"), evidence=g.get("engine_evidence", ""),
                envelope_m=[el.LENGTH_M, el.WIDTH_M, el.HEIGHT_M],
                mounting_face_x_m=float(internal_to_output(g, [(g["eng0"], 0, 0)])[0, 0]),
                flange_face_x_m=float(internal_to_output(g, [(g["eng1"], 0, 0)])[0, 0]),
                propeller_plane_x_m=float(internal_to_output(g, [(g["hub_x"], 0, 0)])[0, 0]),
                exhaust_outlets=outs)


def keypoints(g: dict) -> dict:
    """Named 3D points (internal frame, span units) used by the photo acceptance check."""
    zw, te, tle = g["wing_z"], g["te_x"], g["tip_le_x"]
    b = g["elevon_breaks"][1]
    Rf = g["R_fore"]
    return dict(
        nose=(g["nose_x"], 0, 0), hub=(g["hub_x"], 0, 0),
        tip_le_port=(tle, -0.5, zw), tip_le_stbd=(tle, 0.5, zw),
        tip_te_port=(te, -0.5, zw), tip_te_stbd=(te, 0.5, zw),
        break_port=(te, -b, zw), break_stbd=(te, b, zw),
        fin_top_port=(g["fin_te_top_x"], -0.5, zw + g["fin_up_height"]),
        fin_top_stbd=(g["fin_te_top_x"], 0.5, zw + g["fin_up_height"]),
        fin_bot_port=(g["fin_te_bot_x"], -0.5, zw - g["fin_down_height"]),
        fin_bot_stbd=(g["fin_te_bot_x"], 0.5, zw - g["fin_down_height"]),
        le_root_port=(g["collar_x"], -Rf, zw), le_root_stbd=(g["collar_x"], Rf, zw),
    )


def summary(g: dict) -> dict:
    s = g["span_m"]
    return {
        "span_m": s, "length_m (nose-hub)": g["length"] * s, "length_m (with probe)": g["length_with_probe"] * s,
        "root_chord_m": g["te_x"] * s, "tip_chord_m": g["tip_chord"] * s, "le_sweep_deg": g["sweep_deg"],
        "forebody_dia_m": g["forebody_dia"] * s, "tube_dia_m": 2 * g["R_tube"] * s,
        "fin_height_m": g["fin_total_height"] * s, "fin_root_chord_m": g["fin_root_chord"] * s,
        "prop_dia_m": g["prop_dia_m"], "engine_length_m": el.LENGTH_M, "engine_width_m": el.WIDTH_M,
        "engine_height_m": el.HEIGHT_M, "engine_width_over_span": el.WIDTH_M / s,
    }


def export_obj(mesh: dict, path: str):
    V = to_output_frame(mesh)
    with open(path, "w") as f:
        f.write("# KAL target mesh, metres, +X forward +Y port +Z up\n")
        for v in V:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        cur = None
        for face, s in zip(mesh["faces"], mesh["face_slots"]):
            if s != cur:
                f.write(f"usemtl {SLOTS[s]}\n")
                cur = s
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def edge_report(mesh: dict) -> dict:
    """Per closed part: count of edges not shared by exactly two oppositely wound faces."""
    bad = defaultdict(int)
    by_part = defaultdict(list)
    for face, part in zip(mesh["faces"], mesh["face_parts"]):
        by_part[part].append(face)
    for part, faces in by_part.items():
        if part in DECAL_PARTS:
            continue
        cnt = defaultdict(int)
        for f in faces:
            for k in range(len(f)):
                cnt[(f[k], f[(k + 1) % len(f)])] += 1
        for (a, b), n in cnt.items():
            if n != 1 or cnt.get((b, a), 0) != 1:
                bad[part] += 1
    return dict(bad)


if __name__ == "__main__":
    cfg = load_config()
    mesh = build_mesh(defaults(cfg))
    g = mesh["derived"]
    for k, v in summary(g).items():
        print(f"{k:24s} {v:8.3f}")
    print("faces:", len(mesh["faces"]), " vertices:", len(mesh["vertices_span"]))
    print("non-manifold edges by part:", edge_report(mesh) or "none")
