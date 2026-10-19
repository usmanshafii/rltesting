"""High-detail presentation model of the KAL target (Blender; RGB visualisation only).

    from kal_hero import build_hero
    root = build_hero()                 # canonical defaults, or build_hero(variant_params)

Same parameter set and derived dimensions as kal_geometry.build_mesh, so span, chord, sweep, fuselage
stations, fin root chord and heights, the MD-550 / L550E engine (target/engine_l550e.py: the same parts at
the same places, 300 x 410 x 301 mm) and the propeller diameter are identical to the dataset model. The
dataset renderer keeps using build_target.py; this module only adds what matters at close range.

look="photo" follows the user's instructions of 2026-10-03 to take shape and colour (not dimensions) from
the overhead image P1: matte, gritty charcoal skin; fin tips widened to PHOTO_FIN_TIP of the root chord
(the dataset fins keep their narrow P2 tips); the pitot probe and two blade antennas of the P2 silhouette;
KAL lettering with a crossbar-less A, the fin logo and a spine emblem (assets/decals/hero_decal_atlas.png).
look="survey" keeps the dataset shapes with a satin skin. Both show the same engine. Details at close range:

  * dense tessellation (smooth wing/fuselage silhouettes instead of 10-point airfoils and 24-gon rings)
  * UV-mapped satin finish, panel seams, fasteners, elevon gaps and markings (kal_hero_tex.py)
  * the engine's individual cooling fins (engine_l550e.FIN_PACK count, pitch and thickness), spark plugs
    with boots and leads, carburettor bell-mouths, exhaust port flanges and the knurled drive flange
  * a twisted two-blade propeller (constant geometric pitch through 20 deg at 75 % radius)
  * a time-averaged disc for the spinning propeller, the way a camera with a normal shutter records it;
    `prop_mode="blades"` shows the blades for fast-shutter shots

Detail that the evidence does not fix (seam layout, fasteners, wear, blade airfoil, hub hardware, lead
routing) is a presentation assumption (level C) and is never used to measure anything.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_l550e as el  # noqa: E402
import kal_geometry as kg  # noqa: E402
import kal_hero_tex as kt  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
DECALS = os.path.join(ROOT, "assets", "decals")
TEX_CACHE = os.path.join(ROOT, "out", "hero_textures")
PHOTO_FIN_TIP = 0.75     # fin tip / root chord seen on P1 (top and bottom edges ~0.7-0.8 of the root)


# ============================================================================ mesh helper
def _signed_volume(V, faces):
    vol = 0.0
    for f in faces:
        a = V[f[0]]
        for k in range(1, len(f) - 1):
            vol += float(np.dot(a, np.cross(V[f[k]], V[f[k + 1]]))) / 6.0
    return vol


def _object(name, g, verts_int, faces, uvs=None, mat_idx=None, materials=(), smooth=True, closed=True,
            sharp_deg=None, parent=None, metres=False):
    """verts in the internal span frame (or metres, output frame, if metres=True); uvs: per face a
    list of (u, v) per corner."""
    V = np.asarray(verts_int, float) if metres else kg.internal_to_output(g, verts_int)
    faces = [list(map(int, f)) for f in faces]
    if closed and _signed_volume(V, faces) < 0:
        faces = [f[::-1] for f in faces]
        if uvs is not None:
            uvs = [list(u)[::-1] for u in uvs]
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], faces)
    for m in materials:
        me.materials.append(m)
    if mat_idx is not None:
        me.polygons.foreach_set("material_index", np.asarray(mat_idx, np.int32))
    me.polygons.foreach_set("use_smooth", [bool(smooth)] * len(faces))
    if uvs is not None:
        layer = me.uv_layers.new(name="UVMap")
        flat = [c for u in uvs for c in u]
        layer.data.foreach_set("uv", np.asarray(flat, np.float32).ravel())
    if sharp_deg is not None and hasattr(me, "set_sharp_from_angle"):
        me.set_sharp_from_angle(angle=math.radians(sharp_deg))
    me.validate(clean_customdata=False)
    if not closed:
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def _uv(col, row):
    return (col / kt.N, 1.0 - row / kt.N)


# ============================================================================ materials
def _img(path, colorspace):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


def _skin_material(paths, finish="satin"):
    """finish "satin": the thin clear coat seen in the exhibition close-up (P3). "matte": the flat,
    slightly gritty charcoal of the overhead view (P1): no coat, rougher, a little lighter, with a
    fine grain bump. Both are presentation looks (level C), not measured reflectance."""
    mat = bpy.data.materials.new("KAL_hero_skin")
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    t_alb = nt.nodes.new("ShaderNodeTexImage"); t_alb.image = _img(paths["albedo"], "sRGB")
    t_r = nt.nodes.new("ShaderNodeTexImage"); t_r.image = _img(paths["rough"], "Non-Color")
    t_n = nt.nodes.new("ShaderNodeTexImage"); t_n.image = _img(paths["normal"], "Non-Color")
    for t in (t_alb, t_r, t_n):
        t.interpolation = "Linear"
    nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 1.0
    nt.links.new(t_alb.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(t_r.outputs["Color"], b.inputs["Roughness"])
    nt.links.new(t_n.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    if finish == "matte":
        nm.inputs["Strength"].default_value = 2.0            # panel seams read clearly on P1
        tc = nt.nodes.new("ShaderNodeTexCoord")
        # gritty composite skin: centimetre mottling in albedo plus a millimetre grain bump
        mott = nt.nodes.new("ShaderNodeTexNoise")
        mott.inputs["Scale"].default_value = 45.0
        mott.inputs["Detail"].default_value = 8.0
        mott.inputs["Roughness"].default_value = 0.65
        nt.links.new(tc.outputs["Object"], mott.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.30, 0.70
        mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.70, 1.85
        nt.links.new(mott.outputs["Fac"], mr.inputs["Value"])
        gain = nt.nodes.new("ShaderNodeVectorMath"); gain.operation = "SCALE"
        nt.links.new(t_alb.outputs["Color"], gain.inputs[0])
        nt.links.new(mr.outputs["Result"], gain.inputs["Scale"])
        nt.links.new(gain.outputs["Vector"], b.inputs["Base Color"])
        rough = nt.nodes.new("ShaderNodeMapRange")
        rough.inputs["From Min"].default_value, rough.inputs["From Max"].default_value = 0.30, 0.70
        rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.66, 0.90
        nt.links.new(t_r.outputs["Color"], rough.inputs["Value"])
        nt.links.new(rough.outputs["Result"], b.inputs["Roughness"])
        grain = nt.nodes.new("ShaderNodeTexNoise")
        grain.inputs["Scale"].default_value = 260.0          # ~4 mm cells on the object (metres)
        grain.inputs["Detail"].default_value = 6.0
        nt.links.new(tc.outputs["Object"], grain.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.25
        bump.inputs["Distance"].default_value = 0.0004
        nt.links.new(grain.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(nm.outputs["Normal"], bump.inputs["Normal"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        b.inputs["Specular IOR Level"].default_value = 0.4
        return mat
    # satin two-layer paint: a thin, slightly rough clear coat over the pigmented base (P3 sheen)
    b.inputs["Coat Weight"].default_value = 0.22
    b.inputs["Coat Roughness"].default_value = 0.22
    nt.links.new(nm.outputs["Normal"], b.inputs["Coat Normal"])
    b.inputs["Specular IOR Level"].default_value = 0.5
    return mat


def _pbr(name, color, rough, metal=0.0, coat=0.0, alpha=None, emission=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    if alpha is not None:
        b.inputs["Alpha"].default_value = alpha
        mat.blend_method = "BLEND"
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
    return mat


def _machined_aluminium():
    """Turned aluminium: anisotropic-looking fine circumferential grooves via a wave bump."""
    mat = _pbr("KAL_hero_aluminium", (0.80, 0.80, 0.80), 0.28, metal=1.0)
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 260.0
    wave.inputs["Distortion"].default_value = 0.6
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Anisotropic"].default_value = 0.6
    return mat


def _cast_alloy():
    """Grey cast-aluminium fins and heads (L550E product photo): rough metallic with a fine cast grain."""
    mat = _pbr("KAL_hero_cast_alloy", (0.42, 0.42, 0.43), 0.55, metal=0.85)
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 900.0
    grain.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], grain.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(grain.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return mat


def _knurled_gold():
    """Gold-coloured knurled drive disc (L550E product photo): metallic with a fine diamond knurl bump."""
    mat = _pbr("KAL_hero_flange", (0.72, 0.54, 0.24), 0.32, metal=0.95)
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    bumps = []
    for direction in ("Y", "Z"):
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = direction
        wave.inputs["Scale"].default_value = 700.0
        nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
        bumps.append(wave)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(bumps[0].outputs["Fac"], mul.inputs[0])
    nt.links.new(bumps[1].outputs["Fac"], mul.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0003
    nt.links.new(mul.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return mat


def _blur_disc_material(name, base_rgb, coverage_fn_nodes, band_strength=0.35):
    """Time-averaged rotating part: alpha = blade coverage per radius (from a radial ramp), with a
    faint two-lobed banding that, rotated with the object, reads like a real camera's blade streaks."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    b = nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base_rgb, 1.0)
    b.inputs["Roughness"].default_value = 0.35
    b.inputs["Coat Weight"].default_value = 0.4
    uvn = nodes.new("ShaderNodeUVMap")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(uvn.outputs["UV"], sep.inputs[0])
    ramp = nodes.new("ShaderNodeValToRGB")          # radius (u) -> coverage
    ramp.color_ramp.interpolation = "EASE"
    els = ramp.color_ramp.elements
    pts = coverage_fn_nodes
    els[0].position, els[0].color = pts[0][0], (pts[0][1],) * 3 + (1,)
    els[1].position, els[1].color = pts[-1][0], (pts[-1][1],) * 3 + (1,)
    for pos, val in pts[1:-1]:
        e = els.new(pos)
        e.color = (val,) * 3 + (1,)
    links.new(sep.outputs["X"], ramp.inputs["Fac"])
    # banding: 0.5 + 0.5 cos(2 * 2 pi v), sharpened
    mul = nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 4 * math.pi
    links.new(sep.outputs["Y"], mul.inputs[0])
    cos = nodes.new("ShaderNodeMath"); cos.operation = "COSINE"
    links.new(mul.outputs[0], cos.inputs[0])
    m2 = nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY_ADD"
    m2.inputs[1].default_value = band_strength
    m2.inputs[2].default_value = 1.0
    links.new(cos.outputs[0], m2.inputs[0])
    a = nodes.new("ShaderNodeMath"); a.operation = "MULTIPLY"; a.use_clamp = True
    links.new(ramp.outputs["Color"], a.inputs[0])
    links.new(m2.outputs[0], a.inputs[1])
    links.new(a.outputs[0], b.inputs["Alpha"])
    mat.blend_method = "BLEND"
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    return mat


# ============================================================================ geometry
def _wing(g, mats, parent):
    """Dense closed wing, NACA 00xx sections (wing_tc), straight TE, the elevon hinge as a vertex
    row; UVs into the WU / WL atlas regions."""
    te, ts, tc, zw = g["te_x"], g["tan_sweep"], g["wing_tc"], g["wing_z"]
    ec = g["elevon_c"]
    half = sorted(set(np.round(np.r_[np.linspace(0, 0.5, 46), [0.035, *g["elevon_breaks"]]], 6)))
    ys = [-y for y in reversed(half[1:])] + half
    nf, na = 40, 8

    def section(y):
        le = abs(y) * ts
        c = te - le
        h = 1 - ec / c
        s = np.r_[h * (1 - np.cos(np.linspace(0, math.pi / 2, nf + 1))), np.linspace(h, 1, na + 1)[1:]]
        zt = kg._naca_half(s, tc) * c
        zt[-1] = 0.0
        up = [(le + si * c, y, zw + zi) for si, zi in zip(s, zt)]
        lo = [(le + si * c, y, zw - zi) for si, zi in zip(s[1:-1][::-1], zt[1:-1][::-1])]
        return np.array(up + lo)

    secs = [section(y) for y in ys]
    M = len(secs[0])
    K = nf + na                     # index of the TE point
    verts = np.concatenate(secs)
    faces, uvs = [], []

    def uv_of(j, i, lower):
        x, y = secs[j][i][0], secs[j][i][1]
        c, r = kt.wing_px(g, x, y, lower)
        return _uv(float(c), float(r))
    for j in range(len(ys) - 1):
        for i in range(M):
            i2 = (i + 1) % M
            lower = i >= K
            faces.append((j * M + i, j * M + i2, (j + 1) * M + i2, (j + 1) * M + i))
            uvs.append([uv_of(j, i, lower), uv_of(j, i2, lower), uv_of(j + 1, i2, lower), uv_of(j + 1, i, lower)])
    blank = _uv(4000, 4000)
    faces.append([i for i in range(M)][::-1]); uvs.append([blank] * M)
    faces.append([(len(ys) - 1) * M + i for i in range(M)]); uvs.append([blank] * M)
    return _object("KAL_wing", g, verts, faces, uvs, None, [mats["skin"]], parent=parent)


def _fuselage(g, mats, parent):
    nx, dl = g["nose_x"], g["dome_len"]
    xs = list(nx + dl * (1 - np.cos(np.linspace(0.02, 1, 30) * math.pi / 2)))
    c0, c1 = g["collar_x"], g["collar_x"] + g["collar_step_len"]
    xs += list(np.linspace(nx + dl, g["band0"], 6)) + [g["band0"], g["band1"]]
    xs += list(np.linspace(g["band1"], c0, 3)) + [c0 - 0.0015, c0, c0 + 0.002, c1 - 0.002, c1, c1 + 0.0015]
    xs += list(np.linspace(c1, g["pod0"], 14)) + list(np.linspace(g["pod0"], g["pod1"], 14))
    xs += list(np.linspace(g["pod1"], g["neck_x"], 16)) + list(np.linspace(g["neck_x"], g["eng0"], 6))
    xs = sorted(set(round(float(x), 6) for x in xs if x > nx + 1e-6 and x <= g["eng0"] + 1e-9))
    n = 96
    ang = np.linspace(-math.pi / 2, 3 * math.pi / 2, n, endpoint=False)
    rad = [kg.radius_at(g, x) for x in xs]
    # soften the collar step corners (a 1.5 mm radius look, not a new dimension)
    verts = [(nx, 0.0, 0.0)]
    for x, r in zip(xs, rad):
        verts += [(x, r * math.cos(a), r * math.sin(a)) for a in ang]
    verts = np.array(verts)
    faces, uvs = [], []

    def uv(x, k):
        a = -math.pi / 2 + 2 * math.pi * k / n
        c, r = kt.fus_px(g, x, a)
        return _uv(float(c), float(r))
    for k in range(n):
        k2 = k + 1
        faces.append((0, 1 + (k2 % n), 1 + k))
        uvs.append([uv(nx, k + 0.5), uv(xs[0], k2), uv(xs[0], k)])
    for j in range(len(xs) - 1):
        a0, b0 = 1 + j * n, 1 + (j + 1) * n
        for k in range(n):
            k2 = (k + 1) % n
            faces.append((a0 + k, a0 + k2, b0 + k2, b0 + k))
            uvs.append([uv(xs[j], k), uv(xs[j], k + 1), uv(xs[j + 1], k + 1), uv(xs[j + 1], k)])
    last = 1 + (len(xs) - 1) * n
    faces.append([last + k for k in range(n)])
    uvs.append([_uv(4000, 4000)] * n)
    return _object("KAL_fuselage", g, verts, faces, uvs, None, [mats["skin"]], sharp_deg=35, parent=parent)


def _fin(g, mats, side, parent):
    """Tip fin plate (fin_outline, fin_thickness) with rounded edges via a bevel modifier."""
    poly = kt._fin_outline(g)
    th = g["fin_thickness"]
    y_in, y_out = side * (0.5 - th / 2), side * (0.5 + th / 2)
    verts = [(x, y_out, z) for x, z in poly] + [(x, y_in, z) for x, z in poly]
    n = len(poly)
    out_reg = "FS" if side > 0 else "FP"
    faces, uvs = [], []
    uvo = [_uv(*map(float, kt.fin_px(g, x, z, out_reg))) for x, z in poly]
    uvi = [_uv(*map(float, kt.fin_px(g, x, z, "FI"))) for x, z in poly]
    faces.append(list(range(n))); uvs.append(uvo)
    faces.append([n + i for i in range(n)][::-1]); uvs.append(uvi[::-1])
    for i in range(n):
        i2 = (i + 1) % n
        faces.append((i, n + i, n + i2, i2))
        uvs.append([uvo[i], uvi[i], uvi[i2], uvo[i2]])
    obj = _object(f"KAL_fin_{'stbd' if side > 0 else 'port'}", g, verts, faces, uvs, None, [mats["skin"]],
                  smooth=False, parent=parent)
    bev = obj.modifiers.new("round", "BEVEL")
    bev.width = th * g["span_m"] * 0.45
    bev.segments = 4
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(50)
    bev.harden_normals = False
    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    if hasattr(obj.data, "set_sharp_from_angle"):
        obj.data.set_sharp_from_angle(angle=math.radians(60))
    return obj


def _nose_details(g, mats, parent):
    objs = []
    nx = g["nose_x"]
    # pitot probe: tube with a rounded tip, metres-scale detail from target.yaml (probe_len / probe_dia)
    r = g["probe_dia"] / 2
    L = g["probe_len"]
    n = 20
    xs = [nx - L, nx - L + r * 0.5, nx - L + r * 1.5, nx + 0.012]
    rs = [r * 0.35, r * 0.85, r, r * 1.25]
    verts = [(nx - L - r * 0.3, 0, 0)]
    for x, rr in zip(xs, rs):
        verts += [(x, rr * math.cos(2 * math.pi * k / n), rr * math.sin(2 * math.pi * k / n)) for k in range(n)]
    faces = [(0, 1 + (k + 1) % n, 1 + k) for k in range(n)]
    for j in range(len(xs) - 1):
        faces += [(1 + j * n + k, 1 + j * n + (k + 1) % n, 1 + (j + 1) * n + (k + 1) % n, 1 + (j + 1) * n + k) for k in range(n)]
    faces.append([1 + (len(xs) - 1) * n + k for k in range(n)])
    objs.append(_object("KAL_probe", g, verts, faces, None, None, [mats["probe"]], parent=parent))
    # two blade antennas on the upper nose (P2): swept, tapered, rounded by a bevel
    for k, axn in enumerate(g["antenna_x_from_nose"]):
        x = nx + axn
        c, h = g["antenna_chord"], g["antenna_height"]
        zb = min(kg.radius_at(g, x), kg.radius_at(g, x + c)) - 0.004
        rk = h * math.tan(math.radians(g["antenna_rake_deg"]))
        poly = [(x, zb), (x + c, zb), (x + c + rk * 0.55, zb + h), (x + rk + c * 0.25, zb + h)]
        t = 0.0035
        verts = [(px, -t / 2, pz) for px, pz in poly] + [(px, t / 2, pz) for px, pz in poly]
        faces = [[0, 1, 2, 3], [7, 6, 5, 4]] + [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
        o = _object(f"KAL_antenna_{k}", g, verts, faces, None, None, [mats["antenna"]], smooth=False, parent=parent)
        bev = o.modifiers.new("round", "BEVEL")
        bev.width = t * g["span_m"] * 0.45
        bev.segments = 3
        objs.append(o)
    return objs


def _revolve(profile, n, x_of, r_of):
    """Closed surface of revolution about the internal x axis from (x, r) profile points."""
    verts, faces = [], []
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    for x, r in profile:
        verts += [(x, r * math.cos(a), r * math.sin(a)) for a in ang]
    m = len(profile)
    for j in range(m - 1):
        faces += [(j * n + k, j * n + (k + 1) % n, (j + 1) * n + (k + 1) % n, (j + 1) * n + k) for k in range(n)]
    return verts, faces


def _outward(V, F):
    """Orient a closed part's faces outward (positive signed volume), so parts can be merged."""
    if _signed_volume(np.asarray(V, float), F) < 0:
        F = [list(f)[::-1] for f in F]
    return V, F


def _merge(parts):
    V, F = [], []
    for v, f in parts:
        o = len(V)
        V += list(v)
        F += [[i + o for i in face] for face in f]
    return V, F


def _frustum(p0, p1, r0, r1, n=24, caps=True):
    """Closed frustum between two points (any axis), internal units."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax /= np.linalg.norm(ax)
    ref = np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ax, ref)
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    V = [tuple(c + r * (math.cos(a) * u + math.sin(a) * w)) for c, r in ((p0, r0), (p1, r1)) for a in ang]
    F = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    if caps:
        F += [list(range(n))[::-1], [n + k for k in range(n)]]
    return V, F


def _sweep(points, r, n=12):
    """Closed tube along a polyline (bent pipes, leads), internal units."""
    P = [np.asarray(q, float) for q in points]
    rings = []
    for i, c in enumerate(P):
        t = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)])
        t /= np.linalg.norm(t)
        ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(t, ref); u /= np.linalg.norm(u)
        w = np.cross(t, u)
        rings.append([tuple(c + r * (math.cos(a) * u + math.sin(a) * w)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])
    V = [v for ring in rings for v in ring]
    F = []
    for j in range(len(rings) - 1):
        F += [(j * n + k, j * n + (k + 1) % n, (j + 1) * n + (k + 1) % n, (j + 1) * n + k) for k in range(n)]
    F += [list(range(n))[::-1], [(len(rings) - 1) * n + k for k in range(n)]]
    return V, F


def _rbox(c, size):
    hx, hy, hz = (v / 2 for v in size)
    V = [(c[0] + sx * hx, c[1] + sy * hy, c[2] + sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return V, F


def _rrect_plate(g, xc, zc, y0, y1, dx, dz, corner, n_corner=6):
    """One cooling fin: a closed rounded-rectangle plate in the engine x-z plane between engine y0 < y1
    (metres, engine frame), returned in internal coordinates."""
    r = min(corner, dx / 2, dz / 2)
    pts = []
    for cx, cz, a0 in ((xc + dx / 2 - r, zc + dz / 2 - r, 0), (xc - dx / 2 + r, zc + dz / 2 - r, 90),
                       (xc - dx / 2 + r, zc - dz / 2 + r, 180), (xc + dx / 2 - r, zc - dz / 2 + r, 270)):
        for k in range(n_corner + 1):
            a = math.radians(a0 + 90 * k / n_corner)
            pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    n = len(pts)
    V = [kg.engine_point(g, (px, y, pz)) for y in (y0, y1) for px, pz in pts]
    F = [list(range(n))[::-1], [n + i for i in range(n)]] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return _outward(V, F)


def _engine(g, mats, parent):
    """MD-550 / L550E with the dataset mesh's parts (engine_l550e) at the same places, plus presentation detail
    inside the same envelope: individual cooling fins, spark plugs with boots and leads, carburettor
    bell-mouths, exhaust port flanges and the knurled drive flange. Returns (objects, spinning objects)."""
    s = g["span_m"]

    def P(q):                                       # engine frame (m) -> internal
        return kg.engine_point(g, q)

    grp = {k: [] for k in ("alloy", "black", "heatsink", "carb", "exhaust", "plug", "rubber", "steel")}
    fp, hd, hs, pl, st, cb, ig = el.FIN_PACK, el.HEAD, el.HEATSINK, el.PLUG, el.STUB, el.CARBS, el.IGNITION[0]
    pitch = (fp["y1"] - fp["y0"]) / fp["fins"]
    z_case = el.CASE["h"] / 2
    for k_cyl, (name, side, xc) in enumerate(el.CYLINDERS):
        def Y(y):
            return side * y
        # barrel core, then the fins at the dataset fin pack's pitch and thickness (same outer outline)
        grp["alloy"].append(_frustum(P((xc, Y(fp["y0"]), 0)), P((xc, Y(fp["y1"]), 0)), fp["core_r"] / s, fp["core_r"] / s, 32))
        for k in range(int(fp["fins"])):
            y_a = fp["y0"] + k * pitch + (pitch - fp["fin_t"]) / 2
            lo, hi = sorted((Y(y_a), Y(y_a + fp["fin_t"])))
            grp["alloy"].append(_rrect_plate(g, xc, 0.0, lo, hi, fp["dx"], fp["dz"], fp["corner"]))
        # cylinder head (engine_l550e.HEAD), then Limbach's black heat sink (HEATSINK, as on the datasheet photos):
        # a base plate and thin fins across the crank axis (each in a y-z plane), clipped to the rounded outline and
        # interrupted by the central spark-plug well; then the spark plug: steel hex, white insulator, rubber boot
        lo, hi = sorted((Y(hd["y0"]), Y(hd["y1"])))
        grp["alloy"].append(_rrect_plate(g, xc, 0.0, lo, hi, hd["dx"], hd["dz"], hd["corner"]))
        y_b = hs["y0"] + hs["base_t"]
        lo, hi = sorted((Y(hs["y0"]), Y(y_b)))
        grp["heatsink"].append(_rrect_plate(g, xc, 0.0, lo, hi, hs["dx"], hs["dz"], hs["corner"]))
        y_mid, depth = 0.5 * (y_b + hs["y1"]), hs["y1"] - y_b
        n_hs, rc = int(hs["fins"]), hs["corner"]
        for k in range(n_hs):
            u = -hs["dx"] / 2 + (k + 0.5) * hs["dx"] / n_hs          # fin centre along x, from the block centre
            over = abs(u) - (hs["dx"] / 2 - rc)
            half_h = hs["dz"] / 2 - (rc - math.sqrt(max(0.0, rc ** 2 - over ** 2)) if over > 0 else 0.0)
            spans = [(-half_h, half_h)]
            if abs(u) < hs["plug_well_w"] / 2:                          # spark-plug well in the middle
                spans = [(-half_h, -hs["plug_well_w"] / 2), (hs["plug_well_w"] / 2, half_h)]
            for z0, z1 in spans:
                grp["heatsink"].append(_outward(*_rbox(P((xc + u, Y(y_mid), 0.5 * (z0 + z1))),
                                                       (hs["fin_t"] / s, depth / s, (z1 - z0) / s))))
        y_hex, y_ins = pl["y0"] + 0.007, pl["y0"] + 0.015
        grp["steel"].append(_frustum(P((xc, Y(pl["y0"]), 0)), P((xc, Y(y_hex), 0)), 0.0105 / s, 0.0105 / s, 6))
        grp["plug"].append(_frustum(P((xc, Y(y_hex), 0)), P((xc, Y(y_ins), 0)), 0.006 / s, 0.005 / s, 16))
        grp["rubber"].append(_frustum(P((xc, Y(y_ins), 0)), P((xc, Y(pl["y1"]), 0)), pl["r"] / s, 0.8 * pl["r"] / s, 20))
        # ignition lead: up out of the boot, forward outboard of the fin packs, in to the upper ignition unit
        dz = 0.012 * (k_cyl % 2)                    # rear cylinders' leads run a little higher
        lead = [P((xc, Y(pl["y1"] - 0.004), 0.006)), P((xc, Y(0.188), 0.05 + dz)), P((0.05, Y(0.176), 0.07 + dz)),
                P((0.043, Y(0.06), 0.095)), P((0.043, Y(0.02), 0.098))]
        grp["rubber"].append(_outward(*_sweep(lead, 0.0032 / s, 8)))
        # carburettor on its intake spacer, with a bell-mouth horn
        y = side * cb["y"]
        z1 = z_case + cb["spacer_h"]
        z2 = z1 + cb["h"]
        grp["black"].append(_outward(*_rbox(P((xc, y, z_case + cb["spacer_h"] / 2)),
                                            (0.8 * cb["w"] / s, 0.8 * cb["d"] / s, cb["spacer_h"] / s))))
        grp["carb"].append(_outward(*_rbox(P((xc, y, z1 + cb["h"] / 2)), (cb["w"] / s, cb["d"] / s, cb["h"] / s))))
        grp["carb"].append(_frustum(P((xc, y, z2)), P((xc, y, z2 + 0.7 * cb["horn_h"])), cb["horn_r"] / s, cb["horn_r"] / s, 24))
        grp["carb"].append(_frustum(P((xc, y, z2 + 0.7 * cb["horn_h"])), P((xc, y, z2 + cb["horn_h"])),
                                    cb["horn_r"] / s, 1.2 * cb["horn_r"] / s, 24))
        # exhaust: port flange under the barrel, straight stack to its end (dataset stub and outlet)
        ys_ = side * st["y"]
        grp["steel"].append(_frustum(P((xc, ys_, st["z_flange"] + 0.004)), P((xc, ys_, st["z_flange"] - 0.003)), 0.021 / s, 0.021 / s, 20))
        grp["exhaust"].append(_frustum(P((xc, ys_, st["z_top"])), P((xc, ys_, st["z_end"] + st["outlet_len"])),
                                       st["r"] / s, st["r"] / s, 24))
        grp["exhaust"].append(_frustum(P((xc, ys_, st["z_end"] + st["outlet_len"])), P((xc, ys_, st["z_end"])),
                                       st["outlet_r"] / s, st["outlet_r"] / s, 24))
    # case, covers, mounting face, ignition units (black), rim; the gold drive disc turns with the propeller
    c, fc, mt, fl = el.CASE, el.FRONT_COVER, el.MOUNT, el.FLANGE
    grp["black"].append(_outward(*_rbox(P(((c["x0"] + c["x1"]) / 2, 0, 0)), ((c["x1"] - c["x0"]) / s, c["w"] / s, c["h"] / s))))
    grp["black"].append(_outward(*_rbox(P(((fc["x0"] + fc["x1"]) / 2, 0, 0)), ((fc["x1"] - fc["x0"]) / s, fc["w"] / s, fc["h"] / s))))
    grp["black"].append(_frustum(P((mt["x0"], 0, 0)), P((mt["x1"], 0, 0)), mt["r"] / s, mt["r"] / s, 48))
    for u in el.IGNITION:
        grp["black"].append(_outward(*_rbox(P(((u["x0"] + u["x1"]) / 2, u["y"], (u["z0"] + u["z1"]) / 2)),
                                            ((u["x1"] - u["x0"]) / s, u["w"] / s, (u["z1"] - u["z0"]) / s))))
    grp["black"].append(_frustum(P((fl["x0"] + 0.004, 0, 0)), P((fl["x1"] + 0.004, 0, 0)), fl["rim_r"] / s, fl["rim_r"] / s, 96))
    objs = []
    looks = dict(alloy=("cast_alloy", 50), black=("engine", 40), heatsink=("heatsink", 40), carb=("aluminium", 40),
                 exhaust=("exhaust", 50), plug=("ceramic", 40), rubber=("rubber", 60), steel=("steel", 40))
    for key, parts in grp.items():
        mat, sharp = looks[key]
        o = _object(f"KAL_engine_{key}", g, *_merge(parts), None, None, [mats[mat]], sharp_deg=sharp, parent=parent)
        if key == "black":
            bev = o.modifiers.new("round", "BEVEL")
            bev.width, bev.segments = 0.004, 3
            bev.limit_method, bev.angle_limit = "ANGLE", math.radians(50)
        objs.append(o)
    disc = _frustum(P((fl["x0"], 0, 0)), P((fl["x1"], 0, 0)), fl["r"] / s, fl["r"] / s, 96)
    hub = _frustum(P((0.0, 0, 0)), (g["hub_x"] - g["spinner_len"] / 2, 0.0, 0.0), 0.022 / s, 0.022 / s, 32)
    spin = [_object("KAL_engine_flange", g, *disc, None, None, [mats["flange"]], sharp_deg=40, parent=parent),
            _object("KAL_engine_prop_hub", g, *hub, None, None, [mats["steel"]], sharp_deg=40, parent=parent)]
    return objs + spin, spin


def _prop(g, mats, parent, prop_mode):
    """Spinner, twisted blades and/or the time-averaged disc, grouped under one pivot empty that
    turns about the shaft (output +X)."""
    hub = g["hub_x"]
    s = g["span_m"]
    pivot = bpy.data.objects.new("KAL_prop_pivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.parent = parent
    pivot.location = tuple(kg.internal_to_output(g, [(hub, 0, 0)])[0])
    objs = [pivot]
    to_local = lambda P: kg.internal_to_output(g, P) - kg.internal_to_output(g, [(hub, 0, 0)])[0]  # noqa: E731
    # spinner: an ogive bullet (P4)
    L, R0 = g["spinner_len"], g["spinner_dia"] / 2
    prof = [(hub - L / 2, R0 * 0.98)] + [(hub - L / 2 + L * t, R0 * math.sqrt(max(0.0, 1 - t ** 2)) * (1 - 0.15 * t)) for t in np.linspace(0.05, 1, 14)]
    v, f = _revolve(prof[:-1], 32, None, None)
    v.append((hub + L / 2, 0, 0))
    m = len(prof) - 1
    f += [((m - 1) * 32 + k, (m - 1) * 32 + (k + 1) % 32, len(v) - 1) for k in range(32)]
    f.append([k for k in range(32)][::-1])
    objs.append(_object("KAL_spinner", g, to_local(v), f, None, None, [mats["prop"]], metres=True, parent=pivot))
    Rp = g["prop_dia"] / 2
    r0 = g["spinner_dia"] / 2 * 0.8
    if prop_mode in ("blades", "both"):
        beta75 = math.radians(g["blade_pitch_deg"])
        P = 2 * math.pi * 0.75 * Rp * math.tan(beta75)            # constant geometric pitch
        nr, nc = 18, 14
        rs = r0 + (Rp - r0) * (1 - np.cos(np.linspace(0, math.pi / 2, nr)))
        xc = (1 - np.cos(np.linspace(0, math.pi, nc))) / 2        # chord stations LE->TE
        allv, allf = [], []
        for side in (1, -1):
            base = len(allv)
            for r in rs:
                t = (r - r0) / (Rp - r0)
                chord = g["blade_root_chord"] + (g["blade_tip_chord"] - g["blade_root_chord"]) * t
                chord *= math.sqrt(max(0.05, 1 - max(0.0, (t - 0.82) / 0.18) ** 2))   # rounded tip
                thick = chord * (0.13 - 0.05 * t)
                beta = math.atan(P / (2 * math.pi * max(r, 1e-4)))
                ring = []
                for upper in (True, False):
                    seq = xc if upper else xc[::-1][1:-1]
                    for xx in seq:
                        yt = 5 * thick / chord * (0.2969 * math.sqrt(xx) - 0.126 * xx - 0.3516 * xx ** 2 + 0.2843 * xx ** 3 - 0.1036 * xx ** 4) * chord
                        camber = 0.04 * chord * 4 * xx * (1 - xx)
                        cz = camber + (yt if upper else -yt)
                        cx = (xx - 0.35) * chord
                        # blade frame: radial along +z_int * side; chord in the plane of rotation, tilted by beta
                        tang = cx * math.cos(beta) - cz * math.sin(beta)
                        axial = -(cx * math.sin(beta) + cz * math.cos(beta))
                        ring.append((hub + axial, side * tang, side * r))
                allv += ring
            mlen = 2 * nc - 2
            for j in range(nr - 1):
                for k in range(mlen):
                    k2 = (k + 1) % mlen
                    a, b = base + j * mlen, base + (j + 1) * mlen
                    allf.append((a + k, a + k2, b + k2, b + k))
            allf.append([base + k for k in range(mlen)][::-1])
            allf.append([base + (nr - 1) * mlen + k for k in range(mlen)])
        blades = _object("KAL_prop_blades", g, to_local(allv), allf, None, None, [mats["prop"]], metres=True,
                         parent=pivot)
        objs.append(blades)
    if prop_mode in ("disc", "both"):
        n_r, n_a = 12, 96
        rs = np.linspace(r0 * 1.05, Rp, n_r)
        V, F, U = [], [], []
        for j, r in enumerate(rs):
            for k in range(n_a + 1):
                a = 2 * math.pi * k / n_a
                V.append((hub + 0.002, r * math.cos(a), r * math.sin(a)))
        for j in range(n_r - 1):
            for k in range(n_a):
                a, b = j * (n_a + 1) + k, (j + 1) * (n_a + 1) + k
                F.append((a, a + 1, b + 1, b))
                U.append([((rs[j] - r0) / (Rp - r0), k / n_a), ((rs[j] - r0) / (Rp - r0), (k + 1) / n_a),
                          ((rs[j + 1] - r0) / (Rp - r0), (k + 1) / n_a), ((rs[j + 1] - r0) / (Rp - r0), k / n_a)])
        disc = _object("KAL_prop_disc", g, to_local(V), F, U, None, [mats["prop_disc"]], metres=True,
                       parent=pivot, closed=False, smooth=False)
        objs.append(disc)
    return objs, pivot


# ============================================================================ public entry
def materials(paths, prop_coverage, finish="satin"):
    m = dict(skin=_skin_material(paths, finish),
             aluminium=_machined_aluminium(), cast_alloy=_cast_alloy(), flange=_knurled_gold(),
             steel=_pbr("KAL_hero_steel", (0.08, 0.08, 0.085), 0.45, metal=0.9),
             dark_metal=_pbr("KAL_hero_dark_metal", (0.03, 0.035, 0.04), 0.4, metal=0.7),
             probe=_pbr("KAL_hero_probe", (0.25, 0.25, 0.26), 0.3, metal=1.0),
             antenna=_pbr("KAL_hero_antenna", (0.02, 0.02, 0.022), 0.5, coat=0.2),
             prop=_pbr("KAL_hero_carbon", (0.012, 0.012, 0.013), 0.32, coat=0.6),
             engine=_pbr("KAL_hero_engine", (0.025, 0.025, 0.028), 0.45, metal=0.3),
             heatsink=_pbr("KAL_hero_heatsink", (0.018, 0.018, 0.02), 0.5, metal=0.4),   # black anodised (photo)
             exhaust=_pbr("KAL_hero_exhaust", (0.16, 0.10, 0.07), 0.5, metal=0.8),
             ceramic=_pbr("KAL_hero_ceramic", (0.85, 0.85, 0.82), 0.25),
             rubber=_pbr("KAL_hero_rubber", (0.02, 0.02, 0.022), 0.6))
    m["prop_disc"] = _blur_disc_material("KAL_hero_prop_disc", (0.03, 0.03, 0.032), prop_coverage)
    return m


def build_hero(params: dict | None = None, name: str = "KAL", prop_mode: str = "disc",
               texture_dir: str = TEX_CACHE, look: str = "photo"):
    """Build the presentation model; returns (root empty, handles). The root's frame is the dataset
    model's: metres, +X forward, +Y port, +Z up, origin mid-length on the fuselage axis.

    look="photo" (user requests of 2026-10-03: shape and colour after the P1 image, dimensions from the
    repository): matte charcoal skin, broad fin tips, probe and antennas (P2), P1 lettering and emblems.
    Every dimension (span, chords, fin root chord and heights, fuselage stations, engine, propeller) still
    comes from kal_geometry.derive and engine_l550e; only the fin tip chord, a shape parameter here, is
    widened to PHOTO_FIN_TIP x root chord. look="survey": the dataset model's shapes with a satin skin."""
    if look not in ("photo", "survey"):
        raise ValueError(f"unknown look {look!r}")
    cfg = kg.load_config(kg.DEFAULT_YAML)
    p = {**kg.defaults(cfg), **(params or {})}
    g = kg.derive(p)
    problems = kg.check(g)
    if problems:
        raise ValueError("; ".join(problems))
    if look == "photo":
        g = {**g, "fin_tip_chord": max(g["fin_tip_chord"], PHOTO_FIN_TIP * g["fin_root_chord"])}
    stem = "hero_decal_atlas" if look == "photo" else "decal_atlas"
    js = os.path.join(DECALS, stem + ".json")
    regions = json.load(open(js)) if os.path.exists(js) else {}
    paths = kt.atlas_files(g, texture_dir, os.path.join(DECALS, stem + ".png"), regions, look)
    # blade coverage of the disc per radius: 2 blades x chord / circumference, a bit stronger near
    # the root where the blades are thick
    Rp, r0 = g["prop_dia"] / 2, g["spinner_dia"] / 2 * 0.8
    cov = []
    for u in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0):
        r = r0 + u * (Rp - r0)
        chord = g["blade_root_chord"] + (g["blade_tip_chord"] - g["blade_root_chord"]) * u
        cov.append((u, float(min(0.9, 2 * chord / (2 * math.pi * r) * (2.4 if u < 0.3 else 1.8) * (0.4 if u > 0.97 else 1.0)))))
    mats = materials(paths, cov, "matte" if look == "photo" else "satin")
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    parts = [_wing(g, mats, root), _fuselage(g, mats, root), _fin(g, mats, 1, root), _fin(g, mats, -1, root)]
    parts += _nose_details(g, mats, root)           # probe + two blade antennas (P2 silhouette)
    engine, spinning = _engine(g, mats, root)
    prop, pivot = _prop(g, mats, root, prop_mode)
    root["kal_params"] = json.dumps({k: v for k, v in p.items()}, default=float)
    root["kal_summary"] = json.dumps(kg.summary(g))
    root["kal_model"] = f"presentation (kal_hero.py, look={look}); dimensions from kal_geometry.derive"
    # the drive flange and shaft turn with the propeller: put them on a frame about the shaft (output +X)
    flange_axis = bpy.data.objects.new("KAL_flange_pivot", None)
    bpy.context.scene.collection.objects.link(flange_axis)
    flange_axis.parent = root
    for o in spinning:
        o.parent = flange_axis
    return root, dict(prop_pivot=pivot, flange_pivot=flange_axis, derived=g, parts=parts + engine + prop)


def set_spin(handles, angle_rad: float):
    """Rotate the propeller and the drive flange about the shaft (output +X)."""
    handles["prop_pivot"].rotation_euler = (angle_rad, 0.0, 0.0)
    handles["flange_pivot"].rotation_euler = (angle_rad, 0.0, 0.0)
