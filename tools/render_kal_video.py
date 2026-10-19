"""Eevee renders of the KAL presentation model for the detection video (Blender 4.2).

    set KAL_CONFIG_OVERLAY=configs/m1.yaml
    blender -b -P tools/render_kal_video.py -- --sensor out/video/sensor --out out/video/hero --shot zoom
    blender -b -P tools/render_kal_video.py -- --sensor out/video/sensor --out out/video/hero --shot intro

Shots
  zoom   the sensor site's EO zoom (visual-confirmation) camera, co-located with the S1 sensor of
         render/track_sequence.py: the same flight path, airframe variant, sun, sky and haze. It
         stays centred on the aircraft with small jitter (a rate-aided tracker) and zooms so the
         airframe fills about 40 % of the width (field of view never below 0.4 deg). Pointing is
         scripted from the flight path.
  intro  cinematic chase / orbit over a desert range in the same light (presentation only)
  flyby  ground observer at the runway edge as the aircraft passes (presentation only)

Everything aircraft-related comes from target/kal_hero.py (dimensions from kal_geometry; default
look="photo"). Flight: level, 54 m/s (194 km/h: inside the user's 180-220 km/h request and the
protected 39-56 m/s range), bank only from the coordinated S-turn. The propeller is shown as the
blur disc a normal video shutter records; its speed is the nominal power setting's (5/8 of rated power:
7,500 x 0.625^(1/3) = 6,400 rpm, target/engine_l550e.rpm_at_power_fraction).
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import engine_l550e as el  # noqa: E402
import kal_hero  # noqa: E402
from data import composite as cp  # noqa: E402
from render import placement as pl  # noqa: E402
from render import sampling  # noqa: E402
from render import scene as rs  # noqa: E402
from render import sequence as sq  # noqa: E402

RPM = round(el.rpm_at_power_fraction(0.625), -2)          # 6,400 rpm, nominal power setting
# Video airspeed: the user asked for 180-220 km/h; 54 m/s (194 km/h) also stays inside the protected
# 39-56 m/s range of configs/render.yaml, which is not changed.
SPEED = 54.0


def log(*a):
    print(*a, flush=True)


# ============================================================================ scene setup
def base_scene(plan):
    cfgs = sampling.load_configs()
    args = argparse.Namespace(start=plan["args"]["start"], lighting=plan["args"]["lighting"], cam_el=plan["args"]["cam_el"])
    idx, spec, objs = sq.find_scene(cfgs, args)
    if idx != plan["base_scene"]:
        raise RuntimeError(f"scene sampler drifted: {idx} != {plan['base_scene']}")
    return cfgs, spec, rs.calibrate(spec, cfgs)


def setup_render(res, samples, fps, shutter_s):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.eevee.taa_render_samples = samples
    sc.eevee.use_shadows = True
    sc.eevee.shadow_ray_count = 2
    sc.eevee.shadow_step_count = 6
    if hasattr(sc.eevee, "use_raytracing"):
        sc.eevee.use_raytracing = True
        sc.eevee.ray_tracing_method = "SCREEN"
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.fps = int(round(fps))
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 15
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = shutter_s * fps          # fraction of a frame
    if hasattr(sc.eevee, "motion_blur_steps"):
        sc.eevee.motion_blur_steps = 1
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.film_transparent = False
    return sc


def setup_world(spec, calib, exposure):
    """Exactly the dataset's RGB daylight: Nishita sky (strength 1, the scene's atmosphere) and a
    sun lamp with the calibrated irradiance (render/rgb.py world() / lamps())."""
    sc = bpy.context.scene
    from render import bpy_util as bu
    from render import rgb
    w = rgb.world(spec, calib)
    sc.world = w
    lamps = rgb.lamps(spec, calib)
    for lp in lamps:
        lp.data.use_shadow = True
        lp.data.angle = math.radians(0.53)
    sc.view_settings.exposure = exposure
    return lamps


def haze_group():
    """Aerial perspective for a distant object: out = t * surface + (1 - t) * sky (emission)."""
    g = bpy.data.node_groups.get("KAL_haze")
    if g is not None:
        return g
    g = bpy.data.node_groups.new("KAL_haze", "ShaderNodeTree")
    g.interface.new_socket("Shader", in_out="INPUT", socket_type="NodeSocketShader")
    g.interface.new_socket("Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    gi, go = g.nodes.new("NodeGroupInput"), g.nodes.new("NodeGroupOutput")
    fac = g.nodes.new("ShaderNodeValue"); fac.name = "haze"; fac.outputs[0].default_value = 0.0
    col = g.nodes.new("ShaderNodeRGB"); col.name = "sky"; col.outputs[0].default_value = (0.5, 0.6, 0.8, 1)
    em = g.nodes.new("ShaderNodeEmission")
    g.links.new(col.outputs[0], em.inputs["Color"])
    mix = g.nodes.new("ShaderNodeMixShader")
    g.links.new(fac.outputs[0], mix.inputs[0])
    g.links.new(gi.outputs[0], mix.inputs[1])
    g.links.new(em.outputs[0], mix.inputs[2])
    g.links.new(mix.outputs[0], go.inputs[0])
    return g


def add_haze(materials):
    g = haze_group()
    for m in materials:
        if m is None or not m.use_nodes or m.blend_method == "BLEND":
            continue
        nt = m.node_tree
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        if not out.inputs["Surface"].links:
            continue
        src = out.inputs["Surface"].links[0].from_socket
        node = nt.nodes.new("ShaderNodeGroup")
        node.node_tree = g
        nt.links.new(src, node.inputs[0])
        nt.links.new(node.outputs[0], out.inputs["Surface"])
    return g


def camera(name, lens_mm=50.0, sensor=36.0):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = sensor
    cd.lens = lens_mm
    cd.clip_start, cd.clip_end = 0.5, 60000.0
    obj = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj
    return obj


def look_rot(src, dst, roll=0.0):
    d = Vector(dst) - Vector(src)
    q = d.to_track_quat("-Z", "Y")
    if roll:
        q = q @ Matrix.Rotation(roll, 4, "Z").to_quaternion()
    return q.to_euler()


def _calm_skin():
    """Video-only finish adjustment (target/kal_hero.py is left as is): the photo look's centimetre
    albedo mottling (x0.70-1.85) and grain bump read as stone at close range; a composite airframe
    under matte paint varies far less. Keeps the matte roughness and the seams."""
    m = bpy.data.materials.get("KAL_hero_skin")
    if m is None or not m.use_nodes:
        return
    for n in m.node_tree.nodes:
        if n.type == "MAP_RANGE" and abs(n.inputs["To Max"].default_value - 1.85) < 1e-6:
            n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = 0.78, 0.96
        elif n.type == "BUMP":
            n.inputs["Strength"].default_value = 0.05


def kal(params, mode="disc"):
    """Presentation model (default look="photo"). "disc": the spinning propeller as a camera records
    it with a normal shutter (a translucent blur disc); frozen blades read as animation."""
    root, hd = kal_hero.build_hero(params, prop_mode=mode)
    _calm_skin()
    root.rotation_mode = "QUATERNION"
    for p in (hd["prop_pivot"], hd["flange_pivot"]):
        p.rotation_mode = "XYZ"
    return root, hd


def key_kal(root, hd, frame, matrix, spin):
    M = Matrix([list(r) for r in matrix])
    loc, rot, _ = M.decompose()
    root.location, root.rotation_quaternion = loc, rot
    root.keyframe_insert("location", frame=frame)
    root.keyframe_insert("rotation_quaternion", frame=frame)
    for p in (hd["prop_pivot"], hd["flange_pivot"]):
        p.rotation_euler = (spin, 0, 0)
        p.keyframe_insert("rotation_euler", frame=frame)


def linear_keys(*objs):
    for o in objs:
        ad = o.animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


def spin_at(t):
    """Propeller and drive-flange angle (rad) at time t; direction: clockwise seen from behind (assumed)."""
    return -2 * math.pi * RPM / 60.0 * t


# ============================================================================ zoom shot
def pano_sampler(sensor_dir, plan):
    layer, _ = cp.load_layers(os.path.join(sensor_dir, "pano_rgb.npz"))
    sky = layer["sky"] if layer["sky"].ndim == 3 else layer["sky"][..., None]
    P = plan["panorama"]
    cam = pl.Camera(P["width"], P["height"], P["hfov_deg"], plan["camera"]["position"], P["azimuth_deg"], P["elevation_deg"])

    def at(direction):
        u, v, z = cam.project(np.asarray(cam.C) + np.asarray(direction)[None] * 1000.0)
        x = float(np.clip(u[0] - 0.5, 0, cam.W - 1.001))
        y = float(np.clip(v[0] - 0.5, 0, cam.H - 1.001))
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        s = (sky[y0, x0] * (1 - fx) * (1 - fy) + sky[y0, x0 + 1] * fx * (1 - fy)
             + sky[y0 + 1, x0] * (1 - fx) * fy + sky[y0 + 1, x0 + 1] * fx * fy)
        return np.asarray(s, float)
    return at


def shot_zoom(a, plan, spec, calib):
    frames = plan["frames"]
    fps = plan["fps"]
    span = plan["variant"]["span_m"]
    C = np.asarray(plan["camera"]["position"], float)
    sc = setup_render((a.width, a.height), a.samples, fps, 0.002)
    setup_world(spec, calib, a.exposure)
    root, hd = kal(plan["variant"]["params"], "disc")
    mats = {s.material for o in bpy.data.objects if o.type == "MESH" for s in o.material_slots}
    g = add_haze(mats)
    w = plan["weather"]
    kext = cp.extinction(w["visibility_km"], w["band_ratio"]["rgb"])
    sky_at = pano_sampler(a.sensor, plan)
    cam = camera("zoom_cam")
    cam.rotation_mode = "XYZ"
    cam.location = Vector(C)
    rng = np.random.default_rng(5)
    jit = np.zeros(2)
    zplan = []
    for k, f in enumerate(frames):
        t = f["t"]
        M = np.asarray(f["target_matrix"])
        pos = M[:3, 3]
        key_kal(root, hd, k + 1, M, spin_at(t))
        # rate-aided EO tracker: centred on the aircraft with small residual jitter (a plain
        # first-order lag would trail a crossing target by a third of this narrow field of view)
        cur = np.array(pl.azimuth_elevation(pos - C))
        jit = 0.85 * jit + 0.15 * rng.normal(0, 0.006, 2)
        az, el = float(cur[0] + jit[0]), float(cur[1] + jit[1])
        rng_m = float(np.linalg.norm(pos - C))
        hfov = max(0.40, math.degrees(2 * math.atan(span / (2 * 0.40 * rng_m))))
        cam.data.lens = pl.focal_mm(36.0, hfov)
        cam.data.keyframe_insert("lens", frame=k + 1)
        Rc = pl.camera_rotation(az, el, 0.0)
        cam.rotation_euler = Matrix([list(r) for r in Rc]).to_euler()
        cam.keyframe_insert("rotation_euler", frame=k + 1)
        tr = float(cp.transmittance(rng_m, kext, f["elevation_deg"], w.get("aerosol_scale_height_m")))
        sky = sky_at(pl.direction(az, el))
        g.nodes["haze"].outputs[0].default_value = 1.0 - tr
        g.nodes["haze"].outputs[0].keyframe_insert("default_value", frame=k + 1)
        g.nodes["sky"].outputs[0].default_value = (*[float(v) for v in sky[:3]], 1.0)
        g.nodes["sky"].outputs[0].keyframe_insert("default_value", frame=k + 1)
        zplan.append(dict(k=k, t=t, az=az, el=el, hfov=hfov, width=a.width, height=a.height, range_m=rng_m,
                          transmittance=tr))
    linear_keys(root, hd["prop_pivot"], hd["flange_pivot"], cam)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "zoom_plan.json"), "w") as fh:
        json.dump(dict(frames=zplan, camera_position=C.tolist(), rpm_assumed=RPM, shutter_s=0.001), fh, indent=1)
    render_range(sc, os.path.join(a.out, "zoom"), len(frames), a)


def render_range(sc, out_dir, n, a):
    os.makedirs(out_dir, exist_ok=True)
    first = a.first if a.first is not None else 0
    last = min(n, a.last) if a.last is not None else n
    step = a.step
    for k in range(first, last, step):
        path = os.path.join(out_dir, f"{k:04d}.png")
        if os.path.exists(path) and not a.overwrite:
            continue
        sc.frame_set(k + 1)
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        log(f"VIDEO_FRAME {os.path.basename(out_dir)} {k + 1}/{n}")


# ============================================================================ desert environment
def desert(seed=11, size=6000.0, n=220):
    """Thar-like range (P6, P5): flat sand with scrub, a runway, gentle undulation. Presentation."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs)
    Z = np.zeros_like(X)
    for k in range(6):
        fx, fy, ph = rng.uniform(0.0004, 0.004, 2).tolist() + [rng.uniform(0, 6.28)]
        Z += rng.uniform(0.4, 2.2) / (1 + k) * np.sin(X * fx * 6.28 + ph) * np.cos(Y * fy * 6.28 + 1.3 * ph)
    Z *= np.clip(1 - np.exp(-((np.abs(X) + np.abs(Y)) / 900.0)), 0, 1)
    ax = np.clip((np.abs(X) - 40.0) / 80.0, 0, 1)
    Z *= ax * ax * (3 - 2 * ax)                                              # level strip under the runway
    Z -= 0.3 * (1 - ax)
    verts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(n - 1) for i in range(n - 1)]
    me = bpy.data.meshes.new("desert")
    me.from_pydata(verts.tolist(), [], faces)
    me.polygons.foreach_set("use_smooth", [True] * len(faces))
    obj = bpy.data.objects.new("desert", me)
    bpy.context.scene.collection.objects.link(obj)
    far = bpy.data.meshes.new("desert_far")
    S = 60000.0
    far.from_pydata([(-S, -S, -0.5), (S, -S, -0.5), (S, S, -0.5), (-S, S, -0.5)], [], [(0, 1, 2, 3)])
    fobj = bpy.data.objects.new("desert_far", far)
    bpy.context.scene.collection.objects.link(fobj)
    mat = sand_material()
    obj.data.materials.append(mat)
    fobj.data.materials.append(mat)
    runway()
    scrub(rng, size * 0.45, 6000)
    return obj


def sand_material():
    m = bpy.data.materials.new("sand")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord")
    n1 = N.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 0.004; n1.inputs["Detail"].default_value = 6
    n2 = N.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 0.08; n2.inputs["Detail"].default_value = 4
    vor = N.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 0.35
    for t in (n1, n2, vor):
        L.new(tc.outputs["Object"], t.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.30, (0.20, 0.135, 0.075, 1)
    e[1].position, e[1].color = 0.75, (0.36, 0.26, 0.15, 1)
    L.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    # scrub speckle: Voronoi cells with a small dark core
    sp = N.new("ShaderNodeMath"); sp.operation = "LESS_THAN"; sp.inputs[1].default_value = 0.09
    L.new(vor.outputs["Distance"], sp.inputs[0])
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"
    L.new(sp.outputs[0], mix.inputs["Factor"])
    L.new(ramp.outputs["Color"], mix.inputs["A"])
    mix.inputs["B"].default_value = (0.045, 0.045, 0.025, 1)
    mul = N.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 0.35
    L.new(mix.outputs["Result"], mul.inputs["A"])
    L.new(n2.outputs["Color"], mul.inputs["B"])
    L.new(mul.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.95
    bump = N.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25
    L.new(n2.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def runway():
    """45 m x 3 km asphalt strip with centre-line dashes and edge lines (P6). Along +Y."""
    me = bpy.data.meshes.new("runway")
    W, Lr = 45.0, 3000.0
    me.from_pydata([(-W / 2, -Lr / 2, 0.15), (W / 2, -Lr / 2, 0.15), (W / 2, Lr / 2, 0.15), (-W / 2, Lr / 2, 0.15)], [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new("runway", me)
    bpy.context.scene.collection.objects.link(obj)
    m = bpy.data.materials.new("asphalt")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    ax = N.new("ShaderNodeMath"); ax.operation = "ABSOLUTE"; L.new(sep.outputs["X"], ax.inputs[0])
    centre = N.new("ShaderNodeMath"); centre.operation = "LESS_THAN"; centre.inputs[1].default_value = 0.45
    L.new(ax.outputs[0], centre.inputs[0])
    dash = N.new("ShaderNodeMath"); dash.operation = "PINGPONG"; dash.inputs[1].default_value = 30.0
    L.new(sep.outputs["Y"], dash.inputs[0])
    dsh = N.new("ShaderNodeMath"); dsh.operation = "LESS_THAN"; dsh.inputs[1].default_value = 15.0
    L.new(dash.outputs[0], dsh.inputs[0])
    cl = N.new("ShaderNodeMath"); cl.operation = "MULTIPLY"
    L.new(centre.outputs[0], cl.inputs[0]); L.new(dsh.outputs[0], cl.inputs[1])
    edge = N.new("ShaderNodeMath"); edge.operation = "GREATER_THAN"; edge.inputs[1].default_value = 21.0
    L.new(ax.outputs[0], edge.inputs[0])
    paint = N.new("ShaderNodeMath"); paint.operation = "MAXIMUM"
    L.new(cl.outputs[0], paint.inputs[0]); L.new(edge.outputs[0], paint.inputs[1])
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.6; L.new(tc.outputs["Object"], nz.inputs["Vector"])
    base = N.new("ShaderNodeMix"); base.data_type = "RGBA"
    L.new(nz.outputs["Fac"], base.inputs["Factor"])
    base.inputs["A"].default_value = (0.025, 0.024, 0.023, 1)
    base.inputs["B"].default_value = (0.055, 0.052, 0.048, 1)
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"
    L.new(paint.outputs[0], mix.inputs["Factor"])
    L.new(base.outputs["Result"], mix.inputs["A"])
    mix.inputs["B"].default_value = (0.55, 0.55, 0.52, 1)
    L.new(mix.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85
    obj.data.materials.append(m)
    return obj


def scrub(rng, extent, count):
    """Instanced low bushes (khejri / ker-like scrub), 0.6-2.5 m, scattered with clumping."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    proto = bpy.context.active_object
    proto.name = "bush"
    for v in proto.data.vertices:
        v.co.z = max(v.co.z, -0.2) * 0.7
        v.co *= 1 + 0.25 * (np.sin(v.co.x * 7) * np.cos(v.co.y * 5))
    m = bpy.data.materials.new("bush")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.05, 0.055, 0.025, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    proto.data.materials.append(m)
    centres = rng.uniform(-extent, extent, (count // 6, 2))
    pts = (centres[rng.integers(len(centres), size=count)] + rng.normal(0, 60, (count, 2)))
    pts = pts[np.abs(pts[:, 0]) > 40]                      # keep the runway clear
    me = bpy.data.meshes.new("scrub_points")
    me.from_pydata([(float(x), float(y), 0.0) for x, y in pts], [], [])
    holder = bpy.data.objects.new("scrub", me)
    bpy.context.scene.collection.objects.link(holder)
    mod = holder.modifiers.new("inst", "NODES")
    ng = bpy.data.node_groups.new("scrub_nodes", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    obi = ng.nodes.new("GeometryNodeObjectInfo"); obi.inputs[0].default_value = proto
    rnd = ng.nodes.new("FunctionNodeRandomValue"); rnd.data_type = "FLOAT_VECTOR"
    rnd.inputs["Min"].default_value = (0.6, 0.6, 0.5); rnd.inputs["Max"].default_value = (2.4, 2.4, 1.6)
    ng.links.new(gi.outputs[0], inst.inputs["Points"])
    ng.links.new(obi.outputs["Geometry"], inst.inputs["Instance"])
    ng.links.new(rnd.outputs["Value"], inst.inputs["Scale"])
    ng.links.new(inst.outputs[0], go.inputs[0])
    mod.node_group = ng
    proto.hide_render = True
    proto.hide_viewport = True
    return holder


def clouds(height=2600.0):
    """A broken altocumulus deck far above the action (presentation background)."""
    me = bpy.data.meshes.new("clouds")
    S = 50000.0
    me.from_pydata([(-S, -S, height), (S, -S, height), (S, S, height), (-S, S, height)], [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new("clouds", me)
    bpy.context.scene.collection.objects.link(obj)
    m = bpy.data.materials.new("cloud")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    for n in list(N):
        N.remove(n)
    out = N.new("ShaderNodeOutputMaterial")
    tc = N.new("ShaderNodeTexCoord")
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.00042; nz.inputs["Detail"].default_value = 8
    nz.inputs["Roughness"].default_value = 0.62
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.52, (0, 0, 0, 1)
    e[1].position, e[1].color = 0.68, (1, 1, 1, 1)
    L.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    tr = N.new("ShaderNodeBsdfTransparent")
    em = N.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (1.0, 0.92, 0.82, 1); em.inputs["Strength"].default_value = 6.0
    tl = N.new("ShaderNodeBsdfTranslucent"); tl.inputs["Color"].default_value = (0.9, 0.9, 0.95, 1)
    add = N.new("ShaderNodeAddShader")
    L.new(em.outputs[0], add.inputs[0]); L.new(tl.outputs[0], add.inputs[1])
    mix = N.new("ShaderNodeMixShader")
    L.new(ramp.outputs["Color"], mix.inputs[0])
    L.new(tr.outputs[0], mix.inputs[1]); L.new(add.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    m.blend_method = "BLEND"
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    obj.data.materials.append(m)
    obj.visible_shadow = False
    return obj


def horizon_sky(sensor_dir, plan):
    """Sky radiance 3 deg above the horizon in the scene's own RGB sky (sensor panorama), used as
    the aerial-perspective colour for distant terrain."""
    at = pano_sampler(sensor_dir, plan)
    return [float(v) for v in at(pl.direction(plan["panorama"]["azimuth_deg"], 3.0))[:3]]


def mist_compositing(sc, sky_rgb, start=300.0, depth=14000.0):
    """Aerial perspective for the terrain: mix towards the horizon sky colour with the Mist pass."""
    sc.view_layers[0].use_pass_mist = True
    sc.world.mist_settings.start = start
    sc.world.mist_settings.depth = depth
    sc.world.mist_settings.falloff = "QUADRATIC"
    sc.use_nodes = True
    nt = sc.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    mix = nt.nodes.new("CompositorNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[2].default_value = (*sky_rgb, 1.0)
    pw = nt.nodes.new("CompositorNodeMath"); pw.operation = "MULTIPLY"; pw.inputs[1].default_value = 0.85
    nt.links.new(rl.outputs["Mist"], pw.inputs[0])
    nt.links.new(pw.outputs[0], mix.inputs[0])
    nt.links.new(rl.outputs["Image"], mix.inputs[1])
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(mix.outputs[0], comp.inputs["Image"])


# ============================================================================ cinematic shots
def flight(n, fps, start, heading_deg, alt, weave_deg=8.0, weave_s=14.0, speed=SPEED):
    """Level S-turn with coordinated bank at the video airspeed; returns per-frame (matrix, t, heading, bank)."""
    out = []
    p = np.array([start[0], start[1], alt], float)
    dt = 1.0 / fps
    for k in range(n):
        t = k * dt
        h = heading_deg + weave_deg * math.sin(2 * math.pi * t / weave_s)
        rate = math.radians(weave_deg) * 2 * math.pi / weave_s * math.cos(2 * math.pi * t / weave_s)
        bank = math.degrees(math.atan(speed * rate / 9.81))
        R = pl.object_rotation(h, 2.0, bank)
        out.append((pl.matrix_world(R, p), t, h, bank))
        for s in range(8):
            hs = heading_deg + weave_deg * math.sin(2 * math.pi * (t + (s + 0.5) * dt / 8) / weave_s)
            p = p + speed * pl.direction(hs, 0.0) * dt / 8
    return out


def shot_intro(a, plan, spec, calib):
    """Air-to-air chase on the sunlit side, camera near the aircraft's height so the background is
    sky and hazy distant desert (close-up sand reads as CG). The aircraft banks gently towards the
    camera, showing its upper surface. 1/200 s shutter, shallow depth of field, slight camera drift."""
    fps = plan["fps"]
    n = int(a.seconds * fps)
    sc = setup_render((a.width, a.height), a.samples, fps, 0.005)
    setup_world(spec, calib, a.exposure)
    sun_az = spec["sun"]["azimuth_deg"]
    desert()
    root, hd = kal(plan["variant"]["params"], "disc")
    # heading 110 deg clockwise from the sun: the sun is on the aircraft's left (camera) side
    path = flight(n, fps, (-350.0, -200.0), (sun_az + 110.0) % 360.0, 160.0, weave_deg=-10.0, weave_s=24.0)
    cam = camera("chase", 85.0)
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 2.0
    rng = np.random.default_rng(9)
    drift = np.zeros(3)
    for k, (M, t, h, bank) in enumerate(path):
        key_kal(root, hd, k + 1, M, spin_at(t))
        pos = M[:3, 3]
        fwd = pl.direction(h, 0.0)
        right = np.array([fwd[1], -fwd[0], 0.0])
        u = k / max(1, n - 1)
        # rear-left quarter -> left beam, on the sunlit side
        ang = math.radians(145 - 70 * (u * u * (3 - 2 * u)))
        dist = 17.0 - 2.5 * u
        drift = 0.92 * drift + 0.08 * rng.normal(0, 0.15, 3)
        off = dist * (math.cos(ang) * fwd + math.sin(ang) * (-right)) + np.array([0, 0, 0.8]) + drift
        cpos = pos + off
        cam.location = Vector(cpos)
        cam.rotation_euler = look_rot(cpos, pos + 0.25 * fwd + 0.15 * drift)
        cam.data.dof.focus_distance = float(np.linalg.norm(off))
        for obj, prop in ((cam, "location"), (cam, "rotation_euler")):
            obj.keyframe_insert(prop, frame=k + 1)
        cam.data.dof.keyframe_insert("focus_distance", frame=k + 1)
    linear_keys(root, hd["prop_pivot"], hd["flange_pivot"], cam)
    mist_compositing(sc, tuple(horizon_sky(a.sensor, plan)), start=150.0, depth=9000.0)
    render_range(sc, os.path.join(a.out, "intro"), n, a)


def shot_flyby(a, plan, spec, calib):
    fps = plan["fps"]
    n = int(a.seconds * fps)
    sc = setup_render((a.width, a.height), a.samples, fps, 0.002)
    setup_world(spec, calib, a.exposure)
    sun_az = spec["sun"]["azimuth_deg"]
    desert()
    clouds()
    root, hd = kal(plan["variant"]["params"], "blades")
    head = (sun_az + 200.0) % 360.0
    fwd = pl.direction(head, 0.0)
    start = -fwd[:2] * 44.0 * a.seconds * 0.55 + np.array([25.0, 0.0])
    path = flight(n, fps, tuple(start), head, 70.0, weave_deg=0.0)
    cpos = np.array([60.0, 0.0, 1.7])
    cam = camera("observer", 35.0)
    rng = np.random.default_rng(3)
    shake = np.zeros(3)
    for k, (M, t, h, bank) in enumerate(path):
        key_kal(root, hd, k + 1, M, spin_at(t))
        pos = M[:3, 3]
        shake = 0.9 * shake + 0.1 * rng.normal(0, 0.03, 3)
        cam.location = Vector(cpos + shake * 0.05)
        cam.rotation_euler = look_rot(cpos, pos + np.array([0, 0, 2.0]) + shake * 4.0)
        cam.keyframe_insert("location", frame=k + 1)
        cam.keyframe_insert("rotation_euler", frame=k + 1)
    linear_keys(root, hd["prop_pivot"], hd["flange_pivot"], cam)
    mist_compositing(sc, tuple(horizon_sky(a.sensor, plan)))
    render_range(sc, os.path.join(a.out, "flyby"), n, a)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", default="out/video/sensor")
    ap.add_argument("--out", default="out/video/hero")
    ap.add_argument("--shot", choices=("zoom", "intro", "flyby"), required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--exposure", type=float, default=-3.3)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    a.sensor, a.out = os.path.abspath(a.sensor), os.path.abspath(a.out)
    plan = json.load(open(os.path.join(a.sensor, "plan.json")))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cfgs, spec, calib = base_scene(plan)
    bpy.ops.wm.read_factory_settings(use_empty=True)      # calibrate() leaves a probe scene behind
    {"zoom": shot_zoom, "intro": shot_intro, "flyby": shot_flyby}[a.shot](a, plan, spec, calib)
    log("VIDEO_SHOT_DONE", a.shot)


main()
