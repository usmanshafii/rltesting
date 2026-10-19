"""Air-to-air style 360-degree video of the KAL presentation model (Blender 4.2 Eevee + ffmpeg).

    python tools/kal_orbit_video.py                       # render, finish, encode (resumable)
    python tools/kal_orbit_video.py --frames 0,40,84,126  # test stills only (raw + finished)
    python tools/kal_orbit_video.py --finish-only         # redo the camera finish and encoding

KAL (target/kal_hero.py, look="photo") flies straight and level at the protected default airspeed of
44 m/s, about 420 m above a Thar-like desert, with small turbulence-like attitude motion. A chase
camera orbits it once while moving from above the wing to below it, so the clip shows the front,
both sides, the top, the rear and the underside. The aim is real footage rather than animation:
physical sky (Nishita) and sun, sky-coloured aerial haze, a cumulus deck whose shadows fall on the
ground, a 180-degree shutter (motion blur), depth of field, operator lag and handheld shake, a
time-averaged propeller disc, then bloom, slight chromatic aberration, vignetting, sensor grain and
in-camera sharpening (finish()). No captions.

Presentation only: the terrain, clouds, altitude and camera path are not a sampled flight condition
and nothing here feeds the dataset. Frames go to out/video/orbit/tmp/ (ignored by git); the clip to
out/video/KAL_360_flight.mp4.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out", "video", "orbit")
RAW = os.path.join(OUT, "tmp", "raw")
FIN = os.path.join(OUT, "tmp", "finished")
CLIP = os.path.join(ROOT, "out", "video", "KAL_360_flight.mp4")
STILLS = os.path.join(ROOT, "out", "proposal")
BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"

SPEED = 44.0            # m/s, configs/render.yaml airspeed default (protected)
AGL = 420.0             # m, presentation altitude above the terrain plane
SUN_EL, SUN_AZ = 36.0, 20.0    # deg; Nishita sun_rotation: sun to the east-north-east (KAL heads +Y),
                                # so the low view from below looks away from the sun into blue sky
CLOUD_AGL = 1300.0      # m, fair-weather cumulus bases (~880 m above KAL)
HAZE_M = 7400.0         # m, extinction length (3.912 / 29 km visibility of the M1 scene ~ 1 / 7.4 km)
SKY_E = 0.30            # skylight on the ground as a fraction of the sun's normal irradiance (clear day)
SKY = 0.25              # world strength of the Nishita sky (haze uses the same radiance scale)
LENS = 50.0             # mm, full-frame
FSTOP = 4.0


def args_from(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--frames", default="", help="comma-separated frame indices (test stills)")
    ap.add_argument("--finish-only", action="store_true")
    ap.add_argument("--still", default="", help='proposal stills, "name:az,el,range;..." (deg, deg, m)')
    ap.add_argument("--still-lens", type=float, default=70.0)
    ap.add_argument("--render", action="store_true", help=argparse.SUPPRESS)   # inside Blender
    return ap.parse_args(argv)


# ============================================================================ motion (numpy only)
def _wobble(t, seed, amps, freqs):
    """Smooth pseudo-random signal: a few sines with seeded phases."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return sum(a * math.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi)) for a, f in zip(amps, freqs))


def aircraft_state(t):
    """Position (m) and attitude perturbation (roll, pitch, yaw in deg) at time t. KAL heads +Y."""
    pos = (0.12 * _wobble(t, 1, (1, 0.6), (0.21, 0.47)), SPEED * t, AGL + 0.15 * _wobble(t, 2, (1, 0.5), (0.33, 0.9)))
    att = (_wobble(t, 3, (0.7, 0.35, 0.12), (0.27, 0.71, 1.9)),
           _wobble(t, 4, (0.25, 0.12), (0.4, 1.3)),
           _wobble(t, 5, (0.2, 0.08), (0.23, 0.8)))
    return pos, att


def camera_state(u, t):
    """Orbit around the aircraft's mean path: azimuth from the nose towards the port side, elevation
    above the wing plane, range (m); plus operator lag (aim offset, m) and handheld shake (deg)."""
    az = 35.0 + 360.0 * (u - 0.045 * math.sin(2 * math.pi * u))
    el = 9.0 + 49.0 * math.sin(2 * math.pi * u + 0.33)
    rng_m = 10.0 + 1.0 * math.cos(4 * math.pi * u) - 0.6 * math.sin(2 * math.pi * u)
    aim = (0.22 * _wobble(t, 11, (1, 0.4), (0.17, 0.43)), 0.30 * _wobble(t, 12, (1, 0.5), (0.13, 0.37)),
           0.18 * _wobble(t, 13, (1, 0.5), (0.19, 0.51)))
    shake = (_wobble(t, 21, (0.22, 0.05, 0.02), (0.6, 2.3, 8.7)), _wobble(t, 22, (0.18, 0.05, 0.02), (0.5, 2.9, 10.3)),
             _wobble(t, 23, (0.25, 0.04), (0.35, 3.1)))
    return az, el, rng_m, aim, shake


def parse_stills(spec):
    """'name:az,el,range;...' -> [(name, az, el, range)]"""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        name, vals = part.split(":")
        az, el, rr = (float(v) for v in vals.split(","))
        out.append((name, az, el, rr))
    return out


def still_frame(i, fps):
    """Still i is taken at t = 1 + 2 i seconds (frame index), each pose held for 2 s around it."""
    return int(round((1.0 + 2.0 * i) * fps))


def still_pose(stills, t):
    """Fixed camera pose relative to the aircraft (no orbit, lag or shake): a steady formation shot."""
    _, az, el, rr = stills[min(int(t // 2.0), len(stills) - 1)]
    return az, el, rr, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)


# ============================================================================ Blender side
def _sky_node(nt, sun_rot):
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(SUN_EL)
    sky.sun_rotation = sun_rot
    sky.altitude = AGL
    sky.air_density, sky.dust_density, sky.ozone_density = 1.0, 1.0, 1.0
    sky.sun_disc = False
    return sky


def _haze(nt, surface_socket, sun_rot, strength, length=HAZE_M):
    """out = T * surface + (1 - T) * sky(view direction lifted to the horizon), T = exp(-d / length)."""
    N, L = nt.nodes, nt.links
    geo = N.new("ShaderNodeNewGeometry")
    neg = N.new("ShaderNodeVectorMath"); neg.operation = "SCALE"; neg.inputs["Scale"].default_value = -1.0
    L.new(geo.outputs["Incoming"], neg.inputs[0])
    sep = N.new("ShaderNodeSeparateXYZ"); L.new(neg.outputs[0], sep.inputs[0])
    mz = N.new("ShaderNodeMath"); mz.operation = "MAXIMUM"; mz.inputs[1].default_value = 0.035
    L.new(sep.outputs["Z"], mz.inputs[0])
    comb = N.new("ShaderNodeCombineXYZ")
    L.new(sep.outputs["X"], comb.inputs["X"]); L.new(sep.outputs["Y"], comb.inputs["Y"]); L.new(mz.outputs[0], comb.inputs["Z"])
    nrm = N.new("ShaderNodeVectorMath"); nrm.operation = "NORMALIZE"; L.new(comb.outputs[0], nrm.inputs[0])
    sky = _sky_node(nt, sun_rot)
    L.new(nrm.outputs[0], sky.inputs["Vector"])
    em = N.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = strength
    L.new(sky.outputs["Color"], em.inputs["Color"])
    cam = N.new("ShaderNodeCameraData")
    k = N.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = -1.0 / length
    L.new(cam.outputs["View Distance"], k.inputs[0])
    ex = N.new("ShaderNodeMath"); ex.operation = "EXPONENT"; L.new(k.outputs[0], ex.inputs[0])
    one = N.new("ShaderNodeMath"); one.operation = "SUBTRACT"; one.inputs[0].default_value = 1.0
    L.new(ex.outputs[0], one.inputs[1])
    mix = N.new("ShaderNodeMixShader")
    L.new(one.outputs[0], mix.inputs[0]); L.new(surface_socket, mix.inputs[1]); L.new(em.outputs[0], mix.inputs[2])
    return mix.outputs[0]


def _cloud_cover(nt, vec_socket):
    """Cumulus coverage 0..1 at a point of the deck (shared by the clouds and their ground shadows)."""
    N, L = nt.nodes, nt.links
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.0009
    nz.inputs["Detail"].default_value = 5.0
    nz.inputs["Roughness"].default_value = 0.6
    L.new(vec_socket, nz.inputs["Vector"])
    ramp = N.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = 0.50, 0.58
    L.new(nz.outputs["Fac"], ramp.inputs["Value"])
    return ramp.outputs["Result"], nz


def build_terrain(bpy, sun_rot, sun_dir, sun_e):
    me = bpy.data.meshes.new("desert")
    S = 60000.0
    me.from_pydata([(-S, -S, 0), (S, -S, 0), (S, S, 0), (-S, S, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("desert", me)
    bpy.context.scene.collection.objects.link(ob)
    m = bpy.data.materials.new("desert")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    out = N["Material Output"]
    tc = N.new("ShaderNodeTexCoord")
    pos = tc.outputs["Object"]

    def noise(scale, detail=6.0, rough=0.55, vec=pos):
        n = N.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        L.new(vec, n.inputs["Vector"])
        return n.outputs["Fac"]

    def ramp(fac, stops):
        r = N.new("ShaderNodeValToRGB")
        e = r.color_ramp.elements
        e[0].position, e[0].color = stops[0][0], (*stops[0][1], 1)
        e[1].position, e[1].color = stops[-1][0], (*stops[-1][1], 1)
        for p, c in stops[1:-1]:
            el = e.new(p)
            el.color = (*c, 1)
        L.new(fac, r.inputs["Fac"])
        return r.outputs["Color"]

    def mix(fac, a, b_, blend="MIX"):
        mx = N.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = blend
        if isinstance(fac, float):
            mx.inputs["Factor"].default_value = fac
        else:
            L.new(fac, mx.inputs["Factor"])
        for sock, v in ((mx.inputs[6], a), (mx.inputs[7], b_)):
            if isinstance(v, tuple):
                sock.default_value = (*v, 1)
            else:
                L.new(v, sock)
        return mx.outputs[2]

    # regional albedo (km), dune field streaks (100 m), interdune flats, scrub speckle (m), tracks
    reg = ramp(noise(0.00028, 3, 0.6), [(0.30, (0.22, 0.125, 0.055)), (0.55, (0.30, 0.18, 0.085)), (0.75, (0.36, 0.235, 0.12))])
    warp = N.new("ShaderNodeTexWave"); warp.wave_type = "BANDS"; warp.bands_direction = "DIAGONAL"
    warp.inputs["Scale"].default_value = 0.012
    warp.inputs["Distortion"].default_value = 9.0
    warp.inputs["Detail"].default_value = 2.0
    L.new(pos, warp.inputs["Vector"])
    dunes = ramp(warp.outputs["Fac"], [(0.35, (0.80, 0.80, 0.80)), (0.6, (1.0, 1.0, 1.0)), (0.9, (1.12, 1.10, 1.06))])
    col = mix(1.0, reg, dunes, "MULTIPLY")
    flats = noise(0.0016, 2, 0.5)
    fl = N.new("ShaderNodeMapRange"); fl.inputs["From Min"].default_value, fl.inputs["From Max"].default_value = 0.55, 0.62
    L.new(flats, fl.inputs["Value"])
    col = mix(fl.outputs["Result"], col, (0.34, 0.29, 0.22))            # pale clay pans
    vor = N.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 0.09
    L.new(pos, vor.inputs["Vector"])
    dens = noise(0.0022, 2, 0.5)
    th = N.new("ShaderNodeMapRange"); th.inputs["To Min"].default_value, th.inputs["To Max"].default_value = 0.0, 0.32
    L.new(dens, th.inputs["Value"])
    lt = N.new("ShaderNodeMath"); lt.operation = "LESS_THAN"
    L.new(vor.outputs["Distance"], lt.inputs[0]); L.new(th.outputs["Result"], lt.inputs[1])
    col = mix(lt.outputs[0], col, (0.045, 0.048, 0.022))                # scrub bushes
    fine = noise(0.6, 2, 0.6)
    col = mix(0.25, col, ramp(fine, [(0.3, (0.7, 0.7, 0.7)), (0.7, (1.2, 1.2, 1.2))]), "MULTIPLY")
    # a straight graded track and a wadi line give scale
    sep = N.new("ShaderNodeSeparateXYZ"); L.new(pos, sep.inputs[0])
    ax = N.new("ShaderNodeMath"); ax.operation = "SUBTRACT"; ax.inputs[1].default_value = 180.0
    L.new(sep.outputs["X"], ax.inputs[0])
    ab = N.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"; L.new(ax.outputs[0], ab.inputs[0])
    road = N.new("ShaderNodeMath"); road.operation = "LESS_THAN"; road.inputs[1].default_value = 3.5
    L.new(ab.outputs[0], road.inputs[0])
    col = mix(road.outputs[0], col, (0.25, 0.21, 0.16))
    wadi = N.new("ShaderNodeTexWave"); wadi.inputs["Scale"].default_value = 0.0009
    wadi.inputs["Distortion"].default_value = 14.0
    wadi.inputs["Detail"].default_value = 3.0
    L.new(pos, wadi.inputs["Vector"])
    wd = N.new("ShaderNodeMapRange"); wd.inputs["From Min"].default_value, wd.inputs["From Max"].default_value = 0.94, 0.99
    L.new(wadi.outputs["Fac"], wd.inputs["Value"])
    col = mix(wd.outputs["Result"], col, (0.13, 0.10, 0.06))
    # cloud shadows: the same coverage as the deck, sampled where the sun ray meets it
    sun = N.new("ShaderNodeCombineXYZ")
    sx, sy, sz = sun_dir
    tz = N.new("ShaderNodeMath"); tz.operation = "MULTIPLY_ADD"
    L.new(sep.outputs["Z"], tz.inputs[0]); tz.inputs[1].default_value = -1.0 / sz; tz.inputs[2].default_value = CLOUD_AGL / sz
    for comp, val in (("X", sx), ("Y", sy), ("Z", sz)):
        mm = N.new("ShaderNodeMath"); mm.operation = "MULTIPLY"; mm.inputs[1].default_value = val
        L.new(tz.outputs[0], mm.inputs[0]); L.new(mm.outputs[0], sun.inputs[comp])
    hit = N.new("ShaderNodeVectorMath"); hit.operation = "ADD"
    L.new(pos, hit.inputs[0]); L.new(sun.outputs[0], hit.inputs[1])
    cover, _ = _cloud_cover(nt, hit.outputs[0])
    sh = N.new("ShaderNodeMapRange"); sh.inputs["To Min"].default_value, sh.inputs["To Max"].default_value = 1.0, 0.12
    L.new(cover, sh.inputs["Value"])
    # Lambertian sand lit in the shader: albedo x (E_sun cos(i) x cloud shadow + E_sky) / pi. Exact for
    # this flat, rough surface and it keeps the 120 km plane out of Eevee's light and shadow passes
    # (KAL's own shadow cannot be resolved on the ground from 420 m anyway).
    bump = N.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.35
    L.new(noise(0.05, 3, 0.6), bump.inputs["Height"])
    dot = N.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    L.new(bump.outputs["Normal"], dot.inputs[0]); dot.inputs[1].default_value = sun_dir
    cosi = N.new("ShaderNodeMath"); cosi.operation = "MAXIMUM"; cosi.inputs[1].default_value = 0.0
    L.new(dot.outputs["Value"], cosi.inputs[0])
    direct = N.new("ShaderNodeMath"); direct.operation = "MULTIPLY"
    L.new(cosi.outputs[0], direct.inputs[0]); L.new(sh.outputs["Result"], direct.inputs[1])
    irr = N.new("ShaderNodeMath"); irr.operation = "MULTIPLY_ADD"
    L.new(direct.outputs[0], irr.inputs[0]); irr.inputs[1].default_value = sun_e / math.pi
    irr.inputs[2].default_value = SKY_E * sun_e / math.pi
    lit = N.new("ShaderNodeVectorMath"); lit.operation = "SCALE"
    L.new(col, lit.inputs[0]); L.new(irr.outputs[0], lit.inputs["Scale"])
    em = N.new("ShaderNodeEmission")
    L.new(lit.outputs[0], em.inputs["Color"])
    N.remove(b)
    L.new(_haze(nt, em.outputs[0], sun_rot, SKY), out.inputs["Surface"])
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob


def build_clouds(bpy, sun_rot):
    me = bpy.data.meshes.new("cumulus")
    S, h = 60000.0, CLOUD_AGL
    me.from_pydata([(-S, -S, h), (S, -S, h), (S, S, h), (-S, S, h)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("cumulus", me)
    bpy.context.scene.collection.objects.link(ob)
    m = bpy.data.materials.new("cumulus")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    for n in list(N):
        N.remove(n)
    outn = N.new("ShaderNodeOutputMaterial")
    tc = N.new("ShaderNodeTexCoord")
    cover, _ = _cloud_cover(nt, tc.outputs["Object"])
    # bases are greyer where the cloud is thick (fake self-shadowing), edges bright
    thick = N.new("ShaderNodeTexNoise"); thick.inputs["Scale"].default_value = 0.0021; thick.inputs["Detail"].default_value = 3
    L.new(tc.outputs["Object"], thick.inputs["Vector"])
    shade = N.new("ShaderNodeValToRGB")
    e = shade.color_ramp.elements
    e[0].position, e[0].color = 0.3, (0.55, 0.57, 0.62, 1)
    e[1].position, e[1].color = 0.7, (0.95, 0.95, 0.97, 1)
    L.new(thick.outputs["Fac"], shade.inputs["Fac"])
    em = N.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 1.6
    L.new(shade.outputs["Color"], em.inputs["Color"])
    hz = _haze(nt, em.outputs[0], sun_rot, SKY, 3 * HAZE_M)      # most haze sits below the bases
    tr = N.new("ShaderNodeBsdfTransparent")
    mx = N.new("ShaderNodeMixShader")
    L.new(cover, mx.inputs[0]); L.new(tr.outputs[0], mx.inputs[1]); L.new(hz, mx.inputs[2])
    L.new(mx.outputs[0], outn.inputs["Surface"])
    m.blend_method = "BLEND"
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    ob.data.materials.append(m)
    ob.visible_shadow = False
    return ob


def build(a):
    """Scene, aircraft, camera and all keyframes; returns the scene."""
    sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]
    import bpy
    import numpy as np
    from mathutils import Euler, Matrix, Vector

    import kal_hero

    n = int(round(a.seconds * a.fps))
    w, h = map(int, a.res.split("x"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    ee = sc.eevee
    ee.taa_render_samples = a.samples
    ee.use_shadows = True
    ee.shadow_ray_count = 2
    ee.shadow_step_count = 6
    ee.use_raytracing = False                  # matte skin: light probes suffice, ~2x faster here
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.fps = a.fps
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5                     # 180-degree shutter
    if hasattr(ee, "motion_blur_steps"):
        ee.motion_blur_steps = 1
    if hasattr(ee, "motion_blur_max"):
        ee.motion_blur_max = 48
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.image_settings.compression = 15
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = -0.9           # expose for the sky, as a real camera would (P6)
    sc.frame_start, sc.frame_end = 1, n

    sun_rot = math.radians(SUN_AZ)
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    sc.world = world
    sky = _sky_node(world.node_tree, sun_rot)
    bgn = world.node_tree.nodes["Background"]
    world.node_tree.links.new(sky.outputs["Color"], bgn.inputs["Color"])
    bgn.inputs["Strength"].default_value = SKY
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 4.6
    sun.data.angle = math.radians(0.53)
    sun.data.color = (1.0, 0.96, 0.90)
    sc.collection.objects.link(sun)
    sun.rotation_euler = Euler((math.radians(90 - SUN_EL), 0, sun_rot + math.radians(90)))
    sun_dir = tuple(sun.rotation_euler.to_matrix() @ Vector((0, 0, 1)))
    build_terrain(bpy, sun_rot, sun_dir, sun.data.energy)
    build_clouds(bpy, sun_rot)

    root, hd = kal_hero.build_hero(prop_mode="disc", look="photo")
    root.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    V = np.concatenate([np.array([v.co for v in o.data.vertices]) @ np.array(o.matrix_world)[:3, :3].T
                        + np.array(o.matrix_world)[:3, 3] for o in bpy.data.objects
                        if o.type == "MESH" and o.name.startswith("KAL") and len(o.data.vertices)])
    centre = (V.min(0) + V.max(0)) / 2                    # model frame, metres
    heading = Matrix.Rotation(math.radians(90), 4, "Z")   # model +X (nose) -> world +Y

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cd = cam.data
    cd.sensor_fit, cd.sensor_width, cd.lens = "HORIZONTAL", 36.0, LENS
    stills = parse_stills(getattr(a, "still", ""))
    if stills:
        cd.lens = a.still_lens
    cd.clip_start, cd.clip_end = 0.1, 120000.0
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = FSTOP
    cam.rotation_mode = "QUATERNION"

    fwd, left, up = np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])
    for k in range(n):
        f, t = k + 1, k / a.fps
        pos, (roll, pitch, yaw) = aircraft_state(t)
        R = heading @ Euler((math.radians(roll), math.radians(-pitch), math.radians(yaw)), "XYZ").to_matrix().to_4x4()
        root.location = Vector(pos) - (R.to_3x3() @ Vector(centre))      # bounding-box centre on the path
        root.rotation_quaternion = R.to_quaternion()
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_quaternion", frame=f)
        az, el, rr, aim, shake = camera_state(k / max(1, n - 1), t) if not stills else still_pose(stills, t)
        mean = np.array([0.0, SPEED * t, AGL])
        d = math.cos(math.radians(el)) * (math.cos(math.radians(az)) * fwd + math.sin(math.radians(az)) * left) \
            + math.sin(math.radians(el)) * up
        cpos = mean + rr * d
        target = mean + np.array(aim)
        q = Vector(target - cpos).to_track_quat("-Z", "Y")
        q = q @ Euler((math.radians(shake[0]), math.radians(shake[1]), math.radians(shake[2])), "XYZ").to_quaternion()
        cam.location = Vector(cpos)
        cam.rotation_quaternion = q
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        cd.dof.focus_distance = float(np.linalg.norm(np.array(pos) - cpos))
        cd.dof.keyframe_insert("focus_distance", frame=f)
    for ob in (root, cam, cd):
        ad = ob.animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    return sc


def render(a):
    import bpy
    sc = build(a)
    n = sc.frame_end
    stills = parse_stills(a.still)
    if stills:
        os.makedirs(STILLS, exist_ok=True)
        for i, (name, *_ ) in enumerate(stills):
            sc.frame_set(still_frame(i, a.fps) + 1)
            sc.render.filepath = os.path.join(STILLS, f"{name}_raw.png")
            bpy.ops.render.render(write_still=True)
            print(f"STILL {name}", flush=True)
        return
    only = [int(s) for s in a.frames.split(",") if s.strip()] if a.frames else list(range(n))
    os.makedirs(RAW, exist_ok=True)
    for k in only:
        path = os.path.join(RAW, f"{k:04d}.png")
        if os.path.exists(path) and not a.frames:
            continue
        sc.frame_set(k + 1)
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"ORBIT_FRAME {k + 1}/{n}", flush=True)


# ============================================================================ camera finish + encode
def finish_frame(src, dst, k, grain=1.0):
    """Lens and sensor look of a real camera on a linear-ish copy of the AgX image: bloom from
    highlights, lateral chromatic aberration, vignetting, mild in-camera sharpening and grain."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage as ndi
    im = Image.open(src)
    x = np.asarray(im, np.float32)
    x = x / (65535.0 if x.max() > 255 else 255.0)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, -1)
    x = x[..., :3]
    h, w = x.shape[:2]
    lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hi = np.clip(lum - 0.72, 0, None)[..., None] * x / np.maximum(lum[..., None], 1e-4)
    bloom = np.stack([ndi.gaussian_filter(hi[..., c], w / 90) for c in range(3)], -1)
    x = x + 0.35 * bloom
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    for c, s in ((0, 1.0006), (2, 0.9994)):                       # ~0.6 px at the corners
        x[..., c] = ndi.map_coordinates(x[..., c], [cy + (yy - cy) / s, cx + (xx - cx) / s], order=1, mode="nearest")
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (cx ** 2 + cy ** 2)
    x *= (1 - 0.22 * r2 ** 1.25)[..., None]
    soft = np.stack([ndi.gaussian_filter(x[..., c], 1.1) for c in range(3)], -1)
    x = x + 0.35 * (x - soft)
    rng = np.random.default_rng(1000 + k)
    g = rng.normal(0, 1, (h, w)).astype(np.float32)
    g = ndi.gaussian_filter(g, 0.55) * 1.7
    sig = grain * (0.006 + 0.010 * np.sqrt(np.clip(lum, 0, 1)))
    chroma = rng.normal(0, 0.12, (h, w, 3)).astype(np.float32)
    x = x + (g[..., None] + chroma) * sig[..., None]
    x = 0.012 + 0.988 * x
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(dst)


def finish_all():
    os.makedirs(FIN, exist_ok=True)
    names = sorted(f for f in os.listdir(RAW) if f.endswith(".png"))
    for f in names:
        finish_frame(os.path.join(RAW, f), os.path.join(FIN, f), int(f[:4]))
    return len(names)


def encode(a):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not on PATH")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(FIN, "%04d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    CLIP], check=True)
    print("wrote", CLIP)


def main():
    if "--" in sys.argv:                                            # running inside Blender
        render(args_from(sys.argv[sys.argv.index("--") + 1:]))
        return
    a = args_from(sys.argv[1:])
    if a.still:
        stills = parse_stills(a.still)
        cmd = [BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "-P", os.path.abspath(__file__), "--",
               "--render", "--res", a.res, "--samples", str(a.samples), "--still", a.still,
               "--still-lens", str(a.still_lens), "--seconds", str(2.0 * len(stills) + 1.0)]
        if not a.finish_only:
            subprocess.run(cmd, check=True, cwd=ROOT)
        from PIL import Image
        for name, *_ in stills:
            dst = os.path.join(STILLS, f"KAL_render_{name}.png")
            finish_frame(os.path.join(STILLS, f"{name}_raw.png"), dst, 7, grain=0.45)
            Image.open(dst).save(dst.replace(".png", ".jpg"), quality=95, subsampling=0)
            print("wrote", dst)
        return
    if not a.finish_only:
        cmd = [BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "-P", os.path.abspath(__file__), "--",
               "--render", "--seconds", str(a.seconds), "--fps", str(a.fps), "--res", a.res, "--samples", str(a.samples)]
        cmd += ["--frames", a.frames] if a.frames else []
        subprocess.run(cmd, check=True, cwd=ROOT)
    if a.frames:
        os.makedirs(FIN, exist_ok=True)
        for k in (int(s) for s in a.frames.split(",") if s.strip()):
            finish_frame(os.path.join(RAW, f"{k:04d}.png"), os.path.join(FIN, f"{k:04d}.png"), k)
        return
    print("finished", finish_all(), "frames")
    encode(a)


if __name__ == "__main__":
    main()
