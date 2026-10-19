"""RGB band (Phase 2): per-slot Principled materials, Nishita sky with procedural clouds, a sun lamp
calibrated to the Nishita sun disc, and moonlit / starlit nights. Blender side.

Hot parts add their own glow (incandescence): Planck radiance in each colour band x (1 - albedo), in render
units (render/signatures.rgb_self_emission, signatures.yaml rgb.hot_emission), from the RGB pass's own
temperature draw. It is always there but only shows at night on the hottest exhaust stubs."""
from __future__ import annotations

import math
import os

import bpy
import numpy as np

from render import bpy_util as bu
from render.rotation_blur import per_cap_alpha
from render.signatures import rgb_self_emission

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ATLAS = os.path.join(ROOT, "assets", "decals", "decal_atlas.png")
ATLAS_WHITE = 0.83          # linear value of the atlas lettering (sRGB 235)


# ============================================================================ sun calibration
def calibrate_sun(elevation_deg: float, atmo: dict, samples: int = 256) -> dict:
    """Irradiance of the Nishita sun disc (normal incidence, per RGB channel) and the radiance of a
    white horizontal Lambertian plane with and without the disc, for these sky parameters.

    Runs in a throw-away scene: call before building the real one. Values are in the sky's own
    radiance units, so a sun lamp of this strength lights the target consistently with the sky."""
    bu.reset()
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 0
    sc.cycles.filter_width = 1.0
    sc.render.resolution_x = sc.render.resolution_y = 8
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    im = sc.render.image_settings
    im.file_format, im.color_depth, im.exr_codec = "OPEN_EXR", "32", "ZIP"
    me = bpy.data.meshes.new("calib_plane")
    me.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("calib_plane", me)
    sc.collection.objects.link(ob)
    m, nt, out = bu.new_material("calib_white")
    b = bu.principled(nt, [1.0, 1.0, 1.0], 1.0)
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.0
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    cam = bpy.data.objects.new("calib_cam", bpy.data.cameras.new("calib_cam"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 0.5
    cam.location = (0, 0, 5)
    sc.camera = cam
    from data.imgio import read_exr
    import tempfile
    res = []
    for disc in (True, False):
        w, wnt, bgn = bu.world("calib_world")
        s = bu.sky_node(wnt, max(elevation_deg, -89.0), 0.0, atmo)
        s.sun_disc = disc
        wnt.links.new(s.outputs["Color"], bgn.inputs["Color"])
        sc.world = w
        path = os.path.join(tempfile.gettempdir(), f"fyp_calib_{os.getpid()}_{int(disc)}.exr")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        d = read_exr(path)
        res.append(np.array([d["R"].mean(), d["G"].mean(), d["B"].mean()], float))
        bu.remove_file(path)
    plane_disc, plane_sky = res
    s_el = math.sin(math.radians(elevation_deg))
    E = np.clip(math.pi * (plane_disc - plane_sky) / s_el, 0, None) if s_el > 0.005 else np.zeros(3)
    return dict(sun_irradiance=E.tolist(), plane_disc=plane_disc.tolist(), plane_sky=plane_sky.tolist(),
                elevation_deg=elevation_deg)


# ============================================================================ materials
def _grey(a):
    return [float(a)] * 3


def hot_emission(p_rgb: dict, T_kelvin: float, sig: dict) -> tuple:
    """Glow (r, g, b) in render units (render/signatures.rgb_self_emission)."""
    return rgb_self_emission(p_rgb, T_kelvin, sig)


def _add_glow(nt, shader, emission):
    """Shader + an Emission of colour emission / max and strength max (no-op when it is all zero)."""
    peak = max(emission) if emission else 0.0
    if peak <= 0.0:
        return shader
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = [c / peak for c in emission] + [1.0]
    e.inputs["Strength"].default_value = float(peak)
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(shader, add.inputs[0])
    nt.links.new(e.outputs[0], add.inputs[1])
    return add.outputs[0]


def target_material(name: str, slot: str, p: dict, alpha_img=None, emission=None):
    """alpha_img: (image, half extent m) of the propeller disc's time-averaged rotation coverage.
    emission: the part's glow (r, g, b) in render units (hot_emission)."""
    m, nt, out = bu.new_material(name)
    if slot == "decal":
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(ATLAS, check_existing=True)
        tex.interpolation = "Cubic"
        sc = nt.nodes.new("ShaderNodeVectorMath")
        sc.operation = "SCALE"
        sc.inputs["Scale"].default_value = p["albedo"] / ATLAS_WHITE
        nt.links.new(tex.outputs["Color"], sc.inputs[0])
        b = bu.principled(nt, sc.outputs[0], p["roughness"], alpha=tex.outputs["Alpha"])
    elif slot == "propeller_disc":
        alpha = bu.disc_texture(nt, *alpha_img) if alpha_img else float(per_cap_alpha(p["alpha"]))
        b = bu.principled(nt, _grey(p["albedo"]), p["roughness"], alpha=alpha)
    else:
        color = [p["albedo"] * c for c in p.get("color", [1.0, 1.0, 1.0])]
        b = bu.principled(nt, color, p["roughness"], p.get("metallic", 0.0), p.get("coat", 0.0))
        nt.links.new(_add_glow(nt, b.outputs[0], emission), out.inputs["Surface"])
        return m
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def negative_material(name: str, slot: str, p: dict):
    m, nt, out = bu.new_material(name)
    col = np.asarray(p.get("color", [1, 1, 1]), float) * p["albedo"]
    alpha = float(per_cap_alpha(p["alpha"])) if "alpha" in p else 1.0      # discs are two-capped solids
    b = bu.principled(nt, list(np.clip(col, 0, 1)), p["roughness"], p.get("metallic", 0.0), p.get("coat", 0.0), alpha)
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def environment_material(name: str, kind: str, p: dict, texture: dict | None = None):
    m, nt, out = bu.new_material(name)
    base = np.asarray(p.get("color", [1, 1, 1]), float) * p["albedo"]
    rough = p["roughness"] if not isinstance(p["roughness"], list) else float(np.mean(p["roughness"]))
    if texture:
        f = bu.texture_factor(nt, texture["scale_m"], texture["contrast"], texture.get("seed", 0.0))
        vm = nt.nodes.new("ShaderNodeVectorMath")
        vm.operation = "SCALE"
        vm.inputs[0].default_value = list(np.clip(base, 0, 1))
        nt.links.new(f, vm.inputs["Scale"])
        b = bu.principled(nt, vm.outputs[0], rough)
    else:
        b = bu.principled(nt, list(np.clip(base, 0, 1)), rough)
    if kind == "ground" and "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.0     # matte soil / vegetation: diffuse only
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


# ============================================================================ world and lamps
def world(spec: dict, calib: dict):
    w, nt, bgn = bu.world("rgb_world")
    atmo = spec["atmosphere"]
    if spec["lighting"] == "night":
        glow = [spec["airglow_rgb"]] * 3
        if spec.get("moon"):
            mo = spec["moon"]
            s = bu.sky_node(nt, mo["elevation_deg"], mo["azimuth_deg"], atmo)
            vm = nt.nodes.new("ShaderNodeVectorMath")
            vm.operation = "MULTIPLY_ADD"
            nt.links.new(s.outputs["Color"], vm.inputs[0])
            vm.inputs[1].default_value = (mo["ratio"],) * 3
            vm.inputs[2].default_value = tuple(glow)
            sky = vm.outputs[0]
            cloud = np.asarray(calib["moon"]["plane_disc"]) * mo["ratio"] + np.asarray(glow)
        else:
            sky = None
            cloud = np.asarray(glow) * 1.2
    else:
        s = bu.sky_node(nt, spec["sun"]["elevation_deg"], spec["sun"]["azimuth_deg"], atmo)
        sky = s.outputs["Color"]
        cloud = np.asarray(calib["sun"]["plane_disc"])
    clouds = spec.get("clouds")
    cloud = cloud * (clouds["brightness"] if clouds else 1.0)
    mask = bu.cloud_mask(nt, clouds)
    if sky is None:
        col = bu.mix_rgb(nt, mask, glow, list(cloud))
    elif mask is None:
        col = sky
    else:
        col = bu.mix_rgb(nt, mask, sky, list(cloud))
    if sky is None and mask is None:
        bgn.inputs["Color"].default_value = list(glow) + [1.0]
    else:
        nt.links.new(col, bgn.inputs["Color"])
    return w


def lamps(spec: dict, calib: dict) -> list:
    out = []
    if spec["lighting"] != "night":
        E = np.asarray(calib["sun"]["sun_irradiance"], float)
        if E.max() > 1e-6:
            out.append(bu.sun_lamp("rgb_sun", spec["sun"]["elevation_deg"], spec["sun"]["azimuth_deg"],
                                   E.max(), E / E.max()))
    elif spec.get("moon"):
        mo = spec["moon"]
        E = np.asarray(calib["moon"]["sun_irradiance"], float) * mo["ratio"]
        if E.max() > 0:
            out.append(bu.sun_lamp("rgb_moon", mo["elevation_deg"], mo["azimuth_deg"], E.max(), E / E.max()))
    return out
