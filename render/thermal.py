"""Thermal render passes on the scene's layered atmosphere (CLAUDE.md sensor set rule): LWIR uncooled (8-14 um), LWIR
cooled (7.7-9.3 um) and MWIR (the camera's detector band, 3.6-4.9 um). docs/thermal_bands.md.

As render/lwir.py (the spec's emission-only pass, no bounces), per band:
  * every material emits eps x L_band(T) + (1 - eps) x the surroundings (sky with the normal up, ground with it down),
    L_band from the band's Planck table (thermal/planck_lut.band_lut), temperatures shared with the uncooled LWIR
    draw (render/signatures.add_thermal_bands);
  * the world is the clear sky of the band by elevation from thermal/atmosphere_bands.scene_tables (gas and aerosol
    emission along each ray through the layered atmosphere), with the scene's clouds;
  * sun glints: (1 - eps) x specular fraction x E_sun / (pi sigma^2) x the Gaussian lobe, E_sun the band's direct
    solar irradiance at the ground from the same tables.
MWIR by day adds what the longer bands can neglect: the sun's diffuse reflection, from a sun lamp of the band's direct
irradiance (W m-2) on a diffuse BSDF of albedo (1 - eps)(1 - specular fraction), so shadows follow from the geometry
(the world is hidden from diffuse rays: the sky's reflection stays in the emission term), and the aerosol's single
scattering of sunlight in the sky (thermal/atmosphere_bands.single_scatter_sky, a 2-D table in elevation and
scattering angle). Clouds then reflect the sun as well. Values are W m-2 sr-1 throughout.
"""
from __future__ import annotations

import math

import bpy
import numpy as np

from render import bpy_util as bu
from render import lwir as lw
from render.rotation_blur import per_cap_alpha
from thermal import atmosphere_bands as atb
from thermal.planck_lut import band_lut

K0 = 273.15
LUT_N = 2048
SCAT_N = (256, 64)                 # (elevation, cos scattering angle) samples of the MWIR scattering table


class Band:
    """One thermal camera band of a scene: Planck table, atmosphere table, signatures, sun."""

    def __init__(self, name: str, spec: dict):
        th = spec["thermal"]
        self.name = name
        self.band_um = tuple(th["bands_um"][name])
        self.lut = band_lut(self.band_um)
        self.table = th["tables"]["bands"][name]
        self.Ta = spec["air_temp_c"] + K0
        self.day = bool(spec["sun"]["elevation_deg"] > 0.0 and spec["lighting"] != "night")
        self.e_sun = float(self.table.get("e_sun_ground", 0.0)) if self.day else 0.0
        self.mwir = name == "mwir"
        g = (th.get("glint") or {}).get(name) if name != "lwir" else spec.get("sun_glint")
        self.glint = None
        if self.day and g and self.e_sun > 0:
            fs = float(g["specular_fraction"])
            self.glint = dict(g, specular_fraction=fs, peak_per_reflectance=fs * self.e_sun / (math.pi * g["sigma2_rad2"]))
        self.fs = float(g["specular_fraction"]) if g else 0.0

    def L(self, T_kelvin):
        return self.lut.radiance(np.clip(T_kelvin, self.lut.t_min, self.lut.t_max))

    def dL_dT(self, T_kelvin):
        return self.lut.dL_dT(np.clip(T_kelvin, self.lut.t_min + 0.1, self.lut.t_max - 0.1))


def sky_profile(b: Band, env: dict) -> tuple:
    """(elevation deg, radiance) over -90..90 deg: the band's clear sky above the horizon, the ground below."""
    el = ((np.arange(LUT_N) + 0.5) / LUT_N - 0.5) * 180.0
    rad = atb.sky_radiance_at(b.table, np.clip(el, 0.0, 90.0))
    g = env["ground"]
    rad = np.where(el < 0, g["emissivity"] * float(b.L(b.Ta + g["dT"])) + (1 - g["emissivity"]) * rad[::-1], rad)
    return el, rad


def cloud_radiance(b: Band, env: dict, spec: dict) -> float:
    """Thick cloud: emission at the cloud's temperature, plus in MWIR by day the sun it reflects (diffuse)."""
    c = env["clouds"]
    r = float(c.get("reflectance", 0.0)) if b.mwir and b.day else 0.0
    mu0 = math.sin(math.radians(max(spec["sun"]["elevation_deg"], 0.0)))
    return float((1.0 - r) * b.L(b.Ta + c["dT"]) + r * b.table.get("e_sun_above_aerosol", b.e_sun) * mu0 / math.pi)


def environment_radiance(b: Band, spec: dict, env: dict) -> dict:
    """Hemispherical sky and ground radiances for the reflection term and the haze model (render/lwir.py's
    definitions on the band's tables). The MWIR sky's scattered sunlight is included in L_up by day."""
    el, rad = sky_profile(b, env)
    up = el >= 0
    th = np.radians(90.0 - el[up])
    w = np.cos(th) * np.sin(th)
    L_up = float((rad[up] * w).sum() / w.sum())
    if b.mwir and b.day:
        L_up += float(np.mean(_scatter_table(b, spec)) * 0.5)      # hemispherical mean of the scattering table
    clouds = spec.get("clouds")
    L_cloud = cloud_radiance(b, env, spec)
    if clouds:
        L_up = (1 - clouds["coverage"]) * L_up + clouds["coverage"] * L_cloud
    g = env["ground"]
    T_g = b.Ta + g["dT"]
    L_down = float(g["emissivity"] * b.L(T_g) + (1 - g["emissivity"]) * L_up)
    if b.mwir and b.day:                                            # sunlit ground, diffuse
        mu0 = math.sin(math.radians(spec["sun"]["elevation_deg"]))
        L_down += float((1 - g["emissivity"]) * b.e_sun * mu0 / math.pi)
    return dict(L_up=L_up, L_down=L_down, L_cloud=L_cloud, L_air=float(b.L(b.Ta)),
                L_horizon=float(np.interp(0.5, el, rad)))


def _scatter_table(b: Band, spec: dict) -> np.ndarray:
    """MWIR single-scattering sky [W m-2 sr-1] on (elevation 0..90 deg, cos scattering angle -1..1)."""
    t = b.table
    if not (b.mwir and b.day) or t.get("tau_aerosol_vertical", 0.0) <= 0:
        return np.zeros(SCAT_N, np.float32)
    ne, nc = SCAT_N
    el = (np.arange(ne) + 0.5) / ne * 90.0
    cs = (np.arange(nc) + 0.5) / nc * 2.0 - 1.0
    mu = np.sin(np.radians(el))[:, None]
    mu0 = math.sin(math.radians(spec["sun"]["elevation_deg"]))
    aer = t["aerosol"]
    g = float(spec["thermal"]["atmosphere"]["aerosol_g"]) if spec["thermal"].get("atmosphere") else 0.7
    L = atb.single_scatter_sky(t["e_sun_above_aerosol"], t["tau_aerosol_vertical"], aer["omega0"], g, mu, mu0,
                               cs[None, :])
    return L.astype(np.float32)


def _glint(nt, eps: float, b: Band, spec: dict):
    """render/lwir._glint with the band's lobe and irradiance."""
    if b.glint is None or eps >= 0.999:
        return None
    return lw._glint(nt, eps, dict(spec, sun_glint=b.glint))


def _with_sun(nt, emission_shader, eps: float, b: Band):
    """MWIR by day: add a diffuse BSDF of albedo (1 - eps)(1 - specular fraction), lit by the band's sun lamp."""
    if not (b.mwir and b.day and b.e_sun > 0) or eps >= 0.999:
        return emission_shader
    alb = float((1.0 - eps) * (1.0 - b.fs))
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (alb, alb, alb, 1.0)
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(emission_shader, add.inputs[0])
    nt.links.new(d.outputs[0], add.inputs[1])
    return add.outputs[0]


def slot_material(name: str, p: dict, spec: dict, envr: dict, b: Band, reflection: bool = True, alpha=None,
                  transparent: bool = False):
    m, nt, out = bu.new_material(name)
    if transparent:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        nt.links.new(tr.outputs[0], out.inputs["Surface"])
        return m
    eps = float(p["emissivity"])
    up = eps * float(b.L(b.Ta + p["dT_upper"])) if "dT_upper" in p else None
    strength = lw._strength(nt, eps * float(b.L(b.Ta + p["dT"])), eps, envr, reflection, up)
    glint = _glint(nt, eps, b, spec) if reflection else None
    if glint is not None:
        strength = bu.math_node(nt, "ADD", strength, glint)
    sh = _with_sun(nt, lw._emission(nt, strength), eps, b)
    if alpha is not None:
        if isinstance(alpha, tuple):
            alpha = bu.disc_texture(nt, *alpha)
        sh = lw._with_alpha(nt, sh, alpha)
    nt.links.new(sh, out.inputs["Surface"])
    return m


def target_material(name, slot, p, spec, envr, b: Band, reflection=True, rgb_params=None, alpha_img=None):
    alpha = None
    if alpha_img and slot == "propeller_disc":
        alpha = alpha_img
    elif slot == "propeller_disc":
        alpha = float(per_cap_alpha(rgb_params["alpha"]))
    return slot_material(name, p, spec, envr, b, reflection, alpha=alpha, transparent=(slot == "decal"))


def negative_material(name, slot, p, spec, envr, b: Band, reflection=True, rgb_params=None):
    alpha = (rgb_params or {}).get("alpha")
    alpha = None if alpha is None else float(per_cap_alpha(alpha))
    return slot_material(name, p, spec, envr, b, reflection, alpha=alpha)


def ground_material(name: str, p: dict, spec: dict, envr: dict, texture: dict, b: Band, reflection=True):
    """Ground with the LWIR draw's procedural temperature pattern (same physical surface), linearised in the band."""
    m, nt, out = bu.new_material(name)
    T = b.Ta + p["dT"]
    eps = float(p["emissivity"])
    L0, dLdT = float(b.L(T)), float(b.dL_dT(T))
    f = bu.texture_factor(nt, texture["scale_m"], 1.0, texture.get("seed", 0.0))
    dev = bu.math_node(nt, "SUBTRACT", f, 1.0)
    emis = bu.math_node(nt, "MULTIPLY_ADD", dev, eps * dLdT * p["texture_dT"])
    emis.node.inputs[2].default_value = eps * L0
    refl = (1 - eps) * envr["L_up"] if reflection else 0.0
    s = bu.math_node(nt, "ADD", emis, refl)
    nt.links.new(_with_sun(nt, lw._emission(nt, s), eps, b), out.inputs["Surface"])
    return m


def world(spec: dict, env: dict, envr: dict, b: Band):
    w, nt, bgn = bu.world(f"{b.name}_world")
    el, rad = sky_profile(b, env)
    img = bu.float_image(f"{b.name}_sky_lut", rad)
    vec, z = bu.world_direction(nt)
    asin = bu.math_node(nt, "ARCSINE", z)
    u = bu.math_node(nt, "MULTIPLY_ADD", asin, 1.0 / math.pi)
    u.node.inputs[2].default_value = 0.5
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    comb.inputs["Y"].default_value = 0.5
    nt.links.new(u, comb.inputs["X"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    tex.extension = "EXTEND"
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    col = tex.outputs["Color"]
    scat = _scatter_table(b, spec)
    if scat.any():
        # (u = elevation / 90 above the horizon, v = (cos theta_scatter + 1) / 2); zero below the horizon
        simg = bu.float_image2d(f"{b.name}_scatter", scat.T)          # rows = cos scattering angle
        su = bu.math_node(nt, "MULTIPLY", asin, 2.0 / math.pi)
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        nt.links.new(vec, dot.inputs[0])
        dot.inputs[1].default_value = lw.sun_direction(spec)
        sv = bu.math_node(nt, "MULTIPLY_ADD", dot.outputs["Value"], 0.5)
        sv.node.inputs[2].default_value = 0.5
        c2 = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(su, c2.inputs["X"])
        nt.links.new(sv, c2.inputs["Y"])
        st = nt.nodes.new("ShaderNodeTexImage")
        st.image = simg
        st.interpolation = "Linear"
        st.extension = "EXTEND"
        nt.links.new(c2.outputs[0], st.inputs["Vector"])
        above = bu.math_node(nt, "GREATER_THAN", z, 0.0)
        sc = bu.math_node(nt, "MULTIPLY", st.outputs["Color"], above)
        addn = nt.nodes.new("ShaderNodeVectorMath")
        addn.operation = "ADD"
        nt.links.new(col, addn.inputs[0])
        comb3 = nt.nodes.new("ShaderNodeCombineXYZ")
        for k in ("X", "Y", "Z"):
            nt.links.new(sc, comb3.inputs[k])
        nt.links.new(comb3.outputs[0], addn.inputs[1])
        col = addn.outputs["Vector"]
    mask = bu.cloud_mask(nt, spec.get("clouds"))
    if mask is not None:
        col = bu.mix_rgb(nt, mask, col, [envr["L_cloud"]] * 3)
    nt.links.new(col, bgn.inputs["Color"])
    try:                                    # the sky's reflection is in the emission term: hide it from diffuse rays
        w.cycles_visibility.diffuse = False
        w.cycles_visibility.glossy = False
    except AttributeError:
        pass
    return w


def lamps(spec: dict, b: Band) -> list:
    """MWIR by day: a sun of the band's direct irradiance at the ground (W m-2, Cycles sun strength)."""
    if not (b.mwir and b.day and b.e_sun > 0):
        return []
    return [bu.sun_lamp(f"{b.name}_sun", spec["sun"]["elevation_deg"], spec["sun"]["azimuth_deg"], b.e_sun)]
