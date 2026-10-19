"""Check in Blender that the shader wiring of the night RGB glow and the LWIR sun glint reproduces the pure-NumPy
physics they are tested against (render/signatures.rgb_self_emission, sun_glint_params, glint_radiance).

    blender -b --factory-startup --python-exit-code 1 -P tools/check_glow_glint_blender.py

LWIR: a unit sphere with the real LWIR material (render/lwir.slot_material: emission eps L(T) + the glint nodes) under
a known sun, seen by an orthographic camera; every pixel is compared with the analytic radiance at that point of the
sphere. RGB: a plane with the real RGB material of a 500 C stub (render/rgb.target_material with its glow) in the
dark; its colour must equal rgb_self_emission. Writes docs/reference/engine/data/blender_glow_glint_check.json and
docs/reference/engine/glint_check_sphere.png (rendered | analytic | difference x10). Exit code 1 if a check fails.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]
import bpy  # noqa: E402
import numpy as np  # noqa: E402

from data.imgio import write_png  # noqa: E402
from render import bpy_util as bu  # noqa: E402
from render import lwir, rgb, sampling  # noqa: E402
from render import signatures as sg  # noqa: E402

N_PX, SCALE = 384, 2.2
OUT_DATA = os.path.join(ROOT, "docs", "reference", "engine", "data")
TMP = os.path.join(ROOT, "out", "glow_glint_check")


class _Fixed:
    """Deterministic draws for sun_glint_params: lobe (deg), then specular fraction."""
    def __init__(self, *v):
        self.v = list(v)

    def uniform(self, *a):
        return self.v.pop(0)


def _camera(d_cam):
    d = np.asarray(d_cam, float) / np.linalg.norm(d_cam)
    x = np.cross([0.0, 0.0, 1.0] if abs(d[2]) < 0.99 else [0.0, 1.0, 0.0], d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    cd = bpy.data.cameras.new("cam")
    cd.type = "ORTHO"
    cd.ortho_scale = SCALE
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, d, 10.0 * d       # the camera looks along its local -Z
    bu.set_matrix(cam, M)
    bpy.context.scene.camera = cam
    return x, y, d


def _render(name, grey):
    path = bu.render(os.path.join(TMP, f"{name}.exr"), (N_PX, N_PX))
    d = bu.read_passes(path, grey)
    bu.remove_file(path)
    return d


def check_glint(cfgs):
    sig = cfgs["signatures"]
    bu.reset()
    bu.setup_cycles(cfgs["render"]["render"])
    bu.set_samples(16, adaptive=False)
    bu.set_bounces(0, cfgs["render"]["render"])
    spec = dict(air_temp_c=15.0, sun=dict(azimuth_deg=30.0, elevation_deg=40.0))
    spec["sun_glint"] = sg.sun_glint_params(_Fixed(6.0, 1.0), sig, 40.0, 0.30)
    eps = 0.09                                                        # bare cast aluminium
    envr = dict(L_up=0.0, L_down=0.0)                                 # isolate emission + glint
    mat = lwir.slot_material("glint_test", dict(emissivity=eps, dT=0.0), spec, envr, reflection=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=512, ring_count=256, radius=1.0)
    sphere = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(mat)
    S = np.array(lwir.sun_direction(spec))
    az, el = math.radians(60.0), math.radians(-20.0)                  # camera below, on the sun's side
    x, y, d = _camera([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    img = _render("glint", True)
    L = img["combined"][..., 0]
    a = img["alpha"]
    # analytic radiance at every pixel: the sphere point under the pixel centre, its normal, the mirror direction
    px = SCALE / N_PX
    u = (np.arange(N_PX) + 0.5 - N_PX / 2) * px
    v = (N_PX / 2 - (np.arange(N_PX) + 0.5)) * px
    U, V = np.meshgrid(u, v)
    r2 = U ** 2 + V ** 2
    inside = r2 < 0.85 ** 2
    w = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    Nrm = U[..., None] * x + V[..., None] * y + w[..., None] * d
    ndi = Nrm @ d
    R = 2.0 * ndi[..., None] * Nrm - d
    cos = R @ S
    g = spec["sun_glint"]
    base = eps * float(lwir.L(spec["air_temp_c"] + 273.15))
    pred = base + (1.0 - eps) * g["peak_per_reflectance"] * np.exp(-2.0 * (1.0 - cos) / g["sigma2_rad2"])
    m = inside & (a > 0.999)
    peak_pred, peak_rend = float(pred[m].max()), float(L[m].max())
    rel_rms = float(np.sqrt(np.mean((L[m] - pred[m]) ** 2)) / (peak_pred - base))
    spot = np.unravel_index(np.argmax(np.where(m, L, -1)), L.shape)
    spot_pred = np.unravel_index(np.argmax(np.where(m, pred, -1)), pred.shape)
    hi = max(peak_pred, peak_rend)
    panel = np.concatenate([np.where(m, L, 0) / hi, np.where(m, pred, 0) / hi,
                            np.clip(np.abs(L - pred) * 10 / hi, 0, 1) * m], axis=1)
    write_png(os.path.join(ROOT, "docs", "reference", "engine", "glint_check_sphere.png"),
              (np.clip(panel, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    return dict(epsilon=eps, sun_azimuth_deg=30.0, sun_elevation_deg=40.0, lobe_deg=6.0, specular_fraction=1.0,
                peak_per_reflectance_w_m2_sr=g["peak_per_reflectance"], base_w_m2_sr=base,
                peak_rendered=peak_rend, peak_analytic=peak_pred, peak_ratio=peak_rend / peak_pred,
                rms_error_over_glint_amplitude=rel_rms, spot_px_rendered=[int(i) for i in spot],
                spot_px_analytic=[int(i) for i in spot_pred],
                ok=bool(abs(peak_rend / peak_pred - 1) < 0.03 and rel_rms < 0.02
                        and max(abs(spot[0] - spot_pred[0]), abs(spot[1] - spot_pred[1])) <= 2))


def check_glow(cfgs):
    sig = cfgs["signatures"]
    bu.reset()
    bu.setup_cycles(cfgs["render"]["render"])
    bu.set_samples(16, adaptive=False)
    p = dict(albedo=0.10, roughness=0.6, metallic=0.5)                # an oxidised stub
    T = 773.15
    emission = rgb.hot_emission(p, T, sig)
    mat = rgb.target_material("glow_test", "exhaust_stub", p, emission=emission)
    bpy.ops.mesh.primitive_plane_add(size=4.0)
    plane = bpy.context.active_object
    plane.data.materials.append(mat)
    _camera([0.0, 0.0, 1.0])                                          # no lights, no world: only the glow
    img = _render("glow", False)
    c = img["combined"][N_PX // 4:3 * N_PX // 4, N_PX // 4:3 * N_PX // 4].reshape(-1, 3).mean(axis=0)
    ratio = [float(c[k] / emission[k]) for k in range(3)]
    return dict(temperature_c=T - 273.15, albedo=p["albedo"], expected_rgb=list(emission),
                rendered_rgb=[float(x) for x in c], ratio=ratio, ok=bool(all(abs(r - 1) < 0.01 for r in ratio)))


def main():
    os.makedirs(TMP, exist_ok=True)
    cfgs = sampling.load_configs(overlay="")
    res = dict(model_revision=cfgs["signatures"]["model_revision"], blender=bpy.app.version_string,
               lwir_glint=check_glint(cfgs), rgb_glow=check_glow(cfgs))
    with open(os.path.join(OUT_DATA, "blender_glow_glint_check.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if not (res["lwir_glint"]["ok"] and res["rgb_glow"]["ok"]):
        raise SystemExit(1)


main()
