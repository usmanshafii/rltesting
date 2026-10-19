"""Check in Blender that the MWIR pass's sun wiring (render/thermal.py: a sun lamp of the band's direct irradiance on a
diffuse BSDF of albedo (1 - eps)(1 - specular fraction), added to the emission) gives the analytic radiance
eps L(T) + (1 - eps)(1 - f_s) E cos(i) / pi, with Cycles' sun strength read as irradiance in W m-2.

    blender -b --factory-startup --python-exit-code 1 -P tools/check_thermal_blender.py

A unit sphere with the real MWIR material under a known sun and an orthographic camera; every pixel is compared with
the analytic value at that point of the sphere (the lit side and the shadowed side). Writes
docs/reference/engine/data/blender_thermal_check.json and docs/reference/engine/mwir_sun_check_sphere.png (rendered |
analytic | difference x10). Exit code 1 if the check fails.
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
from render import lwir, sampling, thermal  # noqa: E402

N_PX, SCALE = 320, 2.2
OUT_DATA = os.path.join(ROOT, "docs", "reference", "engine", "data")
TMP = os.path.join(ROOT, "out", "thermal_check")


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
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, d, 10.0 * d
    bu.set_matrix(cam, M)
    bpy.context.scene.camera = cam
    return x, y, d


def check_mwir_sun(cfgs):
    bu.reset()
    bu.setup_cycles(cfgs["render"]["render"])
    bu.set_samples(64, adaptive=False)
    bu.set_bounces(0, cfgs["render"]["render"])
    E, eps, fs = 4.0, 0.5, 0.0                                       # W m-2 in band; no glint, all diffuse
    spec = dict(air_temp_c=15.0, lighting="day", sun=dict(azimuth_deg=30.0, elevation_deg=40.0),
                thermal=dict(bands_um={"mwir": [3.7, 4.8]}, glint={"mwir": dict(lobe_deg=3.0, specular_fraction=fs,
                                                                                 sigma2_rad2=0.003)},
                             tables=dict(bands={"mwir": dict(e_sun_ground=E, sky_elevation_deg=[0.25, 90.0],
                                                             sky_radiance=[0.0, 0.0], L_air=0.0)})))
    b = thermal.Band("mwir", spec)
    envr = dict(L_up=0.0, L_down=0.0)
    mat = thermal.slot_material("mwir_sun_test", dict(emissivity=eps, dT=0.0), spec, envr, b, reflection=True)
    w, nt, bgn = bu.world("black")
    bgn.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bpy.context.scene.world = w
    thermal.lamps(spec, b)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=256, ring_count=128, radius=1.0)
    sphere = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(mat)
    S = np.array(lwir.sun_direction(spec))
    az, el = math.radians(120.0), math.radians(10.0)                 # sees the lit and the dark side
    x, y, d = _camera([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    path = bu.render(os.path.join(TMP, "mwir_sun.exr"), (N_PX, N_PX))
    img = bu.read_passes(path, True)
    bu.remove_file(path)
    L, a = img["combined"][..., 0], img["alpha"]
    px = SCALE / N_PX
    u = (np.arange(N_PX) + 0.5 - N_PX / 2) * px
    v = (N_PX / 2 - (np.arange(N_PX) + 0.5)) * px
    U, V = np.meshgrid(u, v)
    r2 = U ** 2 + V ** 2
    inside = r2 < 0.85 ** 2
    wz = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    N = U[..., None] * x + V[..., None] * y + wz[..., None] * d
    base = eps * float(b.L(spec["air_temp_c"] + 273.15))
    pred = base + (1.0 - eps) * (1.0 - fs) * E * np.clip(N @ S, 0.0, None) / math.pi
    m = inside & (a > 0.999)
    lit = m & ((N @ S) > 0.3)
    ratio = float(np.mean(L[lit] - base) / np.mean(pred[lit] - base))
    rel_rms = float(np.sqrt(np.mean((L[m] - pred[m]) ** 2)) / (pred[m].max() - base))
    dark = m & ((N @ S) < -0.2)
    hi = float(max(pred[m].max(), L[m].max()))
    panel = np.concatenate([np.where(m, L, 0) / hi, np.where(m, pred, 0) / hi,
                            np.clip(np.abs(L - pred) * 10 / hi, 0, 1) * m], axis=1)
    write_png(os.path.join(ROOT, "docs", "reference", "engine", "mwir_sun_check_sphere.png"),
              (np.clip(panel, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    return dict(irradiance_w_m2=E, epsilon=eps, specular_fraction=fs, base_w_m2_sr=base,
                lit_mean_ratio_rendered_over_analytic=ratio, rms_error_over_sun_amplitude=rel_rms,
                dark_side_mean_minus_base=float(np.mean(L[dark]) - base),
                ok=bool(abs(ratio - 1.0) < 0.03 and rel_rms < 0.03 and abs(np.mean(L[dark]) - base) < 1e-3 * hi))


def main():
    os.makedirs(TMP, exist_ok=True)
    cfgs = sampling.load_configs(overlay="")
    res = dict(blender=bpy.app.version_string, mwir_sun=check_mwir_sun(cfgs))
    os.makedirs(OUT_DATA, exist_ok=True)
    with open(os.path.join(OUT_DATA, "blender_thermal_check.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if not res["mwir_sun"]["ok"]:
        raise SystemExit(1)


main()
