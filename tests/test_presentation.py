"""Presentation and video tools: panorama reprojection, video tracker, hero model textures, orbit video."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]





def test_reproject_recovers_direction_function():
    """Resampling a pinhole panorama of a smooth function of direction reproduces it in a
    differently pointed narrow camera (the open-sky background shortcut of the video)."""
    from render import placement as pl
    from kal_video import reproject
    C = np.zeros(3)
    pano = pl.Camera(300, 240, 40.0, C, 100.0, 15.0)
    U, V = np.meshgrid(np.arange(pano.W) + 0.5, np.arange(pano.H) + 0.5)
    d = np.stack([(U - pano.W / 2) / pano.f, -(V - pano.H / 2) / pano.f, -np.ones_like(U)], -1) @ pano.R.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    img = (np.degrees(np.arcsin(d[..., 2])))[..., None]          # elevation of each pixel's ray
    cam = pl.Camera(64, 48, 8.78, C, 104.0, 13.0)
    got = reproject(img, pano, cam)[..., 0]
    Uc, Vc = np.meshgrid(np.arange(cam.W) + 0.5, np.arange(cam.H) + 0.5)
    dc = np.stack([(Uc - cam.W / 2) / cam.f, -(Vc - cam.H / 2) / cam.f, -np.ones_like(Uc)], -1) @ cam.R.T
    dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
    want = np.degrees(np.arcsin(dc[..., 2]))
    assert np.abs(got - want).max() < 0.05


def test_tracker_confirms_and_coasts():
    from kal_video import Tracker
    t = Tracker()
    for k in range(4):
        s = t.step([([100 + k, 50, 108 + k, 54], 0.9)])
    assert s["confirmed"] and s["detected"]
    s = t.step([])
    assert s["confirmed"] and not s["detected"] and s["coasting"] == 1
    s = t.step([([400, 300, 408, 304], 0.95)])          # far outside the gate: not associated
    assert not s["detected"]


def test_hero_texture_maps_inside_atlas():
    import kal_geometry as kg
    import kal_hero_tex as kt
    g = kg.derive(kg.defaults(kg.load_config()))
    c, r = kt.wing_px(g, np.array([0.0, g["te_x"]]), np.array([-0.5, 0.5]))
    assert c.min() >= 0 and c.max() <= kt.WING and r.max() <= kt.WING
    c, r = kt.fus_px(g, np.array([g["nose_x"], g["eng0"]]), np.array([-math.pi / 2, 3 * math.pi / 2 - 1e-6]))
    assert c.min() >= 0 and c.max() <= kt.N and kt.FUS_ROWS[0] <= r.min() and r.max() <= kt.FUS_ROWS[1]
    for reg in ("FS", "FP", "FI"):
        for x, z in kt._fin_outline(g):
            cc, rr = kt.fin_px(g, x, z, reg)
            assert 0 <= cc < kt.N and kt.FIN_ROWS[0] <= rr <= kt.FIN_ROWS[1]


def test_hero_photo_look_keeps_dimensions_and_fits_atlas():
    import kal_geometry as kg
    import kal_hero_tex as kt
    g = kg.derive(kg.defaults(kg.load_config()))
    gp = {**g, "fin_tip_chord": 0.75 * g["fin_root_chord"]}       # kal_hero PHOTO_FIN_TIP
    poly, base = kt._fin_outline(gp), kt._fin_outline(g)
    # shape only: root chord, heights and the trailing edge are the dataset fin's
    assert poly[0] == base[0] and poly[3] == base[3] and poly[2] == base[2] and poly[4] == base[4]
    assert poly[1][1] == base[1][1] and poly[5][1] == base[5][1]
    for reg in ("FS", "FP", "FI"):
        for x, z in poly:
            cc, rr = kt.fin_px(gp, x, z, reg)
            assert 0 <= cc < kt.N and kt.FIN_ROWS[0] <= rr <= kt.FIN_ROWS[1]
    regions = json.load(open(ROOT / "assets" / "decals" / "hero_decal_atlas.json"))
    assert set(regions) == {"wing_text", "fin_text", "flag", "emblem"}
    for u0, v0, u1, v1 in regions.values():
        assert 0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1


def test_orbit_video_keeps_airspeed_and_covers_all_angles():
    import kal_orbit_video as ov
    (x0, y0, _), _ = ov.aircraft_state(0.0)
    (x1, y1, _), att = ov.aircraft_state(1.0)
    assert ov.SPEED == 44.0 and abs((y1 - y0) - 44.0) < 1e-9          # configs/render.yaml default airspeed
    assert max(abs(v) for v in att) < 2.0                               # small turbulence only
    u = np.linspace(0, 1, 169)
    az = np.array([ov.camera_state(k, 0.0)[0] for k in u])
    el = np.array([ov.camera_state(k, 0.0)[1] for k in u])
    assert az[-1] - az[0] >= 360.0 and np.all(np.diff(az) > 0)          # one full orbit, no reversal
    assert el.max() > 45.0 and el.min() < -30.0                         # top and underside views
