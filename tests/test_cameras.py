"""Camera classes and rig (render/cameras.py, configs/sensors.yaml cameras) against the manufacturer readings saved in
docs/sources/cameras/readings.json, and the loader's emulation of each camera on the output grid
(data/camera_emulation.py)."""
import html
import json
import math
import os
import re
import unicodedata

import numpy as np
import pytest

from data import camera_emulation as ce
from render import cameras as cams
from render import sampling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "sources", "cameras")
CFG = sampling.load_configs(overlay="")
CAMS = CFG["sensors"]["cameras"]
with open(os.path.join(SRC, "readings.json"), encoding="utf-8") as _f:
    READ = json.load(_f)


def readings(doc=None, key=None):
    return [r for r in READ["readings"] if (doc is None or r["doc"] == doc) and (key is None or key in r["values"])]


def values(key, doc=None):
    out = []
    for r in readings(doc, key):
        v = r["values"][key]
        out += v if isinstance(v, list) else [v]
    return out


# ---------------------------------------------------------------- the field of view follows from format and lens
def test_fov_formula_reproduces_the_datasheets():
    assert cams.hfov_deg(640, 12.0, 19.0) == pytest.approx(22.9, abs=0.05)           # the user's worked example
    for r in readings("BOSON-DS", "lens_640"):                                      # Boson Table 11, low distortion
        lens = r["values"]["lens_640"]
        if lens["focal_mm"] >= 14.0:
            assert cams.hfov_deg(640, 12.0, lens["focal_mm"]) == pytest.approx(lens["hfov_deg"], rel=0.05)
    lens = readings("BOSON-DS", "lens_320")[0]["values"]["lens_320"]
    assert cams.hfov_deg(320, 12.0, lens["focal_mm"]) == pytest.approx(lens["hfov_deg"], rel=0.02)
    sony = readings("FCB-EV7520-TM", "hfov_deg")[0]["values"]["hfov_deg"]           # 2.3-63.7 deg, f 4.3-129 mm
    f = readings("FCB-EV7520-TM", "focal_mm")[0]["values"]["focal_mm"]
    assert cams.hfov_deg(1920, 2.9, f[0]) == pytest.approx(sony[1], rel=0.05)
    assert cams.hfov_deg(1920, 2.9, f[1]) == pytest.approx(sony[0], rel=0.08)
    oph = readings("OPHIR-ZOOM-WEB")
    wide = next(r for r in oph if "hfov_deg_max" in r["values"])["values"]["hfov_deg_max"]
    tele = next(r for r in oph if "hfov_deg_min" in r["values"])["values"]["hfov_deg_min"]
    assert cams.hfov_deg(640, 15.0, 15.0) == pytest.approx(wide, rel=0.03)
    assert cams.hfov_deg(640, 15.0, 300.0) == pytest.approx(tele, rel=0.03)
    assert cams.focal_for_hfov_mm(640, 12.0, cams.hfov_deg(640, 12.0, 33.3)) == pytest.approx(33.3)


# ---------------------------------------------------------------- configs/sensors.yaml cameras = the readings
def test_camera_classes_are_the_sourced_ones():
    assert set(CAMS) == {"rgb", "lwir_uncooled", "lwir_cooled", "mwir"}                 # SWIR deferred (CLAUDE.md)
    docs = READ["documents"]
    for name, cls in CAMS.items():
        for fmt in cls["formats"]:
            assert fmt["source"] in docs
            src = readings(fmt["source"], "formats")
            assert any([fmt["width"], fmt["height"]] in r["values"]["formats"] for r in src), (name, fmt)
            if "scaled_from" in fmt:                  # a scaled video output of a sourced sensor: same field of view
                n0, p0 = fmt["scaled_from"]
                assert fmt["pitch_um"] == pytest.approx(n0 * p0 / fmt["width"], abs=0.01) and p0 in values("pitch_um")
            else:
                assert fmt["pitch_um"] in values("pitch_um", fmt["source"]), (name, fmt)
    u = CAMS["lwir_uncooled"]
    assert u["band_um"] == values("band_um", "BOSON-DS")
    assert u["focal_mm"][0] == min(r["values"]["lens_640"]["focal_mm"] for r in readings("BOSON-DS", "lens_640"))
    assert u["focal_mm"][1] == max(values("focal_mm", "TAU2-DS"))
    netd_u = values("netd_mk_max", "BOSON-DS") + values("netd_mk_max", "BOSONPLUS-DS") + values("netd_mk_max", "TAU2-DS") \
        + values("netd_mk_max", "DIONE1280-DS") + values("netd_mk_max", "TENUM640-DS")
    assert min(netd_u) <= u["netd_mk"][0] and u["netd_mk"][1] <= max(netd_u)
    t13 = readings("BOSON-DS", "temporal_netd_mk")[0]["values"]["temporal_netd_mk"]
    assert u["row_column_temporal_frac"][1] == pytest.approx(max(c / a for a, c, _ in t13.values()), abs=0.02)
    assert u["thermal_time_constant_ms"] == values("thermal_time_constant_ms", "BOSON-DS")[0]
    c = CAMS["lwir_cooled"]
    assert c["band_um"] == values("band_um", "SCORPIO-LW-DS")
    assert c["band_um"][1] == values("cutoff_um", "PELICAN-D-LW-DS")[0]
    netd_c = values("netd_mk", "SCORPIO-LW-DS") + values("netd_mk", "PELICAN-D-LW-DS")
    assert c["netd_mk"][0] <= min(netd_c) and max(netd_c) <= c["netd_mk"][1]
    assert c["f_number"] == values("f_number", "PELICAN-D-LW-DS")
    assert c["residual_nonuniformity"]["frac_of_dynamic_range"] == values("rnu_frac_of_dr", "PELICAN-D-LW-DS")[0]
    m = CAMS["mwir"]
    for det in m["detectors"]:
        assert det["band_um"] in [r["values"]["band_um"] for r in readings(det["source"], "band_um")]
    netd_m = values("netd_mk", "SCORPIO-MW-DS") + values("netd_mk", "DAPHNIS-HD-MW-DS") + values("netd_mk", "PELICAN-D-640-WEB") \
        + values("netd_mk_max", "MINI-BLACKBIRD-DS")
    assert m["netd_mk"][0] <= min(netd_m) and max(netd_m) <= m["netd_mk"][1]
    assert m["zoom_focal_mm"] == values("focal_mm", "OPHIR-ZOOM-WEB")
    f_m = values("f_number", "DAPHNIS-HD-MW-DS") + values("f_number", "MINI-BLACKBIRD-DS") + values("f_number", "PELICAN-D-640-WEB")
    assert m["f_number"][0] <= min(f_m) and max(f_m) <= m["f_number"][1]
    r = CAMS["rgb"]
    assert r["zoom_focal_mm"] == values("focal_mm", "FCB-EV7520-TM")
    assert max(o for o in r["frame_hz"]["options"]) <= values("frame_hz_max", "IMX327-PI")[0]


def _norm(s):
    s = unicodedata.normalize("NFKC", s).replace("µ", "μ")
    return re.sub(r"[\s|]+", " ", s).strip()


def _doc_pages(path):
    if path.endswith(".pdf"):
        fitz = pytest.importorskip("fitz")
        return [_norm(p.get_text()) for p in fitz.open(path)]
    s = open(path, encoding="utf-8", errors="ignore").read()
    s = s.replace("\\u003C", "<").replace("\\u003E", ">").replace('\\"', '"').replace("\\r\\n", " ").replace("\\t", " ")
    return [_norm(html.unescape(re.sub(r"<[^>]+>", " | ", s)))]


def test_every_quote_is_in_its_saved_document():
    """Each reading's quote appears in the saved copy (on the cited page of a PDF). The copies are git-ignored
    third-party documents (docs/sources/README.md): on a fresh clone the test is skipped."""
    missing = [d["file"] for d in READ["documents"].values() if not os.path.exists(os.path.join(SRC, d["file"]))]
    if missing:
        pytest.skip(f"source documents not present locally: {missing[:3]}")
    pages = {k: _doc_pages(os.path.join(SRC, d["file"])) for k, d in READ["documents"].items()
             if any(r["doc"] == k for r in READ["readings"])}
    for r in READ["readings"]:
        q, pg = _norm(r["quote"]), pages[r["doc"]]
        where = [i + 1 for i, t in enumerate(pg) if q in t]
        assert where, (r["doc"], r["quote"])
        if r["page"]:
            assert r["page"] in where, (r["doc"], r["page"], where)


# ---------------------------------------------------------------- the rig
def test_rig_draws_stay_in_the_sourced_ranges():
    W, H = CFG["sensors"]["image"]["width"], CFG["sensors"]["image"]["height"]
    for k in range(300):
        rig = cams.sample_rig(np.random.default_rng([7, k]), CAMS, W, H)
        g, c = rig["grid"], rig["cameras"]
        assert 5.0 - 1e-6 <= g["hfov_deg"] <= 46.0 + 1e-6
        u = c["lwir_uncooled"]
        assert u["hfov_deg"] == pytest.approx(g["hfov_deg"]) and u["pixel_scale"] == pytest.approx(W / u["width"])
        assert CAMS["lwir_uncooled"]["focal_mm"][0] <= u["focal_mm"] <= CAMS["lwir_uncooled"]["focal_mm"][1]
        for name in ("rgb", "lwir_cooled", "mwir"):
            cam, cls = c[name], CAMS[name]
            assert cls["zoom_focal_mm"][0] - 1e-9 <= cam["focal_mm"] <= cls["zoom_focal_mm"][1] + 1e-9
            assert cam["hfov_deg"] == pytest.approx(cams.hfov_deg(cam["width"], cam["pitch_um"], cam["focal_mm"]))
            widest = cams.hfov_deg(cam["width"], cam["pitch_um"], cls["zoom_focal_mm"][0])
            if cam["coverage"][0] < 1.0:                      # only when even the widest setting is too narrow
                assert cam["focal_mm"] == pytest.approx(cls["zoom_focal_mm"][0]) and widest < g["hfov_deg"]
            assert 0.0 < cam["coverage"][0] <= 1.0 and 0.0 < cam["coverage"][1] <= 1.0
        assert c["rgb"]["coverage"] == [1.0, 1.0]            # the 63.7 deg zoom covers every grid (<= 46 deg)
        assert c["rgb"]["pixel_scale"] < 1.0                 # a 1080p camera is finer than the 640 x 512 grid
        for name in ("lwir_uncooled", "lwir_cooled", "mwir"):
            lo, hi = CAMS[name]["netd_mk"]
            assert lo <= c[name]["netd_mk"] <= hi
        assert 1.0 <= c["rgb"]["read_noise_e"] <= 3.0
        for name in ("lwir_cooled", "mwir"):
            assert 0 < c[name]["integration_ms"] <= 1000.0 / c[name]["frame_hz"] + 1e-9
            assert 0 <= c[name]["pixel_fpn_k"] <= c[name]["residual_fpn_bound_k"]
        assert c["lwir_cooled"]["integration_ms"] < 1.0      # LWIR fills the well in well under a millisecond
    narrow = cams.sample_rig(np.random.default_rng(1), CAMS, W, H, narrow=True)
    u = narrow["cameras"]["lwir_uncooled"]
    assert u["hfov_deg"] == pytest.approx(max(5.0, cams.hfov_deg(u["width"], u["pitch_um"], CAMS["lwir_uncooled"]["focal_mm"][1])))


def test_scene_carries_the_rig_and_m1_uses_it():
    spec, _ = sampling.sample_scene(5, CFG, n_scenes=720)
    rig = spec["cameras"]
    assert rig["grid"]["hfov_deg"] == pytest.approx(spec["camera"]["hfov_deg"])
    assert set(rig["cameras"]) == set(CAMS)
    m1 = sampling.load_configs(overlay="configs/m1.yaml")         # the four-sensor M1 (2026-10-15) uses the same rig
    assert m1["sensors"]["cameras"] == CFG["sensors"]["cameras"]
    spec, _ = sampling.sample_scene(5, m1, n_scenes=600)
    assert set(spec["cameras"]["cameras"]) == set(CAMS)


# ---------------------------------------------------------------- physics of the cooled cameras
def test_cooled_integration_and_residual_pattern():
    lw, mw, hot = (7.7, 9.3), (3.7, 4.8), (3.6, 4.2)
    assert cams.photon_radiance(lw, 310.0) > cams.photon_radiance(lw, 300.0) > cams.photon_radiance(mw, 300.0) \
        > cams.photon_radiance(hot, 300.0)
    t_lw = cams.integration_ms(lw, 2.0, 15.0, 0.5, 6.0e6, 0.65)
    t_mw = cams.integration_ms(mw, 2.0, 15.0, 0.7, 6.5e6, 0.5)
    assert 0.1 < t_lw < 1.0 and 1.0 < t_mw < 10.0
    assert cams.integration_ms(mw, 4.0, 15.0, 0.7, 6.5e6, 0.5) == pytest.approx(4.0 * t_mw)   # ~ F^2
    # Pelican-D LW < 0.04 % STD/DR over 10-90 % fill -> about 34 mK; MWIR is steeper, so fewer kelvin
    assert cams.residual_fpn_k(lw, 0.0004) == pytest.approx(0.034, abs=0.003)
    assert cams.residual_fpn_k(mw, 0.0004) < cams.residual_fpn_k(lw, 0.0004)


# ---------------------------------------------------------------- emulation on the grid
def test_resample_operators_preserve_level():
    for n_out, n_in in ((640, 1391), (640, 320), (320, 640), (512, 512)):
        for mode in ("area", "linear"):
            M = ce.resample_matrix(n_out, n_in, mode)
            assert M.shape == (n_out, n_in) and np.allclose(M.sum(1), 1.0)


def test_emulated_noise_follows_the_camera_pixels():
    rng = np.random.default_rng(3)
    flat = np.full((256, 320), 5.0, np.float32)
    noisy = lambda x: x + rng.standard_normal(x.shape).astype(np.float32)   # noqa: E731  unit noise per camera pixel
    fine, mask = ce.emulate(flat, 0.5, [1.0, 1.0], noisy)                # binned 2 x 2 onto the grid
    assert mask.all() and fine.mean() == pytest.approx(5.0, abs=0.02)
    assert fine.std() == pytest.approx(0.5, rel=0.05)
    coarse, _ = ce.emulate(flat, 2.0, [1.0, 1.0], noisy)                 # each camera pixel spans 2 x 2
    a = coarse - coarse.mean()
    corr = float((a[:, :-1] * a[:, 1:]).mean() / (a * a).mean())
    assert corr > 0.5                                                    # correlated over the coarse pixel
    part, mask = ce.emulate(flat, 1.0, [0.5, 0.75], lambda x: x)        # a zoom that cannot widen to the grid
    y0, y1, x0, x1 = ce.covered_box(256, 320, [0.5, 0.75])
    assert mask.sum() == (y1 - y0) * (x1 - x0) == part.astype(bool).sum()
    assert (x1 - x0, y1 - y0) == (160, 192) and part[y0:y1, x0:x1].min() == pytest.approx(5.0)
    rgb, _ = ce.emulate(np.full((64, 80, 3), 2.0, np.float32), 0.47, [1.0, 1.0], lambda x: x)
    assert rgb.shape == (64, 80, 3) and np.allclose(rgb, 2.0, atol=1e-5)
    assert ce.grid_sigma(0.5, 0.47, 0.3) == 0.3 and ce.grid_sigma(0.5, 2.0, 0.3) == pytest.approx(1.0)
