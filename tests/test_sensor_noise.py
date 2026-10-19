"""Camera models (noise/isp.py, noise/rgb_noise.apply_camera, noise/lwir_fpn.apply_camera): each stage behaves as its
source describes, and configs/sensors.yaml camera_pipelines is tied to the readings of the manufacturer documents
(docs/sources/cameras/readings.json) and the papers (docs/sources/sensor_noise/readings.json)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target")]

from noise import isp, lwir_fpn, rgb_noise  # noqa: E402
from render.sampling import load_configs  # noqa: E402

S = load_configs(overlay="")["sensors"]
PIPE = S["camera_pipelines"]


def _readings(folder):
    return json.loads((ROOT / "docs" / "sources" / folder / "readings.json").read_text(encoding="utf-8"))["readings"]


def _value(readings, key, doc=None):
    vals = [r["values"][key] for r in readings if key in r["values"] and (doc is None or r["doc"] == doc)]
    assert vals, key
    return vals[0]


# ============================================================================ config <-> sources
def test_uncooled_agc_and_defect_ranges_cover_the_boson_documentation():
    cam = _readings("cameras")
    agc = PIPE["lwir_uncooled"]["agc"]
    inside = lambda v, r: r[0] <= v <= r[1]  # noqa: E731
    assert inside(_value(cam, "agc_plateau_default"), agc["plateau"])
    assert inside(_value(cam, "agc_max_gain_default"), agc["max_gain"])
    assert inside(_value(cam, "agc_linear_default"), agc["linear"])
    assert all(inside(v, agc["linear"]) for v in _value(cam, "agc_linear_examples"))
    assert inside(_value(cam, "agc_ace_default"), agc["ace"])
    lo, hi = _value(cam, "agc_ace_range")
    assert lo <= agc["ace"][0] and agc["ace"][1] <= hi
    assert inside(_value(cam, "dde_default"), agc["dde"]) and agc["dde"] == _value(cam, "dde_examples")
    assert agc["tail_pct"] == [_value(cam, "agc_tail_default_pct"), _value(cam, "agc_tail_example_pct")]
    worst = max(v for r in cam if "defects_max_frac" in r["values"] for v in r["values"]["defects_max_frac"].values())
    assert PIPE["lwir_uncooled"]["defects"]["frac"][1] <= worst                  # within the consumer-grade limit
    for cam_name, keys in (("lwir_cooled", ("operability", "operability_min")), ("mwir", ("operability",))):
        best_op = min(r["values"][k] for r in cam if r["sensor"] == cam_name for k in keys if k in r["values"])
        assert PIPE[cam_name]["defects"]["frac"][1] <= 1.0 - best_op + 1e-12


def test_rgb_ranges_follow_the_sony_documents_and_the_noise_paper():
    cam = _readings("cameras")
    rgb = PIPE["rgb"]
    assert rgb["max_gain_db"] == _value(cam, "max_gain_db")
    assert rgb["adc_bits"] == _value(cam, "adc_bits")
    assert rgb["nr_level"] == _value(cam, "nr_levels") and len(rgb["nr_sigma_px"]) == 6 and rgb["nr_sigma_px"][0] == 0
    assert rgb["aperture_level"][1] - rgb["aperture_level"][0] + 1 == _value(cam, "aperture_steps")
    gaussian = _value(_readings("sensor_noise"), "tukey_lambda_gaussian")
    assert rgb["read_tukey_lambda"][1] == gaussian and rgb["read_tukey_lambda"][0] < gaussian    # Gaussian to heavier tails


# ============================================================================ stages
def test_bad_pixel_replacement_uses_good_neighbours():
    rng = np.random.default_rng(0)
    img = np.full((40, 50), 100.0, np.float32) + rng.normal(0, 1, (40, 50)).astype(np.float32)
    m = isp.defect_mask(rng, 40, 50, 0.02, bad_lines=1)
    bad = img.copy()
    bad[m] = 16383.0
    out = isp.replace_bad_pixels(bad, m)
    assert np.abs(out - 100.0).max() < 6.0                                     # no rail value survives, a bad line too
    assert np.array_equal(out[~m], bad[~m])


def test_histogram_agc_respects_max_gain_and_plateau():
    rng = np.random.default_rng(1)
    x = 6000 + np.linspace(0, 20, 64)[None, :] * np.ones((64, 1)) + rng.normal(0, 0.5, (64, 64))
    p = dict(plateau=0.07, tail=0.0, max_gain=1.38, linear=0.2, ace=1.0, dde=1.0, ibe=False, smoothing_px=2.0)
    y = isp.agc_histogram(x, p)
    span = np.percentile(x, 100) - np.percentile(x, 0)
    assert 0 <= y.min() and y.max() <= 1
    assert (y.max() - y.min()) * 255 <= p["max_gain"] * span * 1.25 + 4      # a low-contrast scene is not stretched
    lin = isp.agc_linear(x, 0.0, 100.0)
    assert (lin.max() - lin.min()) > 2 * (y.max() - y.min())
    # monotonic in the low-pass signal, and DDE scales the detail
    flat = np.full((64, 64), 5000.0)
    flat[30:33, 30:33] += 50.0
    flat += rng.normal(0, 3, flat.shape)
    p2 = dict(p, max_gain=10.0, ibe=True)
    lo_d, hi_d = isp.agc_histogram(flat, dict(p2, dde=0.8)), isp.agc_histogram(flat, dict(p2, dde=1.3))
    bg = np.median(lo_d)
    assert (hi_d[31, 31] - np.median(hi_d)) > (lo_d[31, 31] - bg)


def test_tukey_lambda_scale_and_tails():
    rng = np.random.default_rng(2)
    g = isp.tukey_lambda(rng, 0.14, 2.0, 200_000)
    heavy = isp.tukey_lambda(rng, -0.15, 2.0, 200_000)
    assert g.std() == pytest.approx(2.0, rel=0.02) and heavy.std() == pytest.approx(2.0, rel=0.05)
    kurt = lambda a: float(np.mean((a - a.mean()) ** 4) / a.var() ** 2)  # noqa: E731
    assert kurt(g) == pytest.approx(3.0, abs=0.15) and kurt(heavy) > kurt(g) + 0.5


def test_demosaic_and_chromatic_aberration():
    img = np.zeros((32, 40, 3), np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 0.2, 0.5, 0.8
    out = isp.demosaic_bilinear(isp.mosaic_rggb(img))
    assert np.allclose(out[2:-2, 2:-2], img[2:-2, 2:-2], atol=1e-6)
    edge = np.zeros((64, 64, 3), np.float32)
    edge[:, 48:] = 1.0
    ca = isp.chromatic_aberration(edge, 0.05, -0.05)
    assert np.array_equal(ca[..., 1], edge[..., 1]) and not np.allclose(ca[..., 0], ca[..., 2])


# ============================================================================ the two camera models
def _uncooled_cfg():
    cfg = dict(S["lwir_noise"], netd_mk=[40.0, 40.0])
    return cfg


def test_thermal_camera_is_deterministic_bounded_and_has_no_stuck_rails():
    L = np.full((96, 128), 5.0, np.float32)
    L[40:43, 60:64] += 0.3
    cfg = _uncooled_cfg()
    a, pa = lwir_fpn.apply_camera(L, np.random.default_rng(5), cfg, PIPE["lwir_uncooled"], 293.15)
    b, _ = lwir_fpn.apply_camera(L, np.random.default_rng(5), cfg, PIPE["lwir_uncooled"], 293.15)
    assert np.array_equal(a, b) and 0.0 <= a.min() and a.max() <= 1.0
    assert pa["pipeline"]["agc"]["mode"] in ("histogram", "linear")
    pipe = json.loads(json.dumps(PIPE["lwir_uncooled"]))
    pipe["defects"].update(frac=[0.05, 0.05], bad_lines=[1, 1], residual_frac=[0.0, 0.0])   # many mapped defects ...
    pipe["agc"]["histogram_prob"] = 0.0
    clean = json.loads(json.dumps(pipe))
    clean["defects"].update(frac=[0.0, 0.0], bad_lines=[0, 0])
    c, _ = lwir_fpn.apply_camera(L, np.random.default_rng(6), cfg, pipe, 293.15)
    d, _ = lwir_fpn.apply_camera(L, np.random.default_rng(6), cfg, clean, 293.15)
    assert abs(float((c >= 0.999).mean()) - float((d >= 0.999).mean())) < 0.003   # ... are replaced: the 5 % stuck
    assert abs(float((c <= 0.001).mean()) - float((d <= 0.001).mean())) < 0.003   # pixels add no rail values


def test_rgb_camera_noise_grows_at_night_and_noise_reduction_acts_only_at_high_gain():
    cfg = dict(S["rgb_noise"], read_noise_e=[2.0, 2.0])
    pipe = json.loads(json.dumps(PIPE["rgb"]))
    pipe.update(aperture_level=[0, 0], chromatic_aberration=[0.0, 0.0], high_sensitivity_prob=0.0, gamma_standard_prob=1.0)

    def frame(level, nr):
        L = np.full((128, 160, 3), level, np.float32)
        q = dict(pipe, nr_level=[nr, nr])
        out, p = rgb_noise.apply_camera(L, np.random.default_rng(7), cfg, q, 5.0)
        return float(out[8:-8, 8:-8].std() / max(out.mean(), 1e-6)), p
    day_n, day_p = frame(10.0, 0)
    night_n, night_p = frame(0.002, 0)
    night_nr, night_nr_p = frame(0.002, 5)
    day_nr, day_nr_p = frame(10.0, 5)
    assert night_p["gain"] > 10 * day_p["gain"] and night_n > 3 * day_n         # AE gain and relative noise at night
    assert night_nr < 0.8 * night_n and night_nr_p["nr_sigma_px"] > 0           # NR acts at high gain ...
    assert day_nr_p["nr_sigma_px"] == 0 and day_nr == pytest.approx(day_n, rel=1e-6)   # ... and not in bright light
    assert night_p["gain"] <= 10 ** (pipe["max_gain_db"] / 20) + 1e-6


def test_loader_uses_the_camera_pipelines_for_rig_cameras(monkeypatch):
    from data.dataset import SceneDataset
    calls = []
    monkeypatch.setattr(lwir_fpn, "apply_camera", lambda *a, **k: (calls.append("thermal"), (np.zeros_like(a[0]), {}))[1])
    monkeypatch.setattr(rgb_noise, "apply_camera", lambda *a, **k: (calls.append("rgb"), (np.zeros_like(a[0]), {}))[1])
    monkeypatch.setattr(lwir_fpn, "apply", lambda *a, **k: (calls.append("thermal-sensor"), (np.zeros_like(a[0]), {}))[1])
    monkeypatch.setattr(rgb_noise, "apply", lambda *a, **k: (calls.append("rgb-sensor"), (np.zeros_like(a[0]), {}))[1])
    meta = dict(camera=dict(exposure_ms=5.0), conditions=dict(air_temp_c=20.0))
    rng = np.random.default_rng(0)
    on = json.loads(json.dumps(S))
    on["camera_pipelines"]["enabled"] = True
    for sensors in (on, S):                                  # switched on; as shipped (off since the verification)
        ds = SceneDataset(".", None, ["rgb"], dict(sensors=sensors, render={}), metas=[])
        ds._camera_sensor("rgb", dict(read_noise_e=2.0), meta, rng)(np.zeros((4, 4, 3), np.float32))
        for band in ("lwir", "lwir_cooled", "mwir"):
            ds._camera_sensor(band, dict(netd_mk=20.0, band_um=[8.0, 14.0]), meta, rng)(np.zeros((4, 4), np.float32))
    assert S["camera_pipelines"]["enabled"] is False
    assert calls == ["rgb", "thermal", "thermal", "thermal"] + ["rgb-sensor"] + ["thermal-sensor"] * 3
    assert not math.isnan(0.0)
