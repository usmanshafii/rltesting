"""Propeller rotation blur: blade coverage per band integration time."""
import math

import numpy as np
import pytest

import kal_geometry as kg
from render import rotation_blur as rb
from render import sampling

CFGS = sampling.load_configs(overlay="")
G = kg.derive({**kg.defaults(CFGS["target"]), **kg.sample_variant(CFGS["target"], 2)})


def _rot(alpha=0.09, t=None, seed=0):
    t = t or {"rgb": 2.0, "swir": 2.0, "lwir": 20.0}
    return rb.scene_rotation(np.random.default_rng(seed), G, G["prop_rpm"], alpha, t, {"texture_px": 128})


def test_two_caps_compose_to_the_wanted_coverage():
    c = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    a = rb.per_cap_alpha(c)
    assert np.allclose(1 - (1 - a) ** 2, c)
    assert a[1] == pytest.approx(0.0513, abs=1e-3)          # not 0.1, which would double to 0.19


def test_mean_coverage_is_the_sampled_alpha_at_any_exposure():
    rot = _rot(alpha=0.09)
    for sweep in (0.0, 30.0, 400.0):
        cov = rb.blade_coverage(256, rot["propeller"], sweep)
        assert cov.mean() * 4 / math.pi == pytest.approx(0.09, rel=0.06)


def test_short_exposure_freezes_blades_long_exposure_smears_them():
    p = _rot()["propeller"]
    frozen, smeared = rb.blade_coverage(256, p, 0.0), rb.blade_coverage(256, p, 720.0)
    assert frozen.max() == 1.0 and np.mean(frozen[frozen > 0] == 1.0) > 0.95       # two sharp blades
    r = np.hypot(*np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256)))
    ring = (r > 0.55) & (r < 0.6)
    assert smeared[ring].std() < 0.01 and 0 < smeared[ring].mean() < 0.2           # uniform at a given radius
    assert rb.sweep_deg(4000, 5.0) == pytest.approx(120.0) and rb.sweep_deg(1500, 1.0) == pytest.approx(9.0)


def test_only_the_propeller_is_blurred():
    """The drive flange and spinner also turn, but they are bodies of revolution: only the blades need a texture."""
    rot = _rot()
    assert set(rot) == {"rpm", "propeller", "integration_ms", "sweep_deg", "texture_px"}
    assert set(rb.band_alpha_images(rot, "lwir")) == {"propeller_disc"}


def test_band_images_use_each_bands_integration_time():
    rot = _rot(t={"rgb": 1.0, "swir": 1.0, "lwir": 25.0})
    rgb, lwir = rb.band_alpha_images(rot, "rgb"), rb.band_alpha_images(rot, "lwir")
    assert rot["sweep_deg"]["lwir"] > 10 * rot["sweep_deg"]["rgb"]
    assert rgb["propeller_disc"][0].max() > lwir["propeller_disc"][0].max()          # sharper blades in RGB
    assert rgb["propeller_disc"][1] == pytest.approx(G["prop_dia"] * G["span_m"] / 2)


def test_scene_records_rotation_without_shifting_other_draws():
    import copy
    spec, _ = sampling.sample_scene(6, CFGS, 720)
    off = copy.deepcopy(CFGS)
    off["render"]["render"]["rotation_blur"]["enabled"] = False
    spec_off, _ = sampling.sample_scene(6, off, 720)
    assert spec["rotation"] is not None and spec_off["rotation"] is None
    assert spec["signatures"] == spec_off["signatures"] and spec["objects"][0]["bbox_px"] == spec_off["objects"][0]["bbox_px"]
    lo, hi = CFGS["target"]["propeller"]["prop_rpm"]
    assert lo <= spec["rotation"]["rpm"] <= hi
    assert spec["rotation"]["integration_ms"]["rgb"] == pytest.approx(spec["camera"]["exposure_ms"])
