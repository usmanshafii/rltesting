"""Compact CNN detector (models/cnn_detector.py): shapes, targets, losses, decoding, checkpoints."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

torch = pytest.importorskip("torch")
from models import cnn_detector as cd  # noqa: E402




def test_forward_shapes_and_bands():
    for bands in (["rgb"], ["lwir"], ["rgb", "lwir"], ["rgb", "lwir", "swir"]):
        m = cd.CNNDetector(bands, presence=len(bands) > 1).eval()
        x = {b: torch.zeros(2, cd.BAND_CHANNELS[b], 128, 160) for b in bands}
        o = m(x)
        assert o["hm"].shape == (2, 1, 32, 40)
        assert o["reg"].shape == (2, 4, 32, 40)
        assert o["n4"].shape[-2:] == (32, 40) and o["n8"].shape[-2:] == (16, 20)


def test_student_inference_needs_no_swir():
    m = cd.CNNDetector(["rgb", "lwir"], presence=True).eval()
    frames = {"rgb": np.zeros((512, 640, 3), np.float32), "lwir": np.zeros((512, 640, 1), np.float32)}
    out = cd.predict(m, frames)                       # no "swir" key anywhere
    assert set(out) == {"boxes", "scores", "labels"}
    assert "swir" not in m.stems


def test_targets_decode_round_trip():
    """An ideal network output built from the training targets decodes to the label box."""
    for box in ([100.3, 60.2, 103.4, 62.6], [40.0, 30.0, 90.0, 52.0], [10.5, 70.25, 17.0, 74.0]):
        hm, reg, pos, _ = cd.build_targets([np.array([box])], 128, 128)
        logits = torch.logit(hm.clamp(1e-4, 1 - 1e-4))
        dec = cd.decode(dict(hm=logits, reg=reg), conf=0.5)[0]
        assert len(dec["boxes"]) == 1
        assert np.allclose(dec["boxes"][0], box, atol=1e-3)


def test_focal_loss_prefers_correct_heatmap():
    hm, reg, pos, _ = cd.build_targets([np.array([[60, 60, 66, 64]])], 128, 128)
    good = torch.logit(hm.clamp(1e-3, 1 - 1e-3))
    bad = torch.full_like(good, -2.0)
    assert cd.focal_loss(good, hm, pos) < cd.focal_loss(bad, hm, pos)


def test_standardise_is_gain_offset_invariant():
    rng = np.random.default_rng(0)
    img = rng.normal(0.3, 0.01, (64, 80, 1)).astype(np.float32)
    img[30:33, 40:44] += 0.08
    a = cd.standardise(img)
    b = cd.standardise(img * 2.5 + 0.1)
    assert np.allclose(a, b, atol=1e-4)
    assert a[31, 41, 0] > 0.6                       # an 8-sigma blob -> 0.8 (noise units / 10)


def test_nms_and_pyramid_levels():
    boxes = [[10, 10, 20, 20], [11, 11, 21, 21], [100, 100, 110, 104]]
    keep = cd.nms(boxes, [0.9, 0.8, 0.7])
    assert list(keep) == [0, 2]
    assert cd.level_for(5) == 1 and cd.level_for(150) == 2 and cd.level_for(250) == 4
    # every size is covered by at least one pyramid level
    for s in (2, 50, 100, 150, 200, 260):
        assert any(lo <= s < hi for _, lo, hi in cd.LEVELS)


def test_kd_mask_and_cnr_weight():
    _, _, _, gauss = cd.build_targets([np.array([[40, 40, 48, 44]])], 128, 128)
    m = cd.kd_mask(gauss, [[cd.cnr_gap_weight(8.0, 1.0)]], 0.05)
    assert m.shape == (1, 1, 32, 32)
    # w = 1 + 2 * clip(7/5) = 3; the Gaussian is centred on the continuous box centre, so the
    # grid peak is slightly below 1
    assert 0.05 + 3.0 * 0.85 < float(m.max()) <= 0.05 + 3.0 + 1e-6
    assert float(m.min()) == pytest.approx(0.05, abs=1e-3)
    assert cd.cnr_gap_weight(1.0, 6.0) == 1.0


def test_save_load_round_trip(tmp_path):
    m = cd.CNNDetector(["lwir"])
    cd.save(tmp_path / "w.pt", m, name="x", threshold_fa=0.5)
    m2, ck = cd.load(tmp_path / "w.pt")
    assert ck["threshold_fa"] == 0.5 and m2.bands == ["lwir"]
    for (k, v), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)
