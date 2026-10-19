"""M1 detector v2 (tools/m1_det2.py): preprocessing, dense box targets, GIoU and the tracker. The four-sensor
revision's own tests (coverage, parts, manifest re-check) are in tests/test_m1_four_sensor.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

torch = pytest.importorskip("torch")
from models import cnn_detector as cd  # noqa: E402


def _striped_sky(seed=0, H=128, W=160):
    rng = np.random.default_rng(seed)
    sky = np.linspace(0.40, 0.60, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)   # vertical gradient
    sky += np.linspace(0.0, 0.05, W, dtype=np.float32)[None, :]                                # smooth left-right trend
    stripes = rng.normal(0, 0.03, W).astype(np.float32)
    return sky, stripes, sky + stripes[None, :] + rng.normal(0, 0.005, (H, W)).astype(np.float32)


def _stripe_energy(x):
    """Std of the high-pass column profile: how much column structure is left."""
    col = np.median(x, axis=0)
    return float(np.std(col - np.convolve(col, np.ones(31) / 31, mode="same")[None, :][0]))


def test_destripe_removes_column_offsets_and_keeps_sky_and_target():
    sky, stripes, img = _striped_sky()
    img[60:63, 80:83] += 0.2                                                   # 3 px target
    out = cd.destripe(img)
    resid = out - sky
    col = np.median(resid, axis=0)
    assert np.std(col) < 0.3 * np.std(stripes)                                 # stripes reduced > 3x
    assert np.abs(col[-5:]).max() < 0.5 * np.abs(stripes[-5:]).max() + 0.01    # also at the frame edge
    assert np.abs(np.median(out[:, :40]) - np.median(sky[:, :40])) < 0.01       # smooth trend kept
    assert out[61, 81] - np.median(out[55:68, 70:78]) > 0.15                    # target kept
    assert cd.destripe(img[..., None]).shape == img[..., None].shape


def test_destripe_tracks_stripes_that_change_down_the_column_and_row_noise():
    rng = np.random.default_rng(5)
    H, W = 512, 640
    sky = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)   # sky gradient
    gain = rng.normal(0, 0.15, W).astype(np.float32)
    drift = np.linspace(0, 1, H, dtype=np.float32)[:, None] * rng.normal(0, 0.1, W).astype(np.float32)[None, :]
    rows = rng.normal(0, 0.05, H).astype(np.float32)[:, None]
    img = sky + gain[None, :] * sky + drift + rows + rng.normal(0, 0.02, (H, W)).astype(np.float32)
    img[300:370, 200:350] += 0.6                                               # large 150 x 70 px target
    out = cd.destripe(img)
    clean = sky.copy()
    clean[300:370, 200:350] += 0.6
    before = np.abs(img - clean)[:, :].mean()
    after = np.abs(out - clean)[:, :].mean()
    assert after < 0.35 * before                                               # stripe/row structure mostly removed
    target = out[310:360, 210:340].mean() - np.median(out[200:280, 210:340])
    assert target > 0.45                                                       # the large target is not removed


def test_lwir_noise_v1_stream_unchanged_and_realism_optional():
    from noise import lwir_fpn
    cfg = {"netd_mk": [30.0, 60.0], "counts_per_k": [20.0, 80.0], "offset_counts": [3000.0, 9000.0],
           "column_fpn_k": [0.02, 0.15], "pixel_fpn_k": [0.0, 0.04], "bits": 14,
           "agc_low_pct": [0.5, 2.0], "agc_high_pct": [99.0, 99.9]}
    L = np.full((64, 80), 50.0, np.float32)
    L[30:33, 40:43] = 52.0
    got, p = lwir_fpn.apply(L, np.random.default_rng(7), cfg, 300.0)
    # golden: the code before the realism terms were added
    rng = np.random.default_rng(7)
    q = lwir_fpn.sample_params(rng, cfg)
    assert set(q) == {"netd_k", "counts_per_k", "offset", "col_k", "pix_k", "agc_lo", "agc_hi"}
    c = lwir_fpn.to_counts(L, 300.0, q)
    cpk = q["counts_per_k"] * lwir_fpn._ref_scale(300.0)
    c = c + (rng.standard_normal(80) * q["col_k"] * cpk)[None, :]
    c = c + rng.standard_normal((64, 80)) * q["pix_k"] * cpk
    c = c + rng.standard_normal((64, 80)) * q["netd_k"] * cpk
    c = np.clip(np.round(c), 0, 2 ** 14 - 1).astype(np.float32)
    lo, hi = np.percentile(c, [q["agc_lo"], q["agc_hi"]])
    assert np.array_equal(got, np.clip((c - lo) / max(hi - lo, 1.0), 0.0, 1.0).astype(np.float32))
    # realism keys switch on stripes that vary with the scene down a column
    sig = np.tile(np.linspace(-500, 0, 64, dtype=np.float32)[:, None], (1, 80))   # top row far from reference
    s = lwir_fpn.realistic_stripes(sig, np.random.default_rng(1), dict(col_gain=0.01), 50.0)
    assert np.std(s[-1]) < 1e-3                                                # zero signal -> no gain stripe
    assert np.std(s[0]) > 1.0                                                  # strong signal -> stripes
    v2 = dict(cfg, column_gain_frac=[0.001, 0.005], column_drift_k=[0.0, 0.05], row_noise_k=[0.0, 0.02],
              column_temporal_k=[0.0, 0.02])
    out2, p2 = lwir_fpn.apply(L, np.random.default_rng(7), v2, 300.0)
    assert {"col_gain", "col_drift_k", "row_k", "col_t_k"} <= set(p2) and out2.shape == L.shape


def test_preprocess_versions():
    rng = np.random.default_rng(3)
    frames = {"rgb": rng.uniform(0, 1, (64, 80, 3)).astype(np.float32),
              "lwir": rng.uniform(0, 1, (64, 80, 1)).astype(np.float32)}
    v1 = cd.preprocess(frames)
    assert all(np.allclose(v1[b], cd.standardise(frames[b])) for b in frames)   # v1 path unchanged
    v2 = cd.preprocess(frames, cd.PREPROCESS_V2)
    assert v2["lwir"].shape == (64, 80, 1) and v2["rgb"].shape == (64, 80, 3)
    q = cd.quantise8(np.array([0.0, 0.5, 0.999, 1.2]))
    assert np.allclose(q * 255, np.round(q * 255)) and q.max() == 1.0
    with pytest.raises(ValueError):
        cd.preprocess(frames, "unknown")


def test_dense_targets_centre_cell_matches_v1_and_decodes():
    for box in ([100.3, 60.2, 103.4, 62.6], [40.0, 30.0, 90.0, 52.0]):
        hm, reg, pos, _ = cd.build_targets([np.array([box])], 128, 128)
        hm2, reg2, pos2, _, w2, gt2 = cd.build_targets_dense([np.array([box])], 128, 128)
        assert torch.equal(hm, hm2) and torch.equal(pos, pos2)
        c = pos[0, 0].nonzero()[0]
        assert torch.allclose(reg[0, :, c[0], c[1]], reg2[0, :, c[0], c[1]])
        assert float(w2[0, 0, c[0], c[1]]) == 1.0
        # every supervised cell decodes the label box exactly
        boxes = cd.decoded_boxes(reg2)[0].permute(1, 2, 0)[w2[0, 0] > 0]
        assert torch.allclose(boxes, torch.tensor(box).expand_as(boxes), atol=1e-3)
        assert torch.allclose(gt2[0].permute(1, 2, 0)[w2[0, 0] > 0], torch.tensor(box).expand_as(boxes))
    # a 3 px target is supervised at one cell, a 50 px target at several
    _, _, _, _, w_small, _ = cd.build_targets_dense([np.array([[10, 10, 13, 12]])], 64, 64)
    _, _, _, _, w_big, _ = cd.build_targets_dense([np.array([[10, 10, 60, 40]])], 64, 64)
    assert int((w_small > 0).sum()) == 1 and int((w_big > 0).sum()) > 1


def test_giou_values():
    a = torch.tensor([[0.0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]])
    b = torch.tensor([[0.0, 0, 10, 10], [5, 0, 15, 10], [20, 0, 30, 10]])
    g = cd.giou(a, b)
    assert g[0] == pytest.approx(1.0) and g[1] == pytest.approx(1 / 3, abs=1e-6)
    assert g[2] < 0                                                   # disjoint boxes still get a gradient signal


def test_v2_loss_prefers_correct_boxes():
    box = [40.0, 30.0, 90.0, 52.0]
    hm, reg, pos, _, w, gt = cd.build_targets_dense([np.array([box])], 128, 128)
    logits = torch.logit(hm.clamp(1e-3, 1 - 1e-3))
    good, _ = cd.detection_loss_v2(dict(hm=logits, reg=reg), hm, pos, reg, w, gt)
    bad_reg = reg.clone()
    bad_reg[:, 2:] -= 0.7                                             # boxes half the size
    bad, parts = cd.detection_loss_v2(dict(hm=logits, reg=bad_reg), hm, pos, reg, w, gt)
    assert good < bad and parts["giou"] > 0.1
    empty = cd.build_targets_dense([np.zeros((0, 4))], 64, 64)
    loss, _ = cd.detection_loss_v2(dict(hm=torch.zeros(1, 1, 16, 16), reg=torch.zeros(1, 4, 16, 16)),
                                   empty[0], empty[2], empty[1], empty[4], empty[5])
    assert torch.isfinite(loss)


def test_v1_and_v2_checkpoints_load(tmp_path):
    m = cd.CNNDetector(["rgb", "lwir_cooled"])
    cd.save(tmp_path / "v2.pt", m, arch=cd.ARCH_V2, preprocess=cd.PREPROCESS_V2)
    m2, ck = cd.load(tmp_path / "v2.pt")
    assert ck["architecture"] == cd.ARCH_V2 and ck["preprocess"] == cd.PREPROCESS_V2
    assert m2.bands == ["rgb", "lwir_cooled"]
    cd.save(tmp_path / "v1.pt", cd.CNNDetector(["lwir"]))
    _, ck1 = cd.load(tmp_path / "v1.pt")
    assert ck1["architecture"] == cd.ARCH and ck1.get("preprocess") is None


def test_cnr_helper():
    from m1_det2 import cnr
    rng = np.random.default_rng(1)
    img = rng.normal(0.5, 0.01, (40, 40, 1)).astype(np.float32)
    img[18:22, 18:22] += 0.05
    assert cnr(img, [18, 18, 22, 22]) > 3.0
    assert cnr(img, [5, 5, 9, 9]) < 3.0


def test_multitracker_confirms_moving_target_and_drops_one_off_false_alarms():
    from eval.tracking import MultiTracker
    t = MultiTracker()
    for k in range(6):
        dets = [([100 + 2 * k, 50, 106 + 2 * k, 54], 0.9)]
        if k == 2:
            dets.append(([400, 300, 404, 303], 0.8))             # a single-frame false alarm
        t.step(dets)
    assert len(t.confirmed_ids) == 1                             # only the persistent target confirms
    for _ in range(12):
        t.step([])                                                # target lost: coasts, then deleted
    assert t.tracks == []


def test_fa_threshold_and_froc():
    import m1_det2 as D
    absent = [dict(preds=dict(scores=[0.9, 0.2])), dict(preds=dict(scores=[0.5])), dict(preds=dict(scores=[]))]
    assert D.fa_threshold(absent, 0.0) > 0.9                     # no false alarm allowed
    t = D.fa_threshold(absent, 0.34)                              # floor(0.34 * 3) = 1 allowed
    assert 0.5 < t <= 0.9 and sum(s >= t for r in absent for s in r["preds"]["scores"]) == 1
    from m1_det2_eval import froc_from_arrays
    best = np.array([0.95, 0.6, 0.1])
    score, pd = froc_from_arrays(best, [[np.array([0.9, 0.2])], [np.array([0.5])], [np.array([])]], [0.0, 0.34])
    assert pd["0.0"] == pytest.approx(1 / 3)                      # threshold above 0.9: only 0.95 detected
    assert pd["0.34"] == pytest.approx(2 / 3)                     # one false alarm allowed: above 0.5
    assert score == pytest.approx(0.5)
