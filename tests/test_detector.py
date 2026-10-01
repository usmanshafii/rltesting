"""Detector, assignment, loss, distillation (spec 7)."""
import pytest
import torch

from models.detector import Detector
from models.distill import FeatureDistiller, cnr_gap_weights, feature_masks, response_kd
from models.loss import Assigner, DetLoss, assign_batch, nms

W, H = 128, 96
SMALL = dict(width=(8, 16, 16, 32, 32), depth=(1, 1, 1, 1))


def _imgs(bands, B=2):
    return {b: torch.rand(B, 3 if b == "rgb" else 1, H, W) for b in bands}


def _targets():
    return [dict(boxes=torch.tensor([[10.0, 10, 14, 13], [60, 40, 90, 60]]), labels=torch.tensor([0, 1])),
            dict(boxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long))]


@pytest.mark.parametrize("p2", [True, False])
def test_forward_shapes(p2):
    m = Detector(["rgb", "lwir"], 7, p2=p2, **SMALL)
    raw = m(_imgs(m.bands))
    assert [r[0].shape[-1] for r in raw] == ([W // 4] if p2 else []) + [W // 8, W // 16, W // 32]
    f = m.flatten(raw)
    assert f.cls.shape[:2] == f.boxes.shape[:2] and f.cls.shape[-1] == 7 and f.anchors.shape[0] == f.cls.shape[1]


def test_tiny_object_always_gets_a_positive():
    m = Detector(["rgb"], 7, p2=True, **SMALL)
    f = m.flatten(m(_imgs(["rgb"], 1)))
    gt = torch.tensor([[50.2, 30.4, 52.9, 32.3]])          # 2.7 x 1.9 px: no stride-4 anchor centre inside
    ts, tb, fg, _ = Assigner(7)(f.cls[0].sigmoid(), f.boxes[0], f.anchors, f.strides, gt, torch.tensor([0]))
    assert fg.sum() >= 1 and ts[fg, 0].min() > 0
    assert f.strides[fg].max() <= 8                        # tiny objects stay on P2 / P3


def test_loss_backward_and_modality_dropout():
    m = Detector(["rgb", "lwir"], 7, **SMALL).train()
    imgs = _imgs(m.bands)
    for kept in (["rgb", "lwir"], ["lwir"], ["rgb"]):      # spec 7.2 per-batch subsets; a dropped band is None
        m.zero_grad()
        loss, _ = DetLoss(7)(m, m({b: imgs[b] if b in kept else None for b in m.bands}), _targets())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None for p in m.heads.parameters())
        assert all(p.grad is None for b in m.bands if b not in kept for p in m.stems[b].parameters())
    with pytest.raises(ValueError, match="at least one band"):
        m({b: None for b in m.bands})                      # never drops every band


def test_missing_band_and_predict():
    m = Detector(["rgb", "lwir"], 7, **SMALL).eval()
    out = m.predict({"rgb": None, "lwir": torch.rand(1, 1, H, W)}, conf=0.0, max_det=20)
    assert len(out) == 1 and out[0]["boxes"].shape[1] == 4 and len(out[0]["scores"]) <= 20


def test_distillation_loss():
    t = Detector(["rgb", "lwir", "swir"], 7, **SMALL).eval()
    s = Detector(["rgb", "lwir"], 7, **SMALL).train()
    imgs, tg = _imgs(t.bands), _targets()
    with torch.no_grad():
        traw, tfeat = t(imgs, return_feats=True)
    tf = t.flatten(traw)
    sraw, sfeat = s({b: imgs[b] for b in s.bands}, return_feats=True)
    t_pos = assign_batch(Assigner(7), tf, tg)[2]           # teacher-positive locations
    l_resp, _ = response_kd(s.flatten(sraw), tf, t_pos)
    cnr = [torch.tensor([[1.0, 0.5, 9.0], [4.0, 2.0, 6.5]]), torch.zeros(0, 3)]    # per object: rgb, lwir, swir
    w = [cnr_gap_weights(c, t.bands, s.bands) for c in cnr]
    assert w[0].tolist() == [3.0, 2.0]                     # 1 + 2 * clip(gap / 5, 0, 1) for gaps 8 and 2.5
    fd = FeatureDistiller(s.feat_channels, t.feat_channels)
    masks = feature_masks(tg, {lv: tfeat[lv] for lv in fd.levels}, dict(zip(t.feat_channels, t.strides)), w)
    l_feat, parts = fd(sfeat, tfeat, masks)
    loss = l_resp + l_feat                                 # lambda_r = lambda_f = 1
    loss.backward()
    assert torch.isfinite(loss) and sorted(parts) == ["feat_p2", "feat_p3"]
    assert all(p.grad is not None for p in [*s.heads.parameters(), *fd.parameters()])
    # student == teacher: the box KL is 0; the class term is a soft-target BCE, whose minimum is the
    # teacher's entropy rather than 0, so check that its gradient vanishes instead
    cls, box = tf.cls.clone().requires_grad_(), tf.box_logits.clone().requires_grad_()
    l_same, rp = response_kd(tf._replace(cls=cls, box_logits=box), tf, t_pos)
    l_same.backward()
    assert abs(rp["resp_box"]) < 1e-7 and cls.grad.abs().max() < 1e-6 and box.grad.abs().max() < 1e-6
    same = FeatureDistiller(t.feat_channels, t.feat_channels)       # identity adapters: the feature MSE is 0
    for a in same.adapters.values():
        torch.nn.init.dirac_(a.weight)
        torch.nn.init.zeros_(a.bias)
    assert float(same(tfeat, tfeat, masks)[0]) < 1e-10


def test_nms_suppresses_tiny_duplicates():
    b = torch.tensor([[10.0, 10, 13, 13], [11.0, 11, 14, 14], [40, 40, 43, 43]])   # IoU(0, 1) ~ 0.29
    keep = nms(b, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([0, 0, 0]), 0.5, 2.0)
    assert keep.tolist() == [0, 2]
