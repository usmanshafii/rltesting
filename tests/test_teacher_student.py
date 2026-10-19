"""Teacher / student wiring (CLAUDE.md sensor set rule 5; train/engine.py, configs/train.yaml): the teacher trains on RGB
+ LWIR uncooled + LWIR cooled + MWIR with the detection loss alone; the student (RGB + LWIR uncooled) distils from the
teacher's checkpoint with per-batch modality subsets, response KD and masked feature KD. A tiny real detector on
synthetic 64 x 64 scenes, a few iterations each."""
import numpy as np
import pytest
import torch

from train import engine

SMALL = dict(width=[8, 16, 16, 32, 32], depth=[1, 1, 1, 1], p2_head=True)


class FourBandDataset:
    """Synthetic scenes with every band and a per-object CNR array, as data/dataset.SceneDataset returns them."""
    seen = []

    def __init__(self, root, split, bands, cfgs, **kw):
        self.bands, self.epoch = list(bands), 0
        self.metas = [dict(scene_id=f"s{i:06d}") for i in range(4)]
        FourBandDataset.seen.append(tuple(self.bands))

    def __len__(self):
        return 4

    def set_epoch(self, epoch):
        self.epoch = epoch

    def load(self, i):
        rng = np.random.default_rng(i)
        imgs = {b: (0.3 + 0.1 * rng.random((64, 64, 3 if b == "rgb" else 1))).astype(np.float32) for b in self.bands}
        for b in imgs:
            imgs[b][20:26, 30:38] += 0.4                     # the "target"
        cnr = np.array([[2.0, 0.5, 6.0, 9.0, np.nan]], np.float32)   # rgb, lwir, lwir_cooled, mwir, swir
        return dict(images=imgs, boxes=np.array([[30.0, 20.0, 38.0, 26.0]], np.float32), labels=np.array([0]),
                    scene_id=f"s{i:06d}", cnr=cnr)


@pytest.fixture()
def cfg(monkeypatch):
    c = engine.load_all_configs()
    c["train"]["train"].update(epochs=1, batch=2, workers=0, device="cpu", eval_every=1, warmup_epochs=1)
    c["train"]["detector"].update(SMALL)
    monkeypatch.setattr(engine, "SceneDataset", FourBandDataset)
    monkeypatch.setattr("eval.metrics.quick_ap50", lambda *a: 0.1)
    FourBandDataset.seen = []
    return c


def test_config_teacher_and_student_bands():
    t = engine.load_all_configs()["train"]["models"]
    assert t["teacher"]["bands"] == ["rgb", "lwir", "lwir_cooled", "mwir"]
    assert t["student"] == {"bands": ["rgb", "lwir"], "distill_from": "teacher"}
    assert "swir" not in t["teacher"]["bands"]


def test_teacher_then_distilled_student(cfg, tmp_path):
    t_path = engine.train_model("teacher", "unused", str(tmp_path), cfg, max_iters=2)
    tm, tck = engine.load_ckpt(t_path)
    assert tck["bands"] == ["rgb", "lwir", "lwir_cooled", "mwir"] and set(tm.stems) == set(tck["bands"])
    s_path = engine.train_model("student", "unused", str(tmp_path), cfg, teacher_ckpt=t_path, max_iters=3)
    sm, sck = engine.load_ckpt(s_path)
    assert sck["bands"] == ["rgb", "lwir"]                      # the deployable student: no teacher bands, no adapters
    assert not any(k.startswith(("fd.", "teacher")) for k in sck["state_dict"])
    # the student's training set loaded the teacher's bands too (the teacher sees every band of the same scene)
    assert ("rgb", "lwir", "lwir_cooled", "mwir") in FourBandDataset.seen
    out = sm.predict({"rgb": torch.rand(1, 3, 64, 64), "lwir": torch.rand(1, 1, 64, 64)}, conf=0.0, max_det=5)
    assert len(out) == 1
    out = sm.predict({"rgb": None, "lwir": torch.rand(1, 1, 64, 64)}, conf=0.0, max_det=5)   # LWIR alone
    assert len(out) == 1


def test_student_needs_a_matching_teacher(cfg, tmp_path):
    with pytest.raises(ValueError):
        engine.train_model("student", "unused", str(tmp_path), cfg, max_iters=1)
    other = engine.train_model("lwir_only", "unused", str(tmp_path), cfg, max_iters=1)
    with pytest.raises(ValueError):
        engine.train_model("student", "unused", str(tmp_path / "x"), cfg, teacher_ckpt=other, max_iters=1)
