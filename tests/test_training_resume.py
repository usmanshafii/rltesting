"""An interrupted run must resume the optimisation, not restart from EMA weights."""
import copy
import json

import numpy as np
import pytest
import torch

from train import engine


class TinyDataset:
    def __init__(self, root, split, bands, cfgs, **kw):
        self.bands, self.epoch = bands, 0
        self.metas = [dict(scene_id=f"{split}{i}") for i in range(4)]

    def __len__(self):
        return 4

    def set_epoch(self, epoch):
        self.epoch = epoch

    def load(self, i):
        return dict(images={b: np.full((4, 4, 3 if b == "rgb" else 1), (i + self.epoch + 1) / 10,
                                       np.float32) for b in self.bands},
                    boxes=np.array([[0, 0, 2, 2]], np.float32), labels=np.array([0]), scene_id=str(i))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)

    def forward(self, images):
        return self.layer(images["rgb"].mean((2, 3)))


class TinyLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, model, raw, targets):
        loss = (raw - .5).square().mean()
        return loss, {"total": float(loss.detach())}


def config(monkeypatch):
    cfg = engine.load_all_configs()
    cfg["train"]["train"].update(epochs=3, batch=2, workers=0, device="cpu", eval_every=1)
    monkeypatch.setattr(engine, "SceneDataset", TinyDataset)
    monkeypatch.setattr(engine, "build_detector", lambda *a: TinyModel())
    monkeypatch.setattr(engine, "DetLoss", TinyLoss)
    monkeypatch.setattr("eval.metrics.quick_ap50", lambda *a: .25)
    return cfg


def test_resume_reproduces_uninterrupted_run_and_history(tmp_path, monkeypatch):
    cfg = config(monkeypatch)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    engine.train_model("rgb_only", "unused", str(full), cfg, gate_epoch=1, continue_after_gate=True)
    with pytest.raises(RuntimeError, match="AP50 below"):
        engine.train_model("rgb_only", "unused", str(resumed), cfg, gate_epoch=1)
    history = json.loads((resumed / "rgb_only/history.json").read_text())
    assert len(history) == 1 and history[0]["epoch_number"] == 1
    with pytest.raises(RuntimeError, match="gate needs review"):
        engine.train_model("rgb_only", "unused", str(resumed), cfg, resume=True, gate_epoch=1)
    engine.train_model("rgb_only", "unused", str(resumed), cfg, resume=True, gate_epoch=1, continue_after_gate=True)
    a = torch.load(full / "rgb_only/last.pt", weights_only=False)
    b = torch.load(resumed / "rgb_only/last.pt", weights_only=False)
    for key, tensor in a["state_dict"].items():
        assert torch.equal(tensor, b["state_dict"][key])
    for key, tensor in a["training_state"]["model"].items():
        assert torch.equal(tensor, b["training_state"]["model"][key])
    assert a["training_state"]["iters"] == b["training_state"]["iters"] == 6
    assert len(json.loads((resumed / "rgb_only/history.json").read_text())) == 3


def test_resume_rejects_changed_experiment_and_accidental_overwrite(tmp_path, monkeypatch):
    cfg = config(monkeypatch)
    engine.train_model("rgb_only", "unused", str(tmp_path), cfg, epochs=1)
    with pytest.raises(FileExistsError):
        engine.train_model("rgb_only", "unused", str(tmp_path), cfg, epochs=1)
    with pytest.raises(ValueError, match="same dataset"):
        engine.train_model("rgb_only", "unused", str(tmp_path), cfg, epochs=2, resume=True)
    changed = copy.deepcopy(cfg)
    changed["sensors"]["lwir_noise"]["netd_mk"] = [20, 30]
    with pytest.raises(ValueError, match="same dataset"):
        engine.train_model("rgb_only", "unused", str(tmp_path), changed, epochs=1, resume=True)
