"""M1 scope, provenance and completion gates: no Blender/GPU required."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from data.provenance import atomic_json, check_scene, config_identity, render_contract
from render.sampling import assign_splits, load_configs


def test_m1_preserves_protected_baseline_and_scopes_experiment():
    base, m1 = load_configs(overlay=""), load_configs(overlay="configs/m1.yaml")
    assert m1["target"] == base["target"]
    assert m1["render"]["target"] == base["render"]["target"]
    assert m1["signatures"] == base["signatures"]
    assert m1["render"]["dataset"]["families"] == {"open_sky": 1, "horizon": 0}
    assert m1["render"]["dataset"]["lighting"] == {"day": .5, "dusk": 0, "night": .5}
    # the four-sensor M1 (2026-10-15): the project's camera rig and Lahore conditions, no hard negatives
    assert m1["sensors"]["cameras"] == base["sensors"]["cameras"] and m1["sensors"]["cameras"]
    assert m1["render"]["climate"]["enabled"] and m1["render"]["thermal_atmosphere"]["enabled"]
    assert m1["render"]["negatives"]["count"] == [0, 0]
    assert m1["render"]["render"]["engine_samples"] == 128
    splits = assign_splits(600, m1["render"]["dataset"]["splits"])
    assert [splits.count(s) for s in ("train", "val", "test")] == [360, 60, 180]
    assert set(splits[:8]) == {"train", "val", "test"}


def test_render_identity_detects_samples_bands_splits_and_ignores_threads(tmp_path):
    cfg = load_configs(overlay="configs/m1.yaml")
    c = render_contract(cfg, 600, ("rgb", "lwir"))
    for b in ("rgb", "lwir"):
        (tmp_path / f"{b}.npz").touch()
    m = dict(scene_id="s000000", extra=dict(render_contract=c),
             modalities={b: dict(layers=f"{b}.npz") for b in ("rgb", "lwir")})
    assert check_scene(m, tmp_path, c)
    threaded = copy.deepcopy(cfg)
    threaded["render"]["render"]["threads"] = 8
    assert config_identity(threaded) == config_identity(cfg)
    changed = copy.deepcopy(cfg)
    changed["render"]["render"]["engine_samples"] = 16
    for wrong in (render_contract(changed, 600, ("rgb", "lwir")),
                  render_contract(cfg, 400, ("rgb", "lwir")), render_contract(cfg, 600, ("rgb",))):
        with pytest.raises(ValueError, match="configuration"):
            check_scene(m, tmp_path, wrong)
    (tmp_path / "rgb.npz").unlink()
    assert not check_scene(m, tmp_path, c)


def test_render_identity_ignores_loader_only_sensor_settings_but_not_nominal_values():
    """Noise models and camera processing act in the loader: tuning them must not invalidate rendered scenes
    (render_identity), though it changes a trained experiment (config_identity). The nominal sensor, whose
    contrast-to-noise the writer stores in the metadata, stays part of the render identity."""
    from data.provenance import render_identity
    cfg = load_configs(overlay="configs/m1.yaml")
    base, full = render_identity(cfg), config_identity(cfg)
    tuned = copy.deepcopy(cfg)
    tuned["sensors"]["lwir_noise"]["column_fpn_k"] = [0.0, 0.5]
    tuned["sensors"]["rgb_noise"]["read_noise_e"] = [9.0, 9.5]
    tuned["sensors"]["registration"]["lwir"]["shift_px"] = 5.0
    tuned["sensors"]["camera_pipelines"]["rgb"]["nr_level"] = [0, 1]
    assert render_identity(tuned) == base and config_identity(tuned) != full
    assert render_contract(tuned, 600, ("rgb",))["config_sha256"] == base
    nominal = copy.deepcopy(cfg)
    nominal["sensors"]["lwir_noise"]["nominal"]["netd_mk"] = 99.0
    assert render_identity(nominal) != base
    psf = copy.deepcopy(cfg)
    psf["sensors"]["psf_sigma_px"]["lwir"] = [0.5, 0.9]                  # applied partly at render time
    assert render_identity(psf) != base


def test_atomic_json_never_replaces_good_metadata_with_invalid_data(tmp_path):
    path = tmp_path / "scene.json"
    atomic_json(path, dict(done=True))
    with pytest.raises(ValueError):
        atomic_json(path, dict(bad=float("nan")))
    assert json.loads(path.read_text()) == dict(done=True)


def test_audit_rejects_incomplete_production_dataset(tmp_path):
    from tools.m1 import audit, configs
    with pytest.raises(ValueError, match="expected 600"):
        audit(tmp_path, configs(), decode=False)
    assert json.loads((tmp_path / "audit.json").read_text())["status"] == "failed"


def test_bundle_verification_checks_bytes_and_rejects_escape(tmp_path):
    import hashlib
    from tools.m1 import verify_bundle
    path = tmp_path / "config.yaml"
    path.write_text("name: m1")
    atomic_json(tmp_path / "bundle_manifest.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest()})
    assert verify_bundle(tmp_path) == 1
    path.write_text("name: changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)
    atomic_json(tmp_path / "bundle_manifest.json", {"../escape": "x"})
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)


def test_fallback_adapter_uses_exported_pixels_and_original_coordinates():
    import torch
    from types import SimpleNamespace
    from eval.evaluate import predict_yolo_records
    from tools.export_yolo import to_png
    class Dataset:
        W, H = 4, 3
        def __len__(self):
            return 1
        def load(self, i):
            image = np.zeros((3, 4, 3), np.float32)
            image[..., 0], image[..., 1], image[..., 2] = .1, .2, .3
            return dict(scene_id="s0", images={"rgb": image}, visibility_km=40,
                        meta=dict(conditions={}, camera={}, extra={}),
                        objects=[dict(label=0, box=[0, 0, 3, 2])])
    class Model:
        def predict(self, **kwargs):
            expected = np.asarray(to_png(Dataset().load(0)["images"]["rgb"]))[..., ::-1]
            assert np.array_equal(kwargs["source"], expected)
            boxes = SimpleNamespace(xyxy=torch.tensor([[0., 0., 3., 2.]]),
                                    conf=torch.tensor([.9]), cls=torch.tensor([0.]))
            return [SimpleNamespace(boxes=boxes)]
    recs = predict_yolo_records(Model(), Dataset(), ["rgb"], dict(conf=.001, nms_iou=.5, max_det=100), "cpu")
    assert recs[0]["preds"]["boxes"].tolist() == [[0, 0, 3, 2]]
    assert recs[0]["gts"][0]["cls_name"] == "target"


def test_px50_keeps_missing_and_unidentifiable_results_explicit():
    from eval.pd_fit import summarise, px50
    empty = summarise([], n_boot=10)
    assert empty["px50_head"]["status"] == "insufficient_targets"
    misses = [dict(scene=f"s{i}", size_px=2**(1 + i / 5), detected=False, group="beam") for i in range(30)]
    assert summarise(misses, 10)["px50_beam"]["status"] == "no_detections"
    for i, d in enumerate(misses):
        d["detected"] = i > 15
    fit = summarise(misses, 10)
    assert fit["px50_head"]["status"] == "missing_aspect_group"
    assert np.isnan(fit["ratio_head_beam"]["value"])
    assert np.isfinite(fit["px50_beam"]["value"])
    assert np.isinf(px50(np.array([-1e8, 1e-5, 0, 0]), "beam"))


def test_generate_driver_marks_subprocess_failures(tmp_path, monkeypatch):
    from tools import generate_dataset as g
    monkeypatch.setattr(g, "find_blender", lambda _: "blender")
    monkeypatch.setattr(g, "run_chunk", lambda *args: (0, 1, 1, .1, "failed.log"))
    monkeypatch.setattr("sys.argv", ["generate_dataset", "--out", str(tmp_path), "--n", "1"])
    with pytest.raises(RuntimeError, match="Render incomplete"):
        g.main()


def test_generate_driver_detects_false_success_without_metadata(tmp_path, monkeypatch):
    from tools import generate_dataset as g
    monkeypatch.setattr(g, "find_blender", lambda _: "blender")
    monkeypatch.setattr(g, "run_chunk", lambda *args: (0, 1, 0, .1, "empty.log"))
    monkeypatch.setattr("sys.argv", ["generate_dataset", "--out", str(tmp_path), "--n", "1"])
    with pytest.raises(RuntimeError, match="1 missing scenes"):
        g.main()


def test_generate_checks_existing_config_before_launching_blender(tmp_path, monkeypatch):
    from tools import generate_dataset as g
    atomic_json(tmp_path / "meta/s000000.json", dict(scene_id="s000000", extra={}, modalities={}))
    def forbidden(*args):
        pytest.fail("Blender must not launch for incompatible output data")
    monkeypatch.setattr(g, "find_blender", forbidden)
    monkeypatch.setattr("sys.argv", ["generate_dataset", "--out", str(tmp_path), "--n", "1"])
    with pytest.raises(ValueError, match="provenance is missing"):
        g.main()
