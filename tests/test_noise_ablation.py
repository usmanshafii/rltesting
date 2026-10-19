"""The noise-model verification (tools/noise_ablation.py): each variant is what its description says."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

import noise_ablation as A  # noqa: E402


def test_variant_configs():
    cam = A.variant_configs("camera")["sensors"]["camera_pipelines"]
    phys = A.variant_configs("physics")["sensors"]["camera_pipelines"]
    assert cam["enabled"] is True and phys["enabled"] is True                # on for these variants, off by default
    assert A.variant_configs("mixed")["sensors"]["camera_pipelines"]["enabled"] is True
    assert phys["rgb"]["nr_level"] == [0, 0] and phys["rgb"]["aperture_level"] == [0, 0]
    assert phys["rgb"]["chromatic_aberration"] == [0.0, 0.0] and phys["rgb"]["read_tukey_lambda"] == cam["rgb"]["read_tukey_lambda"]
    for c in ("lwir_uncooled", "lwir_cooled", "mwir"):
        assert phys[c]["agc"]["histogram_prob"] == 0.0 and phys[c]["filter_attenuation"] == [0.0, 0.0]
        assert phys[c]["defects"] == cam[c]["defects"]                       # the sensor physics is the same
    for v in ("previous", "gaussian", "foreign"):
        assert "camera_pipelines" not in A.variant_configs(v)["sensors"]
    assert "column_gain_frac" in A.variant_configs("previous")["sensors"]["lwir_noise"]   # the v2 stripe terms


def test_foreign_stage_and_gaussian_sensor():
    rng = np.random.default_rng(0)
    rgb = np.linspace(0, 1, 64 * 80 * 3, dtype=np.float32).reshape(64, 80, 3)
    gray = rgb[..., 0].copy()
    a = A._foreign_post(rgb, np.random.default_rng(1))
    b = A._foreign_post(rgb, np.random.default_rng(1))
    assert a.shape == rgb.shape and np.array_equal(a, b) and 0 <= a.min() and a.max() <= 1
    assert A._foreign_post(gray, rng).shape == gray.shape
    S = A.variant_configs("gaussian")["sensors"]
    meta = dict(camera=dict(exposure_ms=5.0), conditions=dict(air_temp_c=25.0))
    f = A._gaussian_sensor("rgb", dict(read_noise_e=2.0), meta, np.random.default_rng(2), S)
    out = f(np.full((32, 40, 3), 5.0, np.float32))
    assert out.shape == (32, 40, 3) and 0 <= out.min() and out.max() <= 1
    g = A._gaussian_sensor("lwir", dict(netd_mk=40.0, band_um=[8.0, 14.0]), meta, np.random.default_rng(3), S)
    t = g(np.full((32, 40), 5.0, np.float32))
    col = t.mean(0)
    assert t.shape == (32, 40) and col.std() < 0.25 * t.std() + 1e-6                # white noise: no column pattern


def test_pooled_analysis_and_saved_scores(tmp_path, monkeypatch):
    """The seed-pooled comparison finds a clearly better model and the saved scores round-trip exactly."""
    rng = np.random.default_rng(4)
    n = 40
    arrays = {}
    for s in (29, 31):
        for tv in A.TRAIN_VARIANTS:
            for sv in A.TEST_VARIANTS:
                hit = 0.9 if tv == "camera" else 0.3                       # camera finds most targets, others few
                best = np.where(rng.random(n) < hit, rng.uniform(0.6, 0.9, n), rng.uniform(0.0, 0.2, n))
                absent = [[rng.uniform(0.0, 0.3, 3) for _ in range(4)] for _ in range(n)]
                arrays[(s, tv, sv)] = (best, absent)
    P = A.pooled(arrays, (29, 31), [0.005, 0.01, 0.02, 0.05], n_boot=200)
    cam = {k: v for k, v in P["diffs"].items() if k.startswith("camera-")}
    assert len(cam) == 4 * len(A.TEST_VARIANTS) and all(v["significant"] and v["diff"] > 0 for v in cam.values())
    assert not any(k.startswith("mixed-camera") for k in P["diffs"])            # each pair once
    j = P["leave_one_scene_out"]["camera"]
    assert j["worst"][0] <= P["worst"]["camera"] <= j["worst"][1]
    assert P["best_worst_case_share"]["camera"] > 0.95 and abs(sum(P["best_worst_case_share"].values()) - 1) < 1e-9
    monkeypatch.setattr(A, "OUT", tmp_path)
    A.save_scores(arrays)
    back = A.load_scores()
    k = (31, "physics", "foreign")
    assert np.array_equal(back[k][0], arrays[k][0]) and all(np.array_equal(x, y) for x, y in zip(back[k][1][3], arrays[k][1][3]))


def test_mixed_cache_links_the_four_models(tmp_path, monkeypatch):
    """The mixed variant draws each training frame and each validation scene from one of the four models' caches."""
    monkeypatch.setattr(A, "OUT", tmp_path)
    split = dict(train=["s000001", "s000002", "s000003"], val=["s000004"], test=[])
    A.D.atomic_json(tmp_path / "split.json", split)
    for v in A.NOISE_MODELS:
        tr, ev = tmp_path / v / "cache" / "train", tmp_path / v / "cache" / "eval"
        tr.mkdir(parents=True)
        ev.mkdir(parents=True)
        for sid in split["train"]:
            for k in range(A.TRAINING["realisations"]):
                np.save(tr / f"{sid}_r{k}.npy", np.full((2, 2), A.NOISE_MODELS.index(v), np.float16))
                (tr / f"{sid}_r{k}.json").write_text("{}")
            np.save(tr / f"{sid}_a0.npy", np.full((2, 2), A.NOISE_MODELS.index(v), np.float16))
        (ev / "s000004.json").write_text("{}")
        np.save(ev / "s000004_p0.npy", np.full((2, 2), A.NOISE_MODELS.index(v), np.float16))
        np.save(ev / "s000004_a0.npy", np.full((2, 2), A.NOISE_MODELS.index(v), np.float16))
        A.D.atomic_json(tmp_path / v / "manifest.json", dict(experiment=v, training={}))
    A.mix_cache()
    picks = A.D.read(tmp_path / "mixed" / "mix.json")["picks"]
    assert len(picks) == 3 * (A.TRAINING["realisations"] + 1) + 1
    for key, v in picks.items():
        part, stem = key.split("/")
        f = tmp_path / "mixed" / "cache" / part / (stem + ".npy" if part == "train" else stem + "_p0.npy")
        assert np.load(f)[0, 0] == A.NOISE_MODELS.index(v)                    # the frame comes from the drawn model
    assert A.D.read(tmp_path / "mixed" / "manifest.json")["experiment"] == "noise-ablation-mixed"
