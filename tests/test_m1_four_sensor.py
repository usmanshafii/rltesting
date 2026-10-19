"""M1 four-sensor revision (tools/m1_det2.py, tools/member_job.py): coverage planes and masks, coverage-aware
preprocessing, the crop sampler's masks and visible labels, the cooled cameras' noise, the two laptops' part zips and
result fingerprints, the code bundle, and the launcher files."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

torch = pytest.importorskip("torch")
from models import cnn_detector as cd  # noqa: E402


# ============================================================================ model and preprocessing
def test_coverage_planes_reduce_to_the_old_flags_and_see_partial_coverage():
    torch.manual_seed(0)
    m = cd.CNNDetector(["rgb", "lwir", "lwir_cooled", "mwir"], presence=True).eval()
    x = {b: torch.randn(2, cd.BAND_CHANNELS[b], 64, 80) for b in m.bands}
    ones = {b: torch.ones(2, 1, 64, 80) for b in m.bands}
    with torch.no_grad():
        a, b = m(x)["hm"], m(x, None, ones)["hm"]
        part = dict(ones, lwir_cooled=ones["lwir_cooled"].clone())
        part["lwir_cooled"][..., :24] = 0.0
        c = m(x, None, part)["hm"]
        flags = {bd: torch.tensor([1.0, 0.0]) for bd in m.bands}
        d, e = m(x, flags)["hm"], m(x, flags, ones)["hm"]
    assert torch.equal(a, b)                         # full coverage == the constant band-present plane
    assert torch.equal(d, e)                         # with dropout flags as well
    assert not torch.allclose(a, c)                  # a partial mask reaches the network


def test_preprocess_inside_the_covered_box_only():
    rng = np.random.default_rng(0)
    H, W = 128, 160
    box = (20, 100, 30, 130)
    raw = np.zeros((H, W, 1), np.float32)
    raw[box[0]:box[1], box[2]:box[3]] = rng.random((80, 100, 1), dtype=np.float32)
    out = cd.preprocess({"mwir": raw}, cd.PREPROCESS_V2, coverage={"mwir": box})["mwir"]
    alone = cd.preprocess({"mwir": raw[box[0]:box[1], box[2]:box[3]]}, cd.PREPROCESS_V2)["mwir"]
    assert out.shape == (H, W, 1)
    assert np.array_equal(out[box[0]:box[1], box[2]:box[3]], alone)       # the empty border never enters the stats
    outside = np.ones((H, W), bool)
    outside[box[0]:box[1], box[2]:box[3]] = False
    assert not np.any(out[outside, 0])
    full = cd.preprocess({"mwir": raw}, cd.PREPROCESS_V2, coverage={"mwir": (0, H, 0, W)})["mwir"]
    assert np.array_equal(full, cd.preprocess({"mwir": raw}, cd.PREPROCESS_V2)["mwir"])


def test_predict_uses_coverage_masks_only_for_models_with_planes():
    rng = np.random.default_rng(1)
    frames = {"lwir_cooled": rng.normal(0, 0.1, (128, 160, 1)).astype(np.float32)}
    cov = {"lwir_cooled": (32, 96, 40, 120)}
    torch.manual_seed(1)
    m = cd.CNNDetector(["lwir_cooled"], presence=True).eval()
    masks = cd.coverage_masks(cov, (128, 160), m.bands)
    assert masks["lwir_cooled"].sum() == 64 * 80
    a = cd.predict(m, frames, conf=0.0, masks=masks)
    b = cd.predict(m, frames, conf=0.0)
    assert a["scores"] != b["scores"]
    plain = cd.CNNDetector(["lwir"]).eval()                       # no planes: masks are ignored
    f2 = {"lwir": frames["lwir_cooled"]}
    assert cd.predict(plain, f2, conf=0.0, masks={"lwir": masks["lwir_cooled"]}) == cd.predict(plain, f2, conf=0.0)


# ============================================================================ the experiment's rules
def test_models_roles_and_planes():
    import m1_det2 as D
    assert set(D.ORDER) == {"lwir_only", "lwir_cooled_only", "mwir_only", "rgb_lwir", "teacher", "student", "student_ctrl"}
    assert D.SPECS["teacher"][0] == ["rgb", "lwir", "lwir_cooled", "mwir"]          # CLAUDE.md: teacher sees all four
    assert D.SPECS["student"][0] == ["rgb", "lwir"] and D.SPECS["student"][1] == "teacher"
    assert D.SPECS["student_ctrl"] == (["rgb", "lwir"], "rgb_lwir", True)           # the control: no privileged sensor
    assert "swir" not in {b for v in D.SPECS.values() for b in v[0]}
    assert sorted(D.ROLES["A"]["models"] + D.ROLES["B"]["models"]) == sorted(D.ORDER)
    for role in ("A", "B"):                          # a role's teachers are trained on the same laptop, earlier
        done = []
        for n in D.ROLES[role]["models"]:
            assert D.SPECS[n][1] is None or D.SPECS[n][1] in done
            done.append(n)
    assert D.planes("teacher") and D.planes("lwir_cooled_only") and D.planes("mwir_only") and D.planes("student")
    assert not D.planes("lwir_only")
    assert D.PARTS["A"][1] == D.PARTS["B"][0] and D.PARTS["A"][0] == 0 and D.PARTS["B"][1] == D.N_POP


def test_cooled_cameras_do_not_get_the_uncooled_stripe_terms():
    import m1_det2 as D
    from data.dataset import thermal_camera_noise
    S = D.cfgs()["sensors"]["lwir_noise"]
    assert "column_gain_frac" in S                   # the M1 loader adds them for the uncooled camera
    cooled = dict(netd_mk=15.0, band_um=[7.7, 9.3], pixel_fpn_k=0.02, column_fpn_k=0.004)
    c = thermal_camera_noise(S, "lwir_cooled", cooled)
    assert not any(k in c for k in ("column_gain_frac", "column_drift_k", "row_noise_k", "column_temporal_k"))
    assert c["pixel_fpn_k"] == [0.02, 0.02] and c["netd_mk"] == [15.0, 15.0]
    unc = thermal_camera_noise(S, "lwir", dict(netd_mk=40.0, band_um=[8, 14], row_temporal_k=0.01, column_temporal_k=0.005))
    assert unc["column_gain_frac"] == S["column_gain_frac"] and unc["row_noise_k"] == [0.01, 0.01]


def _fake_train_cache(out, cov_cooled, boxes):
    """A one-scene 'train' cache: each band's channel is 1 inside its camera's coverage, 0 outside."""
    import m1_det2 as D
    d = out / "cache" / "train"
    d.mkdir(parents=True)
    H, W = 512, 640
    cov = {b: (0, H, 0, W) for b in D.BANDS}
    cov["lwir_cooled"] = cov_cooled
    arr = np.zeros((H, W, 6), np.float16)
    for b, sl in D.KD_CH.items():
        y0, y1, x0, x1 = cov[b]
        arr[y0:y1, x0:x1, sl] = 1.0
    for name in ("s000001_r0", "s000001_a0"):
        np.save(d / f"{name}.npy", arr)
    info = dict(scene_id="s000001", k=0, boxes=boxes, coverage={b: list(v) for b, v in cov.items()},
                visible={b: [D.centre_inside(bx, cov[b]) for bx in boxes] for b in D.BANDS},
                cnr={b: [3.0] * len(boxes) for b in D.BANDS})
    (d / "s000001_r0.json").write_text(json.dumps(info))
    T = dict(D.TRAINING, realisations=1, absent_realisations=1, batch=24)
    return dict(splits=dict(train=["s000001"]), training=T)


def test_sampler_masks_follow_the_crop_and_hide_unseen_targets(tmp_path):
    import m1_det2 as D
    cov = (100, 412, 120, 520)                      # cooled camera sees the centre only
    m = _fake_train_cache(tmp_path, cov, [[20.0, 20.0, 26.0, 24.0]])     # target outside the cooled view
    s = D.Sampler(tmp_path, m)
    rng = np.random.default_rng(3)
    for _ in range(4):
        imgs, masks, boxes, cnrs, present, kinds = s.batch(rng, ["rgb", "lwir"], True, ("rgb", "lwir", "lwir_cooled", "mwir"))
        for b in ("lwir_cooled", "mwir"):
            assert torch.allclose(masks[b], imgs[b], atol=1e-3)        # mask = where the camera has data, pooled/flipped alike
        assert torch.all(masks["rgb"] == 1) and torch.all(masks["lwir"] == 1)
    # a single-camera cooled model is never taught the target its camera cannot see
    T = dict(m["training"], crop_mix=dict(positive=1.0, background=0.0, absent=0.0))
    s.T = T
    _, _, boxes, _, _, _ = s.batch(rng, ["lwir_cooled"], False, visible_only=True)
    assert all(len(b) == 0 for b in boxes)
    _, _, boxes, _, _, _ = s.batch(rng, ["rgb", "lwir"], False)
    assert all(len(b) == 1 for b in boxes)          # the multi-camera models keep it


def test_fingerprint_comparison_tolerates_last_bit_differences():
    import m1_det2 as D
    a = {"eval/s1_p0.npy": dict(sha256="x", mean=[0.1, 0.2], std=[1.0, 1.0]),
         "eval/s2_p0.npy": dict(sha256="y", mean=[0.1], std=[1.0])}
    b = {"eval/s1_p0.npy": dict(sha256="z", mean=[0.1001, 0.2], std=[1.0, 0.9999]),
         "eval/s2_p0.npy": dict(sha256="w", mean=[0.3], std=[1.0])}
    common, same, diff = D.compare_fingerprints(a, b)
    assert common == ["eval/s1_p0.npy", "eval/s2_p0.npy"] and same == [] and diff == ["eval/s2_p0.npy"]


# ============================================================================ the two laptops' files
def _fake_population(root, D, n, contract):
    src = root / D.SOURCE
    for i in range(n):
        sid = f"s{i:06d}"
        mods = {}
        for b in D.BANDS:
            p = src / "layers" / b / f"{sid}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(bytes([i, len(b)]) * 50)
            mods[b] = dict(layers=f"layers/{b}/{sid}.npz")
        (src / "params").mkdir(parents=True, exist_ok=True)
        (src / "params" / f"{sid}.json").write_text("{}")
        (src / "meta").mkdir(parents=True, exist_ok=True)
        (src / "meta" / f"{sid}.json").write_text(json.dumps(dict(scene_id=sid, modalities=mods,
                                                                  extra=dict(render_contract=contract))))


def test_part_zip_round_trip_and_refusals(tmp_path, monkeypatch):
    import m1_det2 as D
    contract = D.expected_contract()
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setattr(D, "PARTS", {"A": (0, 2), "B": (2, 3)})
    monkeypatch.setattr(D, "ROOT", a)
    _fake_population(a, D, 3, contract)
    z = D.pack_part("A", dest=tmp_path / "m1_part_A.zip")
    assert D.part_info(z)["part"] == "A" and len(D.part_info(z)["files"]) == 2 * (2 + len(D.BANDS))
    monkeypatch.setattr(D, "ROOT", b)                 # the other laptop
    D.unpack_part(z)
    assert D.scene_done(b / D.SOURCE, 0) and D.scene_done(b / D.SOURCE, 1) and not D.scene_done(b / D.SOURCE, 2)
    # damaged in transfer: one layer changed
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(z) as zi, zipfile.ZipFile(bad, "w") as zo:
        for n in zi.namelist():
            data = zi.read(n)
            zo.writestr(n, data[:-1] + b"!" if n.endswith("s000000.npz") and "/rgb/" in n else data)
    with pytest.raises(ValueError, match="damaged"):
        D.unpack_part(bad)
    # rendered with other configs: refused before anything is unpacked
    monkeypatch.setattr(D, "ROOT", a)
    other = dict(contract, config_sha256="0" * 64)
    _fake_population(a, D, 3, other)
    with pytest.raises(ValueError, match="other configs"):
        D.pack_part("A", dest=tmp_path / "x.zip")
    info = D.part_info(z)
    info["render_contract"] = other
    forged = tmp_path / "forged.zip"
    with zipfile.ZipFile(z) as zi, zipfile.ZipFile(forged, "w") as zo:
        for n in zi.namelist():
            zo.writestr(n, json.dumps(info) if n == "m1_part.json" else zi.read(n))
    with pytest.raises(ValueError, match="different configs"):
        D.unpack_part(forged)


def test_manifest_recheck_ignores_the_creation_date(tmp_path):
    import m1_det2 as D
    src = D.ROOT / D.SOURCE
    scenes = {p.stem for p in sorted((src / "meta").glob("s*.json"))[:4] if D.scene_done(src, int(p.stem[1:]))}
    if len(scenes) < 2:
        pytest.skip("fewer than 2 rendered M1 scenes in this checkout")
    D.manifest(tmp_path, scenes=scenes, smoke=True)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["models"]["teacher"]["coverage_planes"] and m["sensors"]["bands"] == list(D.BANDS)
    m["created"] = "2000-01-01"                       # frozen on the other laptop, another day
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    D.manifest(tmp_path, scenes=scenes, smoke=True)  # no exception


def test_code_bundle_holds_what_a_fresh_copy_needs(tmp_path):
    import shutil
    import subprocess
    if shutil.which("git") is None or subprocess.run(["git", "rev-parse"], cwd=ROOT, capture_output=True).returncode:
        pytest.skip("not a git checkout")
    import m1_det2 as D
    z = D.export_code(tmp_path / "code.zip")
    names = set(zipfile.ZipFile(z).namelist())
    for need in ("fyp/member/_job.bat", "fyp/member/RUN_ME.bat", "fyp/tools/m1_det2.py", "fyp/configs/m1.yaml",
                 "fyp/docs/sources/astm/e490_am0_model_grid.csv", "fyp/member/code_bundle.json",
                 "fyp/docs/sources/noaa_isd/lahore_hourly_conditions_2022-08-25_2025-08-24.csv"):
        assert need in names, need
    assert not any(n.startswith("fyp/out/") for n in names)
    assert not any(n.lower().endswith((".pdf", ".xls", ".par")) for n in names)       # third-party documents stay local
    info = json.loads(zipfile.ZipFile(z).read("fyp/member/code_bundle.json"))
    assert info["render_contract"] == D.expected_contract()


def test_batch_files_have_crlf_line_endings():
    """cmd.exe misreads labels in LF-only batch files (.gitattributes); the code bundle copies the working tree."""
    bats = sorted(ROOT.glob("*.bat")) + sorted((ROOT / "member").glob("*.bat"))
    assert bats
    for p in bats:
        b = p.read_bytes()
        assert b.count(b"\n") == b.count(b"\r\n"), p.name


def test_member_job_finds_the_newest_valid_zip(tmp_path, monkeypatch):
    import os
    import member_job as J
    monkeypatch.setattr(J, "search_places", lambda: [tmp_path])
    old, new, bad = tmp_path / "m1_part_A.zip", tmp_path / "m1_part_A (1).zip", tmp_path / "m1_part_A (2).zip"
    for p in (old, new, bad):
        p.write_bytes(b"x")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    os.utime(bad, (3, 3))

    def check(p):
        if p == bad:
            raise ValueError("cut off")
    assert J.find_zip("m1_part_A*.zip", check) == new
    assert J.find_zip("m1_part_B*.zip", check) is None


def test_e490_model_grid_table_matches_the_workbook():
    from thermal import atmosphere_bands as atb
    import csv
    path = ROOT / "docs" / "sources" / "astm" / "e490_am0_model_grid.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert np.array_equal([float(r["wavenumber_cm1"]) for r in rows], atb.W_GRID)
    wb = ROOT / "docs" / "sources" / "astm" / "ASTM_E490_AM0_from_NREL_workbook.csv"
    if not wb.exists():
        pytest.skip("the git-ignored workbook export is not in this checkout")
    import source_measurements as sm
    fresh = [float(r["e490_w_m2_per_cm1"]) for r in sm.e490_model_grid()]
    assert fresh == [float(r["e490_w_m2_per_cm1"]) for r in rows]


# ============================================================================ the case taxonomy (2026-10-17)
def test_teacher_sees_every_sensor_subset_and_the_students_keep_their_dropout(tmp_path):
    import m1_det2 as D
    assert D.SPECS["teacher"][2] == "subsets" and len(D.PROPER_SUBSETS) == 14 and len(D.SUBSETS) == 15
    assert set(D.ROLES["solo"]["models"]) == set(D.ORDER)
    assert D.ROLES["solo"]["models"].index("teacher") < D.ROLES["solo"]["models"].index("student")
    m = _fake_train_cache(tmp_path, (0, 512, 0, 640), [[300.0, 200.0, 306.0, 204.0]])
    s = D.Sampler(tmp_path, m)
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(40):
        _, _, _, _, present, _ = s.batch(rng, list(D.BANDS), "subsets")
        for i in range(m["training"]["batch"]):
            kept = tuple(b for b in D.BANDS if float(present[b][i]) > 0)
            assert kept                                                   # never an empty input
            seen.add(kept)
    assert len(seen) == 15                                                # all four, and every proper subset
    rng = np.random.default_rng(12)
    for _ in range(10):                                                   # two-band models: never both bands dropped
        _, _, _, _, present, _ = s.batch(rng, ["rgb", "lwir"], True)
        assert float((present["rgb"] + present["lwir"]).min()) >= 1.0


def test_signal_classes_and_regime_cases():
    import m1_det2_eval as E
    assert [E.signal_class(v) for v in (0.5, 2.0, 4.9, 5.0, 12.0)] == ["N", "F", "F", "D", "D"]   # d' (eval/detectability)
    c = dict(rgb=1.5, lwir=2.0, lwir_cooled=5.0, mwir=0.2)                # student faint, cooled LWIR detectable
    hits = [k for k, rule in E.REGIME_CASES.items() if rule(c)]
    assert "student faint, privileged detectable" in hits and "one sensor detectable" in hits
    assert "student sensors detectable" not in hits
    assert [k for k, rule in E.REGIME_CASES.items() if rule(dict(rgb=0.1, lwir=0.2, lwir_cooled=0.3, mwir=0.1))] == ["no signal in any sensor"]
    assert set(E.TEACHER_VARIANTS) == {"@rgb_only", "@lwir_only", "@lwir_cooled_only", "@mwir_only", "@rgb_lwir"}
