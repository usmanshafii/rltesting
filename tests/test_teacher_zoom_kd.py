"""The zoomed teacher cameras (configs/sensors.yaml teacher_zoom) and the distillation that leaves out targets the student
cannot see (models/cnn_detector.cnr_gap_weight, kd_keep, response_kd; docs/privileged_teacher_case.md, section 6)."""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

from data import provenance  # noqa: E402
from data.dataset import apply_teacher_zoom, scene_camera  # noqa: E402
from models import cnn_detector as cd  # noqa: E402
from render import cameras  # noqa: E402
from render.sampling import load_configs  # noqa: E402

GRID = dict(width=640, height=512, hfov_deg=30.0, vfov_deg=cameras.other_axis_deg(30.0, 640, 512))
GRID["ifov_mrad"] = 1e3 * 2.0 * math.tan(math.radians(30.0) / 2.0) / 640


def _cooled():
    cam = cameras.zoom_to_cover(dict(width=640, height=512, pitch_um=15.0), [21.7, 550.0], GRID["hfov_deg"], GRID["vfov_deg"])
    cam.update(netd_mk=15.0, pixel_fpn_k=0.01, ifov_mrad=cam["pitch_um"] / cam["focal_mm"])
    cam["pixel_scale"] = cam["ifov_mrad"] / GRID["ifov_mrad"]
    return cam


def test_zoom_in_trades_coverage_for_pixels():
    cam = _cooled()
    z = cameras.zoom_in(cam, 2.0, [21.7, 550.0], GRID)
    assert math.isclose(z["focal_mm"], 2 * cam["focal_mm"]) and math.isclose(z["pixel_scale"], cam["pixel_scale"] / 2)
    for a in (0, 1):                                    # the covered tangent extent halves in both axes
        assert math.isclose(z["coverage"][a], min(1.0, cam["coverage"][a]) / 2, rel_tol=1e-3) or cam["coverage"][a] > 1
    assert z["netd_mk"] == cam["netd_mk"] and z["pixel_fpn_k"] == cam["pixel_fpn_k"]    # same camera, same noise
    long = cameras.zoom_in(cam, 1e3, [21.7, 550.0], GRID)
    assert math.isclose(long["focal_mm"], 550.0) and math.isclose(long["teacher_zoom"], 550.0 / cam["focal_mm"])


def test_apply_teacher_zoom_is_loader_only_deterministic_and_idempotent():
    C = load_configs()
    S = copy.deepcopy(C["sensors"])
    meta = dict(scene_id="s000042", extra=dict(cameras=dict(grid=GRID, cameras=dict(
        lwir_cooled=_cooled(), mwir=dict(_cooled(), zoom=True), lwir_uncooled=dict(width=640, coverage=[1.0, 1.0])))))
    assert apply_teacher_zoom(meta, S) is meta                                  # off by default
    S["teacher_zoom"] = dict(enabled=True, factor=[1.5, 3.0], cameras=["lwir_cooled"], seed=0)
    a, b = apply_teacher_zoom(meta, S), apply_teacher_zoom(meta, S)
    za = scene_camera(a, "lwir_cooled")["teacher_zoom"]
    assert 1.5 <= za <= 3.0 and za == scene_camera(b, "lwir_cooled")["teacher_zoom"]
    assert "teacher_zoom" not in scene_camera(a, "mwir") and scene_camera(a, "lwir") == scene_camera(meta, "lwir")
    assert apply_teacher_zoom(a, S) is a and "teacher_zoom" not in scene_camera(meta, "lwir_cooled")   # copy, once
    C2 = copy.deepcopy(C)
    C2["sensors"]["teacher_zoom"] = S["teacher_zoom"]
    assert provenance.render_identity(C2) == provenance.render_identity(C)     # rendered scenes stay valid


def test_distillation_leaves_out_targets_the_student_cannot_see():
    assert cd.cnr_gap_weight(8.0, 0.2) == 3.0                                 # the old weight peaked there
    assert cd.cnr_gap_weight(8.0, 0.2, student_floor=1.0, unseen=0.0) == 0.0
    assert cd.cnr_gap_weight(8.0, 2.0, student_floor=1.0, unseen=0.0) == cd.cnr_gap_weight(8.0, 2.0)
    yy, xx = np.mgrid[0:32, 0:32]
    g = lambda cy, cx: np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)  # noqa: E731
    keep = cd.kd_keep([np.stack([g(8, 8), g(24, 24)])], [[True, False]])
    assert keep.shape == (1, 1, 32, 32) and float(keep[0, 0, 8, 8]) == 0.0 and float(keep[0, 0, 24, 24]) == 1.0
    student = dict(hm=torch.full((1, 1, 32, 32), -6.0), reg=torch.zeros(1, 4, 32, 32))   # sure it sees only sky
    t_hm = torch.full((1, 1, 32, 32), -6.0)
    t_hm[0, 0, 8, 8] = 6.0                                                        # the teacher fires on the unseen target
    teacher = dict(hm=t_hm, reg=torch.zeros(1, 4, 32, 32))
    quiet = dict(hm=torch.full((1, 1, 32, 32), -6.0), reg=torch.zeros(1, 4, 32, 32))
    a = cd.response_kd(student, teacher, keep=keep)
    b = cd.response_kd(student, quiet, keep=keep)
    assert abs(float(a) - float(b)) < 1e-3                                       # that cell no longer teaches the student
    assert float(cd.response_kd(student, teacher)) > float(cd.response_kd(student, quiet)) + 0.1   # without keep it does
