"""Distillation limited to the labelled targets (models/cnn_detector.kd_gate; docs/privileged_teacher_case.md 6c). Pilot 1's
students inherited the teacher's false alarms, because the response distillation weights the cells the teacher is
confident about. Inside the gate, a teacher false alarm away from the labels changes nothing."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from models import cnn_detector as cd  # noqa: E402


def _gauss():
    boxes = [np.array([[60.0, 60.0, 64.0, 63.0]]), np.zeros((0, 4))]      # a tiny target, and a target-free crop
    return cd.build_targets(boxes, 128, 128)[3]


def test_gate_covers_the_target_and_nothing_far_from_it():
    g = cd.kd_gate(_gauss())
    assert tuple(g.shape) == (2, 1, 32, 32)
    assert g[0, 0, 15, 15] == 1 and g[0, 0, 17, 15] == 1 and g[0, 0, 15, 13] == 1      # centre cell and 2 cells away
    assert g[0, 0, 3, 28] == 0 and float(g[0].sum()) < 60
    assert float(g[1].sum()) == 0                                                     # no target: no distillation


def test_gated_response_distillation_ignores_teacher_false_alarms():
    gate = cd.kd_gate(_gauss())
    torch.manual_seed(0)
    s = dict(hm=torch.randn(2, 1, 32, 32) - 3.0, reg=torch.randn(2, 4, 32, 32))
    t = dict(hm=torch.full((2, 1, 32, 32), -6.0), reg=torch.zeros(2, 4, 32, 32))
    t["hm"][0, 0, 15, 15] = 4.0                          # the teacher finds the target ...
    fa = {k: v.clone() for k, v in t.items()}
    fa["hm"][0, 0, 3, 28] = 5.0                          # ... and fires on clutter far from it and on the empty crop
    fa["hm"][1, 0, 20, 5] = 5.0
    fa["reg"][:, :, 3, 28] = 1.0
    gated = [float(cd.response_kd(s, x, keep=gate)) for x in (t, fa)]
    plain = [float(cd.response_kd(s, x)) for x in (t, fa)]
    assert abs(gated[0] - gated[1]) < 1e-6               # gated: the false alarms are not taught
    assert abs(plain[0] - plain[1]) > 1e-3               # ungated (pilot 1): they are
    assert gated[0] > 0                                  # and the target is still distilled


def test_gated_feature_mask_has_no_background():
    gauss = _gauss()
    gate = cd.kd_gate(gauss)
    m = cd.kd_mask(gauss, [[1.0], []], 0.05, gate=gate)
    assert float(m[1].sum()) == 0 and float(m[0][gate[0] == 0].sum()) == 0
    assert float(cd.kd_mask(gauss, [[1.0], []], 0.05)[1].sum()) > 0      # ungated: the floor covers the background
