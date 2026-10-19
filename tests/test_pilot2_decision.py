"""Section 6d's decision rule (tools/pilot2_decision.py): the 3-point margin read on the hierarchical interval."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools")]

import pilot2_decision as P  # noqa: E402


def test_the_margin_rule():
    assert P.MARGIN == 0.03
    assert P.decide(0.035, 0.09)["verdict"] == "a gain of at least 3 points shown"
    assert P.decide(0.010, 0.07)["verdict"] == "a gain shown, smaller than 3 points or not shown to reach them"
    small = P.decide(0.005, 0.025)                      # a small, real gain: shown, and shown to be below 3 points
    assert small["gain_shown"] and small["no_gain_of_3_points"]
    assert P.decide(-0.02, 0.02)["verdict"] == "no gain of 3 points or more"
    assert P.decide(-0.05, 0.08)["verdict"] == "inconclusive"


def test_section_6c_grades():
    assert P.grade([0.05, 0.04, 0.06, 0.01, 0.03], 0.01, 0.99) == "shown"
    assert P.grade([0.05, -0.04, 0.06, 0.01, -0.03], -0.01, 0.85) == "suggestive"
    assert P.grade([0.05, -0.04, -0.06], -0.10, 0.40) == "not shown"
