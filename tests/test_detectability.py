"""eval/detectability.py: the matched-filter d' recovers a known target's detectability and separates sensor noise
from the scene's own structure."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import detectability as dt  # noqa: E402


def _scene(clutter=0.0, seed=0, amp=3.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:160, 0:160]
    target = amp * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / 2.0)[..., None]
    background = clutter * np.sin(xx / 3.0)[..., None]
    A, A2 = rng.standard_normal((160, 160, 1)) + background, rng.standard_normal((160, 160, 1)) + background
    zero_mean = target[74:87, 73:88] - target[74:87, 73:88].mean()
    return A + target, A, A2, float(np.sqrt((zero_mean ** 2).sum()))


def test_dprime_in_white_noise_is_the_matched_filter_snr():
    P, A, A2, expected = _scene()
    d = dt.dprime(P, A, A2, (79.0, 79.0, 81.0, 81.0))
    assert abs(d["d"] / expected - 1) < 0.1 and abs(d["noise_share"] - 1) < 0.15
    assert dt.signal_class(d["d"]) == "D" and dt.signal_class(1.0) == "N"


def test_clutter_lowers_dprime_and_only_the_noise_share_averages():
    P, A, A2, _ = _scene(clutter=2.0)
    d = dt.dprime(P, A, A2, (79.0, 79.0, 81.0, 81.0))
    assert d["noise_share"] < 0.3 and d["d"] < d["d_noise"]
    assert dt.d_after(d, 1) == np.float64(dt.d_after(d, 1)) and abs(dt.d_after(d, 1) - d["d"]) < 1e-6
    assert d["d"] < dt.d_after(d, 8) < d["d"] * np.sqrt(8)       # averaging helps, but static clutter stays
    none = dt.dprime(A, A, A2, (79.0, 79.0, 81.0, 81.0))
    assert none["d"] == 0.0                                       # no target, no signal


def test_logistic_fit_recovers_the_half_detection_point():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import detector_vs_ideal as V
    rng = np.random.default_rng(3)
    d = np.exp(rng.uniform(np.log(2), np.log(500), 4000))
    p = 1 / (1 + np.exp(-(np.log(d) - np.log(40.0)) * 3.0))      # half detection at d' = 40
    a, b = V._fit(np.log(d), (rng.random(len(d)) < p).astype(float))
    assert b > 0 and abs(np.exp(-a / b) / 40.0 - 1) < 0.1
