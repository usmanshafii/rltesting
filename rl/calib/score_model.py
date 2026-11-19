"""M2-06: Gaussian surrogate of the *historical* frozen detector curve.

The curve was measured for RGB+LWIR jointly, not separately per camera. Applying
it per band is a grade-C surrogate. docs/privileged_teacher_case.md section 3
also corrects the achieved test FA rate: this is not a measured 0.01-FA ROC.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from scipy.special import expit, ndtri
from rl import contracts as C


def pd_from_dprime(band: str, dprime: float) -> float:
    if band not in C.MODELLED_BANDS:
        raise ValueError(f"unmodelled band: {band}")
    d = float(dprime)
    if not math.isfinite(d) or d < 0:
        raise ValueError("dprime must be finite and nonnegative")
    if d == 0:
        return 0.0
    return float(expit(C.FROZEN["detector"]["logistic_a"] + C.FROZEN["detector"]["logistic_b"] * math.log(d)))


class GaussianScoreModel:
    """Null score N(0,1); positive mean chosen to reproduce the frozen logistic.

    A zero-signal frame uses the null distribution. The target-only probability
    pd_from_dprime(0) remains zero for the environment's independent union with FA.
    Scores are surrogate standardized scores, not raw CNN confidences.
    """
    def __init__(self, band: str = "lwir"):
        if band not in C.MODELLED_BANDS:
            raise ValueError(band)
        self.band = band
        self.fa = float(C.FROZEN["detector"]["fa_per_frame"])
        self.threshold = NormalDist().inv_cdf(1-self.fa)

    def mean(self, dprime: float) -> float:
        p = pd_from_dprime(self.band, dprime)
        if dprime == 0:
            return 0.0
        # Exact historical positive-frame curve; the zero-signal case is the null.
        q = p
        return self.threshold + float(ndtri(np.clip(q, 1e-12, 1-1e-12)))

    def detection_probability(self, dprime: float) -> float:
        return pd_from_dprime(self.band, dprime)

    def sample(self, rng: np.random.Generator, dprime: float, size=None):
        return rng.normal(self.mean(dprime), 1.0, size=size)

    def cell_false_alarm_probabilities(self, weights):
        """Partition a whole-frame at-least-one probability exactly, even off-grid.

        weights are nonnegative covered fractions of the *whole* frame, sum <= 1.
        This makes product(1-p_cell)=1-fa for full coverage. SkyEnv v0 instead uses
        fa*g (first-order); switching it needs an M1 change request.
        """
        w = np.asarray(weights, float)
        if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() > 1+1e-12:
            raise ValueError("frame weights must be finite, nonnegative and sum <= 1")
        return -np.expm1(np.log1p(-self.fa) * w)
