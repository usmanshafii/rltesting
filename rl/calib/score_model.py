"""M2-06: detection scores per frame whose detection probability at the threshold for 0.01 false alarms per
frame matches sigmoid(a + b ln d'), a = -5.24, b = 1.40 (rl/config/frozen.yaml detector); frame-level false
alarms are spread over the cells the frame covers (the environment does this with the frame share g).

Done when: the simulated curve lies inside a bootstrap interval of the 55 scenes in detector_vs_ideal.json over
d' 5-500, and drone-free frames give 0.01 +- 0.002 false alarms per frame.
"""
from __future__ import annotations


def pd_from_dprime(band: str, dprime: float) -> float:
    raise NotImplementedError("M2-06: score model")
