"""M3-06: potential-based shaping, F = gamma * Phi(s') - Phi(s), with Phi = 0 at every episode end, including the
time limit (Ng, Harada and Russell, ICML 1999). Invariant I7.

Done when: shaping summed over any full episode equals -Phi(s0) to 1e-6.
"""
from __future__ import annotations


def potential(info: dict) -> float:
    raise NotImplementedError("M3-06: shaping potential")
