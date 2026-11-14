"""M4-04: run a policy on a frozen episode set; reruns at five thresholds; metrics (confirmation rate, range at
confirmation, false confirmations per hour, return), by observability class; the leak test.

Done when: rerunning an evaluation reproduces every number to the last digit.
"""
from __future__ import annotations


def evaluate(policy, episodes: list, threshold_A: float | None = None) -> dict:
    raise NotImplementedError("M4-04: evaluation runner")
