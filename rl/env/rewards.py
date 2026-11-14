"""Reward (task M1-06). A pure function of ground-truth events: it takes no sensor data, privileged or not
(invariant I6, checked by tests/rl/test_invariants.py on this signature).

    correct confirmation   +correct + earliness_weight * (share of the approach still left)
    false confirmation     false_confirmation (each; the episode continues)
    missed drone           missed_confirmable, only if the drone was confirmable at some step
    every step             step
    every slew             slew
"""
from __future__ import annotations

from rl import contracts as C

R = C.FROZEN["reward"]


def reward_terms(*, correct: bool, earliness: float, false_confirmations: int, missed_confirmable: bool,
                 slewed: bool) -> dict[str, float]:
    """Each reward component of one step; their sum is the step's reward. Keys = contracts.REWARD_TERMS."""
    return {
        "correct": float(R["correct"]) if correct else 0.0,
        "earliness": float(R["earliness_weight"]) * max(0.0, min(1.0, earliness)) if correct else 0.0,
        "false": float(R["false_confirmation"]) * int(false_confirmations),
        "missed": float(R["missed_confirmable"]) if missed_confirmable else 0.0,
        "step": float(R["step"]),
        "slew": float(R["slew"]) if slewed else 0.0,
    }
