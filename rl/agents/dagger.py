"""M3-06: DAgger with a step mask on the gate flag o: the student imitates the teacher only where
info["truth"]["confirmable"] is true, and learns from reward elsewhere. Teachers: four-band PPO (ungated) and the
gated teacher on rl.contracts.BAND_SETS["four_shared"] (MWIR without the plume).

Done when: the imitation loss is zero on steps with o = 0.
"""
from __future__ import annotations


def dagger(config: dict) -> dict:
    raise NotImplementedError("M3-06: DAgger")
