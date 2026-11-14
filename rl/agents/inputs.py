"""M3-02, M3-04: what the actor and the critic receive.

Critic variants (configuration switches): symmetric, gated, ungated, cooled LWIR only, MWIR only, shuffled, without h,
truth. Privileged maps come from info["privileged"]; the gate (Filters 1 and 2, rl/PLAN.md Safeguards) uses
info["truth"]["confirmable"] and the shared MWIR variant (evidence_mwir_shared).
"""
from __future__ import annotations

CRITIC_VARIANTS = ("symmetric", "gated", "ungated", "cooled_only", "mwir_only", "shuffled", "without_h", "truth")


def actor_input(obs: dict):
    raise NotImplementedError("M3-02: actor input with the I1 audit")


def critic_input(obs: dict, info: dict, variant: str):
    raise NotImplementedError("M3-04: critic variants")
