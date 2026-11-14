"""M3-01, M3-03: single-file PPO in CleanRL style (credit CleanRL), actor and critic with separate inputs.

The actor's input is built only from the observation (rl.contracts.STUDENT_OBS_KEYS for student arms); the critic's
input comes from rl/agents/inputs.py. Invariant I1: assert at every rollout that no privileged key reaches the actor.
Logs one CSV per run in M4's schema (environment version, episode-set hash, threshold A, configuration hash, seed).

Done when: CartPole-v1 mean return 475 over 100 episodes within 500k steps on 3 seeds (M3-01); a 50k-step run on
rl.env.StubSkyEnv completes and its log passes the schema check (M3-03).
"""
from __future__ import annotations


def train(config: dict) -> dict:
    raise NotImplementedError("M3-01/M3-03: PPO")
