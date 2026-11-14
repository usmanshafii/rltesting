"""M4-02: frozen evaluation sets: lists of rl.contracts.EpisodeSpec drawn by rl.env.scenario.sample_episode_spec
from pre-registered seeds, held-in and held-out conditions; each set has a hash recorded in every result row (I3).

Done when: the generator gives the same episode-set hash on two laptops.
"""
from __future__ import annotations


def episode_set(name: str, seeds: list[int], condition: str = "train") -> list:
    raise NotImplementedError("M4-02: frozen evaluation episodes")


def episode_set_hash(specs: list) -> str:
    raise NotImplementedError("M4-02: episode-set hash")
