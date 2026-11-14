"""M4-03: uniform random actions (the floor)."""
from __future__ import annotations


class RandomPolicy:
    def __call__(self, obs: dict) -> int:
        raise NotImplementedError("M4-03: random policy")
