"""M4-03: Bayesian search: looks at the cell with the highest posterior from the evidence map (2 bands, or 4 for
the non-RL headroom estimate), with the environment's confirmation test.

Done when: a unit test shows it always looks at the highest-posterior cell.
"""
from __future__ import annotations


class BayesSearchPolicy:
    def __init__(self, bands: str = "student"):
        self.bands = bands

    def __call__(self, obs: dict) -> int:
        raise NotImplementedError("M4-03: Bayesian search")
