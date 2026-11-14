"""M4-03: raster scan: visits every cell in a fixed cycle at a fixed zoom; confirmation is the environment's test.

Done when: a unit test shows it visits every cell in a fixed cycle. Then M1-07's "random < raster < oracle" check in
`python -m rl.scripts.check wp1` stops reporting PENDING.
"""
from __future__ import annotations


class RasterPolicy:
    def __call__(self, obs: dict) -> int:
        raise NotImplementedError("M4-03: raster scan")
