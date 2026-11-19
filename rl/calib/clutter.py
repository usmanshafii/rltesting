"""M2-07: measured spread split and a shared-LWIR clutter sampler.

Only sensor noise averages; dwell delegates to the repository's d_after exactly.
The common LWIR standardized spatial pattern is a modelling assumption (grade C),
not a correlation measurement from the census's single noise draw.
"""
from __future__ import annotations

import math
import numpy as np
from eval.detectability import d_after
from rl import contracts as C


def split_spread(dp: dict) -> tuple[float, float]:
    """Clutter and sensor standard deviations in units of signal.

    Finite-sample d_noise can be below d. As d_after does, clip negative clutter
    variance to zero rather than inventing negative variance or changing the data.
    """
    d, dn = float(dp["d"]), float(dp["d_noise"])
    if d <= 0:
        return 0.0, 0.0
    if not math.isfinite(d) or not math.isfinite(dn) or dn <= 0:
        raise ValueError("positive signal needs finite positive d and d_noise")
    n = 1/dn
    return math.sqrt(max(0.0, 1/(d*d)-n*n)), n


def dwell_dprime(dp: dict, frames: int) -> float:
    if not isinstance(frames, (int,np.integer)) or frames < 1:
        raise ValueError("frames must be a positive integer")
    return d_after(dp,frames)


def cloud_false_detection_factor(band: str, is_day: bool) -> float:
    """Grade-C sensitivity setting inherited from stub_sensor.py; not measured FA."""
    if band not in C.MODELLED_BANDS:
        raise ValueError(band)
    return {"rgb":10.0 if is_day else 1.0,"lwir":30.0,"lwir_cooled":30.0,
            "mwir":5.0,"mwir_shared":5.0}[band]


class NoiseClutter:
    def __init__(self, noise_shares: dict):
        self.shares = dict(noise_shares)
        if any(not 0 <= v <= 1 for v in self.shares.values()):
            raise ValueError("noise shares must be in [0,1]")

    def draw(self, rng: np.random.Generator, size):
        """Across spatial positions: common LWIR clutter, independent camera noise."""
        shared = rng.normal(size=size)
        clutter, noise = {}, {}
        for b in C.BANDS:
            c = shared if b in ("lwir","lwir_cooled") else rng.normal(size=size)
            clutter[b] = c * math.sqrt(1-self.shares[b])
            noise[b] = rng.normal(size=size) * math.sqrt(self.shares[b])
        return clutter, noise

    def begin_episode(self, rng: np.random.Generator, shape):
        """One spatial clutter realization held fixed throughout an episode."""
        self._shape=tuple(shape)
        self._clutter,_=self.draw(rng,self._shape)

    def sample_frames(self,rng: np.random.Generator,frames: int):
        if not isinstance(frames,(int,np.integer)) or frames < 1:
            raise ValueError("frames must be a positive integer")
        if not hasattr(self,"_clutter"):
            raise RuntimeError("begin_episode must be called before sample_frames")
        shape=(frames,)+self._shape
        c={b:np.broadcast_to(v,shape).copy() for b,v in self._clutter.items()}
        n={b:rng.normal(size=shape)*math.sqrt(self.shares[b]) for b in C.BANDS}
        return c,n
