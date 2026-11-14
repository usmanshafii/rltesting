"""Stub sensor model (M1), standing in for M2's calibrated model until M2-08 delivers it.

It implements rl.contracts.SensorModel so the environment, the agents and the baselines can be built and tested now;
swapping in M2's model (M1-08) changes no other code. EVERY NUMBER HERE IS A PLACEHOLDER (grade C):
  * d' per band and lighting: a crude least-squares line of log d' on log size_px over the census targets up to
    64 px (docs/reference/privileged/data/census.json; fitted 2026-11-14, M1). Residual spread there is 1.4-1.9 in
    log units; it is represented by a persistent per-episode effect (common + per band). M2-04 replaces all of it.
  * below 1 px the signal falls with the target's share of a pixel, so d' is scaled by size_px^2 there.
  * detection probability per frame: the trained detector's logistic in d' (rl/config/frozen.yaml detector), used for
    every band. M2-06 replaces it with a score model per band.
  * MWIR without the plume (Filter 1): MWIR d' times (1 - plume share), plume share drawn per episode in 0.1-0.7
    (docs/thermal_bands.md section 5: 10-70 % of the exhaust stubs' contrast). M2-05 measures it.
  * low-heat engine: thermal d' scaled down (MWIR 0.3, LWIR bands 0.7). M2-05 measures it.
  * cloud clutter multiplies the per-frame false-detection probability: x30 in both LWIR cameras (shared clutter),
    x5 in MWIR, x10 in RGB by day. M2-07 calibrates it.
"""
from __future__ import annotations

import math

import numpy as np

from rl import contracts as C

# log d' = a + b log(size_px), per band and lighting (crude census fit, see the module docstring)
_LINE = {
    ("rgb", True): (-0.33, 1.61), ("rgb", False): (-2.38, 0.98),
    ("lwir", True): (1.79, 0.93), ("lwir", False): (2.27, 0.77),
    ("lwir_cooled", True): (3.78, 0.23), ("lwir_cooled", False): (1.58, 1.04),
    ("mwir", True): (4.28, 0.11), ("mwir", False): (3.22, 0.48),
}
_SIGMA_COMMON, _SIGMA_BAND = 1.0, 0.5
_PLUME_SHARE = (0.1, 0.7)
_LOW_HEAT = {"mwir": 0.3, "mwir_shared": 0.3, "lwir": 0.7, "lwir_cooled": 0.7}
_CLOUD_PFA_X = {"rgb": 10.0, "lwir": 30.0, "lwir_cooled": 30.0, "mwir": 5.0, "mwir_shared": 5.0}


class StubSensorModel:
    version = "stub-0.1"

    def __init__(self):
        det = C.FROZEN["detector"]
        self._a, self._b = float(det["logistic_a"]), float(det["logistic_b"])
        self._pfa = float(det["fa_per_frame"])
        self._pd_nom = float(C.FROZEN["confirmation"]["pd_nominal"])
        self._eff: dict[str, float] = {}
        self._plume_share = 0.4

    def begin_episode(self, spec: C.EpisodeSpec, rng: np.random.Generator) -> None:
        common = rng.normal(0.0, _SIGMA_COMMON)
        self._eff = {b: common + rng.normal(0.0, _SIGMA_BAND) for b in C.BANDS}
        self._eff["mwir_shared"] = self._eff["mwir"]
        self._plume_share = float(rng.uniform(*_PLUME_SHARE))

    def dprime(self, band: str, size_px: float, spec: C.EpisodeSpec) -> float:
        base = "mwir" if band == "mwir_shared" else band
        a, b = _LINE[(base, bool(spec.is_day))]
        s = max(size_px, 1e-6)
        d = math.exp(a + b * math.log(max(s, 1.0)) + self._eff.get(band, 0.0)) * min(s, 1.0) ** 2
        if band == "mwir_shared":
            d *= 1.0 - self._plume_share
        if spec.engine == "low_heat":
            d *= _LOW_HEAT.get(band, 1.0)
        return d

    def pd_per_frame(self, band: str, size_px: float, spec: C.EpisodeSpec) -> float:
        d = self.dprime(band, size_px, spec)
        if d <= 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-(self._a + self._b * math.log(d))))

    def pfa_per_frame(self, band: str, spec: C.EpisodeSpec, cloud: bool) -> float:
        x = _CLOUD_PFA_X.get(band, 1.0) if cloud else 1.0
        if band == "rgb" and not spec.is_day:
            x = 1.0
        return min(0.5, self._pfa * x)

    def nominal_pd(self, band: str, spec: C.EpisodeSpec) -> float | None:
        if band == "rgb" and not spec.is_day:
            return None                                 # RGB is blind at night: its detections carry no evidence
        return self._pd_nom
