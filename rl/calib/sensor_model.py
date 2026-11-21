"""M2-08: calibrated sensor statistics behind the existing SensorModel contract.

The M1 environment still owns Bernoulli draws. sample_scores and dwell_dprime
are available for the separately reviewed integration; this object cannot change
the environment's per-band independent draws through a probability-only API.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from rl import contracts as C
from rl.calib.fit_dprime import predict_dprime
from rl.calib.score_model import GaussianScoreModel, pd_from_dprime
from rl.calib.clutter import cloud_false_detection_factor, dwell_dprime


def source_hash(path: Path) -> str:
    """Normalize text line endings so checkout policy does not change provenance."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


class CalibratedSensorModel:
    version = "calibrated-1.0-analytic-plume"

    def __init__(self, params_path: str | Path | None = None, *, random_effects: bool = True):
        p=Path(params_path) if params_path else Path(__file__).with_name("params.json")
        self.params=json.loads(p.read_text(encoding="utf-8"))
        if self.params["version"] != self.version:
            raise ValueError("unsupported calibration version")
        self.model=self.params["model"]
        if self.params["frozen_sha256"] != C.frozen_hash():
            raise ValueError("frozen configuration differs from calibration; rebuild parameters")
        self.random_effects=bool(random_effects)
        self._row=None
        self._effects={b:0. for b in C.BANDS}
        self._plume_ratio=float(self.params["values"]["plume_to_stub_ratio_central"]["value"])
        self.verify_sources()

    def verify_sources(self):
        root=C.RL_ROOT.parent
        for name,entry in self.params["sources"].items():
            p=root/entry["path"]
            if source_hash(p) != entry["sha256"]:
                raise ValueError(f"calibration source changed: {name} ({entry['path']})")

        for name,expected in self.params["implementation_sha256"].items():
            if source_hash(Path(__file__).with_name(name)) != expected:
                raise ValueError(f"calibration implementation changed: {name}; rebuild parameters")

    def begin_episode(self, spec: C.EpisodeSpec, rng: np.random.Generator) -> None:
        if spec.engine != "hot":
            raise ValueError("low_heat calibration unavailable: original four-band renders are absent; plan fallback omits this test")
        bank=self.model["residual_bank"]
        candidates=[r for r in bank if (r["lighting"] == "day") == bool(spec.is_day) and bool(r["clouds"]) == bool(spec.cloudy)]
        if not candidates:
            candidates=[r for r in bank if (r["lighting"] == "day") == bool(spec.is_day)]
        self._row=candidates[int(rng.integers(len(candidates)))]
        self._effects={}
        for b in C.BANDS:
            observed=[r["residuals"][b] for r in candidates if b in r["residuals"]]
            value=self._row["residuals"].get(b)
            if value is None:
                value=observed[int(rng.integers(len(observed)))] if observed else 0.
            self._effects[b]=float(value) if self.random_effects else 0.
        lo=self.params["values"]["plume_to_stub_ratio_lower"]["value"]
        hi=self.params["values"]["plume_to_stub_ratio_upper"]["value"]
        self._plume_ratio=float(rng.uniform(lo,hi))

    def _features(self,size_px,spec):
        # EpisodeSpec v0 has no aspect, visibility or power setting. Draw them
        # jointly from TRAINING metadata once and retain them through the episode.
        # No privileged observation is used. CR-M2-01 records this limitation.
        row=self._row
        if row is None:
            bank=self.model["residual_bank"]
            row=next(r for r in bank if (r["lighting"] == "day") == bool(spec.is_day))
        return dict(row,size_px=max(size_px,1.),lighting="day" if spec.is_day else "night",clouds=bool(spec.cloudy))

    def dprime(self, band: str, size_px: float, spec: C.EpisodeSpec) -> float:
        if band not in C.MODELLED_BANDS:
            raise ValueError(band)
        if spec.engine != "hot":
            raise ValueError("low_heat calibration unavailable")
        size=float(size_px)
        if not math.isfinite(size) or size < 0:
            raise ValueError("size_px must be finite and nonnegative")
        if size == 0:
            return 0.
        base="mwir" if band == "mwir_shared" else band
        d=predict_dprime(self.model,base,self._features(size,spec))
        d*=math.exp(self._effects[base])*min(size,1.)**2
        if band == "mwir_shared":
            # Source gives a plume/stub contrast ratio, NOT plume/total. This
            # conservative grade-C removal overestimates the private fraction;
            # r/(1+r) bounds its share even if all non-plume signal were stubs.
            d/=1+self._plume_ratio
        return d

    def pd_per_frame(self,band,size_px,spec):
        return pd_from_dprime(band,self.dprime(band,size_px,spec))

    def pfa_per_frame(self,band,spec,cloud):
        if band not in C.MODELLED_BANDS:
            raise ValueError(band)
        factor=cloud_false_detection_factor(band,bool(spec.is_day)) if cloud else 1.
        return min(.5,float(C.FROZEN["detector"]["fa_per_frame"])*factor)

    def nominal_pd(self,band,spec):
        if band not in C.MODELLED_BANDS:
            raise ValueError(band)
        return None if band == "rgb" and not spec.is_day else float(C.FROZEN["confirmation"]["pd_nominal"])

    def dwell_dprime(self,band,size_px,spec,frames):
        base="mwir" if band == "mwir_shared" else band
        shares=[r["noise_share"][base] for r in self.model["residual_bank"] if r["noise_share"][base] is not None]
        share=float(np.median(shares))
        d=self.dprime(band,size_px,spec)
        return dwell_dprime({"d":d,"d_noise":d/math.sqrt(max(share,1e-12))},frames)

    def sample_scores(self,band,size_px,spec,rng,frames=1):
        return GaussianScoreModel(band).sample(rng,self.dprime(band,size_px,spec),frames)
