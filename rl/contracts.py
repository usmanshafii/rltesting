"""The interface of the RL course project (CSE 482): the one module every other module imports.

Status: DRAFT by M1 (task M1-02), 2026-11-14. All four members approve it at S-03 and the commit is tagged
`interface-v1`; after that it changes only through a change request in rl/PLAN.md (AGENTS.md rule 3).

What is defined here, and nowhere else:
  * the frozen values, loaded from rl/config/frozen.yaml (FROZEN), and their hash (frozen_hash(), invariant I4);
  * the band names and the band sets that say who may see what (invariants I1 and I11);
  * the sky grid, the zoom steps and the action ids;
  * the actor's observation keys and shapes. Privileged data never enter the observation of a student arm: they are
    delivered only in `info["privileged"]`, so a student policy cannot receive them by accident (invariant I1);
  * the keys of `info["truth"]`, `info["events"]` and `info["reward_terms"]`;
  * EpisodeSpec, the complete description of one episode, so evaluation episodes can be frozen and replayed (I3);
  * the SensorModel protocol that M2 implements in rl/calib/ (M1's stub, rl/env/stub_sensor.py, implements it until
    then), so the environment does not change when the real sensor model arrives.

Design decisions this draft makes (each listed in rl/PLAN.md's change log for approval at S-03):
  * confirmation is the sequential probability ratio test in its repeated form, Page's CUSUM, which suits continuous
    monitoring: per cell L <- max(0, L + LLR), confirm when L >= A; under the null, the mean time to a false
    confirmation is at least e^A steps per monitored cell (Lorden 1971);
  * the evidence is built from per-frame detections at the detector's 0.01-false-alarm-per-frame threshold, the
    operating point the repo measured; the log-likelihood ratio uses a nominal detection probability (pd_nominal);
  * an arm confirms on the bands its actor sees: student arms on RGB + uncooled LWIR, four-band arms on all four, with
    the same threshold A for every arm (invariant I2).
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import yaml

RL_ROOT = Path(__file__).resolve().parent
FROZEN_PATH = RL_ROOT / "config" / "frozen.yaml"
INTERFACE_VERSION = "interface-v0-draft"        # becomes "interface-v1" when S-03 approves it
ENV_VERSION = "sky-env-0.1-stub-sensor"         # bumped on every environment change; recorded in every result (I5)


def load_frozen(path: Path = FROZEN_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def frozen_hash(path: Path = FROZEN_PATH) -> str:
    """SHA-256 of frozen.yaml with line endings normalised, so Windows and Linux checkouts agree (invariant I4)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


FROZEN: dict = load_frozen()

# ============================================================================ bands (CLAUDE.md sensor set rule)
BANDS = ("rgb", "lwir", "lwir_cooled", "mwir")          # the four sensors; uncooled and cooled LWIR are separate
SHARED_VARIANT = {"mwir": "mwir_shared"}                # MWIR with the plume's signal removed (Filter 1)
MODELLED_BANDS = BANDS + ("mwir_shared",)
STUDENT_BANDS = ("rgb", "lwir")                         # deployed
PRIVILEGED_BANDS = ("lwir_cooled", "mwir")              # training only
BAND_SETS = {
    "student": STUDENT_BANDS,
    "four": BANDS,                                      # four-band arms and the ungated teacher
    "four_shared": ("rgb", "lwir", "lwir_cooled", "mwir_shared"),   # the gated teacher (Filter 1)
}

# ============================================================================ the sky grid and the zoom steps


@dataclass(frozen=True)
class Grid:
    az_min: float
    az_max: float
    el_min: float
    el_max: float
    cell: float

    @property
    def n_az(self) -> int:
        return int(round((self.az_max - self.az_min) / self.cell))

    @property
    def n_el(self) -> int:
        return int(round((self.el_max - self.el_min) / self.cell))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_el, self.n_az)                   # rows = elevation (row 0 lowest), columns = azimuth


GRID = Grid(*FROZEN["grid"]["az_deg"], *FROZEN["grid"]["el_deg"], FROZEN["grid"]["cell_deg"])
GRID_SHAPE = GRID.shape
ZOOM_HFOV_DEG = tuple(float(v) for v in FROZEN["cameras"]["zoom_hfov_deg"])   # index 0 = widest
N_ZOOM = len(ZOOM_HFOV_DEG)

# ============================================================================ actions
DWELL = 0
SLEW = {                                                # action id -> (d_row, d_col); north = higher elevation
    1: (1, 0), 2: (1, 1), 3: (0, 1), 4: (-1, 1), 5: (-1, 0), 6: (-1, -1), 7: (0, -1), 8: (1, -1),
}
ZOOM_IN = 9                                             # towards a narrower field
ZOOM_OUT = 10
DECLARE = 11                                            # only with learned declaration (a Tier-2 test)
N_ACTIONS = 11
N_ACTIONS_LEARNED_DECLARE = 12
ACTION_NAMES = ("dwell", "slew_n", "slew_ne", "slew_e", "slew_se", "slew_s", "slew_sw", "slew_w", "slew_nw",
                "zoom_in", "zoom_out", "declare")

# ============================================================================ what the actor sees
STUDENT_OBS_KEYS = ("evidence", "since_look", "coverage", "zoom", "time_left", "is_day")


def observation_shapes() -> dict[str, tuple[int, ...]]:
    return {"evidence": GRID_SHAPE, "since_look": GRID_SHAPE, "coverage": GRID_SHAPE, "zoom": (N_ZOOM,),
            "time_left": (1,), "is_day": (1,)}


# ============================================================================ what only training sees: info dict
PRIVILEGED_EVIDENCE_BANDS = ("lwir_cooled", "mwir", "mwir_shared")
PRIVILEGED_KEYS = tuple(f"evidence_{b}" for b in PRIVILEGED_EVIDENCE_BANDS) + tuple(
    f"coverage_{b}" for b in PRIVILEGED_BANDS)
TRUTH_KEYS = ("drone_present", "drone_in_sector", "drone_cell", "az_deg", "el_deg", "range_m", "confirmable",
              "ever_confirmable")
EVENT_KEYS = ("confirmed_cell", "correct", "false_confirmation", "slewed")
REWARD_TERMS = ("correct", "earliness", "false", "missed", "step", "slew")
INFO_KEYS = ("truth", "privileged", "events", "reward_terms", "env_version", "interface_version", "episode_seed",
             "actor_bands", "threshold_A")

# ============================================================================ one episode, completely


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything that defines an episode before the policy acts. Evaluation sets are lists of these (M4)."""
    seed: int
    drone_present: bool
    speed_mps: float
    entry_range_m: float
    entry_az_deg: float
    entry_el_deg: float
    lateral_offset_m: float
    weave_deg: float
    weave_s: float
    span_m: float
    is_day: bool
    cloudy: bool
    cloud_cells: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    engine: str = "hot"                                 # "hot" or "low_heat"
    condition: str = "train"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cloud_cells"] = [list(c) for c in self.cloud_cells]
        return d

    @staticmethod
    def from_dict(d: dict) -> "EpisodeSpec":
        d = dict(d)
        d["cloud_cells"] = tuple(tuple(int(v) for v in c) for c in d.get("cloud_cells", ()))
        return EpisodeSpec(**d)


# ============================================================================ the sensor model M2 implements


@runtime_checkable
class SensorModel(Protocol):
    """Per-band detection statistics. The environment draws the detections itself (so seeding stays in one place);
    the sensor model only says how likely they are. Implemented by rl/env/stub_sensor.StubSensorModel now and by
    M2's model in rl/calib/ from M2-08 on; both must pass tests/rl/test_contracts.py::test_sensor_model_contract."""

    version: str

    def begin_episode(self, spec: EpisodeSpec, rng: np.random.Generator) -> None:
        """Draw the episode's persistent random effects (target clutter, engine state, plume share)."""

    def dprime(self, band: str, size_px: float, spec: EpisodeSpec) -> float:
        """Matched-filter detectability of the drone in one frame of `band`, at this apparent size."""

    def pd_per_frame(self, band: str, size_px: float, spec: EpisodeSpec) -> float:
        """Probability that the detector fires on the drone in one frame, at the 0.01-per-frame threshold."""

    def pfa_per_frame(self, band: str, spec: EpisodeSpec, cloud: bool) -> float:
        """Probability of at least one false detection in one whole frame; `cloud` = the area is cloud clutter."""

    def nominal_pd(self, band: str, spec: EpisodeSpec) -> float | None:
        """Design detection probability used inside the log-likelihood ratio; None = the band is not used
        (for example RGB at night). May depend only on what a deployed system knows (time of day)."""
