"""The camera-post environment (tasks M1-04 to M1-07): a Gymnasium environment, NumPy only, no rendering.

One pan-tilt-zoom head looks at a 60 x 15 deg sky sector in 5-deg cells. Each step (0.2 s, 5 frames) the agent slews
one cell, zooms, or dwells. Every camera on the head that covers a cell reports how many of the 5 frames held a
detection there; the drone, if present and in the frame, raises that probability in its cell. Evidence is the CUSUM of
log-likelihood ratios per cell (rl/env/evidence.py); a cell is confirmed when its evidence reaches threshold A.

Who sees what (rl/contracts.py):
  * the actor's observation holds only the evidence of `actor_bands` ("student" = RGB + uncooled LWIR for every
    student arm), the cells it has covered, its zoom, the time left and whether it is day;
  * confirmation runs on the same bands as the actor, with the same threshold A for every arm (invariant I2);
  * the privileged evidence maps (cooled LWIR, MWIR, MWIR without the plume) and the ground truth, including the
    gate flag `confirmable` (Filter 2), are returned only in `info`, for critics, teachers, gates and evaluation.
Each band draws its detections from its own random stream, so changing what a privileged camera sees can never
change what the student sees (tested by tests/rl/test_invariants.py).

Filter 2 ("confirmable by the student"): at a step, the drone is confirmable if pointing at its cell at the best zoom
for K steps would bring the expected student evidence to A, using the deployed detector's detection probability and
the clear-sky false-detection rate (rl/config/frozen.yaml confirmation.confirmable_K_steps).

Sensor model (M1-08): by default M2's calibrated model, rl/calib (CalibratedSensorModel, version 1). It refuses to load
if its sources or frozen.yaml changed since calibration (rebuild with `python -m rl.calib.calibrate`), and it has no
low-heat calibration, so the `low_heat` condition raises at reset (the plan's fallback omits that test).
`sensor_model=StubSensorModel()` gives the placeholder model of M1-03 to M1-07. `info["sensor_version"]` records which.

False detections: a camera's per-frame false-detection probability `pfa` is for its whole frame; a cell holding a
share g of the frame gets 1 - (1 - pfa)^g, so the cells' "no false detection" probabilities multiply back to 1 - pfa
(M2's exact partition, rl/calib/score_model.py; CR-M2-01). Version 0 used the first-order pfa * g.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl import contracts as C
from rl.env import geometry as geo
from rl.env.evidence import Cusum, llr_counts
from rl.env.flight import Track
from rl.env.rewards import reward_terms
from rl.env.scenario import sample_episode_spec

_SLEW_OF = {v: k for k, v in C.SLEW.items()}            # (d_row, d_col) -> action id


def default_sensor_model():
    """M2's calibrated sensor model (M1-08). Imported here, not at module level: it verifies its sources on load."""
    from rl.calib.sensor_model import CalibratedSensorModel
    return CalibratedSensorModel()


# LLR coefficients per (band, zoom, pointing, nominal Pd, clear-sky pfa): the same in every episode that shares them.
_COEF_CACHE: dict = {}


def frame_share_pfa(pfa, g):
    """Per-cell false-detection probability for a cell holding a share g of a frame whose whole-frame probability is
    pfa: 1 - (1 - pfa)^g (exact partition: the cells' no-detection probabilities multiply to 1 - pfa)."""
    return -np.expm1(np.log1p(-np.asarray(pfa, float)) * np.asarray(g, float))


@dataclass
class DroneNow:
    in_sector: bool
    az: float = math.nan
    el: float = math.nan
    range_m: float = math.nan
    cell: tuple[int, int] | None = None


class SkyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, actor_bands: str = "student", condition: str = "train", sensor_model=None,
                 learned_declaration: bool = False, threshold_A: float | None = None,
                 compute_privileged: bool = True):
        """compute_privileged=False skips the privileged cameras (their maps in `info` stay at 0): faster for arms
        and baselines that never read them. It cannot change what the actor sees (separate random streams)."""
        if actor_bands not in C.BAND_SETS:
            raise ValueError(f"actor_bands must be one of {tuple(C.BAND_SETS)}")
        F = C.FROZEN
        self.actor_bands = actor_bands
        self._bands_actor = C.BAND_SETS[actor_bands]
        self.condition = condition
        self.sensor = sensor_model if sensor_model is not None else default_sensor_model()
        if not isinstance(self.sensor, C.SensorModel):
            raise TypeError("sensor_model must implement rl.contracts.SensorModel")
        self.learned_declaration = bool(learned_declaration)
        self.A = float(threshold_A if threshold_A is not None else F["confirmation"]["threshold_A"])
        self.n_frames = int(F["cameras"]["frames_per_step"])
        self.max_steps = int(F["episode"]["max_steps"])
        self.rho = float(F["confirmation"]["spread_rho"])
        self.K = int(F["confirmation"]["confirmable_K_steps"])
        self.look_cap = float(F["observation"]["since_look_cap_steps"])
        self.clip = float(F["observation"]["evidence_clip_times_A"]) * self.A
        self._tables = geo.coverage_tables()
        self.compute_privileged = bool(compute_privileged)
        self._sim_bands = tuple(dict.fromkeys(self._bands_actor + (C.PRIVILEGED_EVIDENCE_BANDS
                                                                   if self.compute_privileged else ())))
        sh = C.observation_shapes()
        f32 = np.float32
        self.observation_space = spaces.Dict({
            "evidence": spaces.Box(0.0, self.clip, sh["evidence"], f32),
            "since_look": spaces.Box(0.0, 1.0, sh["since_look"], f32),
            "coverage": spaces.Box(0.0, 1.0, sh["coverage"], f32),
            "zoom": spaces.Box(0.0, 1.0, sh["zoom"], f32),
            "time_left": spaces.Box(0.0, 1.0, sh["time_left"], f32),
            "is_day": spaces.Box(0.0, 1.0, sh["is_day"], f32),
        })
        self.action_space = spaces.Discrete(C.N_ACTIONS_LEARNED_DECLARE if self.learned_declaration
                                            else C.N_ACTIONS)

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        spec = options.get("episode_spec")
        if spec is not None:
            spec = spec if isinstance(spec, C.EpisodeSpec) else C.EpisodeSpec.from_dict(spec)
        else:
            ep_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2 ** 31 - 1))
            spec = sample_episode_spec(ep_seed, options.get("condition", self.condition))
        self.episode = spec
        children = np.random.SeedSequence([spec.seed, 1]).spawn(1 + len(C.MODELLED_BANDS))
        self.sensor.begin_episode(spec, np.random.default_rng(children[0]))
        self._rng = {b: np.random.default_rng(children[1 + i]) for i, b in enumerate(C.MODELLED_BANDS)}
        self.track = Track(spec, self.max_steps) if spec.drone_present else None
        self.t = 0
        self.pointing = (C.GRID.n_el // 2, C.GRID.n_az // 2)
        self.zoom = 0
        self.actor_map = Cusum(C.GRID_SHAPE)
        self.priv_maps = {b: Cusum(C.GRID_SHAPE) for b in C.PRIVILEGED_EVIDENCE_BANDS}
        self.since = np.full(C.GRID_SHAPE, self.look_cap)
        self.cloud = np.zeros(C.GRID_SHAPE, dtype=bool)
        for cell in spec.cloud_cells:
            self.cloud[cell] = True
        # per-episode constants of the sensor model, so a step only draws detections
        self._pfa_clear = {b: self.sensor.pfa_per_frame(b, spec, cloud=False) for b in self._sim_bands}
        self._pfa_map = {b: np.where(self.cloud, self.sensor.pfa_per_frame(b, spec, cloud=True), self._pfa_clear[b])
                         for b in self._sim_bands}
        self._pd_nom = {b: self.sensor.nominal_pd(b, spec) for b in self._sim_bands}
        self._gate = {b: (self.sensor.nominal_pd(b, spec), self.sensor.pfa_per_frame(b, spec, cloud=False))
                      for b in C.STUDENT_BANDS}         # constants of the gate flag (_best_zoom)
        self._view = {}                                 # (band, zoom, pointing) -> coverage, null p, LLR coefficients
        self.drone = self._drone_now()
        self._pd_now = {}                               # (band, hfov) -> drone's detection probability, this step
        self.confirmable = self._best_zoom(self.drone)[1] * self.K >= self.A
        self.ever_confirmable = bool(self.confirmable)
        self._cov = self._coverage_union(self._bands_actor)
        events = dict(confirmed_cell=None, correct=False, false_confirmation=False, slewed=False)
        terms = {k: 0.0 for k in C.REWARD_TERMS}
        return self._obs(), self._info(events, terms)

    def step(self, action):
        a = int(action)
        if not self.action_space.contains(a):
            raise ValueError(f"invalid action {action!r}")
        slewed, declared = False, False
        if a in C.SLEW:
            dr, dc = C.SLEW[a]
            r = min(max(self.pointing[0] + dr, 0), C.GRID.n_el - 1)
            c = min(max(self.pointing[1] + dc, 0), C.GRID.n_az - 1)
            slewed = (r, c) != self.pointing
            self.pointing = (r, c)
        elif a == C.ZOOM_IN:
            self.zoom = min(self.zoom + 1, C.N_ZOOM - 1)
        elif a == C.ZOOM_OUT:
            self.zoom = max(self.zoom - 1, 0)
        elif a == C.DECLARE:
            declared = True
        self.t += 1
        self.drone = self._drone_now()
        self._pd_now = {}

        inc_actor = np.zeros(C.GRID_SHAPE)
        cov_actor = np.zeros(C.GRID_SHAPE, dtype=bool)
        for b in self._sim_bands:
            inc, cov = self._sense(b)
            if b in self._bands_actor:
                inc_actor += inc
                cov_actor |= cov
            if b in self.priv_maps:
                self.priv_maps[b].update(inc, cov)
                self.priv_maps[b].spread(self.rho)
        self.actor_map.update(inc_actor, cov_actor)
        self.actor_map.spread(self.rho)
        self.since = np.where(cov_actor, 0.0, self.since + 1.0)
        self._cov = cov_actor

        correct, false_n, confirmed_cell = False, 0, None
        if self.learned_declaration:
            if declared:
                confirmed_cell = self.pointing
                correct = bool(self.drone.in_sector and self.drone.cell == self.pointing)
                false_n = 0 if correct else 1
        else:
            cell = self.actor_map.crossing(self.A, cov_actor)
            if cell is not None:
                confirmed_cell = cell
                correct = bool(self.drone.in_sector and self.drone.cell == cell)
                if not correct:
                    false_n = 1
                    self.actor_map.reset_cell(cell)

        self.confirmable = self._best_zoom(self.drone)[1] * self.K >= self.A
        self.ever_confirmable = self.ever_confirmable or bool(self.confirmable)
        passed = self.track is not None and self.t >= self.track.exit_step
        terminated = bool(correct or passed)
        truncated = bool(not terminated and self.t >= self.max_steps)
        missed = (terminated or truncated) and not correct and self.ever_confirmable
        earliness = (self.track.exit_step - self.t) / self.track.exit_step if correct else 0.0
        terms = reward_terms(correct=correct, earliness=earliness, false_confirmations=false_n,
                             missed_confirmable=missed, slewed=slewed)
        events = dict(confirmed_cell=confirmed_cell, correct=correct, false_confirmation=bool(false_n),
                      slewed=slewed)
        return self._obs(), float(sum(terms.values())), terminated, truncated, self._info(events, terms)

    # ------------------------------------------------------------------ sensing
    def _sense(self, band: str) -> tuple[np.ndarray, np.ndarray]:
        """One step of detections in one band: (LLR increment per cell, covered cells)."""
        key = (band, self.zoom, self.pointing)
        view = self._view.get(key)
        if view is None:                                # constant through the episode for this band, zoom and pointing
            cov, g = self._tables[band][self.zoom][self.pointing[0]][self.pointing[1]]
            p_null = frame_share_pfa(self._pfa_map[band], g).ravel()   # g = 0 outside the frame: nothing drawn there
            pd_nom, coef = self._pd_nom[band], None
            if pd_nom is not None:                      # LLR = k c1 + (n - k) c0 on covered cells, 0 elsewhere
                ckey = key + (pd_nom, self._pfa_clear[band])
                coef = _COEF_CACHE.get(ckey)
                if coef is None:
                    pfa = np.where(cov, frame_share_pfa(self._pfa_clear[band], g), 0.5)
                    coef = _COEF_CACHE[ckey] = (np.where(cov, llr_counts(1, 1, pd_nom, pfa), 0.0),
                                                np.where(cov, llr_counts(0, 1, pd_nom, pfa), 0.0))
            view = self._view[key] = (cov, p_null, coef)
        cov, p, coef = view
        d = self.drone
        hfov = geo.band_hfov(band, self.zoom)
        if d.in_sector and cov[d.cell] and geo.in_frame(d.az, d.el, self.pointing, hfov):
            i = d.cell[0] * C.GRID.n_az + d.cell[1]
            p = p.copy()
            p[i] = 1.0 - (1.0 - self._pd(band, hfov)) * (1.0 - p[i])
        k = (self._rng[band].random((self.n_frames, p.size)) < p).sum(axis=0).reshape(C.GRID_SHAPE)
        if coef is None:
            return np.zeros(C.GRID_SHAPE), cov
        return k * coef[0] + (self.n_frames - k) * coef[1], cov

    def _pd(self, band: str, hfov: float) -> float:
        """The drone's per-frame detection probability in `band` at field `hfov`, this step (asked once per step)."""
        key = (band, hfov)
        pd = self._pd_now.get(key)
        if pd is None:
            size = geo.size_px(self.episode.span_m, self.drone.range_m, hfov)
            pd = self._pd_now[key] = self.sensor.pd_per_frame(band, size, self.episode)
        return pd

    def _drone_now(self) -> DroneNow:
        if self.track is None or self.t >= self.track.exit_step:
            return DroneNow(in_sector=False)
        az, el, rng = self.track.at(self.t)
        cell = geo.cell_of(az, el)
        return DroneNow(in_sector=cell is not None, az=az, el=el, range_m=rng, cell=cell)

    def _best_zoom(self, d: DroneNow) -> tuple[int | None, float]:
        """Zoom step with the most expected student evidence per step when pointed at the drone's cell, and that
        expectation (drone-caused detections, clear-sky null). Used for the gate flag and by the oracle; d is
        always self.drone (the detection probabilities come from this step's cache)."""
        if not d.in_sector:
            return None, -math.inf
        best_z, best_e = None, -math.inf
        n = self.n_frames
        for z in range(C.N_ZOOM):
            e, seen = 0.0, False
            for b in C.STUDENT_BANDS:
                pd_nom, pfa_frame = self._gate[b]
                hfov = geo.band_hfov(b, z)
                cov, g = self._tables[b][z][d.cell[0]][d.cell[1]]
                if pd_nom is None or not cov[d.cell] or not geo.in_frame(d.az, d.el, d.cell, hfov):
                    continue
                seen = True
                pfa = -math.expm1(math.log1p(-pfa_frame) * float(g[d.cell]))   # frame_share_pfa, scalar
                pd = self._pd(b, hfov)
                q = 1.0 - (1.0 - pd) * (1.0 - pfa)      # expected_llr, in scalar arithmetic
                e += n * (q * math.log(pd_nom / pfa) + (1.0 - q) * math.log((1.0 - pd_nom) / (1.0 - pfa)))
            if seen and e > best_e:
                best_z, best_e = z, e
        return best_z, best_e

    def _coverage_union(self, bands) -> np.ndarray:
        cov = np.zeros(C.GRID_SHAPE, dtype=bool)
        for b in bands:
            cov |= self._tables[b][self.zoom][self.pointing[0]][self.pointing[1]][0]
        return cov

    # ------------------------------------------------------------------ outputs
    def _obs(self) -> dict:
        zoom = np.zeros(C.N_ZOOM, dtype=np.float32)
        zoom[self.zoom] = 1.0
        return {
            "evidence": np.minimum(self.actor_map.L, self.clip).astype(np.float32),   # CUSUM evidence is never < 0
            "since_look": np.minimum(self.since / self.look_cap, 1.0).astype(np.float32),
            "coverage": self._cov.astype(np.float32),
            "zoom": zoom,
            "time_left": np.array([(self.max_steps - self.t) / self.max_steps], dtype=np.float32),
            "is_day": np.array([float(self.episode.is_day)], dtype=np.float32),
        }

    def _info(self, events: dict, terms: dict) -> dict:
        d = self.drone
        priv = {f"evidence_{b}": np.minimum(m.L, self.clip).astype(np.float32) for b, m in self.priv_maps.items()}
        for b in C.PRIVILEGED_BANDS:
            priv[f"coverage_{b}"] = self._tables[b][self.zoom][self.pointing[0]][self.pointing[1]][0].astype(
                np.float32)
        return {
            "truth": dict(drone_present=bool(self.episode.drone_present), drone_in_sector=bool(d.in_sector),
                          drone_cell=d.cell if d.cell is not None else (-1, -1),
                          az_deg=d.az if d.in_sector else None, el_deg=d.el if d.in_sector else None,
                          range_m=d.range_m if d.in_sector else None, confirmable=bool(self.confirmable),
                          ever_confirmable=bool(self.ever_confirmable)),
            "privileged": priv,
            "events": events,
            "reward_terms": terms,
            "env_version": C.ENV_VERSION,
            "sensor_version": self.sensor.version,
            "interface_version": C.INTERFACE_VERSION,
            "episode_seed": int(self.episode.seed),
            "actor_bands": self.actor_bands,
            "threshold_A": self.A,
        }

    # ------------------------------------------------------------------ for the oracle and tests only
    def oracle_plan(self) -> tuple[tuple[int, int], int] | None:
        """(drone cell, best zoom) from the simulator's truth. For OraclePolicy and tests; never for a policy."""
        if not self.drone.in_sector:
            return None
        z, _ = self._best_zoom(self.drone)
        return self.drone.cell, (z if z is not None else C.N_ZOOM - 1)


def slew_toward(pointing: tuple[int, int], target: tuple[int, int]) -> int:
    """The slew action that moves one cell from `pointing` towards `target` (DWELL if already there)."""
    dr = int(np.sign(target[0] - pointing[0]))
    dc = int(np.sign(target[1] - pointing[1]))
    return C.DWELL if (dr, dc) == (0, 0) else _SLEW_OF[(dr, dc)]


def make_vec_env(n: int, **kwargs) -> gym.vector.SyncVectorEnv:
    """n environments stepped together (M1-06); reset it with seed=<base> for consecutive episode seeds."""
    return gym.vector.SyncVectorEnv([lambda: SkyEnv(**kwargs) for _ in range(n)])
