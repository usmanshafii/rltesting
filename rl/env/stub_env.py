"""Stub environment (task M1-03): random detections behind the real schema, so agents and evaluation code can be
written before the real environment is ready, and so a failure can be traced to the environment or the agent.

Same observation space, action space and info keys as SkyEnv (rl/contracts.py); nothing in it means anything.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl import contracts as C


class StubSkyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, end_prob: float = 0.005):
        self.end_prob = end_prob
        sh = C.observation_shapes()
        clip = float(C.FROZEN["observation"]["evidence_clip_times_A"]) * float(C.FROZEN["confirmation"]["threshold_A"])
        f32 = np.float32
        self.observation_space = spaces.Dict({
            "evidence": spaces.Box(0.0, clip, sh["evidence"], f32),
            "since_look": spaces.Box(0.0, 1.0, sh["since_look"], f32),
            "coverage": spaces.Box(0.0, 1.0, sh["coverage"], f32),
            "zoom": spaces.Box(0.0, 1.0, sh["zoom"], f32),
            "time_left": spaces.Box(0.0, 1.0, sh["time_left"], f32),
            "is_day": spaces.Box(0.0, 1.0, sh["is_day"], f32),
        })
        self.action_space = spaces.Discrete(C.N_ACTIONS)
        self.max_steps = int(C.FROZEN["episode"]["max_steps"])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), self._info(0.0)

    def step(self, action):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action {action!r}")
        self.t += 1
        terminated = bool(self.np_random.random() < self.end_prob)
        truncated = bool(not terminated and self.t >= self.max_steps)
        reward = float(self.np_random.normal(0.0, 0.1))
        return self._obs(), reward, terminated, truncated, self._info(reward)

    def _obs(self):
        o = {k: self.np_random.uniform(sp.low, sp.high).astype(np.float32)
             for k, sp in self.observation_space.spaces.items()}
        z = np.zeros(C.N_ZOOM, np.float32)
        z[int(self.np_random.integers(C.N_ZOOM))] = 1.0
        o["zoom"] = z
        o["time_left"] = np.array([(self.max_steps - self.t) / self.max_steps], np.float32)
        return o

    def _info(self, reward: float) -> dict:
        priv = {k: self.np_random.random(C.GRID_SHAPE).astype(np.float32) for k in C.PRIVILEGED_KEYS}
        return {
            "truth": dict(drone_present=True, drone_in_sector=True, drone_cell=(0, 0), az_deg=0.0, el_deg=7.5,
                          range_m=5000.0, confirmable=False, ever_confirmable=False),
            "privileged": priv,
            "events": dict(confirmed_cell=None, correct=False, false_confirmation=False, slewed=False),
            "reward_terms": {k: (reward if k == "step" else 0.0) for k in C.REWARD_TERMS},
            "env_version": "stub",
            "interface_version": C.INTERFACE_VERSION,
            "episode_seed": -1,
            "actor_bands": "student",
            "threshold_A": float(C.FROZEN["confirmation"]["threshold_A"]),
        }
