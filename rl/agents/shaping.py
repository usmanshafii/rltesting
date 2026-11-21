"""Potential-based reward shaping with phi(s_terminal) == 0 at every episode end (M3-06)."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from rl.agents.inputs import GateConfig, apply_privileged_gate


class PotentialShaper:
    """Computes r'_t = r_t + gamma * phi(s_{t+1}) - phi(s_t), enforcing phi == 0 at episode ends."""

    def __init__(self, gamma: float = 0.99, scale: float = 0.05, enabled: bool = False) -> None:
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.enabled = bool(enabled)

    def potential(
        self,
        priv_obs: Optional[Dict[str, np.ndarray]],
        pointing_cell: Optional[int] = None,
        episode_ended: bool = False,
    ) -> float:
        # Invariant I7: phi MUST be 0 at every episode end (including truncation/time limit).
        if not self.enabled or episode_ended or priv_obs is None:
            return 0.0

        gated = apply_privileged_gate(
            priv_obs,
            GateConfig(gated=True, allow_cooled_lwir=True, allow_mwir=True),
        )
        total_signal = 0.0
        for key, arr in gated.items():
            a = np.asarray(arr, dtype=np.float32)
            if a.size == 0:
                continue
            if pointing_cell is not None and a.ndim >= 1 and a.shape[0] > pointing_cell:
                total_signal += float(np.maximum(0.0, a[pointing_cell]).sum())
            else:
                total_signal += float(np.maximum(0.0, a).mean())

        return self.scale * total_signal

    def shaped_reward(
        self,
        raw_reward: float,
        phi_current: float,
        phi_next: float,
        terminated_or_truncated: bool,
    ) -> float:
        if not self.enabled:
            return float(raw_reward)
        effective_next_phi = 0.0 if terminated_or_truncated else float(phi_next)
        return float(raw_reward) + self.gamma * effective_next_phi - float(phi_current)