"""Actor input audit, privileged-signal gating, and critic input builders (M3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

try:
    from rl import contracts as C
except ImportError:  # Fallback before M1-02 / M4-02 merges rl/contracts.py
    C = None  # type: ignore[assignment]


DEFAULT_STUDENT_KEYS: tuple[str, ...] = (
    "rgb_llr",
    "uncooled_lwir_llr",
    "time_since_look",
    "pointing",
    "zoom",
    "time_left",
)

DEFAULT_PRIVILEGED_KEYS: tuple[str, ...] = (
    "cooled_lwir_llr",
    "mwir_llr",
    "mwir_no_plume_llr",
    "true_state",
    "drone_pos",
)

CRITIC_VARIANTS: tuple[str, ...] = (
    "symmetric",
    "gated",
    "ungated",
    "cooled_lwir_only",
    "mwir_only",
    "shuffled",
    "without_h",
    "truth",
)


class InputAuditError(AssertionError):
    """Raised when privileged information leaks into the student actor's input."""


def get_student_keys() -> tuple[str, ...]:
    if C is not None and hasattr(C, "STUDENT_OBS_KEYS"):
        return tuple(C.STUDENT_OBS_KEYS)
    return DEFAULT_STUDENT_KEYS


def get_privileged_keys() -> tuple[str, ...]:
    if C is not None and hasattr(C, "PRIVILEGED_KEYS"):
        return tuple(C.PRIVILEGED_KEYS)
    return DEFAULT_PRIVILEGED_KEYS


def audit_actor_inputs(
    obs_dict: Dict[str, Any],
    allowed_keys: Optional[Sequence[str]] = None,
    forbidden_keys: Optional[Sequence[str]] = None,
) -> None:
    """Assert that no privileged channel or key reaches the actor (Invariant I1)."""
    allowed = set(allowed_keys) if allowed_keys is not None else set(get_student_keys())
    forbidden = (
        set(forbidden_keys)
        if forbidden_keys is not None
        else (set(get_privileged_keys()) - allowed)
    )

    leaked = sorted(k for k in obs_dict.keys() if k in forbidden)
    if leaked:
        raise InputAuditError(f"Input audit failed! Privileged keys in actor input: {leaked}")

    unexpected = sorted(k for k in obs_dict.keys() if k not in allowed)
    if unexpected:
        raise InputAuditError(f"Input audit failed! Non-student keys in actor input: {unexpected}")


@dataclass(frozen=True)
class GateConfig:
    """Controls Filter 1 (shared photons, no plume/glint leak) & Filter 2 (confirmable o=1)."""

    gated: bool = True
    allow_cooled_lwir: bool = True
    allow_mwir: bool = True
    use_truth: bool = False


def apply_privileged_gate(
    priv_dict: Dict[str, np.ndarray],
    gate_cfg: GateConfig,
) -> Dict[str, np.ndarray]:
    """Filter privileged maps by band selection, shared-photon mask, and o==1 confirmability."""
    out: Dict[str, np.ndarray] = {}

    # Confirmable mask o in {0, 1} per cell/step (Filter 2: confirmable by student)
    confirmable_mask = priv_dict.get("confirmable_mask", None)

    for key, val in priv_dict.items():
        if key == "confirmable_mask":
            continue
        arr = np.asarray(val, dtype=np.float32).copy()

        # Band filter
        is_cooled = "cooled" in key
        is_mwir = "mwir" in key
        is_truth = key in ("true_state", "drone_pos")

        if is_truth and not gate_cfg.use_truth:
            continue
        if is_cooled and not gate_cfg.allow_cooled_lwir:
            arr = np.zeros_like(arr)
        if is_mwir and not gate_cfg.allow_mwir:
            arr = np.zeros_like(arr)

        # If gated, zero out non-confirmable entries (o == 0) and any raw plume channel
        if gate_cfg.gated and not is_truth:
            if "plume" in key and "no_plume" not in key:
                arr = np.zeros_like(arr)
            if confirmable_mask is not None:
                mask_arr = np.asarray(confirmable_mask, dtype=np.float32)
                # Broadcast mask to arr shape if compatible
                while mask_arr.ndim < arr.ndim:
                    mask_arr = np.expand_dims(mask_arr, axis=-1)
                arr = arr * (mask_arr > 0.5)

        out[key] = arr

    return out


def flatten_dict_tensors(d: Dict[str, Any], keys: Sequence[str], device: torch.device) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for k in keys:
        if k not in d:
            continue
        t = torch.as_tensor(d[k], dtype=torch.float32, device=device)
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(0)
        else:
            t = t.reshape(t.shape[0], -1)
        parts.append(t)
    if not parts:
        raise ValueError("No tensors found to flatten for network input.")
    return torch.cat(parts, dim=-1)


def build_critic_input(
    variant: str,
    student_obs: Dict[str, np.ndarray],
    priv_obs: Dict[str, np.ndarray],
    device: torch.device,
    shuffled_priv_obs: Optional[Dict[str, np.ndarray]] = None,
) -> torch.Tensor:
    """Build critic tensor according to the 8 configured ablation/core switches (M3-04)."""
    if variant not in CRITIC_VARIANTS:
        raise ValueError(f"Unknown critic variant '{variant}'. Expected one of {CRITIC_VARIANTS}")

    student_keys = get_student_keys()
    h_tensor = flatten_dict_tensors(student_obs, student_keys, device=device)

    source_priv = priv_obs
    if variant == "shuffled":
        if shuffled_priv_obs is None:
            raise ValueError("Critic variant 'shuffled' requires shuffled_priv_obs from another episode.")
        source_priv = shuffled_priv_obs

    if variant == "symmetric":
        return h_tensor

    if variant == "gated":
        gated = apply_privileged_gate(source_priv, GateConfig(gated=True, allow_cooled_lwir=True, allow_mwir=True))
        p_tensor = flatten_dict_tensors(gated, sorted(gated.keys()), device=device)
        return torch.cat([h_tensor, p_tensor], dim=-1)

    if variant == "ungated":
        ungated = apply_privileged_gate(source_priv, GateConfig(gated=False, allow_cooled_lwir=True, allow_mwir=True))
        p_tensor = flatten_dict_tensors(ungated, sorted(ungated.keys()), device=device)
        return torch.cat([h_tensor, p_tensor], dim=-1)

    if variant == "cooled_lwir_only":
        cooled = apply_privileged_gate(source_priv, GateConfig(gated=True, allow_cooled_lwir=True, allow_mwir=False))
        cooled_keys = [k for k in sorted(cooled.keys()) if "cooled" in k]
        p_tensor = flatten_dict_tensors(cooled, cooled_keys, device=device)
        return torch.cat([h_tensor, p_tensor], dim=-1)

    if variant == "mwir_only":
        mwir = apply_privileged_gate(source_priv, GateConfig(gated=True, allow_cooled_lwir=False, allow_mwir=True))
        mwir_keys = [k for k in sorted(mwir.keys()) if "mwir" in k]
        p_tensor = flatten_dict_tensors(mwir, mwir_keys, device=device)
        return torch.cat([h_tensor, p_tensor], dim=-1)

    if variant == "shuffled":
        gated_shuf = apply_privileged_gate(source_priv, GateConfig(gated=True, allow_cooled_lwir=True, allow_mwir=True))
        p_tensor = flatten_dict_tensors(gated_shuf, sorted(gated_shuf.keys()), device=device)
        return torch.cat([h_tensor, p_tensor], dim=-1)

    if variant == "without_h":
        gated = apply_privileged_gate(source_priv, GateConfig(gated=True, allow_cooled_lwir=True, allow_mwir=True))
        return flatten_dict_tensors(gated, sorted(gated.keys()), device=device)

    # variant == "truth"
    truth_gated = apply_privileged_gate(
        source_priv,
        GateConfig(gated=False, allow_cooled_lwir=False, allow_mwir=False, use_truth=True),
    )
    truth_keys = [k for k in ("true_state", "drone_pos") if k in truth_gated]
    t_tensor = flatten_dict_tensors(truth_gated, truth_keys, device=device)
    return torch.cat([h_tensor, t_tensor], dim=-1)


class ActorCriticNet(nn.Module):
    """Separate MLP/CNN-ready actor and critic heads (small CPU-friendly size)."""

    def __init__(self, actor_dim: int, critic_dim: int, num_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(actor_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_value(self, critic_x: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_x).squeeze(-1)

    def get_action_and_value(
        self,
        actor_x: torch.Tensor,
        critic_x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(actor_x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(critic_x).squeeze(-1)
        return action, log_prob, entropy, value