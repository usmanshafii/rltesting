"""DAgger imitation with step mask on student confirmability o == 1 (M3-06)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.agents.inputs import audit_actor_inputs, flatten_dict_tensors


def compute_masked_imitation_loss(
    student_logits: torch.Tensor,
    teacher_actions: torch.Tensor,
    confirmable_mask_o: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss applied ONLY on steps where student confirmability o == 1.

    If no steps have o == 1 in the batch, returns 0.0 (zero gradient, zero loss).
    """
    mask = (confirmable_mask_o.reshape(-1) > 0.5).to(dtype=torch.float32, device=student_logits.device)
    active_count = mask.sum()
    if active_count <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=student_logits.device)

    per_step_ce = F.cross_entropy(student_logits, teacher_actions.reshape(-1).long(), reduction="none")
    return (per_step_ce * mask).sum() / active_count


def dagger_step_update(
    student_policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    trajectories: List[Tuple[Dict[str, np.ndarray], int, float]],
    teacher_action_fn: Callable[[Dict[str, np.ndarray], Dict[str, np.ndarray]], int],
    priv_trajectories: List[Dict[str, np.ndarray]],
    device: torch.device = torch.device("cpu"),
) -> float:
    """Relabel visited states with teacher actions and update student only where o == 1."""
    if not trajectories:
        return 0.0

    actor_batches: List[torch.Tensor] = []
    teacher_acts: List[int] = []
    o_masks: List[float] = []

    for actor_obs, _action_taken, _reward in trajectories:
        audit_actor_inputs(actor_obs)
        actor_batches.append(flatten_dict_tensors(actor_obs, sorted(actor_obs.keys()), device=device).squeeze(0))

    for priv_obs in priv_trajectories:
        o_val = float(np.max(priv_obs.get("confirmable_o", priv_obs.get("confirmable_mask", 0.0))))
        o_masks.append(1.0 if o_val > 0.5 else 0.0)

    while len(o_masks) < len(actor_batches):
        o_masks.append(0.0)

    for idx, (actor_obs, _, _) in enumerate(trajectories):
        priv_obs = priv_trajectories[idx] if idx < len(priv_trajectories) else {}
        t_act = teacher_action_fn(actor_obs, priv_obs)
        teacher_acts.append(int(t_act))

    student_policy.train()
    actor_x = torch.stack(actor_batches, dim=0)
    logits = student_policy.actor(actor_x) if hasattr(student_policy, "actor") else student_policy(actor_x)
    t_tensor = torch.as_tensor(teacher_acts, dtype=torch.long, device=device)
    o_tensor = torch.as_tensor(o_masks, dtype=torch.float32, device=device)

    loss = compute_masked_imitation_loss(logits, t_tensor, o_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())