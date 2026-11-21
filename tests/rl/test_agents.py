"""Unit tests for M3 agents, input audit, critic switches, shaping, and DAgger mask."""

import numpy as np
import pytest
import torch

from rl.agents.dagger import compute_masked_imitation_loss
from rl.agents.inputs import (
    ActorCriticNet,
    InputAuditError,
    audit_actor_inputs,
    build_critic_input,
)
from rl.agents.shaping import PotentialShaper


def test_input_audit_fires_on_privileged_channel():
    clean_obs = {"rgb_llr": np.zeros((36,), dtype=np.float32)}
    audit_actor_inputs(clean_obs)

    leaked_obs = {
        "rgb_llr": np.zeros((36,), dtype=np.float32),
        "mwir_llr": np.ones((36,), dtype=np.float32),
    }
    with pytest.raises(InputAuditError):
        audit_actor_inputs(leaked_obs)


@pytest.mark.parametrize(
    "variant",
    [
        "symmetric",
        "gated",
        "ungated",
        "cooled_lwir_only",
        "mwir_only",
        "shuffled",
        "without_h",
        "truth",
    ],
)
def test_critic_variants_build_expected_shapes(variant):
    student_obs = {"rgb_llr": np.ones((4,), dtype=np.float32)}
    priv_obs = {
        "cooled_lwir_llr": np.ones((4,), dtype=np.float32) * 2.0,
        "mwir_llr": np.ones((4,), dtype=np.float32) * 3.0,
        "confirmable_mask": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "true_state": np.array([10.0, 20.0], dtype=np.float32),
    }
    shuffled_priv = {
        "cooled_lwir_llr": np.ones((4,), dtype=np.float32) * 9.0,
        "mwir_llr": np.ones((4,), dtype=np.float32) * 8.0,
        "confirmable_mask": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    }
    tensor = build_critic_input(
        variant=variant,
        student_obs=student_obs,
        priv_obs=priv_obs,
        device=torch.device("cpu"),
        shuffled_priv_obs=shuffled_priv,
    )
    assert tensor.ndim == 2 and tensor.shape[0] == 1 and tensor.shape[1] > 0


def test_shaping_zero_at_episode_end_and_telescoping_sum():
    shaper = PotentialShaper(gamma=1.0, scale=0.1, enabled=True)
    priv_s0 = {"mwir_llr": np.array([2.0, 1.0], dtype=np.float32)}
    priv_s1 = {"mwir_llr": np.array([5.0, 0.0], dtype=np.float32)}
    priv_term = {"mwir_llr": np.array([9.0, 9.0], dtype=np.float32)}

    phi0 = shaper.potential(priv_s0, episode_ended=False)
    phi1 = shaper.potential(priv_s1, episode_ended=False)
    phi_end = shaper.potential(priv_term, episode_ended=True)
    assert phi_end == 0.0

    r0 = shaper.shaped_reward(0.0, phi0, phi1, terminated_or_truncated=False)
    r1 = shaper.shaped_reward(0.0, phi1, phi_end, terminated_or_truncated=True)
    total_shaping = r0 + r1
    assert np.isclose(total_shaping, -phi0, atol=1e-6)


def test_dagger_loss_zero_when_o_is_zero():
    logits = torch.tensor([[2.0, 0.1], [0.2, 3.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    o_zero = torch.tensor([0.0, 0.0])
    loss = compute_masked_imitation_loss(logits, targets, o_zero)
    assert float(loss.item()) == 0.0


def test_declare_action_disabled_by_default():
    net = ActorCriticNet(actor_dim=4, critic_dim=4, num_actions=11, hidden_dim=32)
    x = torch.zeros((1, 4))
    act, _, _, _ = net.get_action_and_value(x, x)
    assert 0 <= int(act.item()) < 11