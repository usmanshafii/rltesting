"""M3-01 to M3-06: agents. Each test skips until its task lands; replace the skip with the real test."""
import pytest


@pytest.mark.skip(reason="pending M3-01: PPO reaches 475 on CartPole-v1 within 500k steps, 3 seeds")
def test_ppo_solves_cartpole():
    ...


@pytest.mark.skip(reason="pending M3-02: the input audit fires when a privileged channel reaches the actor (I1)")
def test_input_audit():
    ...


@pytest.mark.skip(reason="pending M3-04: each critic switch passes exactly its channels; shuffled uses another episode")
def test_critic_variants():
    ...


@pytest.mark.skip(reason="pending M3-06: imitation loss zero where o = 0; shaping sums to -Phi(s0); no declare when off")
def test_dagger_mask_and_shaping():
    ...
