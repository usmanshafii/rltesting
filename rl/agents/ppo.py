"""CleanRL-style single-file PPO with CPU thread control, input audit, and CSV logging."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.agents.inputs import (
    ActorCriticNet,
    audit_actor_inputs,
    build_critic_input,
    flatten_dict_tensors,
)
from rl.agents.shaping import PotentialShaper

try:
    from rl import contracts as C
except ImportError:
    C = None  # type: ignore[assignment]


@dataclass
class PPOConfig:
    seed: int = 1
    total_timesteps: int = 500_000
    num_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    critic_variant: str = "symmetric"
    shaping_enabled: bool = False
    shaping_scale: float = 0.05
    allow_declare_action: bool = False  # Learned declaration stays off unless Tier-2 flag is enabled
    env_version: str = "stub-v0"
    episode_set_hash: str = "none"
    threshold_a: float = 0.0
    log_dir: Optional[str] = None
    torch_num_threads: int = 1


def config_hash(cfg: PPOConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _extract_actor_dict(obs: Any, info: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if isinstance(obs, dict):
        # Filter out any privileged keys accidentally placed in obs dict before auditing
        priv_keys = set(getattr(C, "PRIVILEGED_KEYS", ())) if C is not None else set()
        return {k: v for k, v in obs.items() if k not in priv_keys}
    return {"obs": np.asarray(obs, dtype=np.float32)}


def _extract_priv_dict(obs: Any, info: Dict[str, Any]) -> Dict[str, np.ndarray]:
    priv: Dict[str, np.ndarray] = {}
    if isinstance(info, dict):
        for k, v in info.items():
            if isinstance(v, (np.ndarray, list, int, float)):
                priv[k] = np.asarray(v, dtype=np.float32)
    if isinstance(obs, dict):
        for k, v in obs.items():
            if "cooled" in k or "mwir" in k or "plume" in k or "confirmable" in k:
                priv[k] = np.asarray(v, dtype=np.float32)
    return priv


def _map_action_to_env(action: int, env: gym.Env, allow_declare: bool) -> int:
    n_actions = int(getattr(env.action_space, "n", 11))
    if not allow_declare and n_actions > 11 and action >= 11:
        return 10  # Dwell fallback when declare action is disabled
    return int(np.clip(action, 0, n_actions - 1))


def train_ppo(
    env: gym.Env,
    cfg: PPOConfig,
    shuffled_priv_pool: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Run PPO training loop and return summary metrics + episode/step logs."""
    torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cpu")
    obs, info = env.reset(seed=cfg.seed)
    actor_dict = _extract_actor_dict(obs, info)
    audit_actor_inputs(actor_dict)

    actor_x0 = flatten_dict_tensors(actor_dict, sorted(actor_dict.keys()), device=device)
    actor_dim = actor_x0.shape[-1]

    priv_dict = _extract_priv_dict(obs, info)
    shuf_priv = shuffled_priv_pool[0] if shuffled_priv_pool else None
    critic_x0 = build_critic_input(cfg.critic_variant, actor_dict, priv_dict, device=device, shuffled_priv_obs=shuf_priv)
    critic_dim = critic_x0.shape[-1]

    raw_n_actions = int(getattr(env.action_space, "n", 2))
    num_actions = raw_n_actions if cfg.allow_declare_action else min(raw_n_actions, 11)

    model = ActorCriticNet(actor_dim=actor_dim, critic_dim=critic_dim, num_actions=num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    shaper = PotentialShaper(gamma=cfg.gamma, scale=cfg.shaping_scale, enabled=cfg.shaping_enabled)

    cfg_h = config_hash(cfg)
    csv_path: Optional[Path] = None
    csv_rows: List[Dict[str, Any]] = []
    if cfg.log_dir is not None:
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(cfg.log_dir) / f"ppo_{cfg.critic_variant}_seed{cfg.seed}_{cfg_h}.csv"

    num_updates = max(1, cfg.total_timesteps // cfg.num_steps)
    global_step = 0
    episode_idx = 0
    ep_return = 0.0
    ep_length = 0
    ep_false_conf = 0
    ep_confirmed = 0
    completed_returns: List[float] = []
    start_time = time.perf_counter()

    phi_curr = shaper.potential(priv_dict, episode_ended=False)

    for update in range(1, num_updates + 1):
        b_actor: List[torch.Tensor] = []
        b_critic: List[torch.Tensor] = []
        b_actions: List[torch.Tensor] = []
        b_logprobs: List[torch.Tensor] = []
        b_rewards: List[float] = []
        b_dones: List[float] = []
        b_values: List[torch.Tensor] = []

        last_entropy = 0.0
        last_vloss = 0.0

        for _ in range(cfg.num_steps):
            if global_step >= cfg.total_timesteps:
                break
            global_step += 1

            actor_dict = _extract_actor_dict(obs, info)
            audit_actor_inputs(actor_dict)
            actor_x = flatten_dict_tensors(actor_dict, sorted(actor_dict.keys()), device=device)

            priv_dict = _extract_priv_dict(obs, info)
            if cfg.critic_variant == "shuffled" and shuffled_priv_pool:
                shuf_priv = shuffled_priv_pool[(global_step + cfg.seed) % len(shuffled_priv_pool)]
            critic_x = build_critic_input(
                cfg.critic_variant, actor_dict, priv_dict, device=device, shuffled_priv_obs=shuf_priv
            )

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(actor_x, critic_x)

            env_action = _map_action_to_env(int(action.item()), env, cfg.allow_declare_action)
            next_obs, raw_reward, terminated, truncated, next_info = env.step(env_action)
            done = bool(terminated or truncated)

            next_priv = _extract_priv_dict(next_obs, next_info)
            phi_next = shaper.potential(next_priv, episode_ended=done)
            reward = shaper.shaped_reward(float(raw_reward), phi_curr, phi_next, terminated_or_truncated=done)

            ep_return += float(raw_reward)
            ep_length += 1
            if isinstance(next_info, dict):
                ep_false_conf += int(next_info.get("false_confirmation", 0))
                if next_info.get("confirmed", False):
                    ep_confirmed = 1

            b_actor.append(actor_x.squeeze(0))
            b_critic.append(critic_x.squeeze(0))
            b_actions.append(action.squeeze(0))
            b_logprobs.append(logprob.squeeze(0))
            b_rewards.append(reward)
            b_dones.append(float(done))
            b_values.append(value.squeeze(0))

            obs, info = next_obs, next_info
            phi_curr = 0.0 if done else phi_next

            if done:
                completed_returns.append(ep_return)
                elapsed_hours = max((time.perf_counter() - start_time) / 3600.0, 1e-6)
                row = {
                    "env_version": cfg.env_version,
                    "episode_set_hash": cfg.episode_set_hash,
                    "threshold_a": cfg.threshold_a,
                    "config_hash": cfg_h,
                    "seed": cfg.seed,
                    "critic_variant": cfg.critic_variant,
                    "global_step": global_step,
                    "episode": episode_idx,
                    "ep_return": ep_return,
                    "ep_length": ep_length,
                    "confirmed": ep_confirmed,
                    "false_confirmations": ep_false_conf,
                    "false_conf_per_hour": ep_false_conf / elapsed_hours,
                    "entropy": last_entropy,
                    "value_loss": last_vloss,
                }
                csv_rows.append(row)
                episode_idx += 1
                ep_return = 0.0
                ep_length = 0
                ep_false_conf = 0
                ep_confirmed = 0

                obs, info = env.reset(seed=cfg.seed + episode_idx)
                priv_dict = _extract_priv_dict(obs, info)
                phi_curr = shaper.potential(priv_dict, episode_ended=False)

        if not b_rewards:
            break

        # Bootstrap value
        with torch.no_grad():
            next_actor_dict = _extract_actor_dict(obs, info)
            audit_actor_inputs(next_actor_dict)
            next_actor_x = flatten_dict_tensors(next_actor_dict, sorted(next_actor_dict.keys()), device=device)
            next_priv_dict = _extract_priv_dict(obs, info)
            next_critic_x = build_critic_input(
                cfg.critic_variant, next_actor_dict, next_priv_dict, device=device, shuffled_priv_obs=shuf_priv
            )
            next_value = model.get_value(next_critic_x).item()

        rewards_arr = np.asarray(b_rewards, dtype=np.float32)
        dones_arr = np.asarray(b_dones, dtype=np.float32)
        values_arr = torch.stack(b_values).detach().cpu().numpy()

        advantages = np.zeros_like(rewards_arr, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards_arr))):
            if t == len(rewards_arr) - 1:
                next_non_terminal = 1.0 - dones_arr[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones_arr[t]
                next_val = values_arr[t + 1]
            delta = rewards_arr[t] + cfg.gamma * next_val * next_non_terminal - values_arr[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_arr
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        act_t = torch.stack(b_actor).to(device)
        crit_t = torch.stack(b_critic).to(device)
        actions_t = torch.stack(b_actions).to(device)
        old_logprobs_t = torch.stack(b_logprobs).to(device)

        batch_size = len(rewards_arr)
        mb_size = min(cfg.minibatch_size, batch_size)
        for _ in range(cfg.update_epochs):
            idxs = np.random.permutation(batch_size)
            for start in range(0, batch_size, mb_size):
                mb_idx = idxs[start : start + mb_size]
                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    act_t[mb_idx], crit_t[mb_idx], actions_t[mb_idx]
                )
                logratio = new_logprob - old_logprobs_t[mb_idx]
                ratio = torch.exp(logratio)

                mb_adv = adv_t[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - ret_t[mb_idx]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                last_entropy = float(ent_loss.item())
                last_vloss = float(v_loss.item())

    if csv_path is not None and csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    return {
        "model": model,
        "global_step": global_step,
        "completed_returns": completed_returns,
        "mean_return_last_100": float(np.mean(completed_returns[-100:])) if completed_returns else 0.0,
        "csv_path": str(csv_path) if csv_path else None,
        "rows": csv_rows,
    }