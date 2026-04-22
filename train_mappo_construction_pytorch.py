"""train_mappo_construction_template.py

Fresh shared template for the team to build the full PyTorch MAPPO trainer.

Purpose
-------
This file keeps the parts from the current NumPy trainer that are still useful
right now, while clearly separating:
1) KEEP FOR NOW
2) REFERENCE / LEGACY
3) REPLACE LATER WITH FULL MAPPO
4) TEAM TODO SECTIONS

Important
---------
- This is a WORK TEMPLATE, not the final trainer yet.
- The current NumPy actor/critic update code is kept only as legacy reference.
- The final goal is a PyTorch shared-actor / centralized-critic MAPPO trainer.
- The environment / validation code can stay mostly unchanged for now.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

# ============================================================
# OPTIONAL PYTORCH IMPORTS FOR THE FULL VERSION
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except Exception:
    torch = None
    nn = None
    F = None
    Categorical = None

from construction_scheduling_env import ConstructionSchedulingEnv


# ============================================================
# SECTION C - TRAIN CONFIG
# ============================================================

@dataclass
class TrainConfig:
    """Config for the future full MAPPO trainer."""

    # environment
    grid_rows: int = 10
    grid_cols: int = 10
    robots: int = 6
    heavy_ratio: float = 0.25
    max_steps: int = 200
    travel_speed_cells_per_step: float = 3.0
    travel_reward_weight: float = 0.01

    # training
    seed: int = 7
    episodes: int = 700
    eval_interval: int = 100
    eval_episodes: int = 20
    save_dir: str = "construction_models"
    model_name: str = "construction_mappo_pt.pth"
    plot_interval: int = 10
    plot_path: str = "results/training_curves.png"

    # behavior cloning
    bc_episodes: int = 500
    bc_epochs: int = 30
    bc_batch_size: int = 256

    # network architecture
    actor_hidden_sizes: Tuple[int, int] = (256, 256)
    critic_hidden_sizes: Tuple[int, int] = (256, 256)
    activation: str = "tanh"

    # PPO / MAPPO
    gamma: float = 0.97
    gae_lambda: float = 0.95
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256

    # toggles
    use_gae: bool = True
    normalize_advantages: bool = True
    use_torch: bool = True


# ============================================================
# SECTION E - FULL PYTORCH MODEL SIDE (PERSON 1)
# ============================================================
# TODO [PERSON 1]
# 1. Build PyTorch actor MLP
# 2. Build PyTorch critic MLP
# 3. Add masked action selection
# 4. Return actions, log_probs, entropy, values
# 5. Add save/load for torch model

def get_activation(name: str):
    if nn is None:
        return None
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


class ActorMLP(nn.Module if nn is not None else object):
    """Full PyTorch actor network placeholder."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...], activation: str):
        if nn is None:
            raise RuntimeError("PyTorch is not available.")
        super().__init__()
        act_cls = get_activation(activation)
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), act_cls()])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class CriticMLP(nn.Module if nn is not None else object):
    """Full PyTorch centralized critic placeholder."""

    def __init__(self, state_dim: int, hidden_sizes: Tuple[int, ...], activation: str):
        if nn is None:
            raise RuntimeError("PyTorch is not available.")
        super().__init__()
        act_cls = get_activation(activation)
        layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), act_cls()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class MAPPOAgent(nn.Module if nn is not None else object):
    """Shared actor + centralized critic MAPPO wrapper.

    Person 1 owns this model-side interface:
    - actor: maps each robot observation to action logits
    - critic: maps the global state to a team value estimate
    - action helpers: apply action masks, sample/choose actions, and compute
      log-probabilities needed later by PPO
    """

    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, cfg: TrainConfig):
        if nn is None:
            raise RuntimeError("PyTorch is not available.")
        super().__init__()
        self.action_dim = action_dim
        self.actor = ActorMLP(obs_dim, action_dim, cfg.actor_hidden_sizes, cfg.activation)
        self.critic = CriticMLP(state_dim, cfg.critic_hidden_sizes, cfg.activation)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def _masked_action_distribution(self, obs, action_mask):
        """Build a categorical distribution after removing illegal actions.

        obs shape:
            [N, obs_dim], where N can be num_robots or a flattened rollout batch.

        action_mask shape:
            [N, action_dim], with 1 for legal actions and 0 for illegal actions.
        """
        logits = self.actor(obs)
        masked_logits = logits.masked_fill(action_mask < 0.5, -1.0e9)
        return Categorical(logits=masked_logits), masked_logits

    def get_action_and_logprob(self, obs, action_mask, greedy: bool = False):
        """Select one action for each robot/row in obs.

        Returns:
            actions: [N]
            log_probs: [N], log probability of the selected actions
            entropy: [N], policy entropy for exploration regularization
        """
        dist, masked_logits = self._masked_action_distribution(obs, action_mask)
        if greedy:
            actions = torch.argmax(masked_logits, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, entropy

    def evaluate_actions(self, obs, action_mask, actions):
        """Evaluate already-chosen actions under the current actor.

        PPO needs this during updates: rollout stores old actions, and the
        current actor recomputes their new log-probabilities for the ratio.
        """
        dist, _ = self._masked_action_distribution(obs, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def get_value(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.critic(state)

    def save(self, path: str, cfg: TrainConfig) -> None:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "config": asdict(cfg),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])


# ============================================================
# SECTION F - RETURNS / GAE / DATA SIDE (PERSON 2)
# ============================================================


# TODO [PERSON 2]
# 1. Define rollout buffer fields
# 2. Store obs / states / actions / rewards / dones / masks / log_probs / values
# 3. Compute GAE
# 4. Compute returns
# 5. Flatten / batch / minibatch

class RolloutBuffer:
    """Full rollout buffer placeholder for MAPPO."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.actor_obs = []
        self.actor_actions = []
        self.actor_masks = []
        self.actor_log_probs = []

        self.critic_states = []
        self.critic_rewards = []
        self.critic_dones = []
        self.critic_values = []

        self.advantages = None
        self.returns = None

    def add(self, actor_obs, critic_state, actions, reward, done, action_masks, log_probs, value) -> None:
        self.actor_obs.append(actor_obs)
        self.actor_actions.append(actions)
        self.actor_masks.append(action_masks)
        self.actor_log_probs.append(log_probs)

        self.critic_states.append(critic_state)
        self.critic_rewards.append(reward)
        self.critic_dones.append(done)
        self.critic_values.append(value) 

    def compute_returns_and_advantages(self, last_value, gamma: float, gae_lambda: float, use_gae: bool = True):
        # 将列表转换为 Tensor
        critic_rewards = torch.tensor(np.array(self.critic_rewards), dtype=torch.float32)   # [T]
        critic_values = torch.tensor(np.array(self.critic_values), dtype=torch.float32)     # [T]
        critic_dones = torch.tensor(np.array(self.critic_dones), dtype=torch.float32)       # [T]

        last_value = torch.as_tensor(last_value, dtype=torch.float32).view(())

        if use_gae:
            next_values = torch.cat([critic_values[1:], last_value.unsqueeze(0)], dim=0)
            advantages = torch.zeros_like(critic_rewards)
            last_gae = torch.tensor(0.0, dtype=torch.float32)

            for t in reversed(range(len(critic_rewards))):
                delta = critic_rewards[t] + gamma * next_values[t] * (1.0 - critic_dones[t]) - critic_values[t]
                last_gae = delta + gamma * gae_lambda * (1.0 - critic_dones[t]) * last_gae
                advantages[t] = last_gae

            returns = advantages + critic_values
        else:
            returns = torch.zeros_like(critic_rewards)
            running_return = last_value
            for t in reversed(range(len(critic_rewards))):
                running_return = critic_rewards[t] + gamma * running_return * (1.0 - critic_dones[t])
                returns[t] = running_return
            advantages = returns - critic_values

        num_robots = np.array(self.actor_actions).shape[1]
        self.advantages = advantages.repeat_interleave(num_robots)
        self.returns = returns.repeat_interleave(num_robots)

    def get_training_batches(self, minibatch_size: int):
        """
        一次性处理所有数据，并以列表形式返回所有训练批次。
        """

        def flatten_actor_tensor(data_list, dtype=torch.float32):
            arr = np.array(data_list)          # [T, N, ...]
            t = torch.tensor(arr, dtype=dtype)
            return t.view(-1, *t.shape[2:])    # [T*N, ...]

        obs_f = flatten_actor_tensor(self.actor_obs)
        actions_f = flatten_actor_tensor(self.actor_actions, dtype=torch.long)
        log_probs_f = flatten_actor_tensor(self.actor_log_probs)
        masks_f = flatten_actor_tensor(self.actor_masks)

        critic_states = torch.tensor(np.array(self.critic_states), dtype=torch.float32)  # [T, state_dim]
        num_robots = np.array(self.actor_actions).shape[1]
        states_f = critic_states.repeat_interleave(num_robots, dim=0)                    # [T*N, state_dim]

        advantages_f = self.advantages.view(-1)
        returns_f = self.returns.view(-1)
        # 注意：优势标准化放在 ppo_update(...) 中按 cfg.normalize_advantages 控制，
        # 这里保持原始 advantages，不重复标准化。

        # 2. 随机洗牌
        total_samples = obs_f.size(0)
        indices = torch.randperm(total_samples)

        # 3. 核心改变：创建一个列表，把切好的“肉”都装进去
        all_batches = []

        for start in range(0, total_samples, minibatch_size):
            batch_idx = indices[start: start + minibatch_size]

            # 把这一批数据打包成一个字典，和 ppo_update(...) 的读取方式一致
            batch_data = {
                "obs": obs_f[batch_idx],
                "states": states_f[batch_idx],
                "actions": actions_f[batch_idx],
                "old_log_probs": log_probs_f[batch_idx],
                "advantages": advantages_f[batch_idx],
                "returns": returns_f[batch_idx],
                "action_masks": masks_f[batch_idx],
            }

            # 塞进大列表
            all_batches.append(batch_data)

        # 4. 直接返回这个大列表
        return all_batches

# 调用示例：
# obs = all_batches[0]["obs"]


# ============================================================
# SECTION G - EVALUATION / BASELINE HELPERS WE CAN MOSTLY KEEP
# ============================================================

def update_training_curve_plot(history_path: str, output_path: str) -> None:
    """Save a compact training-curve figure that can be refreshed while training."""
    if not os.path.exists(history_path):
        return

    rows = []
    with open(history_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return

    episodes = np.asarray([int(row["episode"]) for row in rows], dtype=np.int32)
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float32)
    lengths = np.asarray([float(row["length"]) for row in rows], dtype=np.float32)
    successes = np.asarray([float(row["success"]) for row in rows], dtype=np.float32)
    critic_loss = np.asarray([float(row["critic_loss"]) for row in rows], dtype=np.float32)
    policy_loss = np.asarray([float(row["policy_loss"]) for row in rows], dtype=np.float32)

    def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
        if len(values) < 2:
            return values
        window = min(window, len(values))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        prefix = np.full(window - 1, values[0], dtype=np.float32)
        padded = np.concatenate([prefix, values])
        return np.convolve(padded, kernel, mode="valid")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=140)
    fig.patch.set_facecolor("white")

    axes[0, 0].scatter(episodes, rewards, color="#9aa3ad", s=13, alpha=0.7)
    axes[0, 0].plot(episodes, rewards, color="#c4c9cf", linewidth=0.8, alpha=0.7)
    axes[0, 0].plot(episodes, moving_average(rewards), color="#2f6f9f", linewidth=2.0)
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")

    axes[0, 1].scatter(episodes, lengths, color="#9aa3ad", s=13, alpha=0.7)
    axes[0, 1].plot(episodes, lengths, color="#c4c9cf", linewidth=0.8, alpha=0.7)
    axes[0, 1].plot(episodes, moving_average(lengths), color="#7d5ba6", linewidth=2.0)
    axes[0, 1].set_title("Episode Length")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")

    axes[1, 0].plot(episodes, moving_average(successes, window=20), color="#3f8f56", linewidth=2.0)
    axes[1, 0].scatter(episodes, successes, color="#8dc79c", s=13, alpha=0.7)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title("Episode Success")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Success")

    axes[1, 1].plot(episodes, critic_loss, color="#ba4a4a", linewidth=1.4, label="critic")
    axes[1, 1].plot(episodes, policy_loss, color="#4e7fba", linewidth=1.4, label="policy")
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend(frameon=False)

    for ax in axes.reshape(-1):
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("10x10 Multi-Robot Construction Scheduling Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def greedy_baseline_action(env: ConstructionSchedulingEnv) -> np.ndarray:
    """Capability- and distance-aware online greedy scheduler.

    STATUS:
    - KEEP FOR NOW.
    """
    available = env.dependency_mask()
    actions = np.ones(env.num_robots, dtype=np.int64) * env.wait_action

    idle = np.where(env.robot_remaining_time <= 0)[0]
    if len(idle) == 0:
        return actions

    heavy_available = np.where((available > 0.5) & (env.heavy_mask > 0.5))[0]
    normal_available = np.where((available > 0.5) & (env.heavy_mask < 0.5))[0]
    distances = env.robot_module_distances()

    remaining_idle = list(int(x) for x in idle)

    if env.crane_cooldown == 0 and len(heavy_available) > 0 and len(remaining_idle) >= 2:
        best_choice = None
        best_cost = float("inf")
        for module in heavy_available:
            for idx, rid_a in enumerate(remaining_idle):
                for rid_b in remaining_idle[idx + 1 :]:
                    pair = [rid_a, rid_b]
                    travel_time = float(np.max(distances[pair, int(module)])) / env.travel_speed_cells_per_step
                    capability_cost = float(np.mean(env.robot_capabilities[pair, 1]))
                    cost = travel_time + capability_cost + 0.02 * float(np.sum(distances[pair, int(module)]))
                    if cost < best_cost:
                        best_cost = cost
                        best_choice = (int(module), pair)
        if best_choice is not None:
            module, chosen = best_choice
            actions[chosen] = module
            remaining_idle = [rid for rid in remaining_idle if rid not in set(int(x) for x in chosen)]

    if len(normal_available) > 0 and remaining_idle:
        candidate_pairs = []
        for rid in remaining_idle:
            for module in normal_available:
                cost = (
                    float(distances[rid, int(module)]) / env.travel_speed_cells_per_step
                    + float(env.robot_capabilities[rid, 0])
                )
                candidate_pairs.append((cost, rid, int(module)))
        candidate_pairs.sort(key=lambda item: item[0])
        assigned_robots = set()
        assigned_modules = set()
        for _, rid, module in candidate_pairs:
            if rid in assigned_robots or module in assigned_modules:
                continue
            actions[rid] = module
            assigned_robots.add(rid)
            assigned_modules.add(module)
            if len(assigned_robots) == len(remaining_idle):
                break
    return actions

# Returns safer joint action for all robots
def policy_guided_safe_action(env: ConstructionSchedulingEnv, agent, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
    #Feasible decoding helper used mainly in evaluation.
    # Changed to PyTorch ver

    if torch is not None and hasattr(agent, "actor") and isinstance(getattr(agent, "actor"), nn.Module):
        device = next(agent.actor.parameters()).device
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits_t = agent.actor(obs_t)
            masked_logits_t = logits_t.masked_fill(mask_t < 0.5, -1.0e9)
            probs = torch.softmax(masked_logits_t, dim=-1).cpu().numpy()
    else:
        raise NotImplementedError(
            "policy_guided_safe_action(...) requires a PyTorch actor."
        )

    actions = np.ones(env.num_robots, dtype=np.int64) * env.wait_action

    available = env.dependency_mask()
    distances = env.robot_module_distances()
    idle = [int(rid) for rid in np.where(env.robot_remaining_time <= 0)[0]]
    if not idle:
        return actions

    heavy_available = [
        int(module)
        for module in np.where((available > 0.5) & (env.heavy_mask > 0.5))[0]
        if masks[idle, module].max(initial=0.0) > 0.5
    ]
    normal_available = [
        int(module)
        for module in np.where((available > 0.5) & (env.heavy_mask < 0.5))[0]
        if masks[idle, module].max(initial=0.0) > 0.5
    ]

    remaining_idle = set(idle)

    if env.crane_cooldown == 0 and len(remaining_idle) >= 2 and heavy_available:
        best_choice = None
        best_score = -np.inf
        for module in heavy_available:
            ranked = sorted(
                remaining_idle,
                key=lambda rid: float(probs[rid, module])
                / (
                    max(0.25, float(env.robot_capabilities[rid, 1]))
                    * (1.0 + float(distances[rid, module]) / env.max_travel_distance)
                ),
                reverse=True,
            )
            if len(ranked) < 2:
                continue
            pair = ranked[:2]
            score = float(probs[pair[0], module] + probs[pair[1], module])
            if score > best_score:
                best_score = score
                best_choice = (module, pair)
        if best_choice is not None:
            module, pair = best_choice
            for rid in pair:
                actions[rid] = module
                remaining_idle.remove(rid)

    normal_pairs = []
    for rid in remaining_idle:
        for module in normal_available:
            score = float(probs[rid, module]) / (
                max(0.25, float(env.robot_capabilities[rid, 0]))
                * (1.0 + float(distances[rid, module]) / env.max_travel_distance)
            )
            normal_pairs.append((score, rid, module))
    normal_pairs.sort(reverse=True)

    used_modules = set()
    for _, rid, module in normal_pairs:
        if rid not in remaining_idle or module in used_modules:
            continue
        actions[rid] = module
        remaining_idle.remove(rid)
        used_modules.add(module)
        if not remaining_idle:
            break

    return actions


# raw-policy HELPER
def raw_policy_action(env: ConstructionSchedulingEnv, agent, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Use the actor directly with greedy masked argmax, no safe decoder."""
    if torch is None or not hasattr(agent, "actor") or not isinstance(getattr(agent, "actor"), nn.Module):
        raise NotImplementedError("raw_policy_action(...) requires a PyTorch actor.")

    device = next(agent.actor.parameters()).device
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(masks, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits_t = agent.actor(obs_t)
        masked_logits_t = logits_t.masked_fill(mask_t < 0.5, -1.0e9)
        actions_t = torch.argmax(masked_logits_t, dim=-1)

    return actions_t.cpu().numpy()


def pretrain_actor_with_greedy(env: ConstructionSchedulingEnv, agent, episodes: int, cfg: TrainConfig | None = None) -> None:
    """Behavior cloning warm start.

    STATUS:
    - Collect expert actions from the greedy baseline.
    - Train the PyTorch actor with supervised cross-entropy.
    - This gives PPO a feasible initial policy before policy-gradient updates.
    """
    if episodes <= 0:
        return
    if torch is None or F is None:
        raise RuntimeError("PyTorch is not available.")
    if not hasattr(agent, "actor") or not isinstance(getattr(agent, "actor"), nn.Module):
        raise NotImplementedError("Behavior cloning warm start requires the PyTorch MAPPOAgent.")

    # BC Hyperparameters
    # Use TrainConfig when available; otherwise fall back to conservative defaults.
    bc_epochs = cfg.bc_epochs if cfg is not None else 10
    bc_batch_size = cfg.bc_batch_size if cfg is not None else 256
    max_steps = cfg.max_steps if cfg is not None else env.max_steps
    device = next(agent.actor.parameters()).device

    obs_samples = []
    mask_samples = []
    action_samples = []

    # Expert Data Collection
    # The greedy scheduler knows the environment state directly and provides a
    # feasible joint action. Each robot contributes one supervised sample.
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            masks = env.action_mask()
            expert_actions = greedy_baseline_action(env)

            obs_samples.append(obs.astype(np.float32))
            mask_samples.append(masks.astype(np.float32))
            action_samples.append(expert_actions.astype(np.int64))

            obs, _, _, done, _ = env.step(expert_actions)
            step_count += 1

    if not obs_samples:
        return

    # Flatten Episode Data
    # Shapes become [samples * robots, ...], matching the shared actor format.
    obs_tensor = torch.as_tensor(np.asarray(obs_samples, dtype=np.float32), device=device).view(-1, obs_samples[0].shape[-1])
    mask_tensor = torch.as_tensor(np.asarray(mask_samples, dtype=np.float32), device=device).view(-1, mask_samples[0].shape[-1])
    action_tensor = torch.as_tensor(np.asarray(action_samples, dtype=np.int64), device=device).view(-1)

    # Supervised Actor Training
    # Masked logits prevent the cloning loss from assigning probability to
    # unavailable actions.
    total_samples = obs_tensor.shape[0]
    for epoch in range(1, bc_epochs + 1):
        indices = torch.randperm(total_samples, device=device)
        losses = []
        for start in range(0, total_samples, bc_batch_size):
            batch_idx = indices[start:start + bc_batch_size]
            logits = agent.actor(obs_tensor[batch_idx])
            masked_logits = logits.masked_fill(mask_tensor[batch_idx] < 0.5, -1.0e9)
            loss = F.cross_entropy(masked_logits, action_tensor[batch_idx])

            agent.actor_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), cfg.max_grad_norm if cfg is not None else 0.5)
            agent.actor_opt.step()
            losses.append(float(loss.item()))

        print(f"bc epoch {epoch:03d}/{bc_epochs:03d} | loss {float(np.mean(losses)):.4f} | samples {total_samples}")


# ============================================================
# SECTION H - ROLLOUT COLLECTION (PERSON 2 + PERSON 3)
# ============================================================
# TODO [PERSON 2 / PERSON 3]
# Decide exact outputs:
# - actor observation format
# - critic global-state format
# - action representation
# - buffer fields and tensor shapes

def collect_episode(env: ConstructionSchedulingEnv, agent: MAPPOAgent, buffer: RolloutBuffer, cfg: TrainConfig, greedy: bool = False):
    obs, state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    episode_rewards = []
    infos = []

    while not done and step_count < cfg.max_steps:
        # A. 获取动作掩码 (防止选到不可行的任务)
        masks = env.action_mask() # [num_robots, action_dim]

        # B. 构造全局状态
        # state = obs.flatten() # 原写法: [num_robots * obs_dim]，会和 critic 的 state_dim 不匹配

        # C. 将 numpy 转为 torch tensor，并移动到和 agent 一样的 device
        device = next(agent.parameters()).device
        obs_t = torch.from_numpy(obs).float().to(device)
        mask_t = torch.from_numpy(masks).float().to(device)
        state_t = torch.from_numpy(state).float().to(device)

        with torch.no_grad():
            # 拿到动作、动作的 log 概率
            actions_t, log_probs_t, _ = agent.get_action_and_logprob(obs_t, mask_t, greedy=greedy)
            # 拿到 Critic 对局势的评估值
            value_t = agent.get_value(state_t)

        # D. 环境执行
        actions_np = actions_t.cpu().numpy()
        next_obs, next_state, reward, done, info = env.step(actions_np)
        buffer.add(
            actor_obs=obs,
            critic_state=state,
            actions=actions_np,
            reward=float(reward),
            done=float(done),
            action_masks=masks,
            log_probs=log_probs_t.cpu().numpy(),
            value=float(value_t.item()),
        )

        obs = next_obs
        state = next_state
        total_reward += reward
        episode_rewards.append(reward)
        infos.append(info)
        step_count += 1

    return {
        "buffer": buffer,
        "reward": total_reward,
        "length": step_count,
        "rewards": episode_rewards,
        "infos": infos,
        "last_state": state,
        "done": done,
    }


# ============================================================
# SECTION I - PPO / MAPPO UPDATE SIDE (PERSON 3)
# ============================================================
# TODO [PERSON 3]
# 1. PPO clipped objective
# 2. Critic loss
# 3. Entropy regularization
# 4. Optimizer step
# 5. Gradient clipping
# 6. Multiple PPO epochs
# 7. Minibatch updates

#---------------------------- Standardize Model and Data into PyTorch and same Device ---------------------------
# Check which device model is on
def _infer_device(agent: MAPPOAgent):
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    return next(agent.parameters()).device


# Converts data into PyTorch tensor and move to correct device
def _to_torch(x, device, dtype=None):
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    if torch.is_tensor(x):
        tensor = x.to(device)
    else:
        tensor = torch.as_tensor(x, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor
# ---------------------------------------------------------------------------------------------------------------------

# Standardize Advantage Values
def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

# Extracts parts needed in PPO from Model
def _extract_eval_logprob_entropy(eval_out):
    if isinstance(eval_out, tuple):
        if len(eval_out) < 2:
            raise ValueError("agent.evaluate_actions(...) must return at least (log_probs, entropy).")
        return eval_out[0], eval_out[1]
    raise ValueError("agent.evaluate_actions(...) should return a tuple like (log_probs, entropy).")


# ----------------------------------- Actual PPO Learning -------------------------------------------------
def ppo_update(agent: MAPPOAgent, buffer: RolloutBuffer, cfg: TrainConfig) -> Dict[str, float]:
    """
    PPO / MAPPO inner learning loop.

    Assumes:
    - buffer.get_training_batches(cfg.minibatch_size) yields dict minibatches
    - agent.evaluate_actions(obs, action_masks, actions) -> (log_probs, entropy)
    - agent.get_value(states) -> values
    """

    # Initial Checks
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    if not hasattr(buffer, "get_training_batches"):
        raise AttributeError("Buffer must provide get_training_batches(minibatch_size).")

    # get model's device
    device = _infer_device(agent)

    # Metrics List - collect results over epochs and minibatches
    policy_losses = []
    value_losses = []
    entropies = []
    total_losses = []
    ratio_means = []

    # PPO Epoch Loop - reuse same rollout
    for _ in range(cfg.ppo_epochs):

        # Minibatch Loop - split rollout into minibatches
        for batch in buffer.get_training_batches(cfg.minibatch_size):
            # Inputs
            obs = _to_torch(batch["obs"], device, torch.float32)
            states = _to_torch(batch["states"], device, torch.float32)
            actions = _to_torch(batch["actions"], device, torch.long).view(-1)
            old_log_probs = _to_torch(batch["old_log_probs"], device, torch.float32).view(-1)
            returns = _to_torch(batch["returns"], device, torch.float32).view(-1)
            advantages = _to_torch(batch["advantages"], device, torch.float32).view(-1)
            action_masks = _to_torch(batch["action_masks"], device, torch.float32)

            # Optional Advantage Normalization - improve stability
            if getattr(cfg, "normalize_advantages", False):
                advantages = _normalize_advantages(advantages)

            # Recompute current policy quantities
            eval_out = agent.evaluate_actions(obs, action_masks, actions)
            new_log_probs, entropy = _extract_eval_logprob_entropy(eval_out)
            new_log_probs = new_log_probs.view(-1)
            entropy = entropy.view(-1)

            # Recompute current critic values
            values = agent.get_value(states).view(-1)

            ratio = torch.exp(new_log_probs - old_log_probs) # PPO ratio (how much action probability changed)
            # Compute surrogate terms
            surr1 = ratio * advantages # unclipped
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages # clipped


            policy_loss = -torch.min(surr1, surr2).mean() # actor loss (to update action probabilities)
            value_loss = ((values - returns) ** 2).mean() # value loss (critic)
            entropy_mean = entropy.mean() # Entropy: exploration bonus

            # Combine Actor Loss + Critic Loss - entropy Bonus
            total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

            # Clears old Gradients before Backpropagation
            agent.actor_opt.zero_grad()
            agent.critic_opt.zero_grad()

            total_loss.backward() # Backward Pass

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.max_grad_norm)

            # Optimizer (updates weights)
            agent.actor_opt.step()
            agent.critic_opt.step()

            # Store Metrics
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_mean.item()))
            total_losses.append(float(total_loss.item()))
            ratio_means.append(float(ratio.mean().item()))

    if not policy_losses:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "ratio_mean": 0.0,
        }

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "critic_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "total_loss": float(np.mean(total_losses)),
        "ratio_mean": float(np.mean(ratio_means)),
    }

# ============================================================
# SECTION J - EVALUATION
# ============================================================

# evaluate Greedy + Raw + Safe
def evaluate_with_mode(env: ConstructionSchedulingEnv, agent, episodes: int, seed: int, mode: str):
    if episodes <= 0:
        return {
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "mean_completed": 0.0,
        }

    eval_env = ConstructionSchedulingEnv(
        grid_shape=env.grid_shape,
        num_robots=env.num_robots,
        heavy_ratio=env.heavy_ratio,
        max_steps=env.max_steps,
        dependency_prob=env.dependency_prob,
        max_prerequisites=env.max_prerequisites,
        normal_base_duration=env.normal_base_duration,
        heavy_base_duration=env.heavy_base_duration,
        delay_probability=env.delay_probability,
        crane_cooldown_steps=env.crane_cooldown_steps,
        travel_speed_cells_per_step=env.travel_speed_cells_per_step,
        travel_reward_weight=env.travel_reward_weight,
        seed=seed,
    )

    was_training = False
    if torch is not None and hasattr(agent, "training"):
        was_training = bool(agent.training)
        agent.eval()

    rewards = []
    lengths = []
    successes = []
    completed_counts = []

    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        last_info = {}

        while not done and step_count < eval_env.max_steps:
            masks = eval_env.action_mask()

            if mode == "greedy_baseline":
                actions = greedy_baseline_action(eval_env)
            elif mode == "raw_policy":
                actions = raw_policy_action(eval_env, agent, obs, masks)
            elif mode == "safe_decoder":
                actions = policy_guided_safe_action(eval_env, agent, obs, masks)
            else:
                raise ValueError(f"Unknown eval mode: {mode}")

            obs, _, reward, done, info = eval_env.step(actions)
            episode_reward += float(reward)
            step_count += 1
            last_info = info

        rewards.append(episode_reward)
        lengths.append(float(step_count))
        successes.append(float(last_info.get("success", False)))
        completed_counts.append(float(last_info.get("completed", 0)))

    if torch is not None and hasattr(agent, "train"):
        agent.train(was_training)

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_completed": float(np.mean(completed_counts)),
    }

# Comparison printer
def compare_three_policies(env: ConstructionSchedulingEnv, agent, episodes: int, seed: int):
    greedy_metrics = evaluate_with_mode(env, agent, episodes, seed, mode="greedy_baseline")
    raw_metrics = evaluate_with_mode(env, agent, episodes, seed, mode="raw_policy")
    safe_metrics = evaluate_with_mode(env, agent, episodes, seed, mode="safe_decoder")

    print("\n=== POLICY COMPARISON ===")
    print(
        f"A. Greedy baseline      | success {greedy_metrics['success_rate']:.3f} "
        f"| reward {greedy_metrics['mean_reward']:.2f} "
        f"| length {greedy_metrics['mean_length']:.1f} "
        f"| completed {greedy_metrics['mean_completed']:.1f}"
    )
    print(
        f"B. Raw learned policy   | success {raw_metrics['success_rate']:.3f} "
        f"| reward {raw_metrics['mean_reward']:.2f} "
        f"| length {raw_metrics['mean_length']:.1f} "
        f"| completed {raw_metrics['mean_completed']:.1f}"
    )
    print(
        f"C. Safe decoder policy  | success {safe_metrics['success_rate']:.3f} "
        f"| reward {safe_metrics['mean_reward']:.2f} "
        f"| length {safe_metrics['mean_length']:.1f} "
        f"| completed {safe_metrics['mean_completed']:.1f}"
    )

    return {
        "greedy_baseline": greedy_metrics,
        "raw_policy": raw_metrics,
        "safe_decoder": safe_metrics,
    }


# ============================================================
# SECTION K - FULL TRAINING LOOP (PERSON 3)
# ============================================================

# CSV Header String
def _build_history_header() -> str:
    return "episode,reward,length,success,critic_loss,policy_loss,entropy,total_loss\n"

# Extract Episode-Level Metrics from rollout
def _extract_episode_metrics(rollout, buffer) -> Dict[str, float]:
    reward = float("nan")
    length = float("nan")
    success = float("nan")

    if isinstance(rollout, dict):
        if "rewards" in rollout:
            reward = float(np.sum(np.asarray(rollout["rewards"], dtype=np.float32)))
            length = int(len(rollout["rewards"]))
        if "infos" in rollout and rollout["infos"]:
            success = float(rollout["infos"][-1].get("success", np.nan))

    return {"reward": reward, "length": length, "success": success}

# Save Model
def _save_if_possible(agent, path: str, cfg: TrainConfig) -> None:
    try:
        agent.save(path, cfg)
    except NotImplementedError:
        print(f"skip save: agent.save(...) not implemented yet for {path}")

# Training Loop
def train(cfg: TrainConfig) -> str:
    """
    Person 3 training-loop integration.

    Expected external dependencies:
    - collect_episode(...) implemented by Person 2 / shared work
    - RolloutBuffer.compute_returns_and_advantages(...)
    - RolloutBuffer.get_training_batches(...)
    - MAPPOAgent.evaluate_actions(...)
    - MAPPOAgent.save(...)
    """

    # Check PyTorch available
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    
    torch.manual_seed(cfg.seed)


    # Build Environment from ConstructionSchedulingEnv
    env = ConstructionSchedulingEnv(
        grid_shape=(cfg.grid_rows, cfg.grid_cols),
        num_robots=cfg.robots,
        heavy_ratio=cfg.heavy_ratio,
        max_steps=cfg.max_steps,
        travel_speed_cells_per_step=cfg.travel_speed_cells_per_step,
        travel_reward_weight=cfg.travel_reward_weight,
        seed=cfg.seed,
    )

    # Reset once to infer Dimensions
    obs, state = env.reset()

    # Build Agent from MAPPOAgent
    agent = MAPPOAgent(obs.shape[1], state.shape[0], env.action_size, cfg)

    # Create log file: logging progress every episode
    os.makedirs(cfg.save_dir, exist_ok=True)
    history_path = os.path.join(cfg.save_dir, "training_log.csv")
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(_build_history_header())

    # Optional: Behaviour Cloning Warm Start
    if cfg.bc_episodes > 0:
        try:
            pretrain_actor_with_greedy(env, agent, cfg.bc_episodes, cfg)
        except NotImplementedError:
            print("skip BC warm start: pretrain_actor_with_greedy(...) not implemented yet")

    # Model Save Path
    model_path = os.path.join(cfg.save_dir, cfg.model_name)
    best_success = -float("inf")

    # Main Episode Loop
    for episode_id in range(1, cfg.episodes + 1):
        buffer = RolloutBuffer()
        rollout = collect_episode(env, agent, buffer, cfg, greedy=False) # collect rollout

        if isinstance(rollout, RolloutBuffer):
            buffer = rollout
        elif isinstance(rollout, dict) and "buffer" in rollout:
            buffer = rollout["buffer"]
        else:
            raise NotImplementedError(
                "collect_episode(...) must return either a RolloutBuffer "
                "or a dict containing {'buffer': rollout_buffer, ...}."
            )

        if not hasattr(buffer, "compute_returns_and_advantages"):
            raise AttributeError("RolloutBuffer must implement compute_returns_and_advantages(...).")

        last_value = 0.0
        if isinstance(rollout, dict) and not rollout.get("done", True):
            with torch.no_grad():
                device = next(agent.parameters()).device
                last_state_t = torch.from_numpy(rollout["last_state"]).float().to(device)
                value_t = agent.get_value(last_state_t)
            last_value = float(value_t.item())

        # Compute Returns and Advantages
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            use_gae=cfg.use_gae,
        )

        update_metrics = ppo_update(agent, buffer, cfg) # Run PPO update
        ep_metrics = _extract_episode_metrics(rollout, buffer) # Extract Episode Metrics

        # Write CSV
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(
                f"{episode_id},"
                f"{ep_metrics['reward']:.6f},"
                f"{int(ep_metrics['length']) if ep_metrics['length'] == ep_metrics['length'] else -1},"
                f"{ep_metrics['success']:.6f},"
                f"{update_metrics['critic_loss']:.6f},"
                f"{update_metrics['policy_loss']:.6f},"
                f"{update_metrics['entropy']:.6f},"
                f"{update_metrics['total_loss']:.6f}\n"

            )

        if cfg.plot_interval > 0 and (episode_id == 1 or episode_id % cfg.plot_interval == 0):
            update_training_curve_plot(history_path, cfg.plot_path) # refresh plot periodically

        # Print Episode Summary
        print(
            f"ep {episode_id:04d} | "
            f"reward {ep_metrics['reward']:.2f} | "
            f"len {int(ep_metrics['length']) if ep_metrics['length'] == ep_metrics['length'] else -1} | "
            f"success {ep_metrics['success']:.2f} | "
            f"policy {update_metrics['policy_loss']:.4f} | "
            f"value {update_metrics['value_loss']:.4f} | "
            f"entropy {update_metrics['entropy']:.4f}"
        )

        # Save Best Model
        if ep_metrics["success"] == ep_metrics["success"] and ep_metrics["success"] > best_success:
            best_success = ep_metrics["success"]
            _save_if_possible(agent, model_path, cfg)

        # Evaluate every interval
        if cfg.eval_interval > 0 and episode_id % cfg.eval_interval == 0:
            comparison = compare_three_policies(
                env,
                agent,
                episodes=cfg.eval_episodes,
                seed=cfg.seed + episode_id,
            )

    compare_three_policies(env, agent, episodes=cfg.eval_episodes, seed=cfg.seed + 9999)

    # Save Final Model
    final_path = os.path.join(cfg.save_dir, "final_" + cfg.model_name)
    _save_if_possible(agent, final_path, cfg)
    update_training_curve_plot(history_path, cfg.plot_path)
    return model_path


# ============================================================
# SECTION L - CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=700)
    parser.add_argument("--bc-episodes", type=int, default=500)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--heavy-ratio", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", type=str, default="construction_models")
    parser.add_argument("--plot-interval", type=int, default=10)
    parser.add_argument("--plot-path", type=str, default="results/training_curves.png")
    parser.add_argument("--travel-speed", type=float, default=3.0)
    parser.add_argument("--travel-reward-weight", type=float, default=0.01)
    args = parser.parse_args()

    cfg = TrainConfig(
        episodes=args.episodes,
        bc_episodes=args.bc_episodes,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        robots=args.robots,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        heavy_ratio=args.heavy_ratio,
        max_steps=args.max_steps,
        seed=args.seed,
        save_dir=args.save_dir,
        plot_interval=args.plot_interval,
        plot_path=args.plot_path,
        travel_speed_cells_per_step=args.travel_speed,
        travel_reward_weight=args.travel_reward_weight,
    )
    model_path = train(cfg)
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
