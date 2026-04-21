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
    from torch.distributions import Categorical
except Exception:
    torch = None
    nn = None
    Categorical = None

from construction_scheduling_env import ConstructionSchedulingEnv


# ============================================================
# SECTION A - SMALL HELPERS WE CAN KEEP
# ============================================================

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """NumPy masked softmax.

    STATUS:
    - KEEP FOR NOW for legacy/reference code and debugging.
    - The full PyTorch version should have a torch equivalent.
    """
    masked = logits.copy()
    masked[mask < 0.5] = -1.0e9
    masked -= np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(masked) * mask
    total = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.maximum(total, 1.0e-8)


# ============================================================
# SECTION B - LEGACY OPTIMIZER / NETWORKS (REFERENCE ONLY)
# ============================================================

class Adam:
    """Handwritten Adam optimizer from the current file.

    STATUS:
    - FUNCTIONALLY FINE as a real Adam implementation.
    - KEEP AS REFERENCE ONLY.
    - FULL PYTORCH VERSION should use torch.optim.Adam instead.
    """

    def __init__(
        self,
        lr: float = 3.0e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1.0e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: List[np.ndarray] = []
        self.v: List[np.ndarray] = []

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1.0 - self.beta1**self.t)
            v_hat = self.v[i] / (1.0 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class LegacyMLP:
    """Current NumPy MLP from the draft trainer.

    STATUS:
    - VALID MLP, but not the final version we want.
    - KEEP AS REFERENCE for architecture / dimensions / debugging.
    - REPLACE LATER with PyTorch Actor/Critic modules.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        rng: np.random.Generator,
    ):
        dims = (input_dim,) + hidden_dims + (output_dim,)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            self.weights.append(
                rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
            )
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def params(self) -> List[np.ndarray]:
        params: List[np.ndarray] = []
        for weight, bias in zip(self.weights, self.biases):
            params.extend([weight, bias])
        return params

    def forward(self, x: np.ndarray):
        h = x.astype(np.float32)
        caches: List[Tuple[np.ndarray, np.ndarray]] = []
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = h @ weight + bias
            caches.append((h, z))
            if layer < len(self.weights) - 1:
                h = np.tanh(z)
            else:
                h = z
        return h, caches


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
    max_steps: int = 900
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
# SECTION D - LEGACY AGENT (REFERENCE ONLY)
# ============================================================

class LegacySharedActorCentralCritic:
    """Current NumPy shared actor + centralized critic.

    STATUS:
    - MAIN THING TO REPLACE.
    - KEEP here so the team can compare old vs new structure.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        rng: np.random.Generator,
        cfg: TrainConfig,
    ):
        self.action_dim = action_dim
        self.actor = LegacyMLP(obs_dim, cfg.actor_hidden_sizes, action_dim, rng)
        self.critic = LegacyMLP(state_dim, cfg.critic_hidden_sizes, 1, rng)
        self.actor_opt = Adam(cfg.actor_lr)
        self.critic_opt = Adam(cfg.critic_lr)

    def act(self, observations: np.ndarray, masks: np.ndarray, rng: np.random.Generator, greedy: bool = False):
        logits, _ = self.actor.forward(observations)
        probs = masked_softmax(logits, masks)
        actions = []
        log_probs = []
        for prob in probs:
            action = int(np.argmax(prob)) if greedy else int(rng.choice(len(prob), p=prob))
            actions.append(action)
            log_probs.append(float(np.log(max(prob[action], 1.0e-8))))
        return np.asarray(actions, dtype=np.int64), probs, np.asarray(log_probs, dtype=np.float32)


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

def discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """Legacy Monte Carlo discounted returns.

    STATUS:
    - KEEP FOR REFERENCE / fallback.
    - FULL MAPPO should prefer GAE + bootstrapped values.
    """
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


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
        self.obs = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.log_probs = []
        self.values = []

    def add(self, obs, states, actions, rewards, dones, action_masks, log_probs, values) -> None:
        self.obs.append(obs)
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.action_masks.append(action_masks)
        self.log_probs.append(log_probs)
        self.values.append(values)

    def compute_returns_and_advantages(self, last_value, gamma: float, gae_lambda: float):
        # 将列表转换为 Tensor
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32)  # [T, N]
        values = torch.tensor(np.array(self.values), dtype=torch.float32)  # [T, N]
        dones = torch.tensor(np.array(self.dones), dtype=torch.float32)  # [T, N]

        # 扩展 values 数组以包含 last_value (即 V_next)
        # 结果维度: [T+1, N]
        v_next_all = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)

        self.advantages = torch.zeros_like(rewards)
        last_gae = 0

        # 倒序遍历计算 GAE
        for t in reversed(range(len(rewards))):
            # TD Error: delta = r + gamma * V_next * (1-done) - V_now
            delta = rewards[t] + gamma * v_next_all[t] * (1.0 - dones[t]) - values[t]
            # GAE: A_t = delta + gamma * lambda * (1-done) * A_{t+1}
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae

        # Returns = Advantage + Value (作为 Critic 的更新目标)
        self.returns = self.advantages + values

    def get_training_batches(self, minibatch_size: int):
        """
        一次性处理所有数据，并以列表形式返回所有训练批次。
        """

        def flatten_to_tensor(data_list, dtype=torch.float32):
            arr = np.array(data_list)
            t = torch.tensor(arr, dtype=dtype)
            return t.view(-1, *t.shape[2:])

        # 1. 准备大表 (Flattened Tensors)
        obs_f = flatten_to_tensor(self.obs)
        states_f = flatten_to_tensor(self.states)
        actions_f = flatten_to_tensor(self.actions, dtype=torch.long)
        log_probs_f = flatten_to_tensor(self.log_probs)
        masks_f = flatten_to_tensor(self.action_masks)

        # 处理优势和回报
        advantages_f = self.advantages.view(-1)
        returns_f = self.returns.view(-1)
        # 标准化优势函数
        advantages_f = (advantages_f - advantages_f.mean()) / (advantages_f.std() + 1e-8)

        # 2. 随机洗牌
        total_samples = obs_f.size(0)
        indices = torch.randperm(total_samples)

        # 3. 核心改变：创建一个列表，把切好的“肉”都装进去
        all_batches = []

        for start in range(0, total_samples, minibatch_size):
            batch_idx = indices[start: start + minibatch_size]

            # 把这一批数据打包成一个元组 (Tuple)
            batch_data = (
                obs_f[batch_idx],
                states_f[batch_idx],
                actions_f[batch_idx],
                log_probs_f[batch_idx],
                advantages_f[batch_idx],
                returns_f[batch_idx],
                masks_f[batch_idx]
            )

            # 塞进大列表
            all_batches.append(batch_data)

        # 4. 直接返回这个大列表
        return all_batches

#调用示例：
#obs, state, action, log_prob, adv, ret, mask = all_batches[0]


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


def policy_guided_safe_action(env: ConstructionSchedulingEnv, agent, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Feasible decoding helper used mainly in evaluation.

    STATUS:
    - KEEP FOR NOW unless later the team redesigns it.
    """
    if hasattr(agent, "actor") and isinstance(getattr(agent, "actor"), LegacyMLP):
        logits, _ = agent.actor.forward(observations)
        probs = masked_softmax(logits, masks)
    else:
        raise NotImplementedError(
            "TODO: connect policy_guided_safe_action to the new torch actor if needed."
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


def pretrain_actor_with_greedy(env: ConstructionSchedulingEnv, agent, episodes: int, rng: np.random.Generator) -> None:
    """Behavior cloning warm start.

    STATUS:
    - KEEP IDEA for now.
    - IMPLEMENTATION will likely need a PyTorch rewrite.
    """
    if episodes <= 0:
        return
    raise NotImplementedError(
        "TODO: rewrite behavior cloning pretraining for the new torch actor if needed."
    )


# ============================================================
# SECTION H - ROLLOUT COLLECTION (PERSON 2 + PERSON 3)
# ============================================================
# TODO [PERSON 2 / PERSON 3]
# Decide exact outputs:
# - actor observation format
# - critic global-state format
# - action representation
# - buffer fields and tensor shapes

def collect_episode(env: ConstructionSchedulingEnv, agent: MAPPOAgent, buffer: RolloutBuffer, cfg: TrainConfig, rng: np.random.Generator, greedy: bool = False):
    obs = env.reset() 
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < cfg.max_steps:
        # A. 获取动作掩码 (防止选到不可行的任务)
        masks = env.get_action_masks() # [num_robots, action_dim]

        # B. 构造全局状态
        state = obs.flatten() # 形状: [num_robots * obs_dim]

        # C. 将 numpy 转为 torch tensor
        obs_t = torch.from_numpy(obs).float()
        mask_t = torch.from_numpy(masks).float()
        state_t = torch.from_numpy(state).float()

        with torch.no_grad():
            # 拿到动作、动作的 log 概率
            actions_t, log_probs_t, _ = agent.get_action_and_logprob(obs_t, mask_t, greedy=greedy)
            # 拿到 Critic 对局势的评估值
            value_t = agent.get_value(state_t)

        # D. 环境执行
        actions_np = actions_t.cpu().numpy()
        next_obs, rewards, done, info = env.step(actions_np)
        buffer.add(
            obs=obs,
            states=state,
            actions=actions_np,
            rewards=rewards,
            dones=np.array([done] * env.robots), # 记录每个机器人对应的结束标志
            action_masks=masks,
            log_probs=log_probs_t.cpu().numpy(),
            values=value_t.cpu().numpy()
        )

        obs = next_obs
        total_reward += np.mean(rewards)
        step_count += 1

    return total_reward, step_count


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

def evaluate(env: ConstructionSchedulingEnv, agent, episodes: int, seed: int):
    raise NotImplementedError("TODO: reconnect evaluation to the new rollout / new agent")


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
    else:
        rewards = getattr(buffer, "rewards", None)
        if rewards is not None and len(rewards) > 0:
            reward = float(np.sum(np.asarray(rewards, dtype=np.float32)))
            length = int(len(rewards))

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

    # Create RNG (randomness)
    rng = np.random.default_rng(cfg.seed)

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
            pretrain_actor_with_greedy(env, agent, cfg.bc_episodes, rng)
        except NotImplementedError:
            print("skip BC warm start: pretrain_actor_with_greedy(...) not implemented yet")

    # Model Save Path
    model_path = os.path.join(cfg.save_dir, cfg.model_name)
    best_success = -float("inf")

    # Main Episode Loop
    for episode_id in range(1, cfg.episodes + 1):
        rollout = collect_episode(env, agent, rng, greedy=False) # collect rollout

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

        # Compute Returns and Advantages
        buffer.compute_returns_and_advantages(
            last_value=0.0,
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
            try:
                metrics = evaluate(env, agent, episodes=cfg.eval_episodes, seed=cfg.seed + episode_id)
                print(
                    f"eval @ {episode_id:04d} | "
                    f"success {metrics.get('success_rate', float('nan')):.3f}"
                )
            except NotImplementedError:
                print("skip evaluation: evaluate(...) not implemented yet")

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
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--heavy-ratio", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=900)
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
