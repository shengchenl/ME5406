"""Train a lightweight MAPPO-inspired agent for construction scheduling.

The implementation uses NumPy only so that it can run in the provided teaching
environment without installing a large deep learning framework. It still uses
neural-network function approximation: a shared MLP actor for all robots and a
centralized MLP critic for the team state.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np

from construction_scheduling_env import ConstructionSchedulingEnv


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = logits.copy()
    masked[mask < 0.5] = -1.0e9
    masked -= np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(masked) * mask
    total = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.maximum(total, 1.0e-8)


class Adam:
    def __init__(self, lr: float = 3.0e-4, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1.0e-8):
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


class MLP:
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int, rng: np.random.Generator):
        dims = (input_dim,) + hidden_dims + (output_dim,)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def params(self) -> List[np.ndarray]:
        params: List[np.ndarray] = []
        for weight, bias in zip(self.weights, self.biases):
            params.extend([weight, bias])
        return params

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
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

    def backward(self, grad_output: np.ndarray, caches: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        grad = grad_output.astype(np.float32)
        grad_weights: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        grad_biases: List[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        batch = max(1, grad.shape[0])

        for layer in reversed(range(len(self.weights))):
            h_in, _ = caches[layer]
            grad_weights[layer] = (h_in.T @ grad) / batch
            grad_biases[layer] = np.mean(grad, axis=0)
            if layer > 0:
                prev_z = caches[layer - 1][1]
                grad = (grad @ self.weights[layer].T) * (1.0 - np.tanh(prev_z) ** 2)

        grads: List[np.ndarray] = []
        for gw, gb in zip(grad_weights, grad_biases):
            grads.extend([gw, gb])
        return grads

    def state_dict(self, prefix: str) -> Dict[str, np.ndarray]:
        data: Dict[str, np.ndarray] = {}
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            data[f"{prefix}_w{i}"] = weight
            data[f"{prefix}_b{i}"] = bias
        return data

    def load_state_dict(self, data: Dict[str, np.ndarray], prefix: str) -> None:
        for i in range(len(self.weights)):
            self.weights[i][...] = data[f"{prefix}_w{i}"]
            self.biases[i][...] = data[f"{prefix}_b{i}"]


@dataclass
class TrainConfig:
    grid_rows: int = 10
    grid_cols: int = 10
    robots: int = 6
    heavy_ratio: float = 0.25
    max_steps: int = 900
    bc_episodes: int = 500
    bc_epochs: int = 30
    episodes: int = 700
    gamma: float = 0.97
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    seed: int = 7
    save_dir: str = "construction_models"
    model_name: str = "construction_mappo_numpy.npz"
    plot_interval: int = 10
    plot_path: str = "results/training_curves.png"
    travel_speed_cells_per_step: float = 3.0
    travel_reward_weight: float = 0.01


class SharedActorCentralCritic:
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, rng: np.random.Generator, cfg: TrainConfig):
        self.action_dim = action_dim
        self.actor = MLP(obs_dim, (256, 256), action_dim, rng)
        self.critic = MLP(state_dim, (256, 256), 1, rng)
        self.actor_opt = Adam(cfg.actor_lr)
        self.critic_opt = Adam(cfg.critic_lr)

    def act(self, observations: np.ndarray, masks: np.ndarray, rng: np.random.Generator, greedy: bool = False):
        logits, _ = self.actor.forward(observations)
        probs = masked_softmax(logits, masks)
        actions = []
        log_probs = []
        for prob in probs:
            if greedy:
                action = int(np.argmax(prob))
            else:
                action = int(rng.choice(len(prob), p=prob))
            actions.append(action)
            log_probs.append(float(np.log(max(prob[action], 1.0e-8))))
        return np.asarray(actions, dtype=np.int64), probs, np.asarray(log_probs, dtype=np.float32)

    def values(self, states: np.ndarray) -> np.ndarray:
        value, _ = self.critic.forward(states)
        return value.reshape(-1)

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        states = batch["states"]
        returns = batch["returns"]
        values, critic_cache = self.critic.forward(states)
        value_error = values.reshape(-1) - returns
        critic_loss = float(np.mean(value_error**2))
        critic_grad = (2.0 * value_error[:, None]).astype(np.float32)
        critic_grads = self.critic.backward(critic_grad, critic_cache)
        self.critic_opt.step(self.critic.params(), critic_grads)

        observations = batch["actor_obs"]
        masks = batch["actor_masks"]
        actions = batch["actor_actions"]
        advantages = batch["actor_advantages"]
        logits, actor_cache = self.actor.forward(observations)
        probs = masked_softmax(logits, masks)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(actions)), actions] = 1.0
        grad_logits = (probs - one_hot) * advantages[:, None]
        grad_logits *= masks
        actor_grads = self.actor.backward(grad_logits.astype(np.float32), actor_cache)
        self.actor_opt.step(self.actor.params(), actor_grads)

        selected_probs = probs[np.arange(len(actions)), actions]
        policy_loss = float(np.mean(-np.log(np.maximum(selected_probs, 1.0e-8)) * advantages))
        entropy = float(np.mean(-np.sum(probs * np.log(np.maximum(probs, 1.0e-8)), axis=1)))
        return {"critic_loss": critic_loss, "policy_loss": policy_loss, "entropy": entropy}

    def save(self, path: str, cfg: TrainConfig) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}
        data.update(self.actor.state_dict("actor"))
        data.update(self.critic.state_dict("critic"))
        data["config_json"] = np.array(json.dumps(asdict(cfg)))
        np.savez(path, **data)

    def load(self, path: str) -> None:
        data = dict(np.load(path, allow_pickle=True))
        self.actor.load_state_dict(data, "actor")
        self.critic.load_state_dict(data, "critic")


def discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


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
    """Capability- and distance-aware online greedy scheduler."""

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


def policy_guided_safe_action(
    env: ConstructionSchedulingEnv,
    agent: SharedActorCentralCritic,
    observations: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Decode independent policy scores into a feasible multi-robot action.

    The shared actor still decides task preferences. This decoder only enforces
    the construction constraints that are hard for independent argmax to satisfy
    on a 100-module action space: distinct normal-task assignment, heavy-task
    pairing, and no action for busy robots.
    """

    logits, _ = agent.actor.forward(observations)
    probs = masked_softmax(logits, masks)
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


def pretrain_actor_with_greedy(
    env: ConstructionSchedulingEnv,
    agent: SharedActorCentralCritic,
    episodes: int,
    rng: np.random.Generator,
) -> None:
    """Behavior-cloning warm start to make RL training practical for class demos."""

    if episodes <= 0:
        return

    all_obs = []
    all_masks = []
    all_actions = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            masks = env.action_mask()
            actions = greedy_baseline_action(env)
            all_obs.append(obs)
            all_masks.append(masks)
            all_actions.append(actions)
            obs, _, _, done, _ = env.step(actions)

    observations = np.asarray(all_obs, dtype=np.float32).reshape(-1, env.get_observations().shape[1])
    masks = np.asarray(all_masks, dtype=np.float32).reshape(-1, env.action_size)
    actions = np.asarray(all_actions, dtype=np.int64).reshape(-1)

    indices = np.arange(len(actions))
    epochs = getattr(agent, "bc_epochs", 60)
    for epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), 256):
            batch_idx = indices[start : start + 256]
            logits, cache = agent.actor.forward(observations[batch_idx])
            probs = masked_softmax(logits, masks[batch_idx])
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(batch_idx)), actions[batch_idx]] = 1.0
            grad = (probs - one_hot) * masks[batch_idx]
            grads = agent.actor.backward(grad.astype(np.float32), cache)
            agent.actor_opt.step(agent.actor.params(), grads)
        if epoch in (0, epochs - 1):
            logits, _ = agent.actor.forward(observations[:512])
            probs = masked_softmax(logits, masks[:512])
            acc = np.mean(np.argmax(probs, axis=1) == actions[:512])
            print(f"bc epoch {epoch + 1:02d} | demo accuracy {acc:.2f}")


def collect_episode(
    env: ConstructionSchedulingEnv,
    agent: SharedActorCentralCritic,
    rng: np.random.Generator,
    greedy: bool = False,
    render_mode: str = "grid",
):
    obs, state = env.reset()
    done = False
    states = []
    actor_obs = []
    actor_masks = []
    actor_actions = []
    rewards = []
    infos = []
    render_fn = env.render_dag_rgb if render_mode == "dag" else env.render_rgb
    frames = [render_fn()]

    while not done:
        masks = env.action_mask()
        if greedy:
            actions = policy_guided_safe_action(env, agent, obs, masks)
        else:
            actions, _, _ = agent.act(obs, masks, rng, greedy=False)
        next_obs, next_state, reward, done, info = env.step(actions)

        states.append(state)
        actor_obs.append(obs)
        actor_masks.append(masks)
        actor_actions.append(actions)
        rewards.append(reward)
        infos.append(info)
        frames.append(render_fn())

        obs, state = next_obs, next_state

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actor_obs": np.asarray(actor_obs, dtype=np.float32),
        "actor_masks": np.asarray(actor_masks, dtype=np.float32),
        "actor_actions": np.asarray(actor_actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "infos": infos,
        "frames": frames,
    }


def make_batch(episode: Dict, agent: SharedActorCentralCritic, gamma: float) -> Dict[str, np.ndarray]:
    returns = discounted_returns(episode["rewards"].tolist(), gamma)
    values = agent.values(episode["states"])
    advantages = returns - values
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1.0e-6)

    steps, robots = episode["actor_actions"].shape
    return {
        "states": episode["states"],
        "returns": returns,
        "actor_obs": episode["actor_obs"].reshape(steps * robots, -1),
        "actor_masks": episode["actor_masks"].reshape(steps * robots, -1),
        "actor_actions": episode["actor_actions"].reshape(steps * robots),
        "actor_advantages": np.repeat(advantages, robots),
    }


def evaluate(env: ConstructionSchedulingEnv, agent: SharedActorCentralCritic, episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    lengths, rewards, successes, conflicts, violations = [], [], [], [], []
    for _ in range(episodes):
        ep = collect_episode(env, agent, rng, greedy=True)
        total_conflicts = 0
        total_violations = 0
        for info in ep["infos"]:
            stats = info["stats"]
            total_conflicts += stats["task_conflicts"] + stats["resource_conflicts"]
            total_violations += (
                stats["dependency_violations"] + stats["invalid_completed"] + stats["busy_action_violations"]
            )
        lengths.append(len(ep["rewards"]))
        rewards.append(float(np.sum(ep["rewards"])))
        successes.append(float(ep["infos"][-1]["success"]))
        conflicts.append(total_conflicts)
        violations.append(total_violations)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "mean_reward": float(np.mean(rewards)),
        "mean_conflicts": float(np.mean(conflicts)),
        "mean_violations": float(np.mean(violations)),
    }


def train(cfg: TrainConfig) -> str:
    rng = np.random.default_rng(cfg.seed)
    env = ConstructionSchedulingEnv(
        grid_shape=(cfg.grid_rows, cfg.grid_cols),
        num_robots=cfg.robots,
        heavy_ratio=cfg.heavy_ratio,
        max_steps=cfg.max_steps,
        travel_speed_cells_per_step=cfg.travel_speed_cells_per_step,
        travel_reward_weight=cfg.travel_reward_weight,
        seed=cfg.seed,
    )
    obs, state = env.reset()
    agent = SharedActorCentralCritic(obs.shape[1], state.shape[0], env.action_size, rng, cfg)
    agent.bc_epochs = cfg.bc_epochs

    pretrain_actor_with_greedy(env, agent, cfg.bc_episodes, rng)

    history_path = os.path.join(cfg.save_dir, "training_log.csv")
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("episode,reward,length,success,completed,critic_loss,policy_loss,entropy\n")

    best_success = -1.0
    best_length = float("inf")
    model_path = os.path.join(cfg.save_dir, cfg.model_name)

    for episode_id in range(1, cfg.episodes + 1):
        episode = collect_episode(env, agent, rng, greedy=False)
        batch = make_batch(episode, agent, cfg.gamma)
        losses = agent.update(batch)

        ep_reward = float(np.sum(episode["rewards"]))
        ep_len = len(episode["rewards"])
        success = float(episode["infos"][-1]["success"])
        completed = episode["infos"][-1]["completed"]
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(
                f"{episode_id},{ep_reward:.4f},{ep_len},{success:.0f},{completed},"
                f"{losses['critic_loss']:.6f},{losses['policy_loss']:.6f},{losses['entropy']:.6f}\n"
            )
        if cfg.plot_interval > 0 and (episode_id == 1 or episode_id % cfg.plot_interval == 0):
            update_training_curve_plot(history_path, cfg.plot_path)

        if episode_id % 100 == 0 or episode_id == 1:
            metrics = evaluate(env, agent, episodes=20, seed=cfg.seed + episode_id)
            print(
                f"ep {episode_id:04d} | reward {ep_reward:7.2f} | len {ep_len:3d} | "
                f"eval_success {metrics['success_rate']:.2f} | eval_len {metrics['mean_length']:.1f}"
            )
            improved = metrics["success_rate"] > best_success
            tied_but_faster = metrics["success_rate"] == best_success and metrics["mean_length"] < best_length
            if improved or tied_but_faster:
                best_success = metrics["success_rate"]
                best_length = metrics["mean_length"]
                agent.save(model_path, cfg)

    final_path = os.path.join(cfg.save_dir, "final_" + cfg.model_name)
    agent.save(final_path, cfg)
    update_training_curve_plot(history_path, cfg.plot_path)
    return model_path


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
