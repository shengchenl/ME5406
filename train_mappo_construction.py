"""Train a lightweight MAPPO-inspired agent for construction scheduling.

The implementation uses NumPy only so that it can run in the provided teaching
environment without installing a large deep learning framework. It still uses
neural-network function approximation: a shared MLP actor for all robots and a
centralized MLP critic for the team state.
"""

from __future__ import annotations

import argparse
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
    grid_rows: int = 4
    grid_cols: int = 4
    robots: int = 3
    heavy_ratio: float = 0.25
    max_steps: int = 100
    bc_episodes: int = 1000
    bc_epochs: int = 60
    episodes: int = 1200
    gamma: float = 0.97
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    seed: int = 7
    save_dir: str = "construction_models"
    model_name: str = "construction_mappo_numpy.npz"


class SharedActorCentralCritic:
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, rng: np.random.Generator, cfg: TrainConfig):
        self.action_dim = action_dim
        self.actor = MLP(obs_dim, (128, 128), action_dim, rng)
        self.critic = MLP(state_dim, (128, 128), 1, rng)
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


def greedy_baseline_action(env: ConstructionSchedulingEnv) -> np.ndarray:
    """Capability-aware online greedy scheduler used as a baseline/warm start."""

    available = env.dependency_mask()
    actions = np.ones(env.num_robots, dtype=np.int64) * env.wait_action

    idle = np.where(env.robot_remaining_time <= 0)[0]
    if len(idle) == 0:
        return actions

    heavy_available = np.where((available > 0.5) & (env.heavy_mask > 0.5))[0]
    normal_available = np.where((available > 0.5) & (env.heavy_mask < 0.5))[0]

    remaining_idle = list(int(x) for x in idle)

    if env.crane_cooldown == 0 and len(heavy_available) > 0 and len(remaining_idle) >= 2:
        idle_array = np.asarray(remaining_idle, dtype=np.int64)
        heavy_speed = env.robot_capabilities[idle_array, 1]
        pair_order = np.argsort(heavy_speed)[:2]
        chosen = idle_array[pair_order]
        actions[chosen] = int(heavy_available[0])
        remaining_idle = [rid for rid in remaining_idle if rid not in set(int(x) for x in chosen)]

    if len(normal_available) > 0 and remaining_idle:
        idle_array = np.asarray(remaining_idle, dtype=np.int64)
        normal_order = idle_array[np.argsort(env.robot_capabilities[idle_array, 0])]
        for robot_id, module in zip(normal_order, normal_available):
            actions[int(robot_id)] = int(module)
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
        actions, _, _ = agent.act(obs, masks, rng, greedy=greedy)
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
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--bc-episodes", type=int, default=1000)
    parser.add_argument("--bc-epochs", type=int, default=60)
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--heavy-ratio", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", type=str, default="construction_models")
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
    )
    model_path = train(cfg)
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
