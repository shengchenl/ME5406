"""Validate a trained construction scheduling policy and create a result GIF."""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import imageio.v2 as imageio
import numpy as np

from construction_scheduling_env import ConstructionSchedulingEnv
from train_mappo_construction import (
    SharedActorCentralCritic,
    TrainConfig,
    collect_episode,
    evaluate,
    greedy_baseline_action,
)


def load_config(model_path: str) -> TrainConfig:
    data = np.load(model_path, allow_pickle=True)
    if "config_json" not in data:
        return TrainConfig()
    raw = data["config_json"].item()
    return TrainConfig(**json.loads(raw))


def build_agent_and_env(model_path: str) -> Tuple[ConstructionSchedulingEnv, SharedActorCentralCritic, TrainConfig]:
    cfg = load_config(model_path)
    rng = np.random.default_rng(cfg.seed)
    env = ConstructionSchedulingEnv(
        grid_shape=(cfg.grid_rows, cfg.grid_cols),
        num_robots=cfg.robots,
        heavy_ratio=cfg.heavy_ratio,
        max_steps=cfg.max_steps,
        travel_speed_cells_per_step=cfg.travel_speed_cells_per_step,
        travel_reward_weight=cfg.travel_reward_weight,
        seed=cfg.seed + 999,
    )
    obs, state = env.reset()
    agent = SharedActorCentralCritic(obs.shape[1], state.shape[0], env.action_size, rng, cfg)
    agent.load(model_path)
    return env, agent, cfg

def run_greedy_baseline(env: ConstructionSchedulingEnv, render_mode: str = "grid"):
    obs, state = env.reset()
    done = False
    rewards = []
    infos = []
    render_fn = env.render_dag_rgb if render_mode == "dag" else env.render_rgb
    frames = [render_fn()]
    while not done:
        actions = greedy_baseline_action(env)
        obs, state, reward, done, info = env.step(actions)
        rewards.append(reward)
        infos.append(info)
        frames.append(render_fn())
    return {"rewards": np.asarray(rewards), "infos": infos, "frames": frames}


def summarize_episode(name: str, episode) -> None:
    conflicts = 0
    violations = 0
    for info in episode["infos"]:
        stats = info["stats"]
        conflicts += stats["task_conflicts"] + stats["resource_conflicts"]
        violations += stats["dependency_violations"] + stats["invalid_completed"] + stats["busy_action_violations"]
    print(f"{name}:")
    print(f"  success: {episode['infos'][-1]['success']}")
    print(f"  steps: {len(episode['rewards'])}")
    print(f"  reward: {float(np.sum(episode['rewards'])):.2f}")
    print(f"  completed: {episode['infos'][-1]['completed']}/{episode['infos'][-1]['total_modules']}")
    print(f"  conflicts: {conflicts}")
    print(f"  violations: {violations}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="construction_models/construction_mappo_numpy.npz")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--gif", type=str, default="construction_models/validation_episode.gif")
    parser.add_argument("--baseline-gif", type=str, default="construction_models/greedy_baseline_episode.gif")
    parser.add_argument("--dag-gif", type=str, default="")
    parser.add_argument("--baseline-dag-gif", type=str, default="")
    args = parser.parse_args()

    env, agent, cfg = build_agent_and_env(args.model)
    metrics = evaluate(env, agent, episodes=args.episodes, seed=cfg.seed + 2024)
    print("learned policy evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    rng = np.random.default_rng(cfg.seed + 1234)
    learned_episode = collect_episode(env, agent, rng, greedy=True)
    summarize_episode("learned policy example", learned_episode)

    os.makedirs(os.path.dirname(args.gif), exist_ok=True)
    imageio.mimsave(args.gif, learned_episode["frames"], duration=0.35)
    print(f"saved learned policy gif: {args.gif}")

    if args.dag_gif:
        learned_dag_episode = collect_episode(env, agent, rng, greedy=True, render_mode="dag")
        os.makedirs(os.path.dirname(args.dag_gif), exist_ok=True)
        imageio.mimsave(args.dag_gif, learned_dag_episode["frames"], duration=0.35)
        print(f"saved learned policy DAG gif: {args.dag_gif}")

    baseline_env = ConstructionSchedulingEnv(
        grid_shape=(cfg.grid_rows, cfg.grid_cols),
        num_robots=cfg.robots,
        heavy_ratio=cfg.heavy_ratio,
        max_steps=cfg.max_steps,
        travel_speed_cells_per_step=cfg.travel_speed_cells_per_step,
        travel_reward_weight=cfg.travel_reward_weight,
        seed=cfg.seed + 1234,
    )
    baseline_episode = run_greedy_baseline(baseline_env)
    summarize_episode("greedy baseline example", baseline_episode)
    imageio.mimsave(args.baseline_gif, baseline_episode["frames"], duration=0.35)
    print(f"saved greedy baseline gif: {args.baseline_gif}")

    if args.baseline_dag_gif:
        baseline_dag_env = ConstructionSchedulingEnv(
            grid_shape=(cfg.grid_rows, cfg.grid_cols),
            num_robots=cfg.robots,
            heavy_ratio=cfg.heavy_ratio,
            max_steps=cfg.max_steps,
            travel_speed_cells_per_step=cfg.travel_speed_cells_per_step,
            travel_reward_weight=cfg.travel_reward_weight,
            seed=cfg.seed + 1234,
        )
        baseline_dag_episode = run_greedy_baseline(baseline_dag_env, render_mode="dag")
        os.makedirs(os.path.dirname(args.baseline_dag_gif), exist_ok=True)
        imageio.mimsave(args.baseline_dag_gif, baseline_dag_episode["frames"], duration=0.35)
        print(f"saved greedy baseline DAG gif: {args.baseline_dag_gif}")


if __name__ == "__main__":
    main()
