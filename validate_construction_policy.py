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
    greedy_baseline_action,
)

# ============= Load saved config JSON from model file =====================
def load_config(model_path: str) -> TrainConfig:
    data = np.load(model_path, allow_pickle=True)
    if "config_json" not in data:
        return TrainConfig()
    raw = data["config_json"].item()
    return TrainConfig(**json.loads(raw))

# ============== Recreates: Environment + Agent + Config ====================
# - loads trained weights
# - ensures validation uses same setup as training
def build_agent_and_env(model_path: str) -> Tuple[ConstructionSchedulingEnv, SharedActorCentralCritic, TrainConfig]:
    cfg = load_config(model_path) # reads saved config
    rng = np.random.default_rng(cfg.seed)
        
    env = ConstructionSchedulingEnv( # rebuild environment using config
        grid_shape=(cfg.grid_rows, cfg.grid_cols),
        num_robots=cfg.robots,
        heavy_ratio=cfg.heavy_ratio,
        max_steps=cfg.max_steps,
        travel_speed_cells_per_step=cfg.travel_speed_cells_per_step,
        travel_reward_weight=cfg.travel_reward_weight,
        seed=cfg.seed + 999,
    )
    obs, state = env.reset()
    agent = SharedActorCentralCritic(obs.shape[1], state.shape[0], env.action_size, rng, cfg) # rebuild actor/critic structure
    agent.load(model_path) # load saved trained weights
    return env, agent, cfg

# ============= Runs one full baseline episode and collects frames ============
# - Conventional Method (Greedy) for comparison with our RL Model
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

# ============= Validation Metrics of 1 run ======================
"""
success flag
completion time
total cumulative reward
no. of modules completed
conflicts: task + resource conflicts
violations: dependency violations + invalid completed + busy action violations
"""
def summarize_episode(name: str, episode) -> dict:
    conflicts = 0
    violations = 0
    for info in episode["infos"]:
        stats = info["stats"]
        conflicts += stats["task_conflicts"] + stats["resource_conflicts"]
        violations += (
            stats["dependency_violations"]
            + stats["invalid_completed"]
            + stats["busy_action_violations"]
        )

    summary = {
        "success": bool(episode["infos"][-1]["success"]),
        "steps": int(len(episode["rewards"])),
        "reward": float(np.sum(episode["rewards"])),
        "completed": int(episode["infos"][-1]["completed"]),
        "total_modules": int(episode["infos"][-1]["total_modules"]),
        "conflicts": int(conflicts),
        "violations": int(violations),
    }

    print(f"{name}:")
    print(f"  success: {summary['success']}")
    print(f"  steps: {summary['steps']}")
    print(f"  reward: {summary['reward']:.2f}")
    print(f"  completed: {summary['completed']}/{summary['total_modules']}")
    print(f"  conflicts: {summary['conflicts']}")
    print(f"  violations: {summary['violations']}")

    return summary

# ====================== Averaged Validation Metrics after multiple runs =====================
"""
success rate
average completion time
average total reward
average task conflicts
average resource conflicts
average dependency violations
average invalid completed selections
average busy-action violations
average completed modules
standard deviation - how consistent
min and max - extreme outcomes
"""
def evaluate_policy_detailed(
    name: str,
    env: ConstructionSchedulingEnv,
    agent: SharedActorCentralCritic | None,
    episodes: int = 30,
    seed: int = 0,
    use_greedy_baseline: bool = False,
    episode_seeds: list[int] | None = None,
):
    rng = np.random.default_rng(seed)

    if episode_seeds is None:
        episode_seeds = [int(rng.integers(0, 1_000_000_000)) for _ in range(episodes)]
    else:
        episodes = len(episode_seeds)

    successes = []
    completion_times = []
    total_rewards = []
    completed_modules = []

    task_conflicts = []
    resource_conflicts = []
    dependency_violations = []
    invalid_completed = []
    busy_action_violations = []

    for ep_seed in episode_seeds:
        env.seed = ep_seed
        obs, state = env.reset()
        done = False

        rewards = []

        ep_task_conflicts = 0
        ep_resource_conflicts = 0
        ep_dependency_violations = 0
        ep_invalid_completed = 0
        ep_busy_action_violations = 0

        final_info = None

        while not done:
            if use_greedy_baseline:
                actions = greedy_baseline_action(env)
            else:
                masks = env.action_mask()
                actions = agent.act(obs, masks, greedy=True)

            obs, state, reward, done, info = env.step(actions)
            rewards.append(reward)

            stats = info["stats"]
            ep_task_conflicts += stats["task_conflicts"]
            ep_resource_conflicts += stats["resource_conflicts"]
            ep_dependency_violations += stats["dependency_violations"]
            ep_invalid_completed += stats["invalid_completed"]
            ep_busy_action_violations += stats["busy_action_violations"]

            final_info = info

        successes.append(float(final_info["success"]))
        completion_times.append(len(rewards))
        total_rewards.append(float(np.sum(rewards)))
        completed_modules.append(final_info["completed"])

        task_conflicts.append(ep_task_conflicts)
        resource_conflicts.append(ep_resource_conflicts)
        dependency_violations.append(ep_dependency_violations)
        invalid_completed.append(ep_invalid_completed)
        busy_action_violations.append(ep_busy_action_violations)

    metrics = {
        "success_rate": float(np.mean(successes)),

        "avg_completion_time": float(np.mean(completion_times)),
        "std_completion_time": float(np.std(completion_times)),
        "min_completion_time": float(np.min(completion_times)),
        "max_completion_time": float(np.max(completion_times)),

        "avg_total_reward": float(np.mean(total_rewards)),
        "std_total_reward": float(np.std(total_rewards)),
        "min_total_reward": float(np.min(total_rewards)),
        "max_total_reward": float(np.max(total_rewards)),

        "avg_task_conflicts": float(np.mean(task_conflicts)),
        "std_task_conflicts": float(np.std(task_conflicts)),
        "min_task_conflicts": float(np.min(task_conflicts)),
        "max_task_conflicts": float(np.max(task_conflicts)),

        "avg_resource_conflicts": float(np.mean(resource_conflicts)),
        "std_resource_conflicts": float(np.std(resource_conflicts)),
        "min_resource_conflicts": float(np.min(resource_conflicts)),
        "max_resource_conflicts": float(np.max(resource_conflicts)),

        "avg_dependency_violations": float(np.mean(dependency_violations)),
        "std_dependency_violations": float(np.std(dependency_violations)),
        "min_dependency_violations": float(np.min(dependency_violations)),
        "max_dependency_violations": float(np.max(dependency_violations)),

        "avg_invalid_completed": float(np.mean(invalid_completed)),
        "std_invalid_completed": float(np.std(invalid_completed)),
        "min_invalid_completed": float(np.min(invalid_completed)),
        "max_invalid_completed": float(np.max(invalid_completed)),

        "avg_busy_action_violations": float(np.mean(busy_action_violations)),
        "std_busy_action_violations": float(np.std(busy_action_violations)),
        "min_busy_action_violations": float(np.min(busy_action_violations)),
        "max_busy_action_violations": float(np.max(busy_action_violations)),

        "avg_completed_modules": float(np.mean(completed_modules)),
        "std_completed_modules": float(np.std(completed_modules)),
        "min_completed_modules": float(np.min(completed_modules)),
        "max_completed_modules": float(np.max(completed_modules)),
    }

    print(f"{name}:")
    print(f"  success rate: {metrics['success_rate']:.3f}")

    print(
        f"  completion time: {metrics['avg_completion_time']:.3f} ± "
        f"{metrics['std_completion_time']:.3f} "
        f"(min {metrics['min_completion_time']:.0f}, max {metrics['max_completion_time']:.0f})"
    )

    print(
        f"  total reward: {metrics['avg_total_reward']:.3f} ± "
        f"{metrics['std_total_reward']:.3f} "
        f"(min {metrics['min_total_reward']:.3f}, max {metrics['max_total_reward']:.3f})"
    )

    print(
        f"  task conflicts: {metrics['avg_task_conflicts']:.3f} ± "
        f"{metrics['std_task_conflicts']:.3f} "
        f"(min {metrics['min_task_conflicts']:.0f}, max {metrics['max_task_conflicts']:.0f})"
    )

    print(
        f"  resource conflicts: {metrics['avg_resource_conflicts']:.3f} ± "
        f"{metrics['std_resource_conflicts']:.3f} "
        f"(min {metrics['min_resource_conflicts']:.0f}, max {metrics['max_resource_conflicts']:.0f})"
    )

    print(
        f"  dependency violations: {metrics['avg_dependency_violations']:.3f} ± "
        f"{metrics['std_dependency_violations']:.3f} "
        f"(min {metrics['min_dependency_violations']:.0f}, max {metrics['max_dependency_violations']:.0f})"
    )

    print(
        f"  invalid completed selections: {metrics['avg_invalid_completed']:.3f} ± "
        f"{metrics['std_invalid_completed']:.3f} "
        f"(min {metrics['min_invalid_completed']:.0f}, max {metrics['max_invalid_completed']:.0f})"
    )

    print(
        f"  busy-action violations: {metrics['avg_busy_action_violations']:.3f} ± "
        f"{metrics['std_busy_action_violations']:.3f} "
        f"(min {metrics['min_busy_action_violations']:.0f}, max {metrics['max_busy_action_violations']:.0f})"
    )

    print(
        f"  completed modules: {metrics['avg_completed_modules']:.3f} ± "
        f"{metrics['std_completed_modules']:.3f} "
        f"(min {metrics['min_completed_modules']:.0f}, max {metrics['max_completed_modules']:.0f})"
    )

    return metrics

# =============== JSON file to store metrics ================
def save_metrics_json(path: str, payload: dict) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"saved metrics json: {path}")


# ============== Validation Flow ===================
# 1. load trained model
# 2. evaluate learned policy over many episodes
# 3. print metrics
# 4. collect one learned-policy rollout and save GIF
# 5. optionally save DAG GIF
# 6. create a fresh baseline environment
# 7. run greedy baseline rollout
# 8. save baseline GIF
# 9. optionally save baseline DAG GIF
# ===================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="construction_models/construction_mappo_numpy.npz")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--gif", type=str, default="construction_models/validation_episode.gif")
    parser.add_argument("--baseline-gif", type=str, default="construction_models/greedy_baseline_episode.gif")
    parser.add_argument("--dag-gif", type=str, default="")
    parser.add_argument("--baseline-dag-gif", type=str, default="")
    parser.add_argument("--metrics-json", type=str, default="construction_models/validation_metrics.json")
    args = parser.parse_args()

    # Load Trained Model
    env, agent, cfg = build_agent_and_env(args.model)

    # Shared Seed (Single Run)
    single_demo_seed = int(cfg.seed + 1234)

    # Shared Seed list (average runs)
    shared_eval_rng = np.random.default_rng(cfg.seed + 2024)
    shared_episode_seeds = [int(shared_eval_rng.integers(0, 1_000_000_000)) for _ in range(args.episodes)]

    # Average Metrics (Validation)
    learned_avg_metrics = evaluate_policy_detailed(
        "learned policy average evaluation",
        env,
        agent,
        episodes=args.episodes,
        seed=cfg.seed + 2024,
        use_greedy_baseline=False,
        episode_seeds=shared_episode_seeds,
    )

    # Collect 1 learned-policy rollout
    # - qualitative example / GIF visualization
    rng = np.random.default_rng(single_demo_seed)
    env.seed = single_demo_seed
    learned_episode = collect_episode(env, agent, rng, greedy=True)
    learned_single_summary = summarize_episode("learned policy single-run example", learned_episode)

    # Save GIF
    os.makedirs(os.path.dirname(args.gif), exist_ok=True)
    imageio.mimsave(args.gif, learned_episode["frames"], duration=0.35)
    print(f"saved learned policy gif: {args.gif}")

    # Optional save DAG GIF
    if args.dag_gif:
        env.seed = single_demo_seed
        learned_dag_episode = collect_episode(env, agent, rng, greedy=True, render_mode="dag")
        os.makedirs(os.path.dirname(args.dag_gif), exist_ok=True)
        imageio.mimsave(args.dag_gif, learned_dag_episode["frames"], duration=0.35)
        print(f"saved learned policy DAG gif: {args.dag_gif}")

    # Create Fresh Baseline Environment for Greedy Baseline
    baseline_env = ConstructionSchedulingEnv(
        grid_shape=(cfg.grid_rows, cfg.grid_cols),
        num_robots=cfg.robots,
        heavy_ratio=cfg.heavy_ratio,
        max_steps=cfg.max_steps,
        travel_speed_cells_per_step=cfg.travel_speed_cells_per_step,
        travel_reward_weight=cfg.travel_reward_weight,
        seed=cfg.seed + 1234,
    )

    # Average evaluation over many episodes for greedy baseline
    greedy_avg_metrics = evaluate_policy_detailed(
        "greedy baseline average evaluation",
        baseline_env,
        agent=None,
        episodes=args.episodes,
        seed=cfg.seed + 2024,
        use_greedy_baseline=True,
        episode_seeds=shared_episode_seeds,
    )

    # One single Baseline rollout for visualization / GIF
    baseline_env.seed = single_demo_seed
    baseline_episode = run_greedy_baseline(baseline_env)
    greedy_single_summary = summarize_episode("greedy baseline single-run example", baseline_episode)
    os.makedirs(os.path.dirname(args.baseline_gif), exist_ok=True)
    imageio.mimsave(args.baseline_gif, baseline_episode["frames"], duration=0.35)
    print(f"saved greedy baseline gif: {args.baseline_gif}")

    # Save metrics to JSON file
    metrics_payload = {
        "model_path": args.model,
        "episodes": args.episodes,
        "learned_policy_average": learned_avg_metrics,
        "greedy_baseline_average": greedy_avg_metrics,
        "learned_policy_single_run": learned_single_summary,
        "greedy_baseline_single_run": greedy_single_summary,
    }

    save_metrics_json(args.metrics_json, metrics_payload)

    # Optional Save Baseline DAG GIF
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
        baseline_dag_env.seed = single_demo_seed
        baseline_dag_episode = run_greedy_baseline(baseline_dag_env, render_mode="dag")
        os.makedirs(os.path.dirname(args.baseline_dag_gif), exist_ok=True)
        imageio.mimsave(args.baseline_dag_gif, baseline_dag_episode["frames"], duration=0.35)
        print(f"saved greedy baseline DAG gif: {args.baseline_dag_gif}")


if __name__ == "__main__":
    main()
