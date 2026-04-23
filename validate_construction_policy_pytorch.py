"""Validate a trained PyTorch MAPPO construction scheduling policy.

This script is the PyTorch counterpart of validate_construction_policy.py.
It loads a .pth checkpoint, evaluates the learned policy and baselines on
shared episode seeds, and saves GIFs for qualitative inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Callable, Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

from construction_scheduling_env import ConstructionSchedulingEnv
from train_mappo_construction_pytorch import (
    MAPPOAgent,
    TrainConfig,
    greedy_baseline_action,
    naive_greedy_baseline_action,
    policy_guided_safe_action,
    raw_policy_action,
)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_checkpoint_config(model_path: str) -> TrainConfig:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config_data = checkpoint.get("config", {})
    return TrainConfig(**config_data)


def build_agent_and_env(model_path: str) -> Tuple[ConstructionSchedulingEnv, MAPPOAgent, TrainConfig]:
    cfg = load_checkpoint_config(model_path)
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
    coord_context_dim = 4 * env.num_modules + env.num_robots + 2
    actor_input_dim = obs.shape[1] + coord_context_dim
    agent = MAPPOAgent(actor_input_dim, state.shape[0], env.action_size, cfg)
    agent.load(model_path)
    agent.eval()
    return env, agent, cfg


def make_env_like(base_env: ConstructionSchedulingEnv, seed: int) -> ConstructionSchedulingEnv:
    return ConstructionSchedulingEnv(
        grid_shape=base_env.grid_shape,
        num_robots=base_env.num_robots,
        heavy_ratio=base_env.heavy_ratio,
        max_steps=base_env.max_steps,
        dependency_prob=base_env.dependency_prob,
        max_prerequisites=base_env.max_prerequisites,
        normal_base_duration=base_env.normal_base_duration,
        heavy_base_duration=base_env.heavy_base_duration,
        delay_probability=base_env.delay_probability,
        crane_cooldown_steps=base_env.crane_cooldown_steps,
        travel_speed_cells_per_step=base_env.travel_speed_cells_per_step,
        travel_reward_weight=base_env.travel_reward_weight,
        seed=seed,
    )


def select_actions(
    mode: str,
    env: ConstructionSchedulingEnv,
    agent: MAPPOAgent,
    obs: np.ndarray,
) -> np.ndarray:
    masks = env.action_mask()
    if mode == "naive_greedy":
        return naive_greedy_baseline_action(env)
    if mode == "greedy_baseline":
        return greedy_baseline_action(env)
    if mode == "raw_policy":
        return raw_policy_action(env, agent, obs, masks)
    if mode == "safe_decoder":
        return policy_guided_safe_action(env, agent, obs, masks)
    raise ValueError(f"Unknown validation mode: {mode}")


def run_episode(
    env: ConstructionSchedulingEnv,
    agent: MAPPOAgent,
    mode: str,
    collect_frames: bool = False,
) -> Dict:
    obs, _ = env.reset()
    done = False
    rewards: List[float] = []
    infos: List[Dict] = []
    frames = [env.render_rgb()] if collect_frames else []

    while not done:
        actions = select_actions(mode, env, agent, obs)
        obs, _, reward, done, info = env.step(actions)
        rewards.append(float(reward))
        infos.append(info)
        if collect_frames:
            frames.append(env.render_rgb())

    return {
        "rewards": np.asarray(rewards, dtype=np.float32),
        "infos": infos,
        "frames": frames,
    }


def summarize_episode(episode: Dict) -> Dict[str, float | int | bool]:
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

    final_info = episode["infos"][-1]
    return {
        "success": bool(final_info["success"]),
        "steps": int(len(episode["rewards"])),
        "reward": float(np.sum(episode["rewards"])),
        "completed": int(final_info["completed"]),
        "total_modules": int(final_info["total_modules"]),
        "conflicts": int(conflicts),
        "violations": int(violations),
    }


def aggregate_summaries(summaries: List[Dict]) -> Dict[str, float]:
    keys = ["steps", "reward", "completed", "conflicts", "violations"]
    metrics = {"success_rate": float(np.mean([float(s["success"]) for s in summaries]))}
    for key in keys:
        values = np.asarray([float(s[key]) for s in summaries], dtype=np.float32)
        metrics[f"mean_{key}"] = float(np.mean(values))
        metrics[f"std_{key}"] = float(np.std(values))
        metrics[f"min_{key}"] = float(np.min(values))
        metrics[f"max_{key}"] = float(np.max(values))
    return metrics


def evaluate_modes(
    base_env: ConstructionSchedulingEnv,
    agent: MAPPOAgent,
    modes: List[str],
    episode_seeds: List[int],
) -> Dict[str, Dict]:
    results = {}
    for mode in modes:
        summaries = []
        for seed in episode_seeds:
            env = make_env_like(base_env, seed)
            episode = run_episode(env, agent, mode, collect_frames=False)
            summaries.append(summarize_episode(episode))
        results[mode] = aggregate_summaries(summaries)
    return results


def print_comparison(metrics: Dict[str, Dict]) -> None:
    labels = {
        "naive_greedy": "0. Naive greedy",
        "greedy_baseline": "A. Greedy baseline",
        "raw_policy": "B. Raw learned policy",
        "safe_decoder": "C. Safe decoder policy",
    }
    print("\n=== PYTORCH VALIDATION COMPARISON ===")
    for mode, label in labels.items():
        item = metrics[mode]
        print(
            f"{label:24s} | success {item['success_rate']:.3f} "
            f"| reward {item['mean_reward']:.2f} +/- {item['std_reward']:.2f} "
            f"| length {item['mean_steps']:.1f} +/- {item['std_steps']:.1f} "
            f"| completed {item['mean_completed']:.1f}"
        )


def save_demo_outputs(
    base_env: ConstructionSchedulingEnv,
    agent: MAPPOAgent,
    seed: int,
    gif_dir: str,
    gif_duration: float,
) -> Dict[str, Dict]:
    modes = ["raw_policy", "safe_decoder", "greedy_baseline"]
    demo_summaries = {}
    os.makedirs(gif_dir, exist_ok=True)

    for mode in modes:
        env = make_env_like(base_env, seed)
        episode = run_episode(env, agent, mode, collect_frames=True)
        gif_path = os.path.join(gif_dir, f"{mode}.gif")
        png_path = os.path.join(gif_dir, f"{mode}_final.png")
        imageio.mimsave(gif_path, episode["frames"], duration=gif_duration)
        imageio.imwrite(png_path, episode["frames"][-1])
        summary = summarize_episode(episode)
        summary["gif"] = gif_path
        summary["final_png"] = png_path
        demo_summaries[mode] = summary
        print(
            f"saved {mode}: {gif_path} | success {summary['success']} "
            f"| steps {summary['steps']} | reward {summary['reward']:.2f}"
        )

    return demo_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="construction_models_torch_test/construction_mappo_best_raw_eval.pth")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gif-dir", type=str, default="results/pytorch_validation")
    parser.add_argument("--gif-duration", type=float, default=0.55)
    parser.add_argument("--metrics-json", type=str, default="results/pytorch_validation_metrics.json")
    args = parser.parse_args()

    base_env, agent, cfg = build_agent_and_env(args.model)
    rng = np.random.default_rng(args.seed)
    episode_seeds = [int(rng.integers(0, 1_000_000_000)) for _ in range(args.episodes)]
    modes = ["naive_greedy", "greedy_baseline", "raw_policy", "safe_decoder"]

    metrics = evaluate_modes(base_env, agent, modes, episode_seeds)
    print_comparison(metrics)

    demo_seed = int(cfg.seed + 1234)
    demo_summaries = save_demo_outputs(
        base_env,
        agent,
        seed=demo_seed,
        gif_dir=args.gif_dir,
        gif_duration=args.gif_duration,
    )

    payload = {
        "model_path": args.model,
        "config": asdict(cfg),
        "episodes": args.episodes,
        "episode_seeds": episode_seeds,
        "average_metrics": metrics,
        "demo_summaries": demo_summaries,
    }
    _ensure_parent_dir(args.metrics_json)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"saved metrics json: {args.metrics_json}")


if __name__ == "__main__":
    main()
