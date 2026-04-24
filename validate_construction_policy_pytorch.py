# Validate a trained PyTorch MAPPO construction scheduling policy.
# loads a .pth checkpoint, evaluates the learned policy and baselines on shared episode seeds, and saves GIFs for qualitative inspection


from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, fields
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import cv2
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
    valid_fields = {field.name for field in fields(TrainConfig)}
    filtered_config = {
        key: value
        for key, value in config_data.items()
        if key in valid_fields
    }
    return TrainConfig(**filtered_config)


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


PANEL_LABELS = {
    "naive_greedy": "Naive Greedy",
    "greedy_baseline": "Greedy Baseline",
    "raw_policy": "Raw Learned Policy",
    "safe_decoder": "MAPPO + Safety Decoder",
}


def label_frame(
    frame: np.ndarray,
    label: str,
    step_text: str | None = None,
    finished: bool = False,
    finished_text: str | None = None,
) -> np.ndarray:
    labeled = frame.copy()
    bar_height = 34
    labeled[:bar_height, :, :] = np.array([245, 247, 248], dtype=np.uint8)
    cv2.putText(
        labeled,
        label,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (48, 52, 55),
        2,
        cv2.LINE_AA,
    )
    if step_text:
        text_size = cv2.getTextSize(step_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
        x = labeled.shape[1] - text_size[0] - 12
        cv2.putText(
            labeled,
            step_text,
            (x, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (82, 88, 92),
            1,
            cv2.LINE_AA,
        )
    if finished:
        h, w = labeled.shape[:2]
        cv2.rectangle(labeled, (3, 3), (w - 4, h - 4), (94, 166, 110), 5)
        text = finished_text or "Finished"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        extra_chars_width = cv2.getTextSize("MMM", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        tag_w = min(text_size[0] + extra_chars_width + 24, w - 24)
        cv2.rectangle(labeled, (12, bar_height + 10), (12 + tag_w, bar_height + 46), (232, 245, 233), -1)
        cv2.putText(
            labeled,
            text,
            (22, bar_height + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (56, 105, 64),
            2,
            cv2.LINE_AA,
        )
    return labeled


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1.0e-6:
        return frame
    height, width = frame.shape[:2]
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def make_comparison_frame(
    frames_by_mode: Dict[str, List[np.ndarray]],
    summaries_by_mode: Dict[str, Dict],
    index: int,
    scale: float,
) -> np.ndarray:
    order = ["naive_greedy", "greedy_baseline", "raw_policy", "safe_decoder"]
    panels = []
    for mode in order:
        frames = frames_by_mode[mode]
        last_index = len(frames) - 1
        clamped_index = min(index, last_index)
        frame = frames[clamped_index]
        frame = resize_frame(frame, scale)
        finished = index >= last_index
        finished_text = None
        if finished:
            summary = summaries_by_mode[mode]
            if summary["success"]:
                finished_text = f"Finished at step {summary['steps']}"
            else:
                finished_text = f"Stopped at step {summary['steps']}"
        panels.append(
            label_frame(
                frame,
                PANEL_LABELS[mode],
                step_text=f"t = {clamped_index}",
                finished=finished,
                finished_text=finished_text,
            )
        )
    top = np.concatenate([panels[0], panels[1]], axis=1)
    bottom = np.concatenate([panels[2], panels[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def save_comparison_gif(
    frames_by_mode: Dict[str, List[np.ndarray]],
    summaries_by_mode: Dict[str, Dict],
    gif_dir: str,
    gif_duration: float,
    scale: float = 0.55,
    final_hold_frames: int = 8,
) -> Dict[str, str]:
    frame_count = max(len(frames) for frames in frames_by_mode.values())
    comparison_frames = [
        make_comparison_frame(frames_by_mode, summaries_by_mode, index, scale)
        for index in range(frame_count)
    ]
    if comparison_frames:
        comparison_frames.extend([comparison_frames[-1].copy() for _ in range(final_hold_frames)])
    gif_path = os.path.join(gif_dir, "comparison_2x2.gif")
    imageio.mimsave(gif_path, comparison_frames, duration=gif_duration)
    print(f"saved 2x2 comparison gif: {gif_path}")
    return {"gif": gif_path, "frames": comparison_frames}


def save_comparison_mp4(
    comparison_frames: List[np.ndarray],
    gif_dir: str,
    fps: float,
) -> str:
    if not comparison_frames:
        raise ValueError("comparison_frames must not be empty.")
    mp4_path = os.path.join(gif_dir, "comparison_2x2.mp4")
    height, width = comparison_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        mp4_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {mp4_path}")
    for frame in comparison_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved 2x2 comparison mp4: {mp4_path}")
    return mp4_path


def save_comparison_avi(
    comparison_frames: List[np.ndarray],
    gif_dir: str,
    fps: float,
) -> str:
    if not comparison_frames:
        raise ValueError("comparison_frames must not be empty.")
    avi_path = os.path.join(gif_dir, "comparison_2x2.avi")
    height, width = comparison_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        avi_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {avi_path}")
    for frame in comparison_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved 2x2 comparison avi: {avi_path}")
    return avi_path


def save_demo_outputs(
    base_env: ConstructionSchedulingEnv,
    agent: MAPPOAgent,
    seed: int,
    gif_dir: str,
    gif_duration: float,
    mp4_fps: float,
) -> Dict[str, Dict]:
    modes = ["naive_greedy", "greedy_baseline", "raw_policy", "safe_decoder"]
    demo_summaries = {}
    frames_by_mode: Dict[str, List[np.ndarray]] = {}
    os.makedirs(gif_dir, exist_ok=True)

    for mode in modes:
        env = make_env_like(base_env, seed)
        episode = run_episode(env, agent, mode, collect_frames=True)
        frames_by_mode[mode] = episode["frames"]
        gif_path = os.path.join(gif_dir, f"{mode}.gif")
        imageio.mimsave(gif_path, episode["frames"], duration=gif_duration)
        summary = summarize_episode(episode)
        summary["gif"] = gif_path
        demo_summaries[mode] = summary
        print(
            f"saved {mode}: {gif_path} | success {summary['success']} "
            f"| steps {summary['steps']} | reward {summary['reward']:.2f}"
        )

    comparison_outputs = save_comparison_gif(
        frames_by_mode,
        demo_summaries,
        gif_dir=gif_dir,
        gif_duration=gif_duration,
    )
    demo_summaries["comparison_2x2"] = {
        "gif": comparison_outputs["gif"],
        "mp4": save_comparison_mp4(
            comparison_outputs["frames"],
            gif_dir=gif_dir,
            fps=mp4_fps,
        ),
        "avi": save_comparison_avi(
            comparison_outputs["frames"],
            gif_dir=gif_dir,
            fps=mp4_fps,
        ),
    }

    return demo_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models_pytorch/construction_mappo_best_raw_eval.pth")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gif-dir", type=str, default="results/pytorch_validation")
    parser.add_argument("--gif-duration", type=float, default=0.55)
    parser.add_argument("--mp4-fps", type=float, default=2.0)
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
        mp4_fps=args.mp4_fps,
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
