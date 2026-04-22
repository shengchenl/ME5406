
#  Multi-robot cooperative construction scheduling environment.
# Final Optimized Version:
# 1. High-contrast visualization with distinct robot colors.
# 2. Movement arrows showing the path from previous to current step.
# 3. Anti-overlap: Robots at the same grid offset automatically.
# 4. UI: Step counter (Total Time) displayed at top-left.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import cv2


@dataclass
class StepStats:
    started_normal: int = 0
    started_heavy: int = 0
    completed_normal: int = 0
    completed_heavy: int = 0
    dependency_violations: int = 0
    task_conflicts: int = 0
    resource_conflicts: int = 0
    invalid_completed: int = 0
    busy_action_violations: int = 0
    travel_distance: float = 0.0


class ConstructionSchedulingEnv:
    def __init__(
        self,
        grid_shape: Tuple[int, int] = (10, 10),
        num_robots: int = 6,
        heavy_ratio: float = 0.25,
        max_steps: int = 900,
        dependency_prob: float = 0.08,
        max_prerequisites: int = 2,
        normal_base_duration: int = 2,
        heavy_base_duration: int = 4,
        delay_probability: float = 0.25,
        crane_cooldown_steps: int = 1,
        travel_speed_cells_per_step: float = 3.0,
        travel_reward_weight: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.grid_shape = grid_shape
        self.num_robots = num_robots
        self.heavy_ratio = heavy_ratio
        self.max_steps = max_steps
        self.dependency_prob = dependency_prob
        self.max_prerequisites = max_prerequisites
        self.normal_base_duration = normal_base_duration
        self.heavy_base_duration = heavy_base_duration
        self.delay_probability = delay_probability
        self.crane_cooldown_steps = crane_cooldown_steps
        self.travel_speed_cells_per_step = travel_speed_cells_per_step
        self.travel_reward_weight = travel_reward_weight
        self.rng = np.random.default_rng(seed)

        self.num_modules = int(grid_shape[0] * grid_shape[1])
        self.wait_action = self.num_modules
        self.action_size = self.num_modules + 1

        self.robot_capabilities = self._build_robot_capabilities()
        self.module_positions = self._build_module_positions()
        self.robot_start_positions = self._build_robot_start_positions()
        self.robot_positions = self.robot_start_positions.copy()

        # 用于渲染箭头的历史位置记录
        self.last_robot_positions = self.robot_positions.copy()

        # 高对比度机器人颜色 (BGR)
        self.DISTINCT_ROBOT_COLORS = [
            (0, 0, 255),      # 红色
            (255, 0, 0),      # 蓝色
            (0, 255, 0),      # 绿色
            (0, 165, 255),    # 橙色
            (204, 0, 204),    # 紫色
            (0, 255, 255),    # 黄色
        ]

        self.max_travel_distance = float(max(1.0, sum(self.grid_shape)))
        self.dependencies: List[List[int]] = [[] for _ in range(self.num_modules)]
        self.completed = np.zeros(self.num_modules, dtype=np.float32)
        self.in_progress = np.zeros(self.num_modules, dtype=np.float32)
        self.remaining_module_time = np.zeros(self.num_modules, dtype=np.float32)
        self.module_completion_time = np.ones(self.num_modules, dtype=np.int32) * -1
        self.module_completion_order = np.ones(self.num_modules, dtype=np.int32) * -1
        self.heavy_mask = np.zeros(self.num_modules, dtype=np.float32)
        self.module_owners = np.zeros((self.num_modules, self.num_robots), dtype=np.float32)
        self.robot_task = np.ones(self.num_robots, dtype=np.int64) * self.wait_action
        self.robot_remaining_time = np.zeros(self.num_robots, dtype=np.float32)
        self.robot_status = np.zeros(self.num_robots, dtype=np.float32)
        self.cooperation_requests = np.zeros(self.num_robots, dtype=np.float32)
        self.resource_occupied = 0.0
        self.crane_cooldown = 0
        self.steps = 0
        self.last_started_module = self.wait_action
        self.last_completed_modules: List[int] = []
        self.completion_event_count = 0

    def _build_robot_capabilities(self) -> np.ndarray:
        templates = np.array([[0.75, 1.25], [1.00, 0.75], [1.20, 1.00], [0.90, 1.10], [1.10, 0.85], [0.85, 1.15]], dtype=np.float32)
        if self.num_robots <= len(templates):
            return templates[: self.num_robots].copy()
        extra = np.ones((self.num_robots - len(templates), 2), dtype=np.float32)
        return np.vstack([templates, extra])

    def _build_module_positions(self) -> np.ndarray:
        rows, cols = self.grid_shape
        positions = np.zeros((self.num_modules, 2), dtype=np.float32)
        for module in range(self.num_modules):
            row, col = divmod(module, cols)
            positions[module] = np.array([row + 0.5, col + 0.5], dtype=np.float32)
        return positions

    def _build_robot_start_positions(self) -> np.ndarray:
        rows, cols = self.grid_shape
        anchors = [(0.0, 0.0), (0.0, float(cols)), (float(rows), 0.0), (float(rows), float(cols)), (0.0, float(cols) / 2.0), (float(rows), float(cols) / 2.0), (float(rows) / 2.0, 0.0), (float(rows) / 2.0, float(cols))]
        if self.num_robots <= len(anchors):
            return np.asarray(anchors[: self.num_robots], dtype=np.float32)
        positions = list(anchors)
        for idx in range(self.num_robots - len(anchors)):
            frac = (idx + 1) / float(self.num_robots - len(anchors) + 1)
            positions.append((frac * float(rows), frac * float(cols)))
        return np.asarray(positions, dtype=np.float32)

    def robot_module_distances(self) -> np.ndarray:
        delta = np.abs(self.robot_positions[:, None, :] - self.module_positions[None, :, :])
        return np.sum(delta, axis=2).astype(np.float32)

    def _travel_time_for(self, module: int, robots: List[int]) -> int:
        distances = self.robot_module_distances()[robots, module]
        farthest_distance = float(np.max(distances)) if len(distances) else 0.0
        return int(np.ceil(farthest_distance / max(1.0e-6, self.travel_speed_cells_per_step)))

    def _sample_dependencies(self) -> List[List[int]]:
        deps: List[List[int]] = [[] for _ in range(self.num_modules)]
        for module in range(1, self.num_modules):
            candidates = np.arange(module)
            selected = candidates[self.rng.random(module) < self.dependency_prob]
            if len(selected) > self.max_prerequisites:
                selected = self.rng.choice(selected, size=self.max_prerequisites, replace=False)
            deps[module] = sorted(int(x) for x in selected)
        return deps

    def _sample_heavy_modules(self) -> np.ndarray:
        count = max(1, int(round(self.num_modules * self.heavy_ratio)))
        heavy_ids = self.rng.choice(self.num_modules, size=count, replace=False)
        heavy = np.zeros(self.num_modules, dtype=np.float32)
        heavy[heavy_ids] = 1.0
        return heavy

    def dependency_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_modules, dtype=np.float32)
        for module, deps in enumerate(self.dependencies):
            if self.completed[module] > 0.5 or self.in_progress[module] > 0.5:
                continue
            if all(self.completed[d] > 0.5 for d in deps):
                mask[module] = 1.0
        return mask

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.dependencies = self._sample_dependencies()
        self.completed = np.zeros(self.num_modules, dtype=np.float32)
        self.in_progress = np.zeros(self.num_modules, dtype=np.float32)
        self.remaining_module_time = np.zeros(self.num_modules, dtype=np.float32)
        self.module_completion_time = np.ones(self.num_modules, dtype=np.int32) * -1
        self.module_completion_order = np.ones(self.num_modules, dtype=np.int32) * -1
        self.heavy_mask = self._sample_heavy_modules()
        self.robot_positions = self.robot_start_positions.copy()
        self.last_robot_positions = self.robot_positions.copy()
        self.module_owners = np.zeros((self.num_modules, self.num_robots), dtype=np.float32)
        self.robot_task = np.ones(self.num_robots, dtype=np.int64) * self.wait_action
        self.robot_remaining_time = np.zeros(self.num_robots, dtype=np.float32)
        self.robot_status = np.zeros(self.num_robots, dtype=np.float32)
        self.cooperation_requests = np.zeros(self.num_robots, dtype=np.float32)
        self.resource_occupied = 0.0
        self.crane_cooldown = 0
        self.steps = 0
        self.last_started_module = self.wait_action
        self.last_completed_modules = []
        self.completion_event_count = 0
        return self.get_observations(), self.get_global_state()

    def get_global_state(self) -> np.ndarray:
        remaining_norm = self.remaining_module_time / float(max(1, self.heavy_base_duration + 2))
        robot_remaining_norm = self.robot_remaining_time / float(max(1, self.heavy_base_duration + 2))
        capability_flat = self.robot_capabilities.reshape(-1) / 1.5
        position_norm = self.robot_positions.reshape(-1) / float(max(1, max(self.grid_shape)))
        return np.concatenate([self.completed, self.in_progress, remaining_norm, self.heavy_mask, self.dependency_mask(), self.robot_status / 2.0, robot_remaining_norm, capability_flat, position_norm, self.cooperation_requests, np.array([self.resource_occupied], dtype=np.float32), np.array([self.crane_cooldown / float(max(1, self.crane_cooldown_steps))], dtype=np.float32), np.array([self.steps / float(self.max_steps)], dtype=np.float32)]).astype(np.float32)

    def get_observations(self) -> np.ndarray:
        shared = self.get_global_state()
        observations = []
        for rid in range(self.num_robots):
            robot_id = np.zeros(self.num_robots, dtype=np.float32)
            robot_id[rid] = 1.0
            own_capability = self.robot_capabilities[rid] / 1.5
            own_position = self.robot_positions[rid] / float(max(1, max(self.grid_shape)))
            own_distances = self.robot_module_distances()[rid] / self.max_travel_distance
            observations.append(np.concatenate([shared, robot_id, own_capability, own_position, own_distances]).astype(np.float32))
        return np.stack(observations)

    def action_mask(self) -> np.ndarray:
        available = self.dependency_mask()
        mask = np.zeros((self.num_robots, self.action_size), dtype=np.float32)
        normal_available = available * (1.0 - self.heavy_mask)
        heavy_available = available * self.heavy_mask
        for rid in range(self.num_robots):
            if self.robot_remaining_time[rid] <= 0:
                mask[rid, : self.num_modules] = normal_available
                if self.crane_cooldown == 0:
                    mask[rid, : self.num_modules] = np.maximum(mask[rid, : self.num_modules], heavy_available)
            mask[rid, self.wait_action] = 1.0
        return mask

    def _duration_for(self, module: int, robots: List[int], is_heavy: bool) -> int:
        base = self.heavy_base_duration if is_heavy else self.normal_base_duration
        col = 1 if is_heavy else 0
        multiplier = float(np.mean(self.robot_capabilities[robots, col]))
        travel_time = self._travel_time_for(module, robots)
        stochastic_delay = int(self.rng.random() < self.delay_probability)
        return max(1, travel_time + int(np.ceil(base * multiplier)) + stochastic_delay)

    def _advance_active_tasks(self, stats: StepStats) -> float:
        reward = 0.0
        self.last_completed_modules = []
        active_modules = np.where(self.in_progress > 0.5)[0]
        for module in active_modules:
            self.remaining_module_time[module] -= 1.0
            owner_ids = np.where(self.module_owners[module] > 0.5)[0]
            for rid in owner_ids:
                self.robot_remaining_time[rid] = max(0.0, self.robot_remaining_time[rid] - 1.0)
            if self.remaining_module_time[module] <= 0:
                self.in_progress[module] = 0.0
                self.completed[module] = 1.0
                self.module_completion_time[module] = self.steps
                self.remaining_module_time[module] = 0.0
                self.last_completed_modules.append(int(module))
                for rid in owner_ids:
                    self.robot_positions[rid] = self.module_positions[module]
                    self.robot_task[rid] = self.wait_action
                    self.robot_remaining_time[rid] = 0.0
                    self.robot_status[rid] = 0.0
                if self.heavy_mask[module] > 0.5:
                    stats.completed_heavy += 1
                    reward += 3.0
                else:
                    stats.completed_normal += 1
                    reward += 1.0
        if self.last_completed_modules:
            self.completion_event_count += 1
            for module in self.last_completed_modules:
                self.module_completion_order[module] = self.completion_event_count
        return reward

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        self.last_robot_positions = self.robot_positions.copy()
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_robots)
        self.steps += 1
        stats = StepStats()
        reward = -0.05
        self.cooperation_requests[:] = 0.0
        self.last_started_module = self.wait_action
        reward += self._advance_active_tasks(stats)
        if self.crane_cooldown > 0:
            self.crane_cooldown -= 1
        self.resource_occupied = float(self.crane_cooldown > 0)
        available = self.dependency_mask()
        requested_by_module: Dict[int, List[int]] = {}
        for rid, action in enumerate(actions):
            if action == self.wait_action: continue
            if self.robot_remaining_time[rid] > 0: stats.busy_action_violations += 1; reward -= 1.0; continue
            if action < 0 or action >= self.num_modules: stats.dependency_violations += 1; reward -= 1.0; continue
            if self.completed[action] > 0.5 or self.in_progress[action] > 0.5: stats.invalid_completed += 1; reward -= 1.0; continue
            if available[action] < 0.5: stats.dependency_violations += 1; reward -= 1.0; continue
            requested_by_module.setdefault(int(action), []).append(rid)

        heavy_candidates, normal_candidates = [], []
        for module, robots in requested_by_module.items():
            is_heavy = self.heavy_mask[module] > 0.5
            idle_robots = [rid for rid in robots if self.robot_remaining_time[rid] <= 0]
            if is_heavy:
                if self.crane_cooldown > 0:
                    stats.resource_conflicts += len(idle_robots); reward -= 1.0 * len(idle_robots); continue
                if len(idle_robots) >= 2: heavy_candidates.append((module, idle_robots[:2], True))
                elif idle_robots: self.cooperation_requests[idle_robots[0]] = 1.0
            else:
                if len(idle_robots) == 1: normal_candidates.append((module, idle_robots, False))
                elif len(idle_robots) > 1: stats.task_conflicts += len(idle_robots); reward -= 2.0

        start_candidates = list(normal_candidates)
        if self.crane_cooldown == 0 and heavy_candidates:
            heavy_candidates.sort(key=lambda x: x[0]); start_candidates.append(heavy_candidates[0])

        for module, robots, is_heavy in sorted(start_candidates, key=lambda x: x[0]):
            duration = self._duration_for(module, robots, is_heavy)
            stats.travel_distance += float(np.sum(self.robot_module_distances()[robots, module]))
            self.in_progress[module] = 1.0
            self.remaining_module_time[module] = float(duration)
            self.module_owners[module, robots] = 1.0
            self.robot_task[robots] = module
            self.robot_remaining_time[robots] = float(duration)
            self.robot_status[robots] = 2.0 if is_heavy else 1.0
            self.last_started_module = int(module)
            if is_heavy: self.crane_cooldown = self.crane_cooldown_steps; stats.started_heavy += 1; reward += 0.2
            else: stats.started_normal += 1; reward += 0.1

        done = bool(np.all(self.completed > 0.5) or self.steps >= self.max_steps)
        if np.all(self.completed > 0.5): reward += 20.0
        return self.get_observations(), self.get_global_state(), float(reward), done, {"stats": stats.__dict__, "completed": int(np.sum(self.completed)), "total_modules": self.num_modules, "success": bool(np.all(self.completed > 0.5))}

    def render_rgb(self, cell_size: int | None = None) -> np.ndarray:
        rows, cols = self.grid_shape
        if cell_size is None:
            cell_size = 60
        header, legend = 50, 110
        image = np.ones((rows * cell_size + header, cols * cell_size + legend, 3), dtype=np.uint8) * 245

        # 1. Draw Header and Time
        image[:header, :, :] = np.array([236, 238, 242], dtype=np.uint8)
        step_text = f"Step: {self.steps}"
        cv2.putText(image, step_text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

        # 2. Grid
        for module in range(self.num_modules):
            r, c = divmod(module, cols)
            y0, x0 = header + r * cell_size, c * cell_size
            y1, x1 = y0 + cell_size, x0 + cell_size
            if self.completed[module] > 0.5:
                color = (205, 232, 207) if self.heavy_mask[module] < 0.5 else (188, 211, 238)
            elif self.in_progress[module] > 0.5:
                color = (220, 226, 151) if self.heavy_mask[module] < 0.5 else (232, 205, 143)
            else:
                color = (230, 230, 230) if self.heavy_mask[module] < 0.5 else (202, 142, 52)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (40, 40, 40), 1)

        # 3. Draw robots
        occupancy = {}
        for rid, pos in enumerate(self.robot_positions):
            grid_r, grid_c = int(np.clip(pos[0], 0, rows-1)), int(np.clip(pos[1], 0, cols-1))
            occupancy.setdefault((grid_r, grid_c), []).append(rid)

        for (gr, gc), rids in occupancy.items():
            num_here = len(rids)
            base_y, base_x = header + gr * cell_size + cell_size // 2, gc * cell_size + cell_size // 2
            offset_radius = cell_size * 0.22 if num_here > 1 else 0
            for i, rid in enumerate(rids):
                color = self.DISTINCT_ROBOT_COLORS[rid % len(self.DISTINCT_ROBOT_COLORS)]
                angle = (2 * np.pi * i) / num_here
                curr_y = int(base_y + offset_radius * np.sin(angle))
                curr_x = int(base_x + offset_radius * np.cos(angle))

                # 绘制箭头
                prev_pos = self.last_robot_positions[rid]
                if tuple(prev_pos.astype(int)) != (gr, gc):
                    prev_y = int(header + prev_pos[0] * cell_size + cell_size // 2)
                    prev_x = int(prev_pos[1] * cell_size + cell_size // 2)
                    cv2.arrowedLine(image, (prev_x, prev_y), (curr_x, curr_y), color, 2, tipLength=0.25)

                # 绘制机器人圆点
                cv2.circle(image, (curr_x, curr_y), int(cell_size * 0.18), color, -1, cv2.LINE_AA)
                cv2.circle(image, (curr_x, curr_y), int(cell_size * 0.18), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(rid), (curr_x - 4, curr_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Legend
        lx = cols * cell_size + 10
        for rid in range(self.num_robots):
            y = header + 20 + rid * 28
            color = self.DISTINCT_ROBOT_COLORS[rid % len(self.DISTINCT_ROBOT_COLORS)]
            cv2.rectangle(image, (lx, y), (lx + 20, y + 15), color, -1)
            txt = "BUSY" if self.robot_remaining_time[rid] > 0 else "IDLE"
            cv2.putText(image, f"R{rid}:{txt}", (lx + 25, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1)

        return image

    def render_dag_rgb(self, width: int = 1280, height: int = 900) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)
