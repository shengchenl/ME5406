"""Multi-robot cooperative construction scheduling environment.

This enhanced version is still small enough for a course project, but it is no
longer a static one-step DAG toy problem. Each episode samples a new dependency
DAG, modules can take multiple timesteps, robots have heterogeneous speed
profiles, spatial travel costs, and a shared crane has a short cooldown after
task dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


WAIT_ACTION = -1


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
    """CTDE-friendly cooperative construction scheduling task.

    Main mechanics:
    - N x M grid, one module per cell. The project default is now 10 x 10.
    - A fresh random DAG is generated each episode.
    - Some modules are heavy and require two robots to start.
    - Normal and heavy modules take multiple timesteps to finish.
    - Robots have heterogeneous normal/heavy task duration multipliers.
    - Robots start from site boundary/corner points and pay travel time to modules.
    - The shared crane can dispatch at most one new task, then enters cooldown.
    - Rewards are team-shared.
    """

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
        self.max_travel_distance = float(max(1.0, sum(self.grid_shape)))
        self.dependencies: List[List[int]] = [[] for _ in range(self.num_modules)]
        self.completed = np.zeros(self.num_modules, dtype=np.float32)
        self.in_progress = np.zeros(self.num_modules, dtype=np.float32)
        self.remaining_module_time = np.zeros(self.num_modules, dtype=np.float32)
        self.module_completion_time = np.ones(self.num_modules, dtype=np.int32) * -1
        self.module_completion_order = np.ones(self.num_modules, dtype=np.int32) * -1
        self.heavy_mask = np.zeros(self.num_modules, dtype=np.float32)
        self.module_owners = np.zeros((self.num_modules, self.num_robots), dtype=np.float32)
        self.robot_task = np.ones(self.num_robots, dtype=np.int64) * WAIT_ACTION
        self.robot_remaining_time = np.zeros(self.num_robots, dtype=np.float32)
        self.robot_status = np.zeros(self.num_robots, dtype=np.float32)
        self.cooperation_requests = np.zeros(self.num_robots, dtype=np.float32)
        self.resource_occupied = 0.0
        self.crane_cooldown = 0
        self.steps = 0
        self.last_started_module = WAIT_ACTION
        self.last_completed_modules: List[int] = []
        self.completion_event_count = 0

    def _build_robot_capabilities(self) -> np.ndarray:
        """Rows are robots, columns are [normal_multiplier, heavy_multiplier]."""

        templates = np.array(
            [
                [0.75, 1.25],
                [1.00, 0.75],
                [1.20, 1.00],
                [0.90, 1.10],
                [1.10, 0.85],
                [0.85, 1.15],
            ],
            dtype=np.float32,
        )
        if self.num_robots <= len(templates):
            return templates[: self.num_robots].copy()
        extra = np.ones((self.num_robots - len(templates), 2), dtype=np.float32)
        return np.vstack([templates, extra])

    def _build_module_positions(self) -> np.ndarray:
        """Module target positions are grid-cell centers."""

        rows, cols = self.grid_shape
        positions = np.zeros((self.num_modules, 2), dtype=np.float32)
        for module in range(self.num_modules):
            row, col = divmod(module, cols)
            positions[module] = np.array([row + 0.5, col + 0.5], dtype=np.float32)
        return positions

    def _build_robot_start_positions(self) -> np.ndarray:
        """Place robots on grid boundary points, starting from the four corners."""

        rows, cols = self.grid_shape
        anchors = [
            (0.0, 0.0),
            (0.0, float(cols)),
            (float(rows), 0.0),
            (float(rows), float(cols)),
            (0.0, float(cols) / 2.0),
            (float(rows), float(cols) / 2.0),
            (float(rows) / 2.0, 0.0),
            (float(rows) / 2.0, float(cols)),
        ]
        if self.num_robots <= len(anchors):
            return np.asarray(anchors[: self.num_robots], dtype=np.float32)

        positions = list(anchors)
        for idx in range(self.num_robots - len(anchors)):
            frac = (idx + 1) / float(self.num_robots - len(anchors) + 1)
            positions.append((frac * float(rows), frac * float(cols)))
        return np.asarray(positions, dtype=np.float32)

    def robot_module_distances(self) -> np.ndarray:
        """Manhattan distance from every robot to every module center."""

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
        self.module_owners = np.zeros((self.num_modules, self.num_robots), dtype=np.float32)
        self.robot_task = np.ones(self.num_robots, dtype=np.int64) * self.wait_action
        self.robot_remaining_time = np.zeros(self.num_robots, dtype=np.float32)
        self.robot_status = np.zeros(self.num_robots, dtype=np.float32)
        self.cooperation_requests = np.zeros(self.num_robots, dtype=np.float32)
        self.resource_occupied = 0.0
        self.crane_cooldown = 0
        self.steps = 0
        self.last_started_module = WAIT_ACTION
        self.last_completed_modules = []
        self.completion_event_count = 0
        return self.get_observations(), self.get_global_state()

    def get_global_state(self) -> np.ndarray:
        remaining_norm = self.remaining_module_time / float(max(1, self.heavy_base_duration + 2))
        robot_remaining_norm = self.robot_remaining_time / float(max(1, self.heavy_base_duration + 2))
        capability_flat = self.robot_capabilities.reshape(-1) / 1.5
        position_norm = self.robot_positions.reshape(-1) / float(max(1, max(self.grid_shape)))
        return np.concatenate(
            [
                self.completed,
                self.in_progress,
                remaining_norm,
                self.heavy_mask,
                self.dependency_mask(),
                self.robot_status / 2.0,
                robot_remaining_norm,
                capability_flat,
                position_norm,
                self.cooperation_requests,
                np.array([self.resource_occupied], dtype=np.float32),
                np.array([self.crane_cooldown / float(max(1, self.crane_cooldown_steps))], dtype=np.float32),
                np.array([self.steps / float(self.max_steps)], dtype=np.float32),
            ]
        ).astype(np.float32)

    def get_observations(self) -> np.ndarray:
        shared = self.get_global_state()
        observations = []
        for rid in range(self.num_robots):
            robot_id = np.zeros(self.num_robots, dtype=np.float32)
            robot_id[rid] = 1.0
            own_capability = self.robot_capabilities[rid] / 1.5
            own_position = self.robot_positions[rid] / float(max(1, max(self.grid_shape)))
            own_distances = self.robot_module_distances()[rid] / self.max_travel_distance
            observations.append(
                np.concatenate([shared, robot_id, own_capability, own_position, own_distances]).astype(np.float32)
            )
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
            if action == self.wait_action:
                continue
            if self.robot_remaining_time[rid] > 0:
                stats.busy_action_violations += 1
                reward -= 1.0
                continue
            if action < 0 or action >= self.num_modules:
                stats.dependency_violations += 1
                reward -= 1.0
                continue
            if self.completed[action] > 0.5 or self.in_progress[action] > 0.5:
                stats.invalid_completed += 1
                reward -= 1.0
                continue
            if available[action] < 0.5:
                stats.dependency_violations += 1
                reward -= 1.0
                continue
            requested_by_module.setdefault(int(action), []).append(rid)

        normal_candidates: List[Tuple[int, List[int], bool]] = []
        heavy_candidates: List[Tuple[int, List[int], bool]] = []
        for module, robots in requested_by_module.items():
            is_heavy = self.heavy_mask[module] > 0.5
            idle_robots = [rid for rid in robots if self.robot_remaining_time[rid] <= 0]
            if is_heavy:
                if self.crane_cooldown > 0:
                    stats.resource_conflicts += len(idle_robots)
                    reward -= 1.0 * len(idle_robots)
                    continue
                if len(idle_robots) >= 2:
                    heavy_candidates.append((module, idle_robots[:2], True))
                elif idle_robots:
                    self.cooperation_requests[idle_robots[0]] = 1.0
            else:
                if len(idle_robots) == 1:
                    normal_candidates.append((module, idle_robots, False))
                elif len(idle_robots) > 1:
                    stats.task_conflicts += len(idle_robots)
                    reward -= 2.0

        start_candidates: List[Tuple[int, List[int], bool]] = list(normal_candidates)
        if len(heavy_candidates) > 1:
            stats.resource_conflicts += len(heavy_candidates) - 1
            reward -= 2.0 * (len(heavy_candidates) - 1)
        if self.crane_cooldown == 0 and heavy_candidates:
            heavy_candidates.sort(key=lambda item: item[0])
            start_candidates.append(heavy_candidates[0])

        for module, robots, is_heavy in sorted(start_candidates, key=lambda item: item[0]):
            duration = self._duration_for(module, robots, is_heavy)
            travel_distance = float(np.sum(self.robot_module_distances()[robots, module]))
            stats.travel_distance += travel_distance
            reward -= self.travel_reward_weight * travel_distance
            self.in_progress[module] = 1.0
            self.remaining_module_time[module] = float(duration)
            self.module_owners[module, :] = 0.0
            self.module_owners[module, robots] = 1.0
            self.robot_task[robots] = module
            self.robot_remaining_time[robots] = float(duration)
            self.robot_status[robots] = 2.0 if is_heavy else 1.0
            self.last_started_module = int(module)
            if is_heavy:
                self.crane_cooldown = self.crane_cooldown_steps
                self.resource_occupied = 1.0
                stats.started_heavy += 1
                reward += 0.2
            else:
                stats.started_normal += 1
                reward += 0.1

        done = bool(np.all(self.completed > 0.5) or self.steps >= self.max_steps)
        if np.all(self.completed > 0.5):
            reward += 20.0

        obs = self.get_observations()
        state = self.get_global_state()
        info = {
            "stats": stats.__dict__,
            "completed": int(np.sum(self.completed)),
            "in_progress": int(np.sum(self.in_progress)),
            "total_modules": self.num_modules,
            "success": bool(np.all(self.completed > 0.5)),
            "actions": actions.copy(),
            "last_started_module": int(self.last_started_module),
            "last_completed_modules": list(self.last_completed_modules),
            "crane_cooldown": int(self.crane_cooldown),
        }
        return obs, state, float(reward), done, info

    def render_text(self) -> str:
        rows, cols = self.grid_shape
        cells = []
        for module in range(self.num_modules):
            if self.completed[module] > 0.5:
                token = "H" if self.heavy_mask[module] > 0.5 else "X"
            elif self.in_progress[module] > 0.5:
                token = "B" if self.heavy_mask[module] > 0.5 else "b"
            else:
                token = "h" if self.heavy_mask[module] > 0.5 else "."
            cells.append(token)
        lines = [" ".join(cells[r * cols : (r + 1) * cols]) for r in range(rows)]
        return "\n".join(lines)

    def _draw_text(self, image: np.ndarray, text: str, x: int, y: int, color: np.ndarray, scale: int = 2) -> None:
        font = {
            "0": ("111", "101", "101", "101", "111"),
            "1": ("010", "110", "010", "010", "111"),
            "2": ("111", "001", "111", "100", "111"),
            "3": ("111", "001", "111", "001", "111"),
            "4": ("101", "101", "111", "001", "001"),
            "5": ("111", "100", "111", "001", "111"),
            "6": ("111", "100", "111", "101", "111"),
            "7": ("111", "001", "001", "001", "001"),
            "8": ("111", "101", "111", "101", "111"),
            "9": ("111", "101", "111", "001", "111"),
            "t": ("111", "010", "010", "010", "011"),
            "R": ("110", "101", "110", "101", "101"),
            "C": ("111", "100", "100", "100", "111"),
            "D": ("110", "101", "101", "101", "110"),
            "B": ("110", "101", "110", "101", "110"),
            "H": ("101", "101", "111", "101", "101"),
            "N": ("101", "111", "111", "111", "101"),
            ":": ("000", "010", "000", "010", "000"),
            "-": ("000", "000", "111", "000", "000"),
            " ": ("000", "000", "000", "000", "000"),
        }
        cursor = x
        for char in text:
            glyph = font.get(char, font[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        y0 = y + gy * scale
                        y1 = y0 + scale
                        x0 = cursor + gx * scale
                        x1 = x0 + scale
                        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
                            image[y0:y1, x0:x1] = color
            cursor += 4 * scale

    def _text_size(self, text: str, scale: int = 2) -> Tuple[int, int]:
        if not text:
            return 0, 0
        return max(0, (len(text) * 4 - 1) * scale), 5 * scale

    def render_rgb(self, cell_size: int | None = None) -> np.ndarray:
        rows, cols = self.grid_shape
        if cell_size is None:
            cell_size = max(36, min(64, 720 // max(rows, cols)))
        header = 42
        legend = 92
        image = np.ones((rows * cell_size + header, cols * cell_size + legend, 3), dtype=np.uint8) * 245
        robot_colors = np.array(
            [
                [210, 64, 72],
                [58, 126, 196],
                [70, 156, 90],
                [145, 92, 182],
                [213, 122, 46],
                [78, 158, 160],
                [172, 84, 116],
                [105, 118, 190],
            ],
            dtype=np.uint8,
        )

        image[:header, :, :] = np.array([236, 238, 242], dtype=np.uint8)
        progress_width = int((cols * cell_size) * np.mean(self.completed))
        image[header - 12 : header - 4, 8 : 8 + progress_width] = np.array([74, 150, 96], dtype=np.uint8)
        self._draw_text(image, f"t{self.steps}", 8, 10, np.array([40, 40, 40], dtype=np.uint8), scale=2)
        if self.crane_cooldown > 0:
            image[8:28, cols * cell_size - 28 : cols * cell_size - 8] = np.array([202, 142, 52], dtype=np.uint8)
        else:
            image[8:28, cols * cell_size - 28 : cols * cell_size - 8] = np.array([74, 150, 96], dtype=np.uint8)
        self._draw_text(image, "C", cols * cell_size - 22, 12, np.array([255, 255, 255], dtype=np.uint8), scale=2)

        for module in range(self.num_modules):
            row, col = divmod(module, cols)
            y0, y1 = header + row * cell_size, header + (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size
            if self.completed[module] > 0.5 and self.heavy_mask[module] > 0.5:
                color = np.array([188, 211, 238], dtype=np.uint8)
            elif self.completed[module] > 0.5:
                color = np.array([205, 232, 207], dtype=np.uint8)
            elif self.in_progress[module] > 0.5 and self.heavy_mask[module] > 0.5:
                color = np.array([232, 205, 143], dtype=np.uint8)
            elif self.in_progress[module] > 0.5:
                color = np.array([220, 226, 151], dtype=np.uint8)
            elif self.heavy_mask[module] > 0.5:
                color = np.array([202, 142, 52], dtype=np.uint8)
            else:
                color = np.array([230, 230, 230], dtype=np.uint8)
            image[y0:y1, x0:x1] = color
            image[y0 : y0 + 2, x0:x1] = 40
            image[y1 - 2 : y1, x0:x1] = 40
            image[y0:y1, x0 : x0 + 2] = 40
            image[y0:y1, x1 - 2 : x1] = 40

            if self.remaining_module_time[module] > 0:
                max_travel_time = np.ceil(self.max_travel_distance / self.travel_speed_cells_per_step)
                max_time = max(1.0, float(self.heavy_base_duration + max_travel_time + 2))
                bar_width = int((cell_size - 12) * min(1.0, self.remaining_module_time[module] / max_time))
                image[y0 + 6 : y0 + 12, x0 + 6 : x0 + 6 + bar_width] = np.array([40, 40, 40], dtype=np.uint8)
                self._draw_text(
                    image,
                    f"R{int(self.remaining_module_time[module])}",
                    x0 + 18,
                    y0 + 20,
                    np.array([40, 40, 40], dtype=np.uint8),
                    scale=2,
                )
            elif self.completed[module] > 0.5:
                label = str(int(self.module_completion_order[module]))
                label_scale = 3 if cell_size >= 58 and len(label) <= 3 else 2
                text_w, text_h = self._text_size(label, scale=label_scale)
                label_x = x0 + (cell_size - text_w) // 2
                label_y = y0 + (cell_size - text_h) // 2 - 3
                bg_pad = 5
                image[
                    max(y0 + 4, label_y - bg_pad) : min(y1 - 16, label_y + text_h + bg_pad),
                    max(x0 + 4, label_x - bg_pad) : min(x1 - 4, label_x + text_w + bg_pad),
                ] = np.array([248, 248, 248], dtype=np.uint8)
                self._draw_text(
                    image,
                    label,
                    label_x,
                    label_y,
                    np.array([35, 35, 35], dtype=np.uint8),
                    scale=label_scale,
                )

            owners = np.where(self.module_owners[module] > 0.5)[0]
            if len(owners) > 0:
                strip_height = max(6, cell_size // 7)
                strip_y0 = y1 - strip_height - 4
                strip_y1 = y1 - 4
                inner_x0 = x0 + 6
                inner_x1 = x1 - 6
                width = inner_x1 - inner_x0
                for idx, robot_id in enumerate(owners):
                    sx0 = inner_x0 + idx * width // len(owners)
                    sx1 = inner_x0 + (idx + 1) * width // len(owners)
                    image[strip_y0:strip_y1, sx0:sx1] = robot_colors[robot_id % len(robot_colors)]

                marker_size = max(8, cell_size // 5)
                for idx, robot_id in enumerate(owners):
                    my0 = y0 + 6 + idx * (marker_size + 3)
                    my1 = min(my0 + marker_size, y1 - strip_height - 8)
                    mx0 = x1 - 6 - marker_size
                    mx1 = x1 - 6
                    if my1 > my0:
                        image[my0:my1, mx0:mx1] = robot_colors[robot_id % len(robot_colors)]

        for rid, position in enumerate(self.robot_positions):
            py = int(header + position[0] * cell_size)
            px = int(position[1] * cell_size)
            py = int(np.clip(py, header + 4, header + rows * cell_size - 5))
            px = int(np.clip(px, 4, cols * cell_size - 5))
            image[py - 4 : py + 5, px - 4 : px + 5] = np.array([35, 35, 35], dtype=np.uint8)
            image[py - 3 : py + 4, px - 3 : px + 4] = robot_colors[rid % len(robot_colors)]

        lx0 = cols * cell_size + 8
        image[:, cols * cell_size :, :] = np.array([250, 250, 250], dtype=np.uint8)
        for rid in range(self.num_robots):
            y = header + 10 + rid * 24
            image[y : y + 14, lx0 : lx0 + 24] = robot_colors[rid % len(robot_colors)]
            if self.robot_remaining_time[rid] > 0:
                max_travel_time = np.ceil(self.max_travel_distance / self.travel_speed_cells_per_step)
                max_busy_time = float(self.heavy_base_duration + max_travel_time + 2)
                busy_width = int(46 * min(1.0, self.robot_remaining_time[rid] / max_busy_time))
                image[y + 4 : y + 10, lx0 + 32 : lx0 + 32 + busy_width] = np.array([40, 40, 40], dtype=np.uint8)
            else:
                image[y + 4 : y + 10, lx0 + 32 : lx0 + 46] = np.array([170, 210, 170], dtype=np.uint8)
        return image

    def render_dag_rgb(self, width: int = 1280, height: int = 900) -> np.ndarray:
        """Render the dependency DAG as an RGB image for report/video figures.

        Edges point from prerequisite modules to dependent modules. Node fill:
        gray = locked, white = available, yellow = in progress, green/blue =
        completed normal/heavy. Heavy modules use square markers.
        """

        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        depth = np.zeros(self.num_modules, dtype=np.int32)
        for module in range(self.num_modules):
            if self.dependencies[module]:
                depth[module] = 1 + max(depth[d] for d in self.dependencies[module])

        layers: Dict[int, List[int]] = {}
        for module, d in enumerate(depth):
            layers.setdefault(int(d), []).append(module)

        max_depth = max(layers) if layers else 0
        positions: Dict[int, Tuple[float, float]] = {}
        for d, nodes in layers.items():
            x = d / max(1, max_depth)
            for idx, module in enumerate(nodes):
                y = 0.5 if len(nodes) == 1 else 1.0 - idx / (len(nodes) - 1)
                positions[module] = (x, y)

        fig_w = width / 100.0
        fig_h = height / 100.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax.set_facecolor("#f7f7f7")
        fig.patch.set_facecolor("#f7f7f7")

        for module, deps in enumerate(self.dependencies):
            x1, y1 = positions[module]
            for dep in deps:
                x0, y0 = positions[dep]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#9a9a9a", lw=0.8, shrinkA=10, shrinkB=10),
                    zorder=1,
                )

        available = self.dependency_mask()
        for module in range(self.num_modules):
            x, y = positions[module]
            is_heavy = self.heavy_mask[module] > 0.5
            if self.completed[module] > 0.5 and is_heavy:
                color = "#9fc2e6"
            elif self.completed[module] > 0.5:
                color = "#9bd49f"
            elif self.in_progress[module] > 0.5:
                color = "#e8cf73"
            elif available[module] > 0.5:
                color = "#ffffff"
            else:
                color = "#d3d3d3"

            marker = "s" if is_heavy else "o"
            ax.scatter(
                [x],
                [y],
                s=360 if self.num_modules >= 80 else (520 if is_heavy else 440),
                marker=marker,
                c=color,
                edgecolors="#303030",
                linewidths=1.2,
                zorder=3,
            )

            if self.completed[module] > 0.5:
                label = str(int(self.module_completion_order[module]))
            elif self.in_progress[module] > 0.5:
                label = f"R{int(self.remaining_module_time[module])}"
            else:
                label = str(module)
            label_font = 6 if self.num_modules >= 80 else 8
            ax.text(x, y, label, ha="center", va="center", fontsize=label_font, color="#202020", zorder=4)

        ax.text(
            0.01,
            1.03,
            f"DAG view | step {self.steps} | completed {int(np.sum(self.completed))}/{self.num_modules}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            color="#202020",
        )
        ax.text(
            0.99,
            1.03,
            "circle=normal, square=heavy",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#404040",
        )
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.08, 1.08)
        ax.axis("off")

        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return rgb
