# Multi-Robot Cooperative Construction Scheduling

This folder adds a course-project implementation inspired by the original
ME5406 single-agent path-planning example. Instead of physical navigation, the
new task studies high-level multi-robot construction scheduling under random
module dependencies, heavy-module cooperation, stochastic task durations,
heterogeneous robot capabilities, and a shared crane/resource constraint.

## Problem Definition

- The construction site is an `N x M` grid.
- Each cell is one construction module, indexed in row-major order.
- A module is completed only after its multi-step installation duration elapses.
- A fresh random dependency DAG is sampled each episode. Edges are generated only
  from lower-index modules to higher-index modules, so the graph is acyclic.
- Some modules are heavy and require at least two robots to select the same module.
- Robots are heterogeneous: each robot has different speed multipliers for
  normal and heavy modules.
- Normal modules do not require the crane, so multiple idle robots can start
  different normal modules in the same timestep.
- Heavy modules require the shared crane. The crane can dispatch at most one
  heavy module and then enters a short cooldown.
- The team receives a shared global reward.

Reward terms:

- Normal module started: `+0.1`
- Heavy module started cooperatively: `+0.2`
- Normal module completed: `+1`
- Heavy module completed cooperatively: `+3`
- Full structure completed: `+20`
- Dependency violation or selecting completed module: `-1`
- Task/resource conflict: `-2`
- Busy robot taking a non-wait action: `-1`
- Time penalty per step: `-0.05`

## Main Files

- `construction_scheduling_env.py`
  - Gym-style construction scheduling environment.
  - Provides `reset()`, `step(actions)`, `action_mask()`, `render_text()`, and
    `render_rgb()`.

- `train_mappo_construction.py`
  - Trains a MAPPO-inspired centralized-training/decentralized-execution agent.
  - Uses a shared MLP actor for all robots and a centralized MLP critic.
  - Implemented in NumPy so it can run without installing PyTorch or TensorFlow.

- `validate_construction_policy.py`
  - Loads a trained model.
  - Runs validation episodes.
  - Saves a learned-policy GIF and a greedy-baseline GIF.

- `requirements_construction.txt`
  - Minimal Python package list for this project code.

## Quick Start

Create an environment with NumPy, Matplotlib, and ImageIO. The current code does
not require TensorFlow or PyTorch.

```bash
pip install -r requirements_construction.txt
```

Train a small model:

```bash
python3 train_mappo_construction.py \
  --bc-episodes 3000 \
  --bc-epochs 100 \
  --episodes 20 \
  --max-steps 100 \
  --save-dir models_new
```

For a faster smoke test:

```bash
python3 train_mappo_construction.py \
  --bc-episodes 30 \
  --episodes 3 \
  --save-dir models_smoke
```

Validate a trained model and create GIFs:

```bash
python3 validate_construction_policy.py \
  --model models/construction_mappo_numpy.npz \
  --episodes 30 \
  --gif results/validation_episode_completion_order.gif \
  --baseline-gif results/greedy_baseline_completion_order.gif \
  --dag-gif results/validation_episode_dag.gif \
  --baseline-dag-gif results/greedy_baseline_dag.gif
```

Outputs:

- `models/construction_mappo_numpy.npz`
- `models/final_construction_mappo_numpy.npz`
- `results/training_log.csv`
- `results/validation_episode_completion_order.gif`
- `results/greedy_baseline_completion_order.gif`
- `results/validation_episode_dag.gif`
- `results/greedy_baseline_dag.gif`

Recent enhanced-version validation result over 30 episodes:

- Learned policy success rate: `1.000`
- Learned policy mean completion length: `30.933`
- Learned policy mean reward: `44.453`
- Learned policy mean conflicts: `0.000`
- Learned policy mean violations: `0.000`

## Baseline

The validation script includes a conventional greedy online scheduler:

- Busy robots wait. If the crane is available and a heavy module is available,
  assign the two currently idle robots with the best heavy-task capability.
- Assign remaining idle robots to different available normal modules according
  to their normal-task capability. Normal modules can be started in parallel.

This baseline is useful for comparison because it has direct access to the
hand-designed scheduling heuristic. The learned policy is expected to approach
this online scheduler while using only neural-network action selection.

## Suggested Report Description

The learning method can be described as a lightweight MAPPO-inspired CTDE method:

- Decentralized execution: each robot samples its own action from the shared
  actor using its observation.
- Parameter sharing: all robots use the same actor network.
- Centralized training: the critic receives the full construction state and
  estimates the team value.
- Shared reward: all robots optimize the same construction objective.
- A greedy behavior-cloning warm start is used before policy-gradient updates so
  the team quickly learns feasible dependency-respecting construction behavior.
- The environment is intentionally dynamic: random DAGs, stochastic delays,
  busy robots, heterogeneous capabilities, and crane cooldown make it more
  difficult than static topological scheduling.

Because the implementation is NumPy-only, it is best described as an educational
deep RL implementation with MLP function approximation, not a production MAPPO
library.
