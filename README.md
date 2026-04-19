# Multi-Robot Cooperative Construction Scheduling

This folder adds a course-project implementation inspired by the original
ME5406 single-agent path-planning example. Instead of physical navigation, the
new task studies high-level multi-robot construction scheduling under random
module dependencies, heavy-module cooperation, stochastic task durations,
heterogeneous robot capabilities, spatial travel costs, and a shared
crane/resource constraint.

The default project scale is now a `10 x 10` construction grid with `100`
modules and `6` cooperative robots. The grid size is configurable from the
training script, so larger experiments such as `12 x 12` can also be created.

## Problem Definition

- The construction site is an `N x M` grid. The default setting is `10 x 10`.
- Each cell is one construction module, indexed in row-major order.
- Each module is located at the center of its grid cell.
- Robots start from boundary points around the construction site. The default
  six starts are the four corners plus the top and bottom middle boundary
  points.
- Robot travel uses Manhattan distance. Travel time is added to task duration,
  and a small reward penalty is applied for longer travel.
- When a robot completes a module, its position is updated to that module.
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
- Travel distance penalty: `-0.01 x total assigned travel distance`

## Main Files

- `construction_scheduling_env.py`
  - Gym-style construction scheduling environment.
  - Provides `reset()`, `step(actions)`, `action_mask()`, `render_text()`, and
    `render_rgb()`.

- `train_mappo_construction.py`
  - Trains a MAPPO-inspired centralized-training/decentralized-execution agent.
  - Uses a shared MLP actor for all robots and a centralized MLP critic.
  - Implemented in NumPy so it can run without installing PyTorch or TensorFlow.
  - Uses a policy-guided safety decoder during validation so independent robot
    scores become a feasible joint action on the larger `10 x 10` task.

- `validate_construction_policy.py`
  - Loads a trained model.
  - Runs validation episodes.
  - Saves a learned-policy GIF and a greedy-baseline GIF.

- `requirements.txt`
  - Minimal Python package list for this project code.

## Quick Start

Create an environment with NumPy, Matplotlib, and ImageIO. The current code does
not require TensorFlow or PyTorch.

```bash
pip install -r requirements.txt
```

Train the default `10 x 10` model:

```bash
python3 train_mappo_construction.py \
  --grid-rows 10 \
  --grid-cols 10 \
  --robots 6 \
  --max-steps 900 \
  --bc-episodes 500 \
  --bc-epochs 30 \
  --episodes 700 \
  --travel-speed 3.0 \
  --travel-reward-weight 0.01 \
  --plot-interval 10 \
  --plot-path results/training_curves.png \
  --save-dir models
```

For a faster smoke test:

```bash
python3 train_mappo_construction.py \
  --grid-rows 10 \
  --grid-cols 10 \
  --robots 6 \
  --max-steps 900 \
  --bc-episodes 10 \
  --episodes 3 \
  --travel-speed 3.0 \
  --travel-reward-weight 0.01 \
  --plot-interval 1 \
  --plot-path results/training_curves_smoke.png \
  --save-dir models_smoke
```

During training, the script continuously writes the CSV log and refreshes the
curve image:

- `models/training_log.csv`
- `results/training_curves.png`

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
- `results/training_curves.png`
- `results/validation_episode_completion_order.gif`
- `results/greedy_baseline_completion_order.gif`
- `results/validation_episode_dag.gif`
- `results/greedy_baseline_dag.gif`

Recent spatial-version validation result:

- Current `10 x 10` learned policy validation over 5 episodes:
  - Success rate: `1.000`
  - Mean completion length: `133.600`
  - Mean reward: `171.442`
  - Mean conflicts: `0.000`
  - Mean violations: `0.000`
- The action space is now `101` actions: `100` module choices plus `wait`.

## Baseline

The validation script includes a conventional greedy online scheduler:

- Busy robots wait. If the crane is available and a heavy module is available,
  assign an idle robot pair using both heavy-task capability and travel distance.
- Assign remaining idle robots to different available normal modules according
  to a distance-aware capability cost. Normal modules can be started in parallel.

This baseline is useful for comparison because it has direct access to the
hand-designed scheduling heuristic. The learned policy is expected to approach
this online scheduler while using only neural-network action selection.

## Suggested Report Description

The learning method can be described as a lightweight MAPPO-inspired CTDE method:

- Decentralized scoring: each robot evaluates task preferences with the shared
  actor using its own observation.
- Parameter sharing: all robots use the same actor network.
- Centralized training: the critic receives the full construction state and
  estimates the team value.
- For the larger 10 x 10 setting, a policy-guided safety decoder converts
  independent robot task scores into a valid joint action. This prevents
  duplicate normal-task assignment and ensures heavy modules receive two robots.
- The robot observation includes its current normalized position and its
  normalized distance to every module, allowing the shared actor to learn
  spatially efficient assignment preferences.
- Shared reward: all robots optimize the same construction objective.
- A greedy behavior-cloning warm start is used before policy-gradient updates so
  the team quickly learns feasible dependency-respecting construction behavior.
- The environment is intentionally dynamic: random DAGs, stochastic delays,
  busy robots, heterogeneous capabilities, spatial travel, and crane cooldown
  make it more difficult than static topological scheduling.

Because the implementation is NumPy-only, it is best described as an educational
deep RL implementation with MLP function approximation, not a production MAPPO
library.

## Larger Grid Experiments

The same code can generate larger construction sites by changing the CLI
arguments:

```bash
python3 train_mappo_construction.py \
  --grid-rows 12 \
  --grid-cols 12 \
  --robots 8 \
  --max-steps 900 \
  --bc-episodes 700 \
  --bc-epochs 35 \
  --episodes 900 \
  --save-dir models_12x12
```

For the final report, the safest claim is that `10 x 10` is the main learned
setting, while larger grids are scalability experiments.
