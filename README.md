# Multi-Robot Construction Scheduling with MAPPO

This project implements a robotically inspired reinforcement learning system for
high-level multi-robot construction scheduling. A team of heterogeneous robots
must complete a grid of construction modules under random dependency
constraints, normal and heavy task types, travel time, stochastic task duration,
and a shared crane resource.

The final version of the project uses the PyTorch implementation:

- `construction_scheduling_env.py`
- `train_mappo_construction_pytorch.py`
- `validate_construction_policy_pytorch.py`


## Problem Setting

- The construction site is an `N x M` grid. The default setting is `10 x 10`,
  giving 100 construction modules.
- Each module is indexed in row-major order and has a grid-cell location.
- Six robots start from boundary locations around the site.
- A random dependency DAG is sampled each episode. A module can only be assigned
  after its prerequisites are completed.
- Modules can be normal or heavy.
- Normal modules require one robot.
- Heavy modules require two robots and use a shared crane resource.
- Robots are heterogeneous and have different normal-task and heavy-task
  capability multipliers.
- Robot travel is modeled with Manhattan distance. Travel time is included in
  task duration.
- The team receives a shared construction reward, while the actor training also
  uses per-robot shaped rewards to help credit assignment.

The action space is high-level task selection:

- `0 ... num_modules - 1`: assign the robot to a construction module.
- `num_modules`: wait.

This abstraction focuses the project on cooperative task allocation and
construction scheduling rather than low-level motor control.

## Method

The learning method is a MAPPO-inspired centralized-training /
decentralized-execution approach:

- A shared PyTorch actor network scores actions for each robot.
- A centralized critic estimates the value of the global construction state.
- Action masks prevent invalid module selections where possible.
- A greedy behavior-cloning warm start is used before PPO updates.
- PPO-style clipped policy updates train the actor and critic from collected
  rollouts.
- A safety decoder can convert learned task preferences into feasible joint
  actions during evaluation, reducing duplicate normal assignments and ensuring
  heavy tasks receive the required robot pair.

The final comparison reports four policies:

- `Naive greedy`: simple dependency-aware task ordering.
- `Greedy baseline`: hand-designed distance-, capability-, and resource-aware
  scheduler.
- `Raw learned policy`: direct greedy execution of the learned actor.
- `MAPPO + Safety Decoder`: learned actor preferences with feasibility decoding.

## Main Files

- `construction_scheduling_env.py`
  - Environment and visualization.
  - Provides `reset()`, `step(actions)`, `action_mask()`, `get_observations()`,
    `get_global_state()`, and `render_rgb()`.

- `train_mappo_construction_pytorch.py`
  - Final PyTorch MAPPO-inspired training script.
  - Defines the shared actor, centralized critic, rollout buffer, behavior
    cloning, PPO update, training loop, and training-time policy comparison.

- `validate_construction_policy_pytorch.py`
  - Final PyTorch validation script.
  - Loads a trained `.pth` checkpoint, evaluates all policies, saves metrics,
    and generates GIF visualizations.

## Environment Setup

The code was developed and tested on Ubuntu/Linux with a conda environment.
Windows may work with the same dependencies installed, but Ubuntu/Linux is the
recommended and tested platform.

Create and activate an environment, then install dependencies:

```bash
conda create -n me5406-mappo python=3.10 -y
conda activate me5406-mappo
pip install -r requirements.txt
```

If you already have the `me5406-mappo` environment, activate it before running
training or validation:

```bash
conda activate me5406-mappo
```

## Training

Run the final PyTorch trainer:

```bash
python train_mappo_construction_pytorch.py \
  --episodes 300 \
  --bc-episodes 300 \
  --bc-epochs 20 \
  --eval-interval 50 \
  --eval-episodes 10 \
  --save-dir models_pytorch \
  --plot-interval 10 \
  --plot-path results/pytorch_training_curves.png
```

For a quick smoke test:

```bash
python train_mappo_construction_pytorch.py \
  --episodes 3 \
  --bc-episodes 5 \
  --bc-epochs 1 \
  --eval-interval 1 \
  --eval-episodes 2 \
  --save-dir models_pytorch_smoke \
  --plot-interval 1 \
  --plot-path results/pytorch_training_curves_smoke.png
```

Training outputs:

- `models_pytorch/construction_mappo_best_raw_eval.pth`
- `models_pytorch/construction_mappo_final.pth`
- `models_pytorch/training_log.csv`
- `results/pytorch_training_curves.png`

## Validation

Validate the trained PyTorch model:

```bash
python validate_construction_policy_pytorch.py \
  --model models_pytorch/construction_mappo_best_raw_eval.pth \
  --episodes 20 \
  --gif-dir results/pytorch_validation_final \
  --gif-duration 0.65 \
  --metrics-json results/pytorch_validation_final_metrics.json
```

Validation outputs:

- `results/pytorch_validation_final/raw_policy.gif`
- `results/pytorch_validation_final/safe_decoder.gif`
- `results/pytorch_validation_final/greedy_baseline.gif`
- `results/pytorch_validation_final/raw_policy_final.png`
- `results/pytorch_validation_final/safe_decoder_final.png`
- `results/pytorch_validation_final/greedy_baseline_final.png`
- `results/pytorch_validation_final_metrics.json`

The GIF visualization shows robot movement using Manhattan-style paths. Module
colors distinguish normal versus heavy modules and whether each module is
locked, ready, active, or completed.

## Example Final Result

One final policy comparison on the `10 x 10` task produced:

| Method | Success | Reward | Length | Completed |
| --- | ---: | ---: | ---: | ---: |
| Naive greedy | 0.000 | -1121.54 | 900.0 | 18.4 |
| Greedy baseline | 1.000 | 303.31 | 133.2 | 100.0 |
| Raw learned policy | 1.000 | 250.23 | 146.5 | 100.0 |
| MAPPO + Safety Decoder | 1.000 | 302.35 | 138.8 | 100.0 |

These results show that the raw learned policy can complete the task, while the
safety-decoded learned policy approaches the performance of the strong
hand-designed greedy baseline.

## Notes for Submission

For the final project submission, include:

- Source code.
- A trained PyTorch checkpoint in `models_pytorch/`.
- Validation GIF/video results.
- `requirements.txt`.
- This `README.md`.
- The individual report PDF.

Do not include Python cache folders such as `__pycache__/` or temporary
experiment folders that are not part of the final result.
