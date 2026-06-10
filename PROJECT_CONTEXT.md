# PROJECT CONTEXT - SWARM RL FORAGING

## Project Overview

This project is developed for the BIO course.

The objective is to study foraging behavior using Reinforcement Learning in a custom 2D grid-world environment.

The project starts from a single-agent reinforcement learning setup and progressively increases the complexity of the environment. The long-term goal is to extend the system toward simple swarm-like multi-agent behavior.

The project focuses on:

* learning performance;
* resource collection efficiency;
* random baseline comparison;
* environment scaling;
* obstacle navigation;
* state representation;
* robustness to randomized environments;
* training-duration sensitivity;
* future multi-agent scalability.

The goal is not to build a realistic simulator, but to conduct a controlled experimental study of reinforcement learning behavior under increasing environmental complexity.

---

## Development Status

Current status:

```text
Day 1: COMPLETED
Day 2: COMPLETED
Day 3: COMPLETED
Day 4: COMPLETED
Day 5: COMPLETED
Day 6: COMPLETED
Day 7: COMPLETED
Day 8: COMPLETED
Day 9: NEXT
```

---

## Technologies

* Python
* Gymnasium
* Stable-Baselines3
* PPO
* NumPy
* Matplotlib
* PyTorch
* Pygame-ce
* GitHub
* GitHub Desktop
* VS Code

---

## Project Structure

```text
swarm_rl/

├── env/
│   ├── __init__.py
│   └── foraging_env.py
│
├── tests/
│   ├── random_agent.py
│   ├── test_large_grid.py
│   ├── test_obstacles.py
│   ├── test_random_obstacles.py
│   └── test_random_obstacle_reachability.py
│
├── train/
│   ├── train_ppo.py
│   ├── train_ppo_obstacles.py
│   ├── train_ppo_random_obstacles.py
│   └── train_ppo_random_obstacles_100k.py
│
├── evaluation/
│   ├── test_trained_agent.py
│   ├── test_obstacle_agent.py
│   ├── evaluate_model.py
│   ├── evaluate_random_agent.py
│   ├── evaluate_grid_size.py
│   ├── evaluate_obstacle_model.py
│   ├── evaluate_fixed_model_random_obstacles.py
│   ├── evaluate_random_obstacle_model.py
│   ├── compare_random_obstacle_generalization.py
│   ├── compare_agents.py
│   └── generate_plots.py
│
├── models/
│   ├── ppo_foraging.zip
│   ├── ppo_foraging_obstacles.zip
│   ├── ppo_foraging_obstacles_50k.zip
│   ├── ppo_foraging_obstacles_state.zip
│   ├── ppo_foraging_random_obstacles.zip
│   └── ppo_foraging_random_obstacles_100k.zip
│
├── results/
│   ├── success_rate_comparison.png
│   ├── average_steps_comparison.png
│   ├── grid_size_success_rate.png
│   ├── grid_size_average_steps.png
│   ├── obstacle_success_rate.png
│   ├── obstacle_average_steps.png
│   └── report_metrics.txt
│
├── README.md
├── requirements.txt
└── PROJECT_CONTEXT.md
```

---

## Environment Description

The environment is a 2D grid-world foraging task.

Default grid size:

```text
5 x 5
```

Action space:

```text
0 = up
1 = down
2 = left
3 = right
```

Default maximum steps:

```text
50
```

Reward function:

```text
+1 when the agent reaches the food
0 otherwise
```

Episode termination:

* the food is collected;
* the maximum number of steps is reached.

---

## Current Observation Space

After the obstacle state augmentation experiment, the current observation is:

```text
[
agent_x,
agent_y,
food_x,
food_y,
obs1_x,
obs1_y,
obs2_x,
obs2_y,
obs3_x,
obs3_y
]
```

Observation size:

```text
10
```

This allows the PPO policy to observe obstacle positions directly.

---

## Obstacle Modes

The environment currently supports two obstacle modes.

### Fixed Obstacles

Used in the fixed-obstacle experiments.

Static obstacle positions:

```text
[2, 1]
[2, 2]
[2, 3]
```

### Randomized Obstacles

Used in the randomized-obstacle experiments.

The environment can be initialized with:

```python
ForagingEnv(random_obstacles=True)
```

Randomized obstacle generation ensures that:

* obstacles do not overlap with each other;
* obstacles do not spawn on the agent;
* food does not spawn on an obstacle;
* the food is always reachable from the agent.

Reachability is validated using breadth-first search.

---

## Completed Work

### Day 1 - Base Environment

Implemented:

* custom Gymnasium environment;
* reset function;
* step function;
* text rendering;
* random food spawning.

Scientific purpose:

To create a controlled environment for reinforcement learning foraging experiments.

---

### Day 2 - PPO Training

Implemented:

* PPO training pipeline;
* Stable-Baselines3 integration;
* model saving and loading.

Training:

```text
10000 timesteps
```

Model:

```text
models/ppo_foraging.zip
```

Scientific purpose:

To verify that PPO can learn the basic foraging task.

---

### Day 3 - Evaluation and Random Baseline

Implemented:

* PPO evaluation;
* random agent baseline;
* comparison between learned and random behavior.

Results:

| Agent        | Success Rate | Average Reward | Average Steps |
| ------------ | -----------: | -------------: | ------------: |
| Random Agent |          63% |           0.63 |         30.55 |
| PPO Agent    |         100% |           1.00 |          4.26 |

Conclusion:

PPO significantly outperforms the random baseline in the basic foraging environment.

---

### Day 4 - Grid-Size Generalization

Experiment:

Train PPO on a 5x5 grid and evaluate it on a 10x10 grid.

Results:

| Experiment                 | Success Rate | Average Reward | Average Episode Length |
| -------------------------- | -----------: | -------------: | ---------------------: |
| PPO 5x5 evaluated on 10x10 |          96% |           0.96 |                  12.88 |

Conclusion:

The PPO policy generalizes reasonably well from 5x5 to 10x10 in the basic environment, although efficiency decreases.

---

### Day 5 - Reporting and Visualization

Implemented:

* results folder;
* automatic plot generation;
* metrics report generation;
* comparison figures.

Generated files:

* `success_rate_comparison.png`
* `average_steps_comparison.png`
* `grid_size_success_rate.png`
* `grid_size_average_steps.png`
* `report_metrics.txt`

Scientific purpose:

To make the results easier to analyze and report.

---

### Day 6 - Fixed Obstacles and State Representation

Implemented:

* static obstacles;
* obstacle rendering;
* collision handling;
* obstacle testing;
* PPO training with obstacles;
* state augmentation experiment.

Results:

| Experiment                             | Success Rate | Average Reward | Average Episode Length |
| -------------------------------------- | -----------: | -------------: | ---------------------: |
| PPO no obstacles                       |         100% |           1.00 |                   3.86 |
| PPO obstacles 20k                      |          79% |           0.79 |                  13.34 |
| PPO obstacles 50k                      |          70% |           0.70 |                  17.35 |
| PPO obstacles + state augmentation 50k |          87% |           0.87 |                   9.94 |

Conclusion:

Adding obstacle coordinates to the observation improved performance from 70% to 87%.

This suggests that state representation can be more important than simply increasing training duration.

---

### Day 7 - Randomized Obstacles

Implemented:

* randomized obstacle generation;
* obstacle overlap prevention;
* prevention of obstacle spawning on the agent;
* prevention of food spawning on obstacles;
* clean grid rendering;
* reachability validation using breadth-first search.

Validation result:

```text
Episodes tested: 1000
Reachable environments: 1000
Unreachable environments: 0
Reachability Rate: 100.00%
```

Evaluation:

The fixed-obstacle PPO model was evaluated on randomized reachable obstacle environments.

Result:

| Model                                    | Training Environment | Evaluation Environment     | Success Rate | Average Reward | Average Episode Length |
| ---------------------------------------- | -------------------- | -------------------------- | -----------: | -------------: | ---------------------: |
| PPO fixed obstacles + state augmentation | Fixed obstacles      | Random reachable obstacles |       65.50% |           0.66 |                  19.59 |

Conclusion:

The fixed-obstacle PPO policy does not fully generalize to unseen randomized obstacle layouts.

---

### Day 8 - PPO Training on Randomized Obstacles

Implemented:

* PPO training on randomized obstacle environments;
* 50k randomized-obstacle PPO model;
* 100k randomized-obstacle PPO model;
* controlled seeded comparison on the same 1000 randomized environments.

Models:

```text
models/ppo_foraging_random_obstacles.zip
models/ppo_foraging_random_obstacles_100k.zip
```

Final comparison:

| Model                     | Training Environment | Evaluation Environment | Success Rate | Average Reward | Average Episode Length |
| ------------------------- | -------------------- | ---------------------- | -----------: | -------------: | ---------------------: |
| PPO fixed obstacles       | Fixed obstacles      | Random obstacles       |       65.50% |           0.66 |                  19.59 |
| PPO random obstacles 50k  | Random obstacles     | Random obstacles       |       70.90% |           0.71 |                  17.14 |
| PPO random obstacles 100k | Random obstacles     | Random obstacles       |       76.30% |           0.76 |                  14.74 |

Conclusion:

Training directly on randomized obstacle layouts improves generalization to unseen obstacle configurations.

Increasing training duration from 50k to 100k further improves both success rate and efficiency.

---

## Main Scientific Findings So Far

1. PPO significantly outperforms a random policy in the basic foraging task.
2. PPO trained on a 5x5 grid generalizes reasonably well to a 10x10 grid in the basic environment.
3. Adding obstacles substantially increases task difficulty.
4. Increasing training duration alone did not improve fixed-obstacle performance.
5. Adding obstacle information to the observation space improved fixed-obstacle performance from 70% to 87%.
6. A PPO policy trained only on fixed obstacles does not fully generalize to randomized obstacle layouts.
7. Training PPO directly on randomized obstacles improves generalization.
8. Longer training on randomized obstacles improves robustness and efficiency.

---

## Current Best Models

Best fixed-obstacle model:

```text
models/ppo_foraging_obstacles_state.zip
```

Best randomized-obstacle model so far:

```text
models/ppo_foraging_random_obstacles_100k.zip
```

---

## Current Scientific Story

This project investigates how PPO performance changes as the environment becomes more complex.

The progression is:

```text
simple foraging
→ random baseline comparison
→ grid-size generalization
→ fixed obstacles
→ state representation analysis
→ randomized obstacles
→ training-duration sensitivity
→ future multi-agent extension
```

Current key message:

```text
State representation and environment randomization both play important roles in reinforcement learning generalization and robustness.
```

---

## Next Planned Steps

### Day 9 - Randomized Obstacle Training Sensitivity: 200k

Goal:

Train PPO on randomized obstacles for 200k timesteps.

Planned model:

```text
models/ppo_foraging_random_obstacles_200k.zip
```

Scientific question:

Does longer training further improve robustness in randomized obstacle environments?

---

### Day 10 - Random-Obstacle Grid-Size Generalization

Goal:

Evaluate the best randomized-obstacle PPO model trained on 5x5 in a larger 10x10 randomized obstacle environment.

Scientific question:

Does a PPO agent trained on 5x5 randomized obstacles generalize to a larger 10x10 randomized obstacle environment?

---

### Day 11 - Multi-Agent Extension

Goal:

Create a simple multi-agent version of the environment.

Initial target:

```text
2 agents
1 food source
centralized PPO control
joint action space
```

Scientific question:

Does adding multiple agents improve foraging efficiency?

---

### Day 12 - Multi-Agent Scalability and Behavior Analysis

Goal:

Compare single-agent and multi-agent performance and analyze scalability.

Potential experiments:

* 1 agent vs 2 agents;
* PPO vs random baseline;
* possibly 3 agents if time allows;
* trajectory visualization;
* success and failure case analysis.

---

## GitHub Workflow

Repository:

```text
swarm_rl-foraging
```

Owner:

```text
miscigiacomo-sketch
```

Workflow:

```text
1. Edit code in VS Code
2. Test code
3. Commit in GitHub Desktop
4. Push to GitHub
```

Rule:

Whenever a feature works:

```text
Commit + Push
```

Latest completed block:

```text
Train PPO on randomized obstacles for 100k timesteps
```
