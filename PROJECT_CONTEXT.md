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
* oracle validation;
* failure-case analysis;
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
Day 9: COMPLETED
Post-Day 9 validation: COMPLETED
Post-Day 9 policy sampling analysis: COMPLETED
Day 10: COMPLETED
Day 11: NEXT
```

Current next step:

```text
Day 11 - Multi-agent extension
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
│   ├── train_ppo_random_obstacles_100k.py
│   └── train_ppo_random_obstacles_200k.py
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
│   ├── evaluate_oracle_random_obstacles.py
│   ├── analyze_random_obstacle_failures.py
│   ├── compare_random_obstacle_generalization.py
│   ├── evaluate_oracle_random_obstacles.py
│   ├── analyze_random_obstacle_failures.py
│   ├── compare_policy_sampling_modes.py
│   ├── evaluate_stochastic_policy_robustness.py
│   ├── evaluate_random_obstacle_grid_generalization.py
│   ├── compare_agents.py
│   └── generate_plots.py
│
├── models/
│   ├── ppo_foraging.zip
│   ├── ppo_foraging_obstacles.zip
│   ├── ppo_foraging_obstacles_50k.zip
│   ├── ppo_foraging_obstacles_state.zip
│   ├── ppo_foraging_random_obstacles.zip
│   ├── ppo_foraging_random_obstacles_100k.zip
│   └── ppo_foraging_random_obstacles_200k.zip
│
├── results/
│   ├── success_rate_comparison.png
│   ├── average_steps_comparison.png
│   ├── grid_size_success_rate.png
│   ├── grid_size_average_steps.png
│   ├── obstacle_success_rate.png
│   ├── obstacle_average_steps.png
│   ├── random_obstacle_failure_cases.txt
│   ├── policy_sampling_robustness_summary.txt
│   ├── random_obstacle_grid_generalization_summary.txt
│   └── report_metrics.txt
│
├── README.md
├── requirements.txt
├── PROJECT_CONTEXT.md
├── EXPERIMENT_LOG.md
└── ROADMAP.md
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

| Agent | Success Rate | Average Reward | Average Steps |
|---|---:|---:|---:|
| Random Agent | 63% | 0.63 | 30.55 |
| PPO Agent | 100% | 1.00 | 4.26 |

Conclusion:

PPO significantly outperforms the random baseline in the basic foraging environment.

---

### Day 4 - Grid-Size Generalization

Experiment:

Train PPO on a 5x5 grid and evaluate it on a 10x10 grid.

Results:

| Experiment | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| PPO 5x5 evaluated on 10x10 | 96% | 0.96 | 12.88 |

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

| Experiment | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| PPO no obstacles | 100% | 1.00 | 3.86 |
| PPO obstacles 20k | 79% | 0.79 | 13.34 |
| PPO obstacles 50k | 70% | 0.70 | 17.35 |
| PPO obstacles + state augmentation 50k | 87% | 0.87 | 9.94 |

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

| Model | Training Environment | Evaluation Environment | Success Rate | Average Reward | Average Episode Length |
|---|---|---|---:|---:|---:|
| PPO fixed obstacles + state augmentation | Fixed obstacles | Random reachable obstacles | 65.50% | 0.66 | 19.59 |

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

| Model | Training Environment | Evaluation Environment | Success Rate | Average Reward | Average Episode Length |
|---|---|---|---:|---:|---:|
| PPO fixed obstacles | Fixed obstacles | Random obstacles | 65.50% | 0.66 | 19.59 |
| PPO random obstacles 50k | Random obstacles | Random obstacles | 70.90% | 0.71 | 17.14 |
| PPO random obstacles 100k | Random obstacles | Random obstacles | 76.30% | 0.76 | 14.74 |

Conclusion:

Training directly on randomized obstacle layouts improves generalization to unseen obstacle configurations.

Increasing training duration from 50k to 100k further improves both success rate and efficiency.

---

### Day 9 - Randomized Obstacles PPO 200k

Implemented:

* 200k PPO training on randomized reachable obstacle environments;
* new 200k randomized-obstacle model;
* updated seeded comparison including fixed-obstacle, 50k, 100k, and 200k models.

Model:

```text
models/ppo_foraging_random_obstacles_200k.zip
```

Final comparison:

| Model | Training Environment | Evaluation Environment | Success Rate | Average Reward | Average Episode Length |
|---|---|---|---:|---:|---:|
| PPO fixed obstacles | Fixed obstacles | Random obstacles | 65.50% | 0.66 | 19.59 |
| PPO random obstacles 50k | Random obstacles | Random obstacles | 70.90% | 0.71 | 17.14 |
| PPO random obstacles 100k | Random obstacles | Random obstacles | 76.30% | 0.76 | 14.74 |
| PPO random obstacles 200k | Random obstacles | Random obstacles | 76.20% | 0.76 | 14.82 |

Conclusion:

Increasing randomized-obstacle training from 100k to 200k timesteps did not produce a meaningful improvement.

The 100k model achieved a success rate of 76.30%, while the 200k model achieved 76.20%. The average episode length also remained almost unchanged.

This suggests that, under the current sparse reward function and observation design, the randomized-obstacle PPO policy reaches a performance plateau around 100k timesteps.

---

### Post-Day 9 Validation - BFS Oracle and Failure-Case Analysis

Implemented:

* BFS oracle baseline on randomized obstacle environments;
* shortest-path evaluation over 1000 seeded randomized environments;
* PPO failure-case analysis using the best randomized-obstacle model;
* saved failure cases to `results/random_obstacle_failure_cases.txt`.

BFS oracle result:

| Method | Evaluation Environment | Success Rate | Average Shortest Path Length | Average Episode Length |
|---|---|---:|---:|---:|
| BFS oracle | Randomized obstacles | 100.00% | 4.29 | 4.29 |

PPO failure-case analysis:

| Model | Success Rate | Failure Rate | Failures | Average No-Move Steps per Failed Episode | Average Repeated-Position Steps per Failed Episode |
|---|---:|---:|---:|---:|---:|
| PPO random obstacles 100k | 76.30% | 23.70% | 237 | 48.45 | 48.45 |

Conclusion:

The BFS oracle solved 100% of the same randomized-obstacle environments, confirming that all tested environments were solvable.

Failure-case analysis showed that unsuccessful PPO episodes were dominated by repeated no-move behavior. This suggests that PPO failures are not caused by unreachable food positions, but by limitations of the learned deterministic policy under sparse rewards and randomized obstacle layouts.

---

### Post-Day 9 Policy Sampling Robustness Analysis

Implemented:

* deterministic vs stochastic PPO evaluation;
* comparison of the 100k and 200k randomized-obstacle PPO models;
* stochastic robustness evaluation across 10 action seeds;
* saved summary to `results/policy_sampling_robustness_summary.txt`.

Final comparison:

| Model | Deterministic Success | Stochastic Mean Success | Stochastic Std | Mean Episode Length | Mean No-Move Steps |
|---|---:|---:|---:|---:|---:|
| PPO random obstacles 100k | 76.30% | 96.92% | 0.52% | 7.65 | 2.80 |
| PPO random obstacles 200k | 76.20% | 97.58% | 0.29% | 7.17 | 2.51 |

Conclusion:

Deterministic action selection underestimated the learned PPO policy performance because failed episodes were often caused by repeated invalid or ineffective actions in unchanged states.

Stochastic action sampling allowed the policy to escape these local failure states. The 200k randomized-obstacle PPO model reached 97.58% mean success across 10 action seeds, close to the 100% BFS oracle baseline.

The best randomized-obstacle model is now considered to be the 200k PPO model evaluated with stochastic action sampling.

---

### Day 10 - Random-Obstacle Grid-Size Generalization

Implemented:

* evaluation of the best randomized-obstacle PPO model on a larger 10x10 grid;
* comparison between 5x5 and 10x10 randomized-obstacle environments;
* BFS oracle validation on both grid sizes;
* deterministic and stochastic PPO evaluation;
* stochastic robustness evaluation across 10 action seeds;
* results saved to `results/random_obstacle_grid_generalization_summary.txt`.

Model:

```text
models/ppo_foraging_random_obstacles_200k.zip
```

Evaluation mode for the best learned policy:

```python
deterministic=False
```

Final comparison:

| Grid | Oracle Success | PPO Deterministic Success | PPO Stochastic Mean Success | Stochastic Std | PPO Stochastic Average Episode Length |
|---|---:|---:|---:|---:|---:|
| 5x5 | 100.00% | 76.20% | 97.58% | 0.29% | 7.17 |
| 10x10 | 100.00% | 66.90% | 98.08% | 0.19% | 12.60 |

Conclusion:

The stochastic PPO policy trained on 5x5 randomized obstacles generalized successfully to the 10x10 randomized-obstacle environment.

The 10x10 grid required longer trajectories, increasing the average stochastic PPO episode length from 7.17 to 12.60 steps. However, the success rate remained near-oracle, reaching 98.08% on 10x10.

This suggests that the learned stochastic policy can transfer to a larger grid when the number of obstacles remains fixed at 3 and the observation structure remains compatible with the trained model.

---

## Main Scientific Findings So Far

1. PPO significantly outperforms a random policy in the basic foraging task.
2. PPO trained on a 5x5 grid generalizes reasonably well to a 10x10 grid in the basic environment.
3. Adding obstacles substantially increases task difficulty.
4. Increasing training duration alone did not improve fixed-obstacle performance.
5. Adding obstacle information to the observation space improved fixed-obstacle performance from 70% to 87%.
6. A PPO policy trained only on fixed obstacles does not fully generalize to randomized obstacle layouts.
7. Training PPO directly on randomized obstacles improves generalization.
8. Increasing randomized-obstacle training from 50k to 100k improves robustness and efficiency under deterministic evaluation.
9. Deterministic evaluation suggested that randomized-obstacle PPO performance plateaued around 76%.
10. BFS oracle validation confirms that the randomized-obstacle task is solvable, reaching 100% success.
11. Failure-case analysis shows that failed deterministic PPO episodes are dominated by repeated no-move behavior.
12. Stochastic policy sampling greatly improves PPO performance by reducing no-move behavior.
13. The 200k randomized-obstacle PPO model evaluated stochastically achieved the best learned performance on 5x5: 97.58% mean success.
14. The same 200k stochastic PPO policy generalized successfully to 10x10 randomized obstacles, reaching 98.08% mean success.
15. The 10x10 environment increased episode length but not failure rate, suggesting that grid-size scaling mainly affected efficiency rather than task completion under stochastic evaluation.

---

## Current Best Models

Best fixed-obstacle model:

```text
models/ppo_foraging_obstacles_state.zip
```

Best randomized-obstacle model:

```text
models/ppo_foraging_random_obstacles_200k.zip
```

This model should be evaluated with stochastic action sampling:

```python
deterministic=False
```

Under deterministic evaluation, the 100k and 200k models both achieved about 76% success. Under stochastic evaluation, the 200k model achieved the best performance with 97.58% mean success across 10 action seeds.

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
→ oracle validation
→ failure-case analysis
→ policy sampling analysis
→ random-obstacle grid-size generalization
→ future multi-agent extension
```

Current key message:

```text
State representation, environment randomization, and policy deployment mode all strongly affect reinforcement-learning performance. Deterministic evaluation initially suggested a plateau around 76%, but BFS oracle validation, failure-case analysis, and stochastic policy sampling showed that the learned PPO policy was much stronger than the deterministic results suggested. The 200k stochastic PPO policy achieved near-oracle performance on 5x5 randomized obstacles and generalized successfully to 10x10 randomized obstacles.
```

---

## Next Planned Steps

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
Add random-obstacle grid-size generalization
```
