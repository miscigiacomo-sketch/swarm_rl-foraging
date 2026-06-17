# PROJECT CONTEXT - SWARM RL FORAGING

## Project Overview

This project is developed for the BIO / AE4350 course.

The objective is to study **swarm-inspired multi-agent foraging using reinforcement learning** in a custom 2D grid-world environment.

The project starts from a simple single-agent foraging task and progressively increases the complexity of the environment through:

* random baseline comparison;
* PPO training;
* grid-size generalization;
* fixed obstacles;
* randomized obstacles;
* state representation analysis;
* oracle validation;
* failure-case analysis;
* deterministic versus stochastic policy deployment;
* centralized multi-agent control;
* random-obstacle multi-agent generalization;
* 3-agent scalability.

The project is **swarm-inspired**, but not a fully decentralized swarm intelligence system. The current learned controller uses **centralized joint-action PPO**, where one PPO policy selects a joint action for all agents.

The goal is not to build a realistic simulator, but to conduct a controlled experimental study of learning, robustness, generalization, and collective foraging behavior under increasing environmental complexity.

---

## Current Development Status

Current status:

```text
Single-agent baseline experiments: COMPLETED
Fixed-obstacle experiments: COMPLETED
Random-obstacle experiments: COMPLETED
Single-agent failure and stochastic-policy analysis: COMPLETED
Multi-agent no-obstacle experiments: COMPLETED
Multi-agent fixed-obstacle experiments: COMPLETED
Multi-agent random-obstacle experiments: COMPLETED
3-agent scalability experiment: COMPLETED
Final analysis and reporting: NEXT
```

Most recent completed experiment:

```text
Experiment 25 - 3-Agent Random-Obstacle Scalability on 10x10
```

Current next step:

```text
Documentation update, final comparative analysis, plots, README, and final report.
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
│   ├── foraging_env.py
│   └── multi_agent_foraging_env.py
│
├── tests/
│   ├── random_agent.py
│   ├── test_large_grid.py
│   ├── test_multi_agent_env.py
│   ├── test_multi_agent_obstacles.py
│   ├── test_multi_agent_random_obstacles.py
│   ├── test_obstacles.py
│   ├── test_random_obstacles.py
│   └── test_random_obstacle_reachability.py
│
├── train/
│   ├── train_ppo.py
│   ├── train_ppo_obstacles.py
│   ├── train_ppo_random_obstacles.py
│   ├── train_ppo_random_obstacles_100k.py
│   ├── train_ppo_random_obstacles_200k.py
│   ├── train_ppo_multi_agent_2agents.py
│   ├── train_ppo_multi_agent_2agents_obstacles.py
│   ├── train_ppo_multi_agent_2agents_random_obstacles.py
│   └── train_ppo_multi_agent_3agents_random_obstacles_10x10.py
│
├── evaluation/
│   ├── evaluate_random_agent.py
│   ├── evaluate_model.py
│   ├── evaluate_grid_size.py
│   ├── evaluate_obstacle_model.py
│   ├── evaluate_fixed_model_random_obstacles.py
│   ├── evaluate_random_obstacle_model.py
│   ├── evaluate_oracle_random_obstacles.py
│   ├── analyze_random_obstacle_failures.py
│   ├── compare_random_obstacle_generalization.py
│   ├── compare_policy_sampling_modes.py
│   ├── evaluate_stochastic_policy_robustness.py
│   ├── evaluate_random_obstacle_grid_generalization.py
│   ├── compare_agents.py
│   ├── evaluate_multi_agent_baselines.py
│   ├── evaluate_multi_agent_grid_generalization.py
│   ├── evaluate_multi_agent_obstacle_baselines.py
│   ├── evaluate_multi_agent_obstacle_ppo.py
│   ├── evaluate_multi_agent_ppo.py
│   ├── evaluate_multi_agent_random_obstacle_ppo.py
│   ├── evaluate_multi_agent_random_obstacle_grid_generalization.py
│   ├── evaluate_multi_agent_3agents_random_obstacles_10x10.py
│   └── generate_plots.py
│
├── models/
│   ├── ppo_foraging.zip
│   ├── ppo_foraging_obstacles.zip
│   ├── ppo_foraging_obstacles_50k.zip
│   ├── ppo_foraging_obstacles_state.zip
│   ├── ppo_foraging_random_obstacles.zip
│   ├── ppo_foraging_random_obstacles_100k.zip
│   ├── ppo_foraging_random_obstacles_200k.zip
│   ├── ppo_multi_agent_2agents.zip
│   ├── ppo_multi_agent_2agents_obstacles.zip
│   ├── ppo_multi_agent_2agents_random_obstacles.zip
│   └── ppo_multi_agent_3agents_random_obstacles_10x10.zip
│
├── results/
│   ├── report_metrics.txt
│   ├── policy_sampling_robustness_summary.txt
│   ├── random_obstacle_failure_cases.txt
│   ├── random_obstacle_grid_generalization_summary.txt
│   ├── multi_agent_baseline_summary.txt
│   ├── multi_agent_grid_generalization_summary.txt
│   ├── multi_agent_obstacle_baseline_summary.txt
│   ├── multi_agent_ppo_summary.txt
│   ├── multi_agent_random_obstacle_ppo_summary.txt
│   ├── multi_agent_random_obstacle_grid_generalization_summary.txt
│   └── multi_agent_3agents_random_obstacle_10x10_summary.txt
│
├── README.md
├── requirements.txt
├── PROJECT_CONTEXT.md
├── EXPERIMENT_LOG.md
└── ROADMAP.md
```

Note: model `.zip` files are ignored by `.gitignore` and are not intended to be committed. They can be regenerated by running the corresponding training scripts.

---

## Environment Description

The project contains two main environments:

1. `ForagingEnv` for the single-agent experiments;
2. `MultiAgentForagingEnv` for the centralized multi-agent experiments.

Both environments are 2D grid-world foraging tasks.

Default action encoding:

```text
0 = up
1 = down
2 = left
3 = right
```

Reward function:

```text
+1 when an agent reaches the food
0 otherwise
```

Episode termination:

* the food is collected;
* the maximum number of steps is reached.

---

## Single-Agent Observation Space

After obstacle state augmentation, the single-agent observation is:

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

## Multi-Agent Observation Space

The multi-agent environment uses centralized joint-action control.

For 2 agents and 3 obstacles, the observation is:

```text
[
agent1_x,
agent1_y,
agent2_x,
agent2_y,
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
12
```

For 3 agents and 3 obstacles, the observation is:

```text
[
agent1_x,
agent1_y,
agent2_x,
agent2_y,
agent3_x,
agent3_y,
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
14
```

Joint action space:

```text
2 agents: 4^2 = 16 actions
3 agents: 4^3 = 64 actions
```

---

## Obstacle Modes

The environments support fixed and randomized obstacle configurations.

### Fixed Obstacles

Fixed-obstacle experiments use the static obstacle positions:

```text
(2, 1)
(2, 2)
(2, 3)
```

### Randomized Obstacles

Randomized obstacle experiments use:

```python
random_obstacles=True
num_obstacles=3
```

Randomized obstacle generation ensures that:

* obstacles do not overlap with each other;
* obstacles do not spawn on agents;
* food does not spawn on obstacles;
* food does not spawn on agents;
* the food is reachable from at least one agent.

Reachability is validated using breadth-first search.

---

## Completed Experimental Progression

### Basic Single-Agent Experiments

The initial experiments established the base environment, PPO training, random baseline comparison, grid-size generalization, plotting, fixed obstacles, randomized obstacles, oracle validation, failure-case analysis, and deterministic-versus-stochastic policy sampling.

Important single-agent findings:

* PPO solved the basic 5x5 foraging task with 100% success.
* The 5x5 PPO policy generalized reasonably well to 10x10 without obstacles, reaching 96% success.
* Fixed obstacles increased task difficulty.
* Adding obstacle positions to the observation improved fixed-obstacle performance from 70% to 87%.
* A fixed-obstacle PPO policy did not fully generalize to randomized obstacle layouts.
* Training directly on randomized obstacles improved robustness.
* BFS oracle validation confirmed that randomized-obstacle environments were solvable.
* Deterministic PPO failures were dominated by repeated no-move behavior.
* Stochastic PPO evaluation revealed much stronger learned-policy performance than deterministic argmax evaluation.

Best single-agent randomized-obstacle model:

```text
models/ppo_foraging_random_obstacles_200k.zip
```

Best deployment mode:

```python
deterministic=False
```

---

## Multi-Agent Experiments

### Experiment 16 - Multi-Agent No-Obstacle Baselines

| Setup | Method | Success Rate | Average Reward | Average Episode Length |
|---|---|---:|---:|---:|
| 2 agents | Random baseline | 84.30% | 0.84 | 21.30 |
| 2 agents | Greedy decentralized baseline | 100.00% | 1.00 | 2.41 |
| 3 agents | Random baseline | 92.10% | 0.92 | 16.46 |
| 3 agents | Greedy decentralized baseline | 99.00% | 0.99 | 2.52 |

### Experiment 17 - Multi-Agent PPO, No Obstacles

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| PPO deterministic | 98.90% | 0.99 | 2.94 |
| PPO stochastic | 100.00% | 1.00 | 2.58 |

### Experiment 18 - Multi-Agent Grid-Size Generalization, No Obstacles

| Grid | Random | Greedy | PPO Deterministic | PPO Stochastic |
|---|---:|---:|---:|---:|
| 5x5 | 84.30% | 100.00% | 98.90% | 100.00% |
| 10x10 | 37.90% | 100.00% | 96.00% | 99.88% |

Conclusion:

Multi-agent PPO generalized well from 5x5 to 10x10 in the no-obstacle setting. Stochastic PPO again performed slightly better than deterministic PPO.

---

## Recent Completed Experiments

### Experiment 19 - Multi-Agent Fixed-Obstacle Environment

Fixed-obstacle support was added to `MultiAgentForagingEnv`.

Implemented features:

* fixed obstacles;
* obstacle-aware spawning;
* obstacle collision handling;
* rendering support for obstacles;
* collision prevention between agents;
* position-swap prevention;
* compatibility with the no-obstacle multi-agent setup.

Tests passed:

```text
tests/test_multi_agent_obstacles.py
tests/test_multi_agent_env.py
```

---

### Experiment 20 - Multi-Agent Fixed-Obstacle Baselines

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 76.40% | 0.76 | 24.63 |
| Greedy obstacle-aware baseline | 93.10% | 0.93 | 5.83 |

Conclusion:

The greedy obstacle-aware baseline strongly outperformed the random baseline, but remained a hand-coded local heuristic rather than a learned controller.

---

### Experiment 21 - Multi-Agent Fixed-Obstacle PPO

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 76.40% | 0.76 | 24.63 |
| Greedy obstacle-aware baseline | 93.10% | 0.93 | 5.83 |
| PPO deterministic | 92.80% | 0.93 | 6.08 |
| PPO stochastic | 100.00% | 1.00 | 3.32 |

Conclusion:

PPO learned the fixed-obstacle multi-agent task. Deterministic PPO performed close to the greedy baseline, while stochastic PPO achieved perfect success and the shortest average episode length.

---

### Experiment 22 - Multi-Agent Random-Obstacle Environment

The multi-agent environment was extended with randomized reachable obstacles.

Implemented features:

* `random_obstacles=True`;
* `num_obstacles=3`;
* randomized obstacle generation at reset;
* no overlap between obstacles;
* no overlap between obstacles, agents, and food;
* food reachability validation using breadth-first search;
* compatibility with no-obstacle and fixed-obstacle multi-agent environments;
* terminal visualization for random-obstacle layouts.

Validation:

```text
tests/test_multi_agent_random_obstacles.py: 8 passed
fixed-obstacle + random-obstacle tests: 14 passed
no-obstacle multi-agent test: passed
```

Conclusion:

Experiment 22 created the basis for randomized-obstacle multi-agent PPO training and evaluation.

---

### Experiment 23 - Multi-Agent Random-Obstacle PPO, 5x5

| Field | Value |
|---|---|
| Grid Size | 5x5 |
| Number of Agents | 2 |
| Random Obstacles | True |
| Number of Obstacles | 3 |
| Joint Action Space | `4^2 = 16` |
| Observation Size | 12 |
| Training Timesteps | 100,000 |
| Training Script | `train/train_ppo_multi_agent_2agents_random_obstacles.py` |
| Evaluation Script | `evaluation/evaluate_multi_agent_random_obstacle_ppo.py` |
| Results File | `results/multi_agent_random_obstacle_ppo_summary.txt` |

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 81.50% | 0.81 | 22.24 |
| Greedy obstacle-aware baseline | 93.50% | 0.94 | 5.45 |
| PPO deterministic | 82.40% | 0.82 | 10.77 |
| PPO stochastic | 99.40% | 0.99 | 3.91 |

Conclusion:

PPO learned useful behavior in the 2-agent randomized-obstacle environment. Deterministic PPO remained weak compared with the greedy heuristic, but stochastic PPO achieved the best performance.

---

### Experiment 24 - Multi-Agent Random-Obstacle Grid Generalization

The 2-agent PPO model trained on 5x5 randomized obstacles was evaluated on both 5x5 and 10x10 grids.

| Grid | Method | Success Rate | Average Reward | Average Episode Length |
|---|---|---:|---:|---:|
| 5x5 | Random baseline | 81.50% | 0.81 | 22.24 |
| 5x5 | Greedy obstacle-aware baseline | 93.50% | 0.94 | 5.45 |
| 5x5 | PPO deterministic | 82.40% | 0.82 | 10.77 |
| 5x5 | PPO stochastic | 99.40% | 0.99 | 3.91 |
| 10x10 | Random baseline | 37.60% | 0.38 | 38.81 |
| 10x10 | Greedy obstacle-aware baseline | 95.10% | 0.95 | 7.03 |
| 10x10 | PPO deterministic | 86.30% | 0.86 | 11.51 |
| 10x10 | PPO stochastic | 99.98% | 1.00 | 7.15 |

Conclusion:

The 10x10 environment increased the spatial search space while keeping the same number of agents and obstacles. The random baseline dropped strongly, but stochastic PPO generalized extremely well, reaching 99.98% success on 10x10.

---

### Experiment 25 - 3-Agent Random-Obstacle Scalability on 10x10

| Field | Value |
|---|---|
| Grid Size | 10x10 |
| Number of Agents | 3 |
| Random Obstacles | True |
| Number of Obstacles | 3 |
| Joint Action Space | `4^3 = 64` |
| Observation Size | 14 |
| Training Timesteps | 200,000 |
| Training Script | `train/train_ppo_multi_agent_3agents_random_obstacles_10x10.py` |
| Evaluation Script | `evaluation/evaluate_multi_agent_3agents_random_obstacles_10x10.py` |
| Results File | `results/multi_agent_3agents_random_obstacle_10x10_summary.txt` |

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 50.00% | 0.50 | 35.12 |
| Greedy obstacle-aware baseline | 96.20% | 0.96 | 5.66 |
| PPO deterministic | 65.00% | 0.65 | 21.72 |
| PPO stochastic | 99.82% | 1.00 | 9.92 |

Conclusion:

Experiment 25 tested scalability from two to three agents in a 10x10 randomized-obstacle environment.

Increasing the number of agents increased the centralized joint action space from `4^2 = 16` to `4^3 = 64`. Deterministic PPO became much less robust, reaching only 65.00% success. In contrast, stochastic PPO achieved 99.82% mean success.

This strengthens the swarm-inspired framing of the project by showing that the framework can extend from two to three agents while maintaining high stochastic-policy performance.

---

## Main Scientific Findings

1. PPO significantly outperforms a random policy in the basic single-agent foraging task.
2. PPO trained on a 5x5 grid generalizes reasonably well to 10x10 in the basic no-obstacle environment.
3. Obstacles substantially increase task difficulty.
4. Obstacle-aware state representation improves PPO performance.
5. Fixed-obstacle PPO does not fully generalize to randomized obstacle layouts.
6. Training directly on randomized obstacles improves robustness.
7. BFS oracle validation confirms that randomized obstacle environments are solvable.
8. Deterministic PPO failures are often caused by repeated ineffective or invalid action patterns.
9. Stochastic PPO deployment reveals stronger policy performance than deterministic argmax deployment.
10. PPO stochastic is not equivalent to random behavior; actions are sampled from the learned PPO policy distribution.
11. Multi-agent PPO performs well under centralized joint-action control.
12. Random-obstacle PPO generalizes well from 5x5 to 10x10 when evaluated stochastically.
13. Scaling from 2 to 3 agents increases the joint action space from 16 to 64 actions.
14. Deterministic PPO becomes fragile in the 3-agent 10x10 scenario.
15. Stochastic PPO remains highly robust in the 3-agent 10x10 scenario, reaching 99.82% mean success.

---

## Current Best Models

Best single-agent fixed-obstacle model:

```text
models/ppo_foraging_obstacles_state.zip
```

Best single-agent randomized-obstacle model:

```text
models/ppo_foraging_random_obstacles_200k.zip
```

Best 2-agent randomized-obstacle model:

```text
models/ppo_multi_agent_2agents_random_obstacles.zip
```

Best 3-agent randomized-obstacle model:

```text
models/ppo_multi_agent_3agents_random_obstacles_10x10.zip
```

Important deployment note:

```python
deterministic=False
```

The strongest learned-policy results are obtained through stochastic deployment, where actions are sampled from the trained PPO policy distribution.

---

## Current Scientific Story

This project investigates how PPO performance changes as the foraging task becomes more complex.

The progression is:

```text
simple single-agent foraging
→ random baseline comparison
→ grid-size generalization
→ fixed obstacles
→ state representation analysis
→ randomized obstacles
→ training-duration sensitivity
→ oracle validation
→ failure-case analysis
→ deterministic versus stochastic policy deployment
→ centralized multi-agent foraging
→ fixed-obstacle multi-agent PPO
→ randomized-obstacle multi-agent PPO
→ 5x5 to 10x10 multi-agent generalization
→ 3-agent random-obstacle scalability
```

Current key message:

```text
The learned PPO policy distribution is more robust than deterministic argmax deployment suggests. Across randomized-obstacle multi-agent experiments, stochastic PPO consistently outperforms deterministic PPO, random baselines, and often hand-coded greedy heuristics. Scaling from 2 to 3 agents increases the joint action space and makes deterministic deployment more fragile, but stochastic PPO remains highly successful.
```

---

## Swarm-Inspired Framing

The project should be described as:

```text
swarm-inspired multi-agent foraging using reinforcement learning
```

It should not be described as a fully decentralized swarm intelligence system.

Reasons it is swarm-inspired:

* multiple agents operate in a shared environment;
* the task is collective foraging;
* agents share a common reward objective;
* agents interact through collisions, position constraints, and shared food collection;
* robustness is tested under randomized environmental layouts;
* scalability is tested by moving from 2 agents to 3 agents.

Main limitation:

```text
The PPO controller is centralized and selects joint actions for all agents.
```

This limitation should be stated clearly in the report.

---

## Next Planned Steps

The main numerical experiments are now complete.

The remaining work should focus on:

1. updating `EXPERIMENT_LOG.md`, `PROJECT_CONTEXT.md`, and `ROADMAP.md`;
2. final comparative analysis;
3. final plots and tables;
4. learned-behavior and failure analysis;
5. README update;
6. final report preparation.

No additional experiments should be added unless there is a clear reason and enough time.

---

## GitHub Workflow

Repository:

```text
swarm_rl-foraging
```

Workflow:

```text
1. Edit code in VS Code
2. Test code
3. Commit in GitHub Desktop
4. Push to GitHub
```

Rule:

```text
Whenever a feature works, commit and push it.
```

Do not commit:

```text
models/*.zip
venv/
__pycache__/
media/
Visualization/
*.gif
*.mp4
```

Latest completed block:

```text
Train PPO for 3-agent random-obstacle foraging
```
