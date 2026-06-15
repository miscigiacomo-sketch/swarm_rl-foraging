# PROJECT CONTEXT - SWARM RL FORAGING

## Project Overview

This project is developed for the BIO / AE4350 course.

The objective is to study foraging behavior using Reinforcement Learning in a custom 2D grid-world environment.

The project starts from a single-agent reinforcement learning setup and progressively increases the complexity of the environment. It then extends the system toward simple swarm-like multi-agent behavior.

The project focuses on:

* learning performance;
* resource collection efficiency;
* random baseline comparison;
* greedy baseline comparison;
* environment scaling;
* obstacle navigation;
* state representation;
* robustness to randomized environments;
* training-duration sensitivity;
* oracle validation;
* failure-case analysis;
* deterministic vs stochastic policy deployment;
* centralized multi-agent control;
* future multi-agent scalability.

The goal is not to build a realistic simulator, but to conduct a controlled experimental study of reinforcement learning behavior under increasing environmental complexity.

Important framing:

```text
This project implements a bio-inspired reinforcement-learning controller based on PPO for a custom foraging/navigation environment, and extends it toward swarm-like multi-agent coordination.
```

Do not present the work as inventing PPO. Present it as implementing, adapting, and analyzing PPO in a bio-inspired foraging/navigation task.

---

## Development Status

Current status:

```text
Single-agent baseline experiments: COMPLETED
Single-agent fixed-obstacle experiments: COMPLETED
Single-agent randomized-obstacle experiments: COMPLETED
Single-agent oracle and failure analysis: COMPLETED
Single-agent stochastic policy analysis: COMPLETED
Single-agent random-obstacle 10x10 generalization: COMPLETED
Multi-agent no-obstacle environment and baselines: COMPLETED
Multi-agent no-obstacle PPO: COMPLETED
Multi-agent no-obstacle 10x10 generalization: COMPLETED
Multi-agent fixed-obstacle environment: COMPLETED
Multi-agent fixed-obstacle baselines: COMPLETED
Multi-agent fixed-obstacle PPO: COMPLETED
```

Current next step:

```text
Experiment 22 - Multi-Agent Random-Obstacle Environment
```

Do not work on 3 agents yet.

Do not train multi-agent randomized-obstacle PPO until the randomized obstacle environment and tests are implemented.

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

Current important folders and files:

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
│   ├── test_obstacles.py
│   ├── test_random_obstacles.py
│   ├── test_random_obstacle_reachability.py
│   ├── test_multi_agent_env.py
│   └── test_multi_agent_obstacles.py
│
├── train/
│   ├── train_ppo.py
│   ├── train_ppo_obstacles.py
│   ├── train_ppo_random_obstacles.py
│   ├── train_ppo_random_obstacles_100k.py
│   ├── train_ppo_random_obstacles_200k.py
│   ├── train_ppo_multi_agent_2agents.py
│   └── train_ppo_multi_agent_2agents_obstacles.py
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
│   ├── evaluate_multi_agent_ppo.py
│   ├── evaluate_multi_agent_grid_generalization.py
│   ├── evaluate_multi_agent_obstacle_baselines.py
│   ├── evaluate_multi_agent_obstacle_ppo.py
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
│   └── ppo_multi_agent_2agents_obstacles.zip
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
│   ├── multi_agent_baseline_summary.txt
│   ├── multi_agent_ppo_summary.txt
│   ├── multi_agent_grid_generalization_summary.txt
│   ├── multi_agent_obstacle_baseline_summary.txt
│   ├── multi_agent_obstacle_ppo_summary.txt
│   └── report_metrics.txt
│
├── README.md
├── requirements.txt
├── PROJECT_CONTEXT.md
├── EXPERIMENT_LOG.md
└── ROADMAP.md
```

Important GitHub note:

The `models/` directory and `*.zip` files are ignored by `.gitignore`. This is acceptable because the training scripts regenerate the models.

---

## Single-Agent Environment Description

The single-agent environment is a 2D grid-world foraging task.

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

Current single-agent observation after state augmentation:

```text
agent_x, agent_y,
food_x, food_y,
obs1_x, obs1_y,
obs2_x, obs2_y,
obs3_x, obs3_y
```

Observation size:

```text
10
```

---

## Single-Agent Obstacle Modes

### Fixed Obstacles

Fixed obstacle positions:

```text
(2, 1), (2, 2), (2, 3)
```

### Randomized Obstacles

The single-agent environment supports:

```python
ForagingEnv(random_obstacles=True)
```

Randomized obstacle generation ensures:

* obstacles do not overlap with each other;
* obstacles do not spawn on the agent;
* food does not spawn on an obstacle;
* the food is always reachable from the agent.

Reachability is validated using breadth-first search.

---

## Multi-Agent Environment Description

The multi-agent environment is implemented in:

```text
env/multi_agent_foraging_env.py
```

It uses centralized control:

* one policy controls all agents;
* one food source;
* reward is 1 if any agent reaches the food;
* maximum episode length is 50 steps;
* collision prevention is included;
* position-swap prevention is included.

Action space:

```text
4 ** num_agents
```

For 2 agents:

```text
4^2 = 16 joint actions
```

For 3 agents:

```text
4^3 = 64 joint actions
```

Render convention:

```text
A, B, C = agents
F = food
X = obstacle
. = empty cell
```

---

## Multi-Agent No-Obstacle Observation

For 2 agents without obstacles, the observation is:

```text
A_x, A_y, B_x, B_y, food_x, food_y
```

Observation size:

```text
6
```

---

## Multi-Agent Fixed-Obstacle Observation

For 2 agents with 3 fixed obstacles, the observation is:

```text
A_x, A_y, B_x, B_y,
food_x, food_y,
obs1_x, obs1_y,
obs2_x, obs2_y,
obs3_x, obs3_y
```

Observation size:

```text
12
```

Fixed obstacles:

```text
(2, 1), (2, 2), (2, 3)
```

Environment rules:

* agents cannot spawn on obstacle cells;
* food cannot spawn on obstacle cells;
* agents cannot move into obstacle cells;
* collisions between agents are prevented;
* direct position swaps are prevented;
* if `obstacles=None`, the environment remains backward compatible with the previous no-obstacle scripts.

---

# Completed Experimental Progression

## Single-Agent Experiments

| Experiment | Topic | Key Result |
|---:|---|---|
| 1 | Random baseline | 63.00% success, avg length 30.55 |
| 2 | PPO baseline | 100.00% success, avg length 4.26 |
| 3 | 5x5 to 10x10 generalization | 96.00% success on 10x10 |
| 4 | Fixed obstacles PPO 20k | 79.00% success |
| 5 | Fixed obstacles PPO 50k | 70.00% success |
| 6 | Fixed obstacles + state augmentation | 87.00% success |
| 7 | Random obstacle reachability | 1000 / 1000 reachable |
| 8 | Fixed-obstacle PPO on random obstacles | 65.50% success |
| 9 | Random-obstacle PPO 50k | 70.90% success |
| 10 | Random-obstacle PPO 100k | 76.30% deterministic success |
| 11 | Random-obstacle PPO 200k | 76.20% deterministic success |
| 12 | BFS oracle | 100.00% success |
| 13 | Failure-case analysis | deterministic failures dominated by repeated no-move behavior |
| 14 | Policy sampling analysis | 200k stochastic PPO reached 97.58% success |
| 15 | Random-obstacle 10x10 generalization | 200k stochastic PPO reached 98.08% success on 10x10 |

---

## Multi-Agent Experiments

| Experiment | Topic | Key Result |
|---:|---|---|
| 16 | Multi-agent no-obstacle baselines | 2-agent random 84.30%, greedy 100.00% |
| 17 | Multi-agent PPO, 2 agents | deterministic 98.90%, stochastic 100.00% |
| 18 | Multi-agent no-obstacle 10x10 generalization | stochastic PPO 99.88% on 10x10 |
| 19 | Multi-agent fixed-obstacle environment | obstacle support added and tested |
| 20 | Multi-agent fixed-obstacle baselines | random 76.40%, greedy 93.10% |
| 21 | Multi-agent fixed-obstacle PPO | deterministic 92.80%, stochastic 100.00% |

---

# Current Best Models

Best single-agent fixed-obstacle model:

```text
models/ppo_foraging_obstacles_state.zip
```

Best single-agent randomized-obstacle model:

```text
models/ppo_foraging_random_obstacles_200k.zip
```

Recommended evaluation mode for the best single-agent randomized-obstacle model:

```python
deterministic=False
```

Best multi-agent no-obstacle model:

```text
models/ppo_multi_agent_2agents.zip
```

Best multi-agent fixed-obstacle model:

```text
models/ppo_multi_agent_2agents_obstacles.zip
```

Recommended evaluation mode for the fixed-obstacle multi-agent model:

```python
deterministic=False
```

---

# Key Results and Interpretations

1. PPO strongly outperformed the random baseline in the basic single-agent foraging task.
2. PPO generalized reasonably well from 5x5 to 10x10 in the no-obstacle single-agent environment.
3. Fixed obstacles increased task difficulty.
4. Increasing training duration alone did not improve fixed-obstacle performance.
5. Adding obstacle coordinates to the observation improved fixed-obstacle performance from 70% to 87%.
6. A fixed-obstacle PPO policy did not fully generalize to randomized obstacle layouts.
7. Training directly on randomized obstacles improved robustness.
8. BFS oracle validation confirmed that randomized-obstacle environments were solvable.
9. Deterministic PPO failures were dominated by repeated no-move behavior.
10. Stochastic policy sampling strongly improved randomized-obstacle PPO performance.
11. The best single-agent stochastic randomized-obstacle PPO generalized successfully to 10x10.
12. The multi-agent no-obstacle environment worked for both 2 and 3 agents.
13. Centralized PPO successfully learned the 2-agent no-obstacle task.
14. The 2-agent no-obstacle PPO generalized well from 5x5 to 10x10.
15. Fixed-obstacle support was added to the multi-agent environment without breaking backward compatibility.
16. In the 2-agent fixed-obstacle environment, the random baseline achieved 76.40% success.
17. The greedy obstacle-aware baseline achieved 93.10% success.
18. The fixed-obstacle multi-agent PPO model achieved 92.80% deterministic success.
19. The same fixed-obstacle multi-agent PPO model achieved 100.00% stochastic success with an average episode length of 3.32.
20. Stochastic deployment is again important: it can reveal stronger performance than deterministic argmax action selection.

---

# Current Scientific Story

This project investigates how PPO performance changes as the foraging environment becomes more complex.

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
→ policy sampling analysis
→ random-obstacle grid-size generalization
→ multi-agent no-obstacle foraging
→ multi-agent grid-size generalization
→ multi-agent fixed-obstacle foraging
→ multi-agent fixed-obstacle PPO
→ next: multi-agent randomized obstacles
```

Current key message:

```text
State representation, environment randomization, policy deployment mode, and multi-agent coordination all strongly affect reinforcement-learning performance. Deterministic PPO evaluation can underestimate policy performance, while stochastic action sampling often reveals stronger learned behavior. The current best multi-agent fixed-obstacle PPO policy achieves 100% success under stochastic deployment.
```

---

# Next Planned Step

## Experiment 22 - Multi-Agent Random-Obstacle Environment

Goal:

Extend the multi-agent environment to support variable/random obstacles.

Planned features:

* `random_obstacles=True`;
* `num_obstacles=3`;
* randomized obstacle generation at reset;
* no overlap between obstacles, agents, and food;
* BFS reachability validation so at least one agent can reach the food;
* updated tests for randomized obstacle behavior.

Planned test file:

```text
tests/test_multi_agent_random_obstacles.py
```

Important constraints:

* do not start 3-agent experiments yet;
* do not train PPO on randomized multi-agent obstacles until the environment and tests are finished;
* preserve backward compatibility with the no-obstacle and fixed-obstacle multi-agent scripts.

---

# GitHub Workflow

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

```text
Whenever a feature works: commit + push
```

Latest completed commit:

```text
Train PPO for fixed-obstacle multi-agent foraging
```

Next expected commit:

```text
Add randomized obstacles to multi-agent environment
```
