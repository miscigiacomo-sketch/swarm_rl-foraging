# ROADMAP - SWARM RL FORAGING

This roadmap defines the planned development path for the `swarm_rl-foraging` project.

The project is structured as a progressive reinforcement learning study. The environment complexity is increased step by step, starting from a single-agent grid-world and progressing toward simple swarm-like multi-agent foraging.

---

## Overall Scientific Direction

The project investigates how PPO performance changes under increasing environment complexity.

Main research themes:

* baseline comparison;
* learning efficiency;
* grid-size generalization;
* obstacle navigation;
* state representation;
* randomized environment robustness;
* training-duration sensitivity;
* oracle validation;
* failure-case analysis;
* deterministic vs stochastic policy deployment;
* multi-agent scalability.

The final goal is to present a controlled experimental study of reinforcement learning for foraging behavior, progressing from a single-agent system toward a simple swarm-like multi-agent setup.

Important framing:

```text
This project implements a bio-inspired reinforcement-learning controller based on PPO for a custom foraging/navigation environment, and extends it toward swarm-like multi-agent coordination.
```

The project should not be presented as inventing PPO. It should be presented as implementing, adapting, and experimentally analyzing PPO in a custom bio-inspired foraging task.

---

# Completed Work

## Experiment 1 - Random Agent Baseline

Status: completed

Evaluated a random policy in the basic 5x5 single-agent foraging environment.

Key result:

| Agent | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random Agent | 63.00% | 0.63 | 30.55 |

---

## Experiment 2 - PPO Baseline

Status: completed

Trained PPO on the basic 5x5 single-agent foraging environment.

Key result:

| Agent | Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|---:|
| PPO | 10,000 | 100.00% | 1.00 | 4.26 |

Main finding:

PPO strongly outperformed the random baseline in the basic foraging task.

---

## Experiment 3 - Grid-Size Generalization

Status: completed

Evaluated the PPO model trained on 5x5 in a larger 10x10 no-obstacle grid.

Key result:

| Training Grid | Evaluation Grid | Success Rate | Average Reward | Average Episode Length |
|---|---|---:|---:|---:|
| 5x5 | 10x10 | 96.00% | 0.96 | 12.88 |

Main finding:

The PPO policy generalized reasonably well to a larger grid, but the average episode length increased.

---

## Experiment 4 - Fixed Obstacles, PPO 20k

Status: completed

Introduced fixed obstacles into the single-agent environment.

Key result:

| Environment | Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|---:|
| 5x5 fixed obstacles | 20,000 | 79.00% | 0.79 | 13.34 |

---

## Experiment 5 - Fixed Obstacles, PPO 50k

Status: completed

Trained PPO longer in the fixed-obstacle single-agent environment.

Key result:

| Environment | Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|---:|
| 5x5 fixed obstacles | 50,000 | 70.00% | 0.70 | 17.35 |

Main finding:

Increasing training time alone did not improve performance.

---

## Experiment 6 - Fixed Obstacles with State Augmentation

Status: completed

Added obstacle coordinates to the observation space.

Key result:

| Environment | Observation | Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---|---|---:|---:|---:|---:|
| 5x5 fixed obstacles | agent, food, obstacles | 50,000 | 87.00% | 0.87 | 9.94 |

Main finding:

Adding obstacle coordinates to the state representation improved fixed-obstacle PPO performance from 70% to 87%.

---

## Experiment 7 - Reachability Validation for Randomized Obstacles

Status: completed

Implemented randomized obstacle generation with BFS reachability validation.

Key result:

| Episodes Tested | Reachable Environments | Unreachable Environments | Reachability Rate |
|---:|---:|---:|---:|
| 1000 | 1000 | 0 | 100.00% |

Main finding:

Randomized obstacle environments were guaranteed to be solvable, making later evaluations scientifically cleaner.

---

## Experiment 8 - Fixed-Obstacle PPO Evaluated on Randomized Obstacles

Status: completed

Evaluated the best fixed-obstacle PPO model on randomized reachable obstacle environments.

Key result:

| Training Environment | Evaluation Environment | Success Rate | Average Reward | Average Episode Length |
|---|---|---:|---:|---:|
| Fixed obstacles | Randomized obstacles | 65.50% | 0.66 | 19.59 |

Main finding:

The fixed-obstacle PPO policy did not fully generalize to randomized obstacle layouts.

---

## Experiment 9 - Randomized Obstacles PPO 50k

Status: completed

Trained PPO directly on randomized reachable obstacle environments.

Key result:

| Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---:|---:|---:|---:|
| 50,000 | 70.90% | 0.71 | 17.14 |

---

## Experiment 10 - Randomized Obstacles PPO 100k

Status: completed

Increased randomized-obstacle training to 100k timesteps.

Key result:

| Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---:|---:|---:|---:|
| 100,000 | 76.30% | 0.76 | 14.74 |

Main finding:

Increasing training from 50k to 100k improved robustness and efficiency.

---

## Experiment 11 - Randomized Obstacles PPO 200k

Status: completed

Increased randomized-obstacle training to 200k timesteps.

Key result:

| Training Timesteps | Deterministic Success Rate | Average Reward | Average Episode Length |
|---:|---:|---:|---:|
| 200,000 | 76.20% | 0.76 | 14.82 |

Main finding:

The deterministic success rate plateaued around 76%.

---

## Experiment 12 - BFS Oracle Baseline on Randomized Obstacles

Status: completed

Evaluated a BFS shortest-path oracle on randomized obstacle environments.

Key result:

| Method | Success Rate | Average Shortest Path Length | Failed Seeds |
|---|---:|---:|---|
| BFS oracle | 100.00% | 4.29 | `[]` |

Main finding:

The task was solvable; PPO failures were not caused by impossible environments.

---

## Experiment 13 - PPO Failure-Case Analysis on Randomized Obstacles

Status: completed

Analyzed failed deterministic PPO episodes.

Key result:

| Model | Success Rate | Failure Rate | Failures | Avg No-Move Steps per Failed Episode |
|---|---:|---:|---:|---:|
| PPO random obstacles 100k | 76.30% | 23.70% | 237 | 48.45 |

Main finding:

Failed deterministic PPO episodes were dominated by repeated no-move behavior.

---

## Experiment 14 - PPO Policy Sampling Mode Analysis

Status: completed

Compared deterministic and stochastic policy deployment for randomized-obstacle PPO models.

Key result:

| Model | Deterministic Success | Stochastic Mean Success | Stochastic Std | Mean Episode Length |
|---|---:|---:|---:|---:|
| PPO Random Obstacles 100k | 76.30% | 96.92% | 0.52% | 7.65 |
| PPO Random Obstacles 200k | 76.20% | 97.58% | 0.29% | 7.17 |

Main finding:

Stochastic policy sampling greatly improved performance and reduced repeated no-move failures.

---

## Experiment 15 - Random-Obstacle Grid-Size Generalization

Status: completed

Evaluated the best randomized-obstacle PPO policy on 5x5 and 10x10 randomized-obstacle environments.

Key result:

| Grid | Oracle Success | PPO Deterministic Success | PPO Stochastic Mean Success | PPO Stochastic Average Episode Length |
|---|---:|---:|---:|---:|
| 5x5 | 100.00% | 76.20% | 97.58% | 7.17 |
| 10x10 | 100.00% | 66.90% | 98.08% | 12.60 |

Main finding:

The stochastic PPO policy trained on 5x5 randomized obstacles generalized successfully to a larger 10x10 randomized-obstacle environment.

---

## Experiment 16 - Multi-Agent No-Obstacle Baselines

Status: completed

Created and evaluated no-obstacle multi-agent baselines for 2 and 3 agents.

Key result:

| Agents | Method | Success Rate | Average Reward | Average Episode Length |
|---:|---|---:|---:|---:|
| 2 | Random baseline | 84.30% | 0.84 | 21.30 |
| 2 | Greedy decentralized baseline | 100.00% | 1.00 | 2.41 |
| 3 | Random baseline | 92.10% | 0.92 | 16.46 |
| 3 | Greedy decentralized baseline | 99.00% | 0.99 | 2.52 |

Main finding:

Adding more agents improved random exploration, while the greedy baseline was very strong without obstacles.

---

## Experiment 17 - Multi-Agent PPO, 2 Agents

Status: completed

Trained centralized PPO with 2 agents in the no-obstacle multi-agent environment.

Key result:

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 84.30% | 0.84 | 21.30 |
| Greedy decentralized baseline | 100.00% | 1.00 | 2.41 |
| PPO deterministic | 98.90% | 0.99 | 2.94 |
| PPO stochastic | 100.00% | 1.00 | 2.58 |

Main finding:

The centralized PPO controller successfully learned the two-agent no-obstacle foraging task.

---

## Experiment 18 - Multi-Agent Grid-Size Generalization

Status: completed

Evaluated the two-agent PPO policy trained on 5x5 in both 5x5 and 10x10 no-obstacle environments.

Key result:

| Grid | Random Success | Greedy Success | PPO Deterministic Success | PPO Stochastic Success | PPO Stochastic Average Episode Length |
|---|---:|---:|---:|---:|---:|
| 5x5 | 84.30% | 100.00% | 98.90% | 100.00% | 2.57 |
| 10x10 | 37.90% | 100.00% | 96.00% | 99.88% | 5.64 |

Main finding:

The two-agent PPO policy trained on 5x5 generalized very well to a larger 10x10 no-obstacle environment.

---

## Experiment 19 - Multi-Agent Fixed-Obstacle Environment

Status: completed

Added optional fixed-obstacle support to the centralized multi-agent environment.

Implemented:

* optional `obstacles` argument in `MultiAgentForagingEnv`;
* obstacle coordinates in the observation;
* obstacle-aware agent and food spawning;
* movement blocking into obstacles;
* preservation of collision prevention and position-swap prevention;
* backward compatibility when `obstacles=None`.

Setup:

| Field | Value |
|---|---|
| Environment | `MultiAgentForagingEnv` |
| Grid Size | 5x5 |
| Number of Agents | 2 |
| Fixed Obstacles | `(2, 1)`, `(2, 2)`, `(2, 3)` |
| Joint Action Space | `4^2 = 16` |
| Observation Size | 12 |
| Updated File | `env/multi_agent_foraging_env.py` |
| Test File | `tests/test_multi_agent_obstacles.py` |

Testing:

| Test | Result |
|---|---|
| `tests/test_multi_agent_obstacles.py` | 6 passed |
| `tests/test_multi_agent_env.py` | passed |

Main finding:

Fixed-obstacle support was added without breaking the previous no-obstacle multi-agent workflow.

---

## Experiment 20 - Multi-Agent Fixed-Obstacle Baselines

Status: completed

Evaluated random and greedy obstacle-aware baselines in the 2-agent fixed-obstacle environment.

Key result:

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 76.40% | 0.76 | 24.63 |
| Greedy obstacle-aware baseline | 93.10% | 0.93 | 5.83 |

Main finding:

Fixed obstacles made the multi-agent task harder than the no-obstacle case. The greedy obstacle-aware baseline was strong but not optimal because it uses local Manhattan-distance decisions rather than full path planning.

---

## Experiment 21 - Multi-Agent Fixed-Obstacle PPO

Status: completed

Trained and evaluated a centralized PPO controller in the same 2-agent fixed-obstacle environment used for the Experiment 20 baselines.

Setup:

| Field | Value |
|---|---|
| Environment | `MultiAgentForagingEnv` |
| Grid Size | 5x5 |
| Number of Agents | 2 |
| Max Steps | 50 |
| Fixed Obstacles | `(2, 1)`, `(2, 2)`, `(2, 3)` |
| Control Type | Centralized PPO |
| Joint Action Space | `4^2 = 16` |
| Observation Size | 12 |
| Training Timesteps | 100,000 |
| Model | `models/ppo_multi_agent_2agents_obstacles.zip` |
| Training Script | `train/train_ppo_multi_agent_2agents_obstacles.py` |
| Evaluation Script | `evaluation/evaluate_multi_agent_obstacle_ppo.py` |
| Output File | `results/multi_agent_obstacle_ppo_summary.txt` |

Key result:

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 76.40% | 0.76 | 24.63 |
| Greedy obstacle-aware baseline | 93.10% | 0.93 | 5.83 |
| PPO deterministic | 92.80% | 0.93 | 6.08 |
| PPO stochastic | 100.00% | 1.00 | 3.32 |

Main finding:

The centralized PPO controller successfully learned the fixed-obstacle multi-agent foraging task. Deterministic PPO performed close to the greedy obstacle-aware baseline, while stochastic PPO achieved the best result with 100% success and shorter average episode length.

---

# Next Planned Experiments

## Experiment 22 - Multi-Agent Random-Obstacle Environment

Status: next

### Goal

Extend the multi-agent environment to support randomized obstacle layouts.

### Planned Features

* `random_obstacles=True`;
* `num_obstacles=3`;
* obstacle generation at reset;
* no overlap between obstacles, agents, and food;
* BFS reachability validation so that at least one agent can reach the food;
* tests for randomized obstacle behavior.

### Planned Files

| Purpose | File |
|---|---|
| Environment update | `env/multi_agent_foraging_env.py` |
| Tests | `tests/test_multi_agent_random_obstacles.py` |

### Scientific Motivation

Fixed obstacles test learning in a controlled layout. Random obstacles test whether the multi-agent system can handle changing spatial constraints.

---

## Experiment 23 - Multi-Agent Random-Obstacle PPO, 5x5

Status: planned

### Goal

Train and evaluate a centralized PPO policy with 2 agents in a 5x5 environment with randomized reachable obstacles.

### Planned Outputs

| Output | Path |
|---|---|
| Training script | `train/train_ppo_multi_agent_2agents_random_obstacles.py` |
| Evaluation script | `evaluation/evaluate_multi_agent_random_obstacle_ppo.py` |
| Model | `models/ppo_multi_agent_2agents_random_obstacles.zip` |
| Results summary | `results/multi_agent_random_obstacle_ppo_summary.txt` |

### Planned Evaluation

Evaluate:

* random baseline;
* greedy obstacle-aware baseline;
* PPO deterministic;
* PPO stochastic.

---

## Experiment 24 - Multi-Agent Random-Obstacle Grid-Size Generalization

Status: planned

### Goal

Evaluate whether the PPO policy trained on 5x5 randomized obstacles generalizes to a larger 10x10 randomized-obstacle environment with 2 agents.

### Planned Output

`results/multi_agent_random_obstacle_grid_generalization_summary.txt`

### Scientific Motivation

This experiment tests whether the learned multi-agent policy transfers to larger spatial environments while still handling variable obstacle layouts.

---

## Optional Experiment - 3-Agent Scalability

Status: optional

### Goal

Test scalability from 2 agents to 3 agents after the 2-agent random-obstacle experiments are stable and documented.

Important difference:

| Agents | Joint Action Space |
|---:|---:|
| 2 | `4^2 = 16` |
| 3 | `4^3 = 64` |

The 3-agent setting is optional because the larger centralized joint action space may require additional training and interpretation.

---

## Final Documentation Phase

Status: planned

### Goal

Prepare the final project submission.

Planned work:

* update final plots;
* update `README.md`;
* update `PROJECT_CONTEXT.md`;
* update `ROADMAP.md`;
* clean `EXPERIMENT_LOG.md`;
* prepare final report tables;
* write final scientific interpretation.

The final report should emphasize:

* reproducible experimental setup;
* clear comparison against baselines;
* sensitivity analysis;
* analysis of learned behavior;
* deterministic vs stochastic PPO deployment;
* multi-agent scalability and limitations.

---

# Current Priority

The next concrete development step is:

```text
Experiment 22 - Multi-Agent Random-Obstacle Environment
```

Do not start 3-agent experiments yet. Do not train random-obstacle multi-agent PPO until the randomized obstacle environment and its tests are completed.
