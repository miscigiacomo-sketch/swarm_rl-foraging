# ROADMAP - SWARM RL FORAGING

This roadmap defines the development path for the `swarm_rl-foraging` project.

The project is structured as a progressive reinforcement-learning study. The environment complexity is increased step by step, starting from a single-agent grid-world and progressing toward swarm-inspired multi-agent foraging with obstacles, randomized environments, and scalability tests.

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
* multi-agent scalability;
* swarm-inspired collective foraging.

The project should be framed as:

```text
Swarm-inspired multi-agent foraging using reinforcement learning.
```

The learned controller is a centralized joint-action PPO policy. Therefore, the project should not be described as a fully decentralized swarm-intelligence system. Instead, it studies a simplified swarm-inspired foraging task with multiple agents, shared rewards, obstacle constraints, and scalability from single-agent to multi-agent behavior.

The project also does not invent PPO. It implements, adapts, and experimentally analyzes PPO in a custom bio-inspired foraging environment.

---

# Completed Work

## Single-Agent Experiments

### Experiment 1 - Random Agent Baseline

Status: completed

| Agent | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random Agent | 63.00% | 0.63 | 30.55 |

---

### Experiment 2 - PPO Baseline

Status: completed

| Agent | Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|---:|
| PPO | 10,000 | 100.00% | 1.00 | 4.26 |

Main finding:

PPO strongly outperformed the random baseline in the basic 5x5 foraging task.

---

### Experiment 3 - Grid-Size Generalization

Status: completed

| Training Grid | Evaluation Grid | Success Rate | Average Reward | Average Episode Length |
|---|---|---:|---:|---:|
| 5x5 | 10x10 | 96.00% | 0.96 | 12.88 |

Main finding:

The PPO policy trained on 5x5 generalized reasonably well to a larger no-obstacle 10x10 grid, although average episode length increased.

---

### Experiment 4 - Fixed Obstacles, PPO 20k

Status: completed

| Environment | Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|---:|
| 5x5 fixed obstacles | 20,000 | 79.00% | 0.79 | 13.34 |

---

### Experiment 5 - Fixed Obstacles, PPO 50k

Status: completed

| Environment | Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|---:|
| 5x5 fixed obstacles | 50,000 | 70.00% | 0.70 | 17.35 |

Main finding:

Increasing training time alone did not improve fixed-obstacle performance.

---

### Experiment 6 - Fixed Obstacles with State Augmentation

Status: completed

| Environment | Observation | Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---|---|---:|---:|---:|---:|
| 5x5 fixed obstacles | agent, food, obstacles | 50,000 | 87.00% | 0.87 | 9.94 |

Main finding:

Adding obstacle coordinates to the observation improved fixed-obstacle PPO performance from 70% to 87%.

---

### Experiment 7 - Reachability Validation for Randomized Obstacles

Status: completed

| Episodes Tested | Reachable Environments | Unreachable Environments | Reachability Rate |
|---:|---:|---:|---:|
| 1000 | 1000 | 0 | 100.00% |

Main finding:

Randomized obstacle environments were guaranteed to be solvable using BFS reachability validation.

---

### Experiment 8 - Fixed-Obstacle PPO Evaluated on Randomized Obstacles

Status: completed

| Training Environment | Evaluation Environment | Success Rate | Average Reward | Average Episode Length |
|---|---|---:|---:|---:|
| Fixed obstacles | Randomized obstacles | 65.50% | 0.66 | 19.59 |

Main finding:

The fixed-obstacle PPO policy did not fully generalize to randomized obstacle layouts.

---

### Experiment 9 - Randomized Obstacles PPO 50k

Status: completed

| Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---:|---:|---:|---:|
| 50,000 | 70.90% | 0.71 | 17.14 |

---

### Experiment 10 - Randomized Obstacles PPO 100k

Status: completed

| Training Timesteps | Success Rate | Average Reward | Average Episode Length |
|---:|---:|---:|---:|
| 100,000 | 76.30% | 0.76 | 14.74 |

Main finding:

Increasing randomized-obstacle training from 50k to 100k improved robustness and efficiency.

---

### Experiment 11 - Randomized Obstacles PPO 200k

Status: completed

| Training Timesteps | Deterministic Success Rate | Average Reward | Average Episode Length |
|---:|---:|---:|---:|
| 200,000 | 76.20% | 0.76 | 14.82 |

Main finding:

The deterministic success rate plateaued around 76%.

---

### Experiment 12 - BFS Oracle Baseline on Randomized Obstacles

Status: completed

| Method | Success Rate | Average Shortest Path Length | Failed Seeds |
|---|---:|---:|---|
| BFS oracle | 100.00% | 4.29 | `[]` |

Main finding:

The task was solvable; PPO failures were not caused by impossible environments.

---

### Experiment 13 - PPO Failure-Case Analysis on Randomized Obstacles

Status: completed

| Model | Success Rate | Failure Rate | Failures | Avg No-Move Steps per Failed Episode |
|---|---:|---:|---:|---:|
| PPO random obstacles 100k | 76.30% | 23.70% | 237 | 48.45 |

Main finding:

Failed deterministic PPO episodes were dominated by repeated no-move behavior.

---

### Experiment 14 - PPO Policy Sampling Mode Analysis

Status: completed

| Model | Deterministic Success | Stochastic Mean Success | Stochastic Std | Mean Episode Length |
|---|---:|---:|---:|---:|
| PPO Random Obstacles 100k | 76.30% | 96.92% | 0.52% | 7.65 |
| PPO Random Obstacles 200k | 76.20% | 97.58% | 0.29% | 7.17 |

Main finding:

Stochastic policy sampling greatly improved performance and reduced repeated no-move failures.

---

### Experiment 15 - Random-Obstacle Grid-Size Generalization

Status: completed

| Grid | Oracle Success | PPO Deterministic Success | PPO Stochastic Mean Success | PPO Stochastic Average Episode Length |
|---|---:|---:|---:|---:|
| 5x5 | 100.00% | 76.20% | 97.58% | 7.17 |
| 10x10 | 100.00% | 66.90% | 98.08% | 12.60 |

Main finding:

The stochastic PPO policy trained on 5x5 randomized obstacles generalized successfully to a larger 10x10 randomized-obstacle environment.

---

## Multi-Agent Experiments

### Experiment 16 - Multi-Agent No-Obstacle Baselines

Status: completed

| Agents | Method | Success Rate | Average Reward | Average Episode Length |
|---:|---|---:|---:|---:|
| 2 | Random baseline | 84.30% | 0.84 | 21.30 |
| 2 | Greedy decentralized baseline | 100.00% | 1.00 | 2.41 |
| 3 | Random baseline | 92.10% | 0.92 | 16.46 |
| 3 | Greedy decentralized baseline | 99.00% | 0.99 | 2.52 |

Main finding:

Adding more agents improved random exploration, while the greedy baseline was very strong without obstacles.

---

### Experiment 17 - Multi-Agent PPO, 2 Agents

Status: completed

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 84.30% | 0.84 | 21.30 |
| Greedy decentralized baseline | 100.00% | 1.00 | 2.41 |
| PPO deterministic | 98.90% | 0.99 | 2.94 |
| PPO stochastic | 100.00% | 1.00 | 2.58 |

Main finding:

The centralized PPO controller successfully learned the two-agent no-obstacle foraging task.

---

### Experiment 18 - Multi-Agent Grid-Size Generalization

Status: completed

| Grid | Random Success | Greedy Success | PPO Deterministic Success | PPO Stochastic Success | PPO Stochastic Average Episode Length |
|---|---:|---:|---:|---:|---:|
| 5x5 | 84.30% | 100.00% | 98.90% | 100.00% | 2.57 |
| 10x10 | 37.90% | 100.00% | 96.00% | 99.88% | 5.64 |

Main finding:

The two-agent PPO policy trained on 5x5 generalized very well to a larger 10x10 no-obstacle environment.

---

### Experiment 19 - Multi-Agent Fixed-Obstacle Environment

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

Main finding:

Fixed-obstacle support was added without breaking the previous no-obstacle multi-agent workflow.

---

### Experiment 20 - Multi-Agent Fixed-Obstacle Baselines

Status: completed

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 76.40% | 0.76 | 24.63 |
| Greedy obstacle-aware baseline | 93.10% | 0.93 | 5.83 |

Main finding:

The greedy obstacle-aware baseline was strong but not optimal because it uses local Manhattan-distance decisions rather than full path planning.

---

### Experiment 21 - Multi-Agent Fixed-Obstacle PPO

Status: completed

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 76.40% | 0.76 | 24.63 |
| Greedy obstacle-aware baseline | 93.10% | 0.93 | 5.83 |
| PPO deterministic | 92.80% | 0.93 | 6.08 |
| PPO stochastic | 100.00% | 1.00 | 3.32 |

Main finding:

The centralized PPO controller learned the fixed-obstacle multi-agent foraging task. Stochastic PPO achieved the best result with 100% success and the shortest average episode length.

---

### Experiment 22 - Multi-Agent Random-Obstacle Environment

Status: completed

Extended the multi-agent environment to support randomized reachable obstacle layouts.

Implemented:

* `random_obstacles=True`;
* `num_obstacles=3`;
* obstacle generation at reset;
* no overlap between obstacles, agents, and food;
* BFS reachability validation so that at least one agent can reach the food;
* tests for randomized obstacle behavior;
* terminal visualization for random-obstacle layouts.

Planned and completed files:

| Purpose | File |
|---|---|
| Environment update | `env/multi_agent_foraging_env.py` |
| Tests | `tests/test_multi_agent_random_obstacles.py` |
| Visualization | `visualization/visualize_multi_agent_random_obstacles.py` |

Validation:

| Test | Result |
|---|---|
| Random-obstacle tests | 8 passed |
| Fixed + random obstacle tests | 14 passed |
| No-obstacle multi-agent test | passed |

Main finding:

The environment can generate randomized obstacle layouts while preserving reachability and backward compatibility with previous fixed-obstacle and no-obstacle setups.

---

### Experiment 23 - Multi-Agent Random-Obstacle PPO, 5x5

Status: completed

Trained and evaluated centralized PPO with 2 agents in a 5x5 environment with randomized reachable obstacles.

Setup:

| Field | Value |
|---|---|
| Environment | `MultiAgentForagingEnv` |
| Grid Size | 5x5 |
| Number of Agents | 2 |
| Random Obstacles | True |
| Number of Obstacles | 3 |
| Algorithm | PPO |
| Control Type | Centralized joint-action control |
| Joint Action Space | `4^2 = 16` |
| Observation Size | 12 |
| Training Timesteps | 100,000 |
| Training Script | `train/train_ppo_multi_agent_2agents_random_obstacles.py` |
| Evaluation Script | `evaluation/evaluate_multi_agent_random_obstacle_ppo.py` |
| Results File | `results/multi_agent_random_obstacle_ppo_summary.txt` |

Key result:

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 81.50% | 0.81 | 22.24 |
| Greedy obstacle-aware baseline | 93.50% | 0.94 | 5.45 |
| PPO deterministic | 82.40% | 0.82 | 10.77 |
| PPO stochastic | 99.40% | 0.99 | 3.91 |

Main finding:

PPO stochastic strongly outperformed all other methods. PPO deterministic was much weaker, showing that deterministic argmax deployment can underuse the learned PPO policy distribution.

---

### Experiment 24 - Multi-Agent Random-Obstacle Grid-Size Generalization

Status: completed

Evaluated whether the 2-agent PPO policy trained on 5x5 randomized obstacles generalizes to a larger 10x10 randomized-obstacle environment.

Setup:

| Field | Value |
|---|---|
| Training Grid Size | 5x5 |
| Evaluation Grid Sizes | 5x5 and 10x10 |
| Number of Agents | 2 |
| Random Obstacles | True |
| Number of Obstacles | 3 |
| Control Type | Centralized joint-action control |
| Evaluation Script | `evaluation/evaluate_multi_agent_random_obstacle_grid_generalization.py` |
| Results File | `results/multi_agent_random_obstacle_grid_generalization_summary.txt` |

Key result:

| Grid | Random Success | Greedy Success | PPO Deterministic Success | PPO Stochastic Success | PPO Stochastic Average Episode Length |
|---|---:|---:|---:|---:|---:|
| 5x5 | 81.50% | 93.50% | 82.40% | 99.40% | 3.91 |
| 10x10 | 37.60% | 95.10% | 86.30% | 99.98% | 7.15 |

Main finding:

The 10x10 grid increased the spatial search space while keeping the same number of agents and obstacles. PPO stochastic generalized extremely well, reaching 99.98% success on 10x10.

---

### Experiment 25 - 3-Agent Random-Obstacle Scalability on 10x10

Status: completed

Trained and evaluated centralized PPO with 3 agents in a 10x10 randomized-obstacle environment.

Setup:

| Field | Value |
|---|---|
| Environment | `MultiAgentForagingEnv` |
| Grid Size | 10x10 |
| Number of Agents | 3 |
| Random Obstacles | True |
| Number of Obstacles | 3 |
| Algorithm | PPO |
| Control Type | Centralized joint-action control |
| Joint Action Space | `4^3 = 64` |
| Observation Size | 14 |
| Training Timesteps | 200,000 |
| Training Script | `train/train_ppo_multi_agent_3agents_random_obstacles_10x10.py` |
| Evaluation Script | `evaluation/evaluate_multi_agent_3agents_random_obstacles_10x10.py` |
| Results File | `results/multi_agent_3agents_random_obstacle_10x10_summary.txt` |

Key result:

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 50.00% | 0.50 | 35.12 |
| Greedy obstacle-aware baseline | 96.20% | 0.96 | 5.66 |
| PPO deterministic | 65.00% | 0.65 | 21.72 |
| PPO stochastic | 99.82% | 1.00 | 9.92 |

Main finding:

Increasing from 2 to 3 agents increased the centralized joint action space from `4^2 = 16` to `4^3 = 64`. Deterministic PPO became much less robust, while stochastic PPO still achieved 99.82% mean success.

This strengthens the swarm-inspired scalability story while also showing the limitation of deterministic argmax deployment in larger joint action spaces.

---

# Current Status

The main numerical experiments are complete.

The project has progressed through:

* single-agent foraging;
* single-agent fixed and randomized obstacle handling;
* BFS oracle validation;
* deterministic vs stochastic PPO analysis;
* grid-size generalization;
* multi-agent no-obstacle control;
* multi-agent fixed-obstacle control;
* multi-agent randomized-obstacle control;
* 2-agent to 3-agent scalability.

The strongest current learned results are obtained with PPO stochastic deployment.

---

# Next Planned Work

## Experiment 26 - Final Comparative Analysis

Status: next

### Goal

Create final comparative tables and plots across the main scenarios.

### Planned Comparisons

* random baseline;
* greedy obstacle-aware baseline;
* PPO deterministic;
* PPO stochastic;
* fixed obstacles vs randomized obstacles;
* 5x5 vs 10x10;
* 2 agents vs 3 agents.

### Planned Outputs

| Output | Suggested Path |
|---|---|
| Final comparative CSV | `results/final_analysis/final_comparative_results.csv` |
| Final comparative summary | `results/final_analysis/final_comparative_summary.txt` |
| Success-rate plot | `plots/final_success_rate_comparison.png` |
| Episode-length plot | `plots/final_episode_length_comparison.png` |
| Deterministic vs stochastic plot | `plots/ppo_deterministic_vs_stochastic_success.png` |

### Scientific Motivation

This analysis will convert the experiment sequence into a compact final result set for the README and final report.

---

## Experiment 27 - Learned-Behavior and Failure Analysis

Status: planned

### Goal

Explain the observed performance differences between deterministic and stochastic PPO deployment.

### Key Questions

* Why does deterministic PPO perform poorly in some randomized-obstacle settings?
* How does stochastic policy sampling help avoid repeated suboptimal joint-action patterns?
* Why is PPO stochastic not equivalent to random exploration?
* How does the joint action space grow from 16 actions for 2 agents to 64 actions for 3 agents?
* What limitations remain because the controller is centralized rather than decentralized?

### Planned Output

A written analysis section for the README and report, possibly supported by local qualitative visualizations.

---

## Final Documentation Phase

Status: planned

### Goal

Prepare the final project submission.

Planned work:

1. update final plots;
2. update `README.md`;
3. update `PROJECT_CONTEXT.md`;
4. update `ROADMAP.md`;
5. clean and finalize `EXPERIMENT_LOG.md`;
6. prepare final report tables;
7. write final scientific interpretation;
8. write final report.

The final report should emphasize:

* reproducible experimental setup;
* clear comparison against baselines;
* sensitivity analysis;
* analysis of learned behavior;
* deterministic vs stochastic PPO deployment;
* multi-agent scalability and limitations;
* swarm-inspired framing without claiming full decentralized swarm intelligence.

---

# Priority List for a High-Quality Final Project

## Essential

* Complete final comparative tables and plots.
* Write clear learned-behavior analysis.
* Update README with project goal, setup, commands, and main results.
* Prepare final report with a clear scientific story.

## Completed Quality Improvements

* Completed randomized-obstacle training sensitivity analysis.
* Added BFS oracle baseline for environment solvability validation.
* Added PPO failure-case analysis for learned-behavior interpretation.
* Added policy sampling robustness analysis showing near-oracle stochastic PPO performance.
* Completed random-obstacle 10x10 generalization.
* Completed multi-agent fixed-obstacle PPO.
* Completed multi-agent randomized-obstacle PPO.
* Completed 3-agent randomized-obstacle scalability.

## Optional Extensions

These are not recommended unless there is extra time:

* decentralized/local policy baseline;
* obstacle-density robustness;
* larger-grid training from scratch;
* reward shaping;
* invalid-action penalty.

At this stage, additional experiments are less important than documentation, analysis, plots, README quality, and final report clarity.

---

# Current Best Models

Best single-agent fixed-obstacle model:

`models/ppo_foraging_obstacles_state.zip`

Best single-agent randomized-obstacle model:

`models/ppo_foraging_random_obstacles_200k.zip`

Best 2-agent randomized-obstacle model:

`models/ppo_multi_agent_2agents_random_obstacles.zip`

Best 3-agent randomized-obstacle model:

`models/ppo_multi_agent_3agents_random_obstacles_10x10.zip`

Recommended learned-policy evaluation mode:

```text
deterministic=False
```

Important distinction:

* Random baseline uses uniform random actions and is not learned.
* PPO stochastic samples from the trained PPO policy distribution and is still reinforcement learning.
* PPO deterministic selects the highest-probability action from the trained PPO policy distribution.

---

# Current Main Conclusion

The current results show that state representation, environment randomization, policy deployment mode, grid-size scaling, and number of agents all affect PPO performance.

Adding obstacle coordinates to the observation improved single-agent fixed-obstacle performance from 70% to 87%.

Randomized obstacle training improved robustness compared to fixed-obstacle training, but deterministic PPO evaluation revealed repeated no-move failure modes. BFS oracle validation confirmed that the environments were solvable, so PPO failures were caused by policy limitations rather than impossible tasks.

Stochastic PPO deployment consistently improved performance by sampling from the learned policy distribution. This is not random exploration; it uses the trained PPO probabilities. The stochastic policy reached near-oracle performance in single-agent randomized-obstacle settings and continued to perform strongly in multi-agent environments.

In the 2-agent randomized-obstacle setting, stochastic PPO achieved 99.40% success on 5x5 and 99.98% success on 10x10.

In the 3-agent randomized-obstacle 10x10 setting, increasing the joint action space from 16 to 64 made deterministic PPO much less robust, with only 65.00% success. However, stochastic PPO still achieved 99.82% success.

The project therefore supports the conclusion that PPO can learn effective swarm-inspired multi-agent foraging behavior in the custom environment, but the learned policy is most robust when deployed stochastically and when the limitations of centralized joint-action control are clearly acknowledged.
