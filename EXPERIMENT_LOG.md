# EXPERIMENT LOG - SWARM RL FORAGING

This file summarizes the main experiments performed in the `swarm_rl-foraging` project.

The goal is to keep a clear record of the training setup, evaluation setup, metrics, and scientific conclusions for each experiment.

---

## Metrics

The main metrics used throughout the project are:

* **Success Rate**: percentage of episodes in which at least one agent reaches the food.
* **Average Reward**: average reward obtained across evaluation episodes.
* **Average Episode Length**: average number of steps per episode.

Unless otherwise specified, each experiment reports its own evaluation episode count. Early baseline evaluations used 100 episodes, while later robustness and multi-agent evaluations generally used 1000 seeded episodes for fair comparison.

For randomized-obstacle and stochastic-policy evaluations, models were evaluated on seeded environments to ensure fair comparisons.

---

# Completed Experiments

## Experiment 1 - Random Agent Baseline

| Field                  | Value         |
| ---------------------- | ------------- |
| Environment            | 5x5 grid      |
| Obstacles              | None          |
| Agent                  | Random policy |
| Episodes               | 100           |
| Success Rate           | 63%           |
| Average Reward         | 0.63          |
| Average Episode Length | 30.55         |

### Conclusion

The random agent provides a simple baseline for comparison. It succeeds in some episodes by chance, but it is inefficient and requires many steps on average.

---

## Experiment 2 - PPO Baseline

| Field                  | Value                     |
| ---------------------- | ------------------------- |
| Environment            | 5x5 grid                  |
| Obstacles              | None                      |
| Algorithm              | PPO                       |
| Training Timesteps     | 10,000                    |
| Model                  | `models/ppo_foraging.zip` |
| Success Rate           | 100%                      |
| Average Reward         | 1.00                      |
| Average Episode Length | 4.26                      |

### Conclusion

PPO significantly outperformed the random baseline in the basic foraging environment.

---

## Experiment 3 - Grid-Size Generalization

| Field                  | Value                     |
| ---------------------- | ------------------------- |
| Training Environment   | 5x5 grid                  |
| Evaluation Environment | 10x10 grid                |
| Obstacles              | None                      |
| Model                  | `models/ppo_foraging.zip` |
| Success Rate           | 96%                       |
| Average Reward         | 0.96                      |
| Average Episode Length | 12.88                     |

### Conclusion

The PPO policy trained on 5x5 generalized reasonably well to a larger 10x10 grid. However, the average episode length increased, showing reduced efficiency in the larger environment.

---

## Experiment 4 - Fixed Obstacles, PPO 20k

| Field                  | Value                               |
| ---------------------- | ----------------------------------- |
| Environment            | 5x5 grid                            |
| Obstacles              | Fixed obstacles                     |
| Obstacle Positions     | `(2, 1)`, `(2, 2)`, `(2, 3)`        |
| Algorithm              | PPO                                 |
| Training Timesteps     | 20,000                              |
| Model                  | `models/ppo_foraging_obstacles.zip` |
| Success Rate           | 79%                                 |
| Average Reward         | 0.79                                |
| Average Episode Length | 13.34                               |

### Conclusion

Adding obstacles increased the difficulty of the task. The PPO agent still learned useful behavior, but performance decreased compared to the no-obstacle baseline.

---

## Experiment 5 - Fixed Obstacles, PPO 50k

| Field                  | Value                                   |
| ---------------------- | --------------------------------------- |
| Environment            | 5x5 grid                                |
| Obstacles              | Fixed obstacles                         |
| Obstacle Positions     | `(2, 1)`, `(2, 2)`, `(2, 3)`            |
| Algorithm              | PPO                                     |
| Training Timesteps     | 50,000                                  |
| Model                  | `models/ppo_foraging_obstacles_50k.zip` |
| Success Rate           | 70%                                     |
| Average Reward         | 0.70                                    |
| Average Episode Length | 17.35                                   |

### Conclusion

Increasing the number of training timesteps alone did not improve fixed-obstacle performance. In this experiment, performance decreased compared to the 20k model.

---

## Experiment 6 - Fixed Obstacles with State Augmentation

| Field                  | Value                                             |
| ---------------------- | ------------------------------------------------- |
| Environment            | 5x5 grid                                          |
| Obstacles              | Fixed obstacles                                   |
| Observation Space      | Agent position, food position, obstacle positions |
| Algorithm              | PPO                                               |
| Training Timesteps     | 50,000                                            |
| Model                  | `models/ppo_foraging_obstacles_state.zip`         |
| Success Rate           | 87%                                               |
| Average Reward         | 0.87                                              |
| Average Episode Length | 9.94                                              |

### Conclusion

Adding obstacle coordinates to the observation space improved performance from 70% to 87%.

This suggests that state representation was more important than simply increasing training duration in the fixed-obstacle environment.

---

## Experiment 7 - Reachability Validation for Randomized Obstacles

| Field                    | Value                |
| ------------------------ | -------------------- |
| Environment              | 5x5 grid             |
| Obstacles                | Randomized obstacles |
| Number of Obstacles      | 3                    |
| Episodes Tested          | 1000                 |
| Reachable Environments   | 1000                 |
| Unreachable Environments | 0                    |
| Reachability Rate        | 100%                 |

### Conclusion

Randomized obstacle environments were validated using breadth-first search.

The generator ensures that the food is always reachable from the agent. This prevents impossible episodes and makes randomized-obstacle evaluations scientifically cleaner.

---

## Experiment 8 - Fixed-Obstacle PPO Evaluated on Randomized Obstacles

| Field                  | Value                                     |
| ---------------------- | ----------------------------------------- |
| Training Environment   | Fixed obstacles                           |
| Evaluation Environment | Randomized reachable obstacles            |
| Model                  | `models/ppo_foraging_obstacles_state.zip` |
| Evaluation Episodes    | 1000 seeded episodes                      |
| Success Rate           | 65.50%                                    |
| Average Reward         | 0.66                                      |
| Average Episode Length | 19.59                                     |

### Conclusion

The PPO policy trained only on fixed obstacles did not fully generalize to randomized obstacle layouts.

This suggests that the policy learned behavior adapted to the fixed obstacle configuration, but was less robust when obstacle positions changed.

---

## Experiment 9 - Randomized Obstacles PPO 50k

| Field                  | Value                                      |
| ---------------------- | ------------------------------------------ |
| Training Environment   | Randomized reachable obstacles             |
| Evaluation Environment | Randomized reachable obstacles             |
| Algorithm              | PPO                                        |
| Training Timesteps     | 50,000                                     |
| Model                  | `models/ppo_foraging_random_obstacles.zip` |
| Evaluation Episodes    | 1000 seeded episodes                       |
| Success Rate           | 70.90%                                     |
| Average Reward         | 0.71                                       |
| Average Episode Length | 17.14                                      |

### Conclusion

Training directly on randomized obstacle layouts improved generalization compared to the fixed-obstacle PPO model.

---

## Experiment 10 - Randomized Obstacles PPO 100k

| Field                  | Value                                           |
| ---------------------- | ----------------------------------------------- |
| Training Environment   | Randomized reachable obstacles                  |
| Evaluation Environment | Randomized reachable obstacles                  |
| Algorithm              | PPO                                             |
| Training Timesteps     | 100,000                                         |
| Model                  | `models/ppo_foraging_random_obstacles_100k.zip` |
| Evaluation Episodes    | 1000 seeded episodes                            |
| Success Rate           | 76.30%                                          |
| Average Reward         | 0.76                                            |
| Average Episode Length | 14.74                                           |

### Conclusion

Increasing randomized-obstacle training from 50k to 100k timesteps improved both success rate and efficiency.

---

## Experiment 11 - Randomized Obstacles PPO 200k

| Field                  | Value                                           |
| ---------------------- | ----------------------------------------------- |
| Training Environment   | Randomized reachable obstacles                  |
| Evaluation Environment | Randomized reachable obstacles                  |
| Algorithm              | PPO                                             |
| Training Timesteps     | 200,000                                         |
| Model                  | `models/ppo_foraging_random_obstacles_200k.zip` |
| Evaluation Episodes    | 1000 seeded episodes                            |
| Success Rate           | 76.20%                                          |
| Average Reward         | 0.76                                            |
| Average Episode Length | 14.82                                           |

### Conclusion

Increasing randomized-obstacle training from 100k to 200k timesteps did not produce a meaningful deterministic improvement.

The deterministic success rate remained close to 76%, suggesting a performance plateau under deterministic action selection.

---

## Experiment 12 - BFS Oracle Baseline on Randomized Obstacles

| Field                        | Value                                            |
| ---------------------------- | ------------------------------------------------ |
| Environment                  | 5x5 randomized reachable obstacles               |
| Agent Type                   | BFS oracle / shortest-path planner               |
| Learning Algorithm           | None                                             |
| Evaluation Episodes          | 1000 seeded episodes                             |
| Success Rate                 | 100.00%                                          |
| Average Reward               | 1.00                                             |
| Average Shortest Path Length | 4.29                                             |
| Average Episode Length       | 4.29                                             |
| Failed Seeds                 | `[]`                                             |
| Evaluation Script            | `evaluation/evaluate_oracle_random_obstacles.py` |

### Conclusion

The BFS oracle solved 100% of the randomized-obstacle environments.

This confirms that the randomized environments used for evaluation were solvable and that PPO failures were not caused by impossible environments.

---

## Experiment 13 - PPO Failure-Case Analysis on Randomized Obstacles

| Field                                              | Value                                            |
| -------------------------------------------------- | ------------------------------------------------ |
| Environment                                        | 5x5 randomized reachable obstacles               |
| Model                                              | `models/ppo_foraging_random_obstacles_100k.zip`  |
| Evaluation Episodes                                | 1000 seeded episodes                             |
| Success Rate                                       | 76.30%                                           |
| Failure Rate                                       | 23.70%                                           |
| Failures                                           | 237                                              |
| Saved Failure Cases                                | 10                                               |
| Average No-Move Steps per Failed Episode           | 48.45                                            |
| Average Repeated-Position Steps per Failed Episode | 48.45                                            |
| Output File                                        | `results/random_obstacle_failure_cases.txt`      |
| Analysis Script                                    | `evaluation/analyze_random_obstacle_failures.py` |

### Conclusion

Failure-case analysis showed that failed deterministic PPO episodes were dominated by repeated no-move behavior.

This suggests that the deterministic policy often selected invalid or ineffective actions repeatedly, especially when blocked by walls or obstacles.

---

## Experiment 14 - PPO Policy Sampling Mode Analysis

| Field               | Value                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| Environment         | 5x5 randomized reachable obstacles                                                               |
| Evaluation Episodes | 1000 seeded episodes per action seed                                                             |
| Action Seeds        | `0, 1, 2, 3, 4, 42, 100, 123, 999, 2024`                                                         |
| Models              | `models/ppo_foraging_random_obstacles_100k.zip`, `models/ppo_foraging_random_obstacles_200k.zip` |
| Analysis Script     | `evaluation/evaluate_stochastic_policy_robustness.py`                                            |
| Output File         | `results/policy_sampling_robustness_summary.txt`                                                 |

### Results

| Model                     | Deterministic Success | Stochastic Mean Success | Stochastic Std | Mean Average Reward | Mean Average Episode Length | Mean No-Move Steps |
| ------------------------- | --------------------: | ----------------------: | -------------: | ------------------: | --------------------------: | -----------------: |
| PPO Random Obstacles 100k |                76.30% |                  96.92% |          0.52% |                0.97 |                        7.65 |               2.80 |
| PPO Random Obstacles 200k |                76.20% |                  97.58% |          0.29% |                0.98 |                        7.17 |               2.51 |

### Conclusion

Stochastic action sampling greatly improved performance.

The 200k randomized-obstacle PPO model achieved the best stochastic performance, reaching 97.58% success.

This suggests that deterministic evaluation can underestimate PPO performance when the learned policy contains useful alternative actions that are ignored by argmax action selection.

---

## Experiment 15 - Random-Obstacle Grid-Size Generalization

| Field                   | Value                                                        |
| ----------------------- | ------------------------------------------------------------ |
| Training Environment    | 5x5 randomized reachable obstacles                           |
| Evaluation Environments | 5x5 and 10x10 randomized reachable obstacles                 |
| Model                   | `models/ppo_foraging_random_obstacles_200k.zip`              |
| Evaluation Episodes     | 1000 seeded episodes                                         |
| Stochastic Action Seeds | `0, 1, 2, 3, 4, 42, 100, 123, 999, 2024`                     |
| Evaluation Script       | `evaluation/evaluate_random_obstacle_grid_generalization.py` |
| Output File             | `results/random_obstacle_grid_generalization_summary.txt`    |

### Results

| Grid  | Oracle Success | PPO Deterministic Success | PPO Stochastic Mean Success | Stochastic Std | PPO Stochastic Average Episode Length |
| ----- | -------------: | ------------------------: | --------------------------: | -------------: | ------------------------------------: |
| 5x5   |        100.00% |                    76.20% |                      97.58% |          0.29% |                                  7.17 |
| 10x10 |        100.00% |                    66.90% |                      98.08% |          0.19% |                                 12.60 |

### Conclusion

The stochastic PPO policy trained on 5x5 randomized obstacles generalized successfully to a larger 10x10 randomized-obstacle environment.

The average episode length increased on the larger grid, as expected, but the success rate remained near-oracle.

---

## Experiment 16 - Multi-Agent No-Obstacle Baselines

| Field             | Value                                          |
| ----------------- | ---------------------------------------------- |
| Environment       | Multi-agent 5x5 grid                           |
| Obstacles         | None                                           |
| Food Sources      | 1                                              |
| Control Type      | Centralized joint action space                 |
| Evaluation Script | `evaluation/evaluate_multi_agent_baselines.py` |
| Output File       | `results/multi_agent_baseline_summary.txt`     |

### Results

| Agents | Method                        | Success Rate | Average Episode Length |
| -----: | ----------------------------- | -----------: | ---------------------: |
|      2 | Random baseline               |       84.30% |                  21.30 |
|      2 | Greedy decentralized baseline |      100.00% |                   2.41 |
|      3 | Random baseline               |       92.10% |                  16.46 |
|      3 | Greedy decentralized baseline |       99.00% |                   2.52 |

### Conclusion

Adding multiple agents increased the probability of finding the food under random exploration.

The greedy decentralized baseline was very strong in the no-obstacle setting because each agent could move directly toward the food without needing obstacle-aware path planning.

---

## Experiment 17 - Multi-Agent PPO, 2 Agents

| Field              | Value                                    |
| ------------------ | ---------------------------------------- |
| Environment        | Multi-agent 5x5 grid                     |
| Obstacles          | None                                     |
| Number of Agents   | 2                                        |
| Control Type       | Centralized PPO                          |
| Joint Action Space | `4^2 = 16`                               |
| Training Timesteps | 100,000                                  |
| Model              | `models/ppo_multi_agent_2agents.zip`     |
| Training Script    | `train/train_ppo_multi_agent_2agents.py` |
| Evaluation Script  | `evaluation/evaluate_multi_agent_ppo.py` |
| Output File        | `results/multi_agent_ppo_summary.txt`    |

### Results

| Method                        | Success Rate | Average Episode Length |
| ----------------------------- | -----------: | ---------------------: |
| Random baseline               |       84.30% |                  21.30 |
| Greedy decentralized baseline |      100.00% |                   2.41 |
| PPO deterministic             |       98.90% |                   2.94 |
| PPO stochastic                |      100.00% |                   2.58 |

### Conclusion

The centralized PPO controller successfully learned the two-agent foraging task.

Stochastic PPO reached 100% success and performed close to the greedy decentralized baseline in the no-obstacle environment.

---

## Experiment 18 - Multi-Agent Grid-Size Generalization

| Field                   | Value                                                    |
| ----------------------- | -------------------------------------------------------- |
| Training Environment    | Multi-agent 5x5 grid                                     |
| Evaluation Environments | 5x5 and 10x10 grids                                      |
| Obstacles               | None                                                     |
| Number of Agents        | 2                                                        |
| Model                   | `models/ppo_multi_agent_2agents.zip`                     |
| Evaluation Script       | `evaluation/evaluate_multi_agent_grid_generalization.py` |
| Output File             | `results/multi_agent_grid_generalization_summary.txt`    |

### Results

| Grid  | Random Success | Greedy Success | PPO Deterministic Success | PPO Stochastic Success | PPO Stochastic Average Episode Length |
| ----- | -------------: | -------------: | ------------------------: | ---------------------: | ------------------------------------: |
| 5x5   |         84.30% |        100.00% |                    98.90% |                100.00% |                                  2.57 |
| 10x10 |         37.90% |        100.00% |                    96.00% |                 99.88% |                                  5.64 |

### Conclusion

The two-agent PPO policy trained on 5x5 generalized very well to a larger 10x10 no-obstacle environment.

Random exploration became much weaker on the larger grid, while stochastic PPO remained almost perfect.

---

## Experiment 19 - Multi-Agent Fixed-Obstacle Environment

| Field              | Value                                 |
| ------------------ | ------------------------------------- |
| Environment        | `MultiAgentForagingEnv`               |
| Grid Size          | 5x5                                   |
| Number of Agents   | 2                                     |
| Max Steps          | 50                                    |
| Fixed Obstacles    | `(2, 1)`, `(2, 2)`, `(2, 3)`          |
| Joint Action Space | `4^2 = 16`                            |
| Updated File       | `env/multi_agent_foraging_env.py`     |
| Test File          | `tests/test_multi_agent_obstacles.py` |

### Implementation Summary

The centralized multi-agent environment was extended with optional fixed-obstacle support.

If `obstacles=None`, the environment remains backward compatible with the previous no-obstacle multi-agent experiments.

If fixed obstacles are provided, the observation includes:

* agent positions;
* food position;
* obstacle positions.

For 2 agents and 3 obstacles, the observation size is:

| Component            | Values |
| -------------------- | -----: |
| 2 agents             |      4 |
| food position        |      2 |
| 3 obstacle positions |      6 |
| Total                |     12 |

### Environment Rules

* Agents cannot spawn on obstacle cells.
* Food cannot spawn on obstacle cells.
* Agents cannot move into obstacle cells.
* Existing collision prevention is preserved.
* Existing position-swap prevention is preserved.
* Render displays agents as `A`, `B`, `C`, food as `F`, and obstacles as `X`.

### Testing

The new obstacle test file verified:

* Stable-Baselines3 environment compatibility with `check_env`;
* observation shape is correct;
* obstacle coordinates are included in the observation;
* agents do not spawn on obstacles;
* food does not spawn on obstacles;
* movement into obstacle cells is blocked.

| Test                                  | Result   |
| ------------------------------------- | -------- |
| `tests/test_multi_agent_obstacles.py` | 6 passed |
| `tests/test_multi_agent_env.py`       | passed   |

### Conclusion

Fixed-obstacle support was successfully added to the centralized multi-agent environment while preserving backward compatibility with the previous no-obstacle multi-agent scripts.

This prepared the project for obstacle-aware multi-agent baseline evaluation and PPO training.

---

## Experiment 20 - Multi-Agent Fixed-Obstacle Baselines

| Field               | Value                                                   |
| ------------------- | ------------------------------------------------------- |
| Environment         | `MultiAgentForagingEnv`                                 |
| Grid Size           | 5x5                                                     |
| Number of Agents    | 2                                                       |
| Max Steps           | 50                                                      |
| Fixed Obstacles     | `(2, 1)`, `(2, 2)`, `(2, 3)`                            |
| Control Type        | Centralized joint action space                          |
| Evaluation Episodes | 1000                                                    |
| Evaluation Script   | `evaluation/evaluate_multi_agent_obstacle_baselines.py` |
| Output File         | `results/multi_agent_obstacle_baseline_summary.txt`     |

### Methods

Two non-learning baselines were evaluated.

The **random baseline** selects a random centralized joint action at each step.

The **greedy obstacle-aware baseline** is a simple non-learning policy. Each agent selects a movement action that reduces Manhattan distance to the food while avoiding:

* obstacle cells;
* already proposed cells by other agents;
* direct position swaps when possible;
* no-move boundary actions when better alternatives exist.

### Results

| Method                         | Success Rate | Average Reward | Average Episode Length |
| ------------------------------ | -----------: | -------------: | ---------------------: |
| Random baseline                |       76.40% |           0.76 |                  24.63 |
| Greedy obstacle-aware baseline |       93.10% |           0.93 |                   5.83 |

### Conclusion

The random baseline achieved 76.40% success, lower than the no-obstacle multi-agent random baseline. This confirms that the fixed obstacle barrier makes the task harder.

The greedy obstacle-aware baseline reached 93.10% success and was much more efficient, with an average episode length of 5.83 steps.

The greedy baseline is strong but not optimal because it makes local Manhattan-distance decisions rather than performing full path planning. It can fail when moving around the obstacle barrier requires temporarily moving away from the food or resolving agent interactions.

These results provide reference baselines for training PPO in the same fixed-obstacle multi-agent environment.

---

## Experiment 21 - Multi-Agent Fixed-Obstacle PPO

| Field               | Value                                                   |
| ------------------- | ------------------------------------------------------- |
| Environment         | `MultiAgentForagingEnv`                                 |
| Grid Size           | 5x5                                                     |
| Number of Agents    | 2                                                       |
| Max Steps           | 50                                                      |
| Fixed Obstacles     | `(2, 1)`, `(2, 2)`, `(2, 3)`                            |
| Control Type        | Centralized PPO                                         |
| Joint Action Space  | `4^2 = 16`                                              |
| Observation Size    | 12                                                      |
| Training Timesteps  | 100,000                                                 |
| Model               | `models/ppo_multi_agent_2agents_obstacles.zip`          |
| Training Script     | `train/train_ppo_multi_agent_2agents_obstacles.py`      |
| Evaluation Episodes | 1000 seeded episodes                                    |
| Evaluation Script   | `evaluation/evaluate_multi_agent_obstacle_ppo.py`       |
| Output File         | `results/multi_agent_obstacle_ppo_summary.txt`          |

### Results

| Method                         | Success Rate | Average Reward | Average Episode Length |
| ------------------------------ | -----------: | -------------: | ---------------------: |
| Random baseline                |       76.40% |           0.76 |                  24.63 |
| Greedy obstacle-aware baseline |       93.10% |           0.93 |                   5.83 |
| PPO deterministic              |       92.80% |           0.93 |                   6.08 |
| PPO stochastic                 |      100.00% |           1.00 |                   3.32 |

### Conclusion

The centralized PPO controller successfully learned the fixed-obstacle multi-agent foraging task.

Under deterministic evaluation, PPO achieved 92.80% success, performing very close to the greedy obstacle-aware baseline of 93.10%. However, deterministic PPO was slightly less successful and slightly less efficient than the greedy baseline.

Under stochastic action sampling, PPO achieved 100.00% success with an average episode length of 3.32 steps. This outperformed both the random baseline and the greedy obstacle-aware baseline.

These results show that the learned PPO policy can exploit the fixed-obstacle layout effectively. As in the previous single-agent randomized-obstacle experiments, stochastic policy deployment can reveal stronger performance than deterministic argmax action selection.

---

# Next Planned Experiments

## Experiment 22 - Multi-Agent Random-Obstacle Environment

Status: planned.

### Goal

Extend the multi-agent environment to support variable/random obstacles.

### Planned Features

* `random_obstacles=True`;
* `num_obstacles=3`;
* obstacle generation at reset;
* no overlap between obstacles, agents, and food;
* BFS reachability validation to avoid impossible environments;
* updated tests for randomized obstacle behavior.

### Planned Test File

`tests/test_multi_agent_random_obstacles.py`

### Scientific Motivation

Fixed obstacles test learning in a controlled layout.

Random obstacles test whether a policy can become robust to changing spatial constraints.

---

## Experiment 23 - Multi-Agent Random-Obstacle PPO, 5x5

Status: planned.

### Goal

Train and evaluate a centralized PPO policy with 2 agents in a 5x5 environment with randomized reachable obstacles.

### Planned Outputs

| Output            | Path                                                      |
| ----------------- | --------------------------------------------------------- |
| Training script   | `train/train_ppo_multi_agent_2agents_random_obstacles.py` |
| Evaluation script | `evaluation/evaluate_multi_agent_random_obstacle_ppo.py`  |
| Model             | `models/ppo_multi_agent_2agents_random_obstacles.zip`     |
| Results summary   | `results/multi_agent_random_obstacle_ppo_summary.txt`     |

### Planned Evaluation

Evaluate:

* deterministic PPO;
* stochastic PPO;
* random baseline;
* greedy obstacle-aware baseline.

---

## Experiment 24 - Multi-Agent Random-Obstacle Grid-Size Generalization

Status: planned.

### Goal

Evaluate whether the PPO policy trained on 5x5 randomized obstacles generalizes to a larger 10x10 randomized-obstacle environment with 2 agents.

### Planned Output

`results/multi_agent_random_obstacle_grid_generalization_summary.txt`

### Scientific Motivation

This experiment tests whether the learned multi-agent policy transfers to larger spatial environments while still handling variable obstacle layouts.

---

## Optional Experiment - 3-Agent Scalability

Status: optional.

### Goal

Test scalability from 2 agents to 3 agents.

### Important Difference

The centralized joint action space grows from:

| Agents | Joint Action Space |
| -----: | -----------------: |
|      2 |         `4^2 = 16` |
|      3 |         `4^3 = 64` |

This makes the problem more difficult and may require additional training.

### Planned Use

This experiment should only be attempted after the 2-agent random-obstacle experiments are stable and documented.

---

## Final Documentation Phase

Status: planned.

### Goal

Prepare the project for final submission.

### Planned Work

* update final plots;
* update `README.md`;
* update `PROJECT_CONTEXT.md`;
* update `ROADMAP.md`;
* clean `EXPERIMENT_LOG.md`;
* prepare final report tables;
* write final scientific interpretation.

---

# Summary Table

| Experiment / Task                          | Training Environment             | Evaluation Environment              | Timesteps | Success Rate | Avg Reward | Avg Episode Length |
| ------------------------------------------ | -------------------------------- | ----------------------------------- | --------: | -----------: | ---------: | -----------------: |
| Random Agent                               | N/A                              | 5x5 no obstacles                    |       N/A |          63% |       0.63 |              30.55 |
| PPO Baseline                               | 5x5 no obstacles                 | 5x5 no obstacles                    |       10k |         100% |       1.00 |               4.26 |
| Grid Generalization                        | 5x5 no obstacles                 | 10x10 no obstacles                  |       10k |          96% |       0.96 |              12.88 |
| PPO Fixed Obstacles                        | 5x5 fixed obstacles              | 5x5 fixed obstacles                 |       20k |          79% |       0.79 |              13.34 |
| PPO Fixed Obstacles                        | 5x5 fixed obstacles              | 5x5 fixed obstacles                 |       50k |          70% |       0.70 |              17.35 |
| PPO Fixed Obstacles + State                | 5x5 fixed obstacles              | 5x5 fixed obstacles                 |       50k |          87% |       0.87 |               9.94 |
| Fixed PPO on Random Obstacles              | 5x5 fixed obstacles              | 5x5 randomized obstacles            |       50k |       65.50% |       0.66 |              19.59 |
| PPO Random Obstacles                       | 5x5 randomized obstacles         | 5x5 randomized obstacles            |       50k |       70.90% |       0.71 |              17.14 |
| PPO Random Obstacles                       | 5x5 randomized obstacles         | 5x5 randomized obstacles            |      100k |       76.30% |       0.76 |              14.74 |
| PPO Random Obstacles                       | 5x5 randomized obstacles         | 5x5 randomized obstacles            |      200k |       76.20% |       0.76 |              14.82 |
| PPO Random Obstacles 100k Stochastic       | 5x5 randomized obstacles         | 5x5 randomized obstacles            |      100k |       96.92% |       0.97 |               7.65 |
| PPO Random Obstacles 200k Stochastic       | 5x5 randomized obstacles         | 5x5 randomized obstacles            |      200k |       97.58% |       0.98 |               7.17 |
| BFS Oracle                                 | N/A                              | 5x5 randomized obstacles            |       N/A |      100.00% |       1.00 |               4.29 |
| PPO Random Obstacles 200k Stochastic       | 5x5 randomized obstacles         | 10x10 randomized obstacles          |      200k |       98.08% |       0.98 |              12.60 |
| Multi-Agent Random Baseline                | N/A                              | 5x5 no obstacles, 2 agents          |       N/A |       84.30% |       0.84 |              21.30 |
| Multi-Agent Greedy Baseline                | N/A                              | 5x5 no obstacles, 2 agents          |       N/A |      100.00% |       1.00 |               2.41 |
| Multi-Agent PPO Deterministic              | 5x5 no obstacles, 2 agents       | 5x5 no obstacles, 2 agents          |      100k |       98.90% |       0.99 |               2.94 |
| Multi-Agent PPO Stochastic                 | 5x5 no obstacles, 2 agents       | 5x5 no obstacles, 2 agents          |      100k |      100.00% |       1.00 |               2.58 |
| Multi-Agent PPO Stochastic Generalization  | 5x5 no obstacles, 2 agents       | 10x10 no obstacles, 2 agents        |      100k |       99.88% |        N/A |               5.64 |
| Multi-Agent Fixed-Obstacle Random Baseline | N/A                              | 5x5 fixed obstacles, 2 agents       |       N/A |       76.40% |       0.76 |              24.63 |
| Multi-Agent Fixed-Obstacle Greedy Baseline | N/A                              | 5x5 fixed obstacles, 2 agents       |       N/A |       93.10% |       0.93 |               5.83 |
| Multi-Agent Fixed-Obstacle PPO Det.        | 5x5 fixed obstacles, 2 agents    | 5x5 fixed obstacles, 2 agents       |      100k |       92.80% |       0.93 |               6.08 |
| Multi-Agent Fixed-Obstacle PPO Stoch.      | 5x5 fixed obstacles, 2 agents    | 5x5 fixed obstacles, 2 agents       |      100k |      100.00% |       1.00 |               3.32 |

---

# Main Findings

1. PPO strongly outperformed the random baseline in the basic single-agent environment.
2. The PPO policy generalized reasonably well from 5x5 to 10x10 in the no-obstacle single-agent setting.
3. Fixed obstacles increased task difficulty.
4. Increasing training duration alone did not improve fixed-obstacle performance.
5. Adding obstacle coordinates to the observation space improved fixed-obstacle performance from 70% to 87%.
6. A fixed-obstacle PPO policy did not fully generalize to randomized obstacle layouts.
7. Training directly on randomized obstacles improved robustness.
8. The BFS oracle solved 100% of randomized-obstacle environments, confirming that PPO failures were not caused by impossible environments.
9. PPO failure-case analysis showed that deterministic failures were dominated by repeated no-move behavior.
10. Stochastic PPO evaluation greatly improved randomized-obstacle performance, reaching 97.58% for the 200k model.
11. Random-obstacle grid-size generalization was successful under stochastic action sampling.
12. The multi-agent no-obstacle environment worked for both 2 and 3 agents.
13. Centralized PPO successfully learned the 2-agent no-obstacle task.
14. The 2-agent PPO policy generalized well from 5x5 to 10x10 in the no-obstacle setting.
15. Fixed-obstacle support was successfully added to the multi-agent environment without breaking backward compatibility.
16. In the multi-agent fixed-obstacle setting, the random baseline achieved 76.40% success.
17. The greedy obstacle-aware baseline achieved 93.10% success and was much more efficient than random exploration.
18. The greedy obstacle-aware baseline is strong but not optimal because it uses local Manhattan-distance decisions rather than full path planning.
19. Deterministic PPO in the multi-agent fixed-obstacle environment achieved 92.80% success, performing very close to the greedy obstacle-aware baseline.
20. Stochastic PPO in the multi-agent fixed-obstacle environment achieved 100.00% success and outperformed both the random and greedy baselines.
21. Stochastic policy deployment continued to be important: it improved PPO performance not only in single-agent randomized obstacles, but also in the multi-agent fixed-obstacle setting.
22. The final planned progression is randomized multi-agent obstacles, randomized-obstacle PPO, 10x10 randomized-obstacle generalization, and then final documentation/reporting.
