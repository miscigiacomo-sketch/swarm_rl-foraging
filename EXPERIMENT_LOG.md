# EXPERIMENT LOG - SWARM RL FORAGING

This file summarizes the main experiments performed in the `swarm_rl-foraging` project.

The goal is to keep a clear record of the training setup, evaluation setup, metrics, and scientific conclusions for each experiment.

---

## Metrics

The main metrics used throughout the project are:

* **Success Rate**: percentage of episodes in which the agent reaches the food.
* **Average Reward**: average reward obtained across evaluation episodes.
* **Average Episode Length**: average number of steps per episode.

Unless otherwise specified, evaluations were performed over 100 episodes.

For the randomized-obstacle comparison, models were evaluated on the same 1000 seeded randomized environments to ensure a fair comparison.

---

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

The random agent provides a baseline for comparison. It succeeds in some episodes by chance, but it is inefficient and requires many steps on average.

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

PPO significantly outperforms the random baseline in the basic foraging environment.

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

The PPO policy trained on 5x5 generalizes reasonably well to a larger 10x10 grid. However, the average episode length increases, showing reduced efficiency in the larger environment.

---

## Experiment 4 - Fixed Obstacles, PPO 20k

| Field                  | Value                               |
| ---------------------- | ----------------------------------- |
| Environment            | 5x5 grid                            |
| Obstacles              | Fixed obstacles                     |
| Obstacle Positions     | `[2,1]`, `[2,2]`, `[2,3]`           |
| Algorithm              | PPO                                 |
| Training Timesteps     | 20,000                              |
| Model                  | `models/ppo_foraging_obstacles.zip` |
| Success Rate           | 79%                                 |
| Average Reward         | 0.79                                |
| Average Episode Length | 13.34                               |

### Conclusion

Adding obstacles increases the difficulty of the task. The PPO agent still learns useful behavior, but performance decreases compared to the no-obstacle baseline.

---

## Experiment 5 - Fixed Obstacles, PPO 50k

| Field                  | Value                                   |
| ---------------------- | --------------------------------------- |
| Environment            | 5x5 grid                                |
| Obstacles              | Fixed obstacles                         |
| Obstacle Positions     | `[2,1]`, `[2,2]`, `[2,3]`               |
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

Randomized obstacle environments were validated using breadth-first search. The generator ensures that the food is always reachable from the agent.

This prevents impossible episodes and makes randomized-obstacle evaluations scientifically cleaner.

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

The PPO policy trained only on fixed obstacles does not fully generalize to randomized obstacle layouts.

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

The success rate increased from 70.90% to 76.30%, while the average episode length decreased from 17.14 to 14.74 steps.

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

Increasing randomized-obstacle training from 100k to 200k timesteps did not produce a meaningful improvement.

The 100k model achieved a success rate of 76.30%, while the 200k model achieved 76.20%. The average episode length also remained almost unchanged, increasing slightly from 14.74 to 14.82 steps.

This suggests that, under deterministic action selection, performance appears to reach a plateau around 100k timesteps. Later policy-sampling analysis showed that the learned policy performs much better when evaluated stochastically, indicating that deterministic action selection can underestimate PPO performance in this environment.

---

## Experiment 12 - BFS Oracle Baseline on Randomized Obstacles

| Field                        | Value                                     |
| ---------------------------- | ----------------------------------------- |
| Environment                  | 5x5 randomized reachable obstacles        |
| Agent Type                   | BFS oracle / shortest-path planner        |
| Learning Algorithm           | None                                      |
| Evaluation Episodes          | 1000 seeded episodes                      |
| Success Rate                 | 100.00%                                   |
| Average Reward               | 1.00                                      |
| Average Shortest Path Length | 4.29                                      |
| Average Episode Length       | 4.29                                      |
| Failed Seeds                 | `[]`                                      |
| Evaluation Script            | `evaluation/evaluate_oracle_random_obstacles.py` |

### Conclusion

The BFS oracle solved 100% of the randomized-obstacle environments.

This confirms that the randomized environments used for evaluation were solvable and that the food was reachable in every tested episode. Therefore, PPO failures are not caused by impossible environments, but by limitations of the learned policy under the current sparse reward and observation design.

The oracle also provides an upper-bound reference for efficiency. On the same 5x5 randomized-obstacle setting, the oracle required an average shortest path length of 4.29 steps, while the best PPO randomized-obstacle model required 14.74 steps on average.

---

## Experiment 13 - PPO Failure-Case Analysis on Randomized Obstacles

| Field                                      | Value                                             |
| ------------------------------------------ | ------------------------------------------------- |
| Environment                                | 5x5 randomized reachable obstacles                |
| Model                                      | `models/ppo_foraging_random_obstacles_100k.zip`   |
| Evaluation Episodes                        | 1000 seeded episodes                              |
| Success Rate                               | 76.30%                                            |
| Failure Rate                               | 23.70%                                            |
| Failures                                   | 237                                               |
| Saved Failure Cases                        | 10                                                |
| Average No-Move Steps per Failed Episode   | 48.45                                             |
| Average Repeated-Position Steps per Failed Episode | 48.45                                      |
| Output File                                | `results/random_obstacle_failure_cases.txt`       |
| Analysis Script                            | `evaluation/analyze_random_obstacle_failures.py`  |

### Conclusion

Failure-case analysis showed that the PPO policy failed in 237 out of 1000 randomized-obstacle episodes.

The most important observation is that failed episodes were dominated by repeated no-move behavior. The model produced an average of 48.45 no-move steps per failed episode, close to the 50-step episode limit.

This suggests that in many failed episodes the deterministic PPO policy repeatedly selected invalid or ineffective actions, such as attempting to move into a wall or obstacle. Since the current reward function does not penalize invalid moves and only gives reward when the food is reached, the policy can become trapped in states where it repeatedly selects the same ineffective action.

This supports the interpretation that the 76.30% success rate reflects a learned-policy limitation rather than an environment-generation bug.

---

## Experiment 14 - PPO Policy Sampling Mode Analysis

| Field | Value |
| --- | --- |
| Environment | 5x5 randomized reachable obstacles |
| Evaluation Episodes | 1000 seeded episodes per action seed |
| Action Seeds | `0, 1, 2, 3, 4, 42, 100, 123, 999, 2024` |
| Models | `models/ppo_foraging_random_obstacles_100k.zip`, `models/ppo_foraging_random_obstacles_200k.zip` |
| Analysis Script | `evaluation/evaluate_stochastic_policy_robustness.py` |
| Output File | `results/policy_sampling_robustness_summary.txt` |

### Results

| Model | Deterministic Success | Stochastic Mean Success | Stochastic Std | Mean Average Reward | Mean Average Episode Length | Mean No-Move Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO Random Obstacles 100k | 76.30% | 96.92% | 0.52% | 0.97 | 7.65 | 2.80 |
| PPO Random Obstacles 200k | 76.20% | 97.58% | 0.29% | 0.98 | 7.17 | 2.51 |

### Conclusion

The policy sampling analysis showed that deterministic evaluation substantially underestimated the capability of the learned PPO policy.

Under deterministic action selection, both the 100k and 200k randomized-obstacle models achieved only about 76% success. This matched the earlier failure-case analysis, where failed episodes were dominated by repeated no-move behavior.

When the same trained policies were evaluated using stochastic action sampling, performance improved dramatically. The 100k model achieved a mean success rate of 96.92%, while the 200k model achieved the best performance with a mean success rate of 97.58% across 10 action seeds.

This suggests that the learned PPO policy contains useful action probabilities even when the deterministic argmax action can lead to repeated invalid or ineffective movements. Stochastic sampling allows the agent to escape these local failure states and reach near-oracle performance without changing the environment, reward function, or trained model.

The best randomized-obstacle policy for future experiments is therefore the 200k PPO model evaluated with stochastic action sampling.

---

## Experiment 15 - Random-Obstacle Grid-Size Generalization

| Field | Value |
| --- | --- |
| Training Environment | 5x5 randomized reachable obstacles |
| Evaluation Environments | 5x5 and 10x10 randomized reachable obstacles |
| Model | `models/ppo_foraging_random_obstacles_200k.zip` |
| Evaluation Episodes | 1000 seeded episodes |
| Stochastic Action Seeds | `0, 1, 2, 3, 4, 42, 100, 123, 999, 2024` |
| Evaluation Script | `evaluation/evaluate_random_obstacle_grid_generalization.py` |
| Output File | `results/random_obstacle_grid_generalization_summary.txt` |

### Results

| Grid | Oracle Success | PPO Deterministic Success | PPO Stochastic Mean Success | Stochastic Std | PPO Stochastic Average Episode Length |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5x5 | 100.00% | 76.20% | 97.58% | 0.29% | 7.17 |
| 10x10 | 100.00% | 66.90% | 98.08% | 0.19% | 12.60 |

### Conclusion

The stochastic PPO policy trained on 5x5 randomized obstacles generalized successfully to a larger 10x10 randomized-obstacle environment.

The BFS oracle achieved 100% success on both grid sizes, confirming that both evaluation settings were solvable. The PPO policy also achieved near-oracle success under stochastic action sampling, reaching 97.58% on 5x5 and 98.08% on 10x10.

The average episode length increased from 7.17 steps on 5x5 to 12.60 steps on 10x10, which is expected because the larger grid requires longer paths on average. However, the success rate remained high, showing that the learned stochastic policy transferred well to the larger grid.

The deterministic evaluation remained weaker, especially on the 10x10 grid, where success dropped to 66.90%. This supports the earlier conclusion that deterministic action selection is vulnerable to repeated no-move behavior, while stochastic action sampling better reflects the useful action distribution learned by PPO.

---

# Summary Table

| Experiment                    | Training Environment     | Evaluation Environment   | Timesteps | Success Rate | Avg Reward | Avg Episode Length |
| ----------------------------- | ------------------------ | ------------------------ | --------: | -----------: | ---------: | -----------------: |
| Random Agent                  | None                     | 5x5 no obstacles         |       N/A |          63% |       0.63 |              30.55 |
| PPO Baseline                  | 5x5 no obstacles         | 5x5 no obstacles         |       10k |         100% |       1.00 |               4.26 |
| Grid Generalization           | 5x5 no obstacles         | 10x10 no obstacles       |       10k |          96% |       0.96 |              12.88 |
| PPO Fixed Obstacles           | 5x5 fixed obstacles      | 5x5 fixed obstacles      |       20k |          79% |       0.79 |              13.34 |
| PPO Fixed Obstacles           | 5x5 fixed obstacles      | 5x5 fixed obstacles      |       50k |          70% |       0.70 |              17.35 |
| PPO Fixed Obstacles + State   | 5x5 fixed obstacles      | 5x5 fixed obstacles      |       50k |          87% |       0.87 |               9.94 |
| Fixed PPO on Random Obstacles | 5x5 fixed obstacles      | 5x5 randomized obstacles |       50k |       65.50% |       0.66 |              19.59 |
| PPO Random Obstacles          | 5x5 randomized obstacles | 5x5 randomized obstacles |       50k |       70.90% |       0.71 |              17.14 |
| PPO Random Obstacles          | 5x5 randomized obstacles | 5x5 randomized obstacles |      100k |       76.30% |       0.76 |              14.74 |
| PPO Random Obstacles          | 5x5 randomized obstacles | 5x5 randomized obstacles |      200k |       76.20% |       0.76 |              14.82 |
| PPO Random Obstacles 100k Stochastic | 5x5 randomized obstacles | 5x5 randomized obstacles | 100k | 96.92% | 0.97 | 7.65 |
| PPO Random Obstacles 200k Stochastic | 5x5 randomized obstacles | 5x5 randomized obstacles | 200k | 97.58% | 0.98 | 7.17 |
| BFS Oracle                    | N/A                      | 5x5 randomized obstacles |       N/A |      100.00% |       1.00 |               4.29 |
| PPO Random Obstacles 200k Stochastic | 5x5 randomized obstacles | 10x10 randomized obstacles | 200k | 98.08% | 0.98 | 12.60 |
| BFS Oracle                    | N/A                      | 10x10 randomized obstacles |       N/A |      100.00% |       1.00 |               N/A |

---

# Main Findings

1. PPO strongly outperformed the random baseline in the basic environment.
2. The PPO policy generalized reasonably well from 5x5 to 10x10 in the no-obstacle setting.
3. Obstacles increased task difficulty.
4. Increasing training duration alone did not improve fixed-obstacle performance.
5. Adding obstacle information to the observation space improved fixed-obstacle performance from 70% to 87%.
6. A fixed-obstacle PPO policy did not fully generalize to randomized obstacle layouts.
7. Training directly on randomized obstacles improved generalization.
8. Increasing randomized-obstacle training from 50k to 100k further improved robustness and efficiency under deterministic evaluation, but deterministic performance plateaued around 76% at 200k timesteps.
9. The BFS oracle solved 100% of the randomized-obstacle environments, confirming that the task was solvable.
10. PPO failure-case analysis showed that failed deterministic episodes were dominated by repeated no-move behavior, suggesting limitations of deterministic action selection rather than impossible environments.
11. Stochastic policy sampling greatly improved performance, reaching 96.92% for the 100k model and 97.58% for the 200k model.
12. The 200k randomized-obstacle PPO model evaluated stochastically is the best randomized-obstacle policy so far.
13. Random-obstacle grid-size generalization was successful: the 200k stochastic PPO policy achieved 97.58% success on 5x5 and 98.08% success on 10x10.
14. The 10x10 randomized-obstacle environment required longer paths, increasing the stochastic PPO average episode length from 7.17 to 12.60 steps, but the success rate remained near-oracle.

---

# Development Tasks and Next Experiments

This section records implementation tasks separately from quantitative experiments.  
The experiment sections above remain focused on training/evaluation results, while the tasks below track environment and project-structure changes.

---

## Task 1 - Multi-Agent Fixed-Obstacle Environment

| Field | Value |
| --- | --- |
| Status | Completed |
| Environment | Centralized multi-agent foraging |
| Grid Size | 5x5 |
| Agents | 2 |
| Obstacles | Fixed |
| Obstacle Positions | `(2, 1)`, `(2, 2)`, `(2, 3)` |
| Joint Action Space | `4^2 = 16` |
| Observation Size | 12 |
| Main File | `env/multi_agent_foraging_env.py` |
| Test File | `tests/test_multi_agent_obstacles.py` |

### Implementation Summary

The multi-agent environment was extended with optional fixed-obstacle support while preserving backward compatibility with the previous no-obstacle version.

The updated environment now:
- accepts an optional `obstacles` argument;
- includes obstacle coordinates in the observation when obstacles are provided;
- prevents agents from spawning on obstacle cells;
- prevents the food from spawning on obstacle cells;
- blocks movements into obstacle cells;
- preserves the existing collision and position-swap prevention logic;
- keeps the no-obstacle setup unchanged when `obstacles=None`.

The text renderer was also simplified to use one-character agent labels (`A`, `B`, `C`, ...), which keeps the printed grid aligned during debugging.

### Validation

The obstacle environment was manually checked with 2 agents and 3 fixed obstacles.

Observed configuration:
- observation shape: `(12,)`;
- action space: `Discrete(16)`;
- obstacles visible in the rendered grid as `X`.

Automated tests were added in `tests/test_multi_agent_obstacles.py`.

Test coverage:
- Stable-Baselines3 `check_env` compatibility;
- correct observation shape;
- obstacle coordinates included in the observation;
- agents do not spawn on obstacles;
- food does not spawn on obstacles;
- movement into obstacle cells is blocked.

Result:
- `6 passed` for the obstacle-specific test file;
- existing no-obstacle multi-agent tests still pass.

### Conclusion

The centralized multi-agent environment is now ready for obstacle-aware baseline evaluation and PPO training.

---

## Task 2 - Fixed-Obstacle Multi-Agent Baselines

| Field | Planned Value |
| --- | --- |
| Environment | 5x5 multi-agent fixed-obstacle grid |
| Agents | 2 |
| Obstacles | Fixed: `(2, 1)`, `(2, 2)`, `(2, 3)` |
| Baselines | Random policy and obstacle-aware greedy policy |
| Planned Script | `evaluation/evaluate_multi_agent_obstacle_baselines.py` |
| Planned Output | `results/multi_agent_obstacle_baseline_summary.txt` |

### Objective

Evaluate how random and greedy non-learning policies perform in the fixed-obstacle multi-agent environment.

This will provide a reference for judging whether PPO learns useful obstacle-aware behavior.

---

## Task 3 - Fixed-Obstacle Multi-Agent PPO

| Field | Planned Value |
| --- | --- |
| Environment | 5x5 multi-agent fixed-obstacle grid |
| Agents | 2 |
| Algorithm | PPO |
| Planned Training Script | `train/train_ppo_multi_agent_2agents_obstacles.py` |
| Planned Evaluation Script | `evaluation/evaluate_multi_agent_obstacle_ppo.py` |
| Planned Model | `models/ppo_multi_agent_2agents_obstacles.zip` |

### Objective

Train and evaluate a centralized PPO controller in the fixed-obstacle multi-agent environment.

The result should be compared against the random and greedy obstacle-aware baselines from Task 2.

---

## Task 4 - Random-Obstacle Multi-Agent Environment

| Field | Planned Value |
| --- | --- |
| Environment | 5x5 multi-agent randomized-obstacle grid |
| Agents | 2 |
| Obstacles | Randomized reachable obstacles |
| Main File | `env/multi_agent_foraging_env.py` |

### Objective

Extend the multi-agent environment from fixed obstacles to randomized obstacles.

The generator should avoid placing obstacles on agents or food and should preserve reachability so that generated episodes are solvable.

---

## Task 5 - Random-Obstacle Multi-Agent PPO on 5x5

| Field | Planned Value |
| --- | --- |
| Environment | 5x5 randomized-obstacle multi-agent grid |
| Agents | 2 |
| Algorithm | PPO |
| Evaluation Modes | Deterministic and stochastic action selection |

### Objective

Train PPO directly on randomized multi-agent obstacle layouts and evaluate whether it becomes more robust than a policy trained only on fixed obstacles.

---

## Task 6 - Random-Obstacle Multi-Agent Generalization to 10x10

| Field | Planned Value |
| --- | --- |
| Training Environment | 5x5 randomized-obstacle multi-agent grid |
| Evaluation Environment | 10x10 randomized-obstacle multi-agent grid |
| Agents | 2 |

### Objective

Evaluate whether a PPO policy trained on 5x5 randomized-obstacle environments transfers to a larger 10x10 grid with variable obstacles.

This is the main final generalization experiment for the 2-agent setting.

---

## Task 7 - Optional 3-Agent Scalability Experiment

| Field | Planned Value |
| --- | --- |
| Environment | Multi-agent foraging with obstacles |
| Agents | 3 |
| Joint Action Space | `4^3 = 64` |
| Priority | Optional |

### Objective

Test whether the centralized-control approach remains effective when increasing from 2 to 3 agents.

This task is optional because it increases the action-space size substantially and may require a separate training setup.
