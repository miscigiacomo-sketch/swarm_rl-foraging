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

This suggests that, under the current sparse reward function and observation design, performance reaches a plateau around 100k timesteps. Further improvements may require changes such as reward shaping, different hyperparameters, curriculum learning, or a modified observation/action design.

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
| BFS Oracle                    | N/A                      | 5x5 randomized obstacles |       N/A |      100.00% |       1.00 |               4.29 |

---

# Main Findings

1. PPO strongly outperformed the random baseline in the basic environment.
2. The PPO policy generalized reasonably well from 5x5 to 10x10 in the no-obstacle setting.
3. Obstacles increased task difficulty.
4. Increasing training duration alone did not improve fixed-obstacle performance.
5. Adding obstacle information to the observation space improved fixed-obstacle performance from 70% to 87%.
6. A fixed-obstacle PPO policy did not fully generalize to randomized obstacle layouts.
7. Training directly on randomized obstacles improved generalization.
8. Increasing randomized-obstacle training from 50k to 100k further improved robustness and efficiency, but performance plateaued at 200k timesteps.
9. The BFS oracle solved 100% of the randomized-obstacle environments, confirming that the task was solvable.
10. PPO failure-case analysis showed that failed episodes were dominated by repeated no-move behavior, suggesting limitations of the learned policy rather than impossible environments.

---

# Next Experiments

## Random-Obstacle Grid-Size Generalization

Planned evaluation:

Train on 5x5 randomized obstacles and evaluate on 10x10 randomized obstacles.

The current best randomized-obstacle model is:

`models/ppo_foraging_random_obstacles_100k.zip`

Although the 200k model was also trained, it did not improve over the 100k model. Therefore, the 100k model remains the best randomized-obstacle policy for the next generalization experiment.

Scientific question:

Does the randomized-obstacle PPO policy generalize to a larger grid?

---

## Multi-Agent Extension

Planned setup:

* 2 agents;
* 1 food source;
* centralized PPO control;
* joint action space.

Scientific question:

Does adding multiple agents improve foraging efficiency?
