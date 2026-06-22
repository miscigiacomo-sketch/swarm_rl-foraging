# EXPERIMENT LOG - SWARM RL FORAGING

This file summarizes the main experiments performed in the `swarm_rl-foraging` project.

The goal is to keep a clear record of the training setup, evaluation setup, metrics, and scientific conclusions for each experiment.

---

## Metrics

The main metrics used throughout the project are:

* **Success Rate**: percentage of episodes in which at least one agent reaches the food.
* **Average Reward**: average reward obtained across evaluation episodes.
* **Average Episode Length**: average number of steps per episode.

Unless otherwise specified, evaluations were performed over 100 episodes.

For randomized-obstacle evaluations, models were evaluated on seeded environments to ensure fair comparisons.

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

# Recent Completed Experiments

## Experiment 19 - Multi-Agent Fixed-Obstacle Environment

| Field | Value |
|---|---|
| Environment | `MultiAgentForagingEnv` |
| Grid Size | 5x5 |
| Number of Agents | 2 |
| Obstacles | Fixed obstacles |
| Obstacle Positions | `(2, 1)`, `(2, 2)`, `(2, 3)` |
| Joint Action Space | `4^2 = 16` |
| Observation Size | 12 |
| Test File | `tests/test_multi_agent_obstacles.py` |

### Conclusion

Fixed-obstacle support was successfully added to the centralized multi-agent environment.

The implementation preserved compatibility with the previous no-obstacle multi-agent setup, while adding obstacle collision handling, obstacle-aware spawning, rendering support, collision prevention, and position-swap prevention.

---

## Experiment 20 - Multi-Agent Fixed-Obstacle Baselines

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 76.40% | 0.76 | 24.63 |
| Greedy obstacle-aware baseline | 93.10% | 0.93 | 5.83 |

### Conclusion

The random baseline achieved 76.40% success, while the greedy obstacle-aware baseline reached 93.10% success.

The greedy baseline was much more efficient because it used local Manhattan-distance decisions while avoiding obstacles and invalid moves. However, it is still a hand-coded local heuristic and does not perform full path planning.

---

## Experiment 21 - Multi-Agent Fixed-Obstacle PPO

| Field | Value |
|---|---|
| Environment | `MultiAgentForagingEnv` |
| Grid Size | 5x5 |
| Number of Agents | 2 |
| Obstacles | Fixed obstacles |
| Algorithm | PPO |
| Control Type | Centralized joint-action control |
| Joint Action Space | `4^2 = 16` |
| Training Timesteps | 100,000 |
| Training Script | `train/train_ppo_multi_agent_2agents_obstacles.py` |
| Evaluation Script | `evaluation/evaluate_multi_agent_obstacle_ppo.py` |
| Model | `models/ppo_multi_agent_2agents_obstacles.zip` |
| Results File | `results/multi_agent_obstacle_ppo_summary.txt` |

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 76.40% | 0.76 | 24.63 |
| Greedy obstacle-aware baseline | 93.10% | 0.93 | 5.83 |
| PPO deterministic | 92.80% | 0.93 | 6.08 |
| PPO stochastic | 100.00% | 1.00 | 3.32 |

### Conclusion

The centralized PPO controller successfully learned the fixed-obstacle multi-agent foraging task.

Deterministic PPO performed close to the greedy obstacle-aware baseline, while stochastic PPO achieved perfect success and the shortest average episode length. This reinforced the earlier observation that sampling from the learned PPO policy distribution can improve robustness compared to deterministic argmax deployment.

---

## Experiment 22 - Multi-Agent Random-Obstacle Environment

| Field | Value |
|---|---|
| Environment | `MultiAgentForagingEnv` |
| Grid Size | 5x5 |
| Number of Agents | 2 |
| Random Obstacles | True |
| Number of Obstacles | 3 |
| Reachability Validation | Breadth-first search |
| Joint Action Space | `4^2 = 16` |
| Observation Size | 12 |
| Test File | `tests/test_multi_agent_random_obstacles.py` |

### Implemented Features

* randomized obstacle generation at environment reset;
* no overlap between obstacles;
* no overlap between obstacles, agents, and food;
* food reachability validation using breadth-first search;
* compatibility with no-obstacle and fixed-obstacle multi-agent environments;
* terminal visualization for random-obstacle layouts.

### Validation

The randomized-obstacle tests passed successfully.

The fixed-obstacle and no-obstacle multi-agent tests were also rerun and passed, confirming backward compatibility.

### Conclusion

Experiment 22 extended the multi-agent environment from fixed obstacles to randomized reachable obstacle layouts.

This created the foundation for training and evaluating policies under changing spatial constraints.

---

## Experiment 23 - Multi-Agent Random-Obstacle PPO, 5x5

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
| Model | `models/ppo_multi_agent_2agents_random_obstacles.zip` |
| Results File | `results/multi_agent_random_obstacle_ppo_summary.txt` |

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 81.50% | 0.81 | 22.24 |
| Greedy obstacle-aware baseline | 93.50% | 0.94 | 5.45 |
| PPO deterministic | 82.40% | 0.82 | 10.77 |
| PPO stochastic | 99.40% | 0.99 | 3.91 |

### Conclusion

PPO successfully learned useful behavior in the randomized-obstacle multi-agent environment.

The deterministic PPO result was only slightly better than the random baseline and below the greedy heuristic. However, stochastic PPO strongly outperformed all other methods, reaching 99.40% success and the shortest average episode length.

This shows that the learned PPO policy distribution contained useful alternative actions, and that stochastic deployment was more robust than deterministic argmax action selection.

---

## Experiment 24 - Multi-Agent Random-Obstacle Grid Generalization

| Field | Value |
|---|---|
| Training Grid Size | 5x5 |
| Evaluation Grid Sizes | 5x5 and 10x10 |
| Number of Agents | 2 |
| Random Obstacles | True |
| Number of Obstacles | 3 |
| Algorithm | PPO |
| Control Type | Centralized joint-action control |
| Model | `models/ppo_multi_agent_2agents_random_obstacles.zip` |
| Evaluation Script | `evaluation/evaluate_multi_agent_random_obstacle_grid_generalization.py` |
| Results File | `results/multi_agent_random_obstacle_grid_generalization_summary.txt` |

### 5x5 Results

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 81.50% | 0.81 | 22.24 |
| Greedy obstacle-aware baseline | 93.50% | 0.94 | 5.45 |
| PPO deterministic | 82.40% | 0.82 | 10.77 |
| PPO stochastic | 99.40% | 0.99 | 3.91 |

### 10x10 Results

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 37.60% | 0.38 | 38.81 |
| Greedy obstacle-aware baseline | 95.10% | 0.95 | 7.03 |
| PPO deterministic | 86.30% | 0.86 | 11.51 |
| PPO stochastic | 99.98% | 1.00 | 7.15 |

### Conclusion

The 10x10 environment increased the spatial search space while keeping the same number of agents and obstacles.

The random baseline dropped strongly from 81.50% on 5x5 to 37.60% on 10x10, showing that the larger grid made uninformed exploration much less effective. The greedy baseline remained strong because it directly followed a local obstacle-aware distance heuristic.

PPO stochastic generalized extremely well to 10x10, reaching 99.98% success. This indicates that the learned policy remained effective even when evaluated outside the original 5x5 training grid.

---

## Experiment 25 - 3-Agent Random-Obstacle Scalability on 10x10

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
| Model | `models/ppo_multi_agent_3agents_random_obstacles_10x10.zip` |
| Results File | `results/multi_agent_3agents_random_obstacle_10x10_summary.txt` |

| Method | Success Rate | Average Reward | Average Episode Length |
|---|---:|---:|---:|
| Random baseline | 50.00% | 0.50 | 35.12 |
| Greedy obstacle-aware baseline | 96.20% | 0.96 | 5.66 |
| PPO deterministic | 65.00% | 0.65 | 21.72 |
| PPO stochastic | 99.82% | 1.00 | 9.92 |

### Conclusion

Experiment 25 tested scalability from 2 to 3 agents in a 10x10 randomized-obstacle environment.

Increasing the number of agents increased the centralized joint action space from `4^2 = 16` to `4^3 = 64`. This made deterministic PPO deployment much less robust, with only 65.00% success.

However, stochastic PPO still achieved 99.82% mean success. This suggests that the learned PPO policy distribution contained useful alternative joint actions, and that stochastic sampling helped avoid repeated suboptimal action patterns in the larger joint action space.

The experiment strengthens the swarm-inspired framing of the project by showing that the framework can be extended from two agents to three agents while maintaining high stochastic-policy performance.

---

## Experiment 26 - Final Comparative Analysis

| Field | Value |
|---|---|
| Analysis Type | Final comparative analysis |
| Compared Methods | Random baseline, greedy obstacle-aware baseline, PPO deterministic, PPO stochastic |
| Compared Scenarios | 2-agent fixed obstacles 5x5, 2-agent random obstacles 5x5, 2-agent random obstacles 10x10, 3-agent random obstacles 10x10 |
| Analysis Script | `evaluation/generate_final_comparative_analysis.py` |
| CSV Output | `results/final_analysis/final_comparative_results.csv` |
| Summary Output | `results/final_analysis/final_comparative_summary.txt` |
| Plot Outputs | `plots/final_success_rate_comparison.png`, `plots/final_episode_length_comparison.png`, `plots/ppo_deterministic_vs_stochastic_success.png`, `plots/ppo_stochastic_gain_over_deterministic.png` |

### Conclusion

Experiment 26 consolidated the main final results into a single comparative analysis.

The analysis compared random baselines, greedy obstacle-aware baselines, deterministic PPO, and stochastic PPO across the key final multi-agent scenarios. This made it easier to evaluate the effect of obstacles, randomized layouts, grid-size generalization, agent-count scaling, and PPO deployment mode.

The main conclusion was that stochastic PPO achieved the highest success rate in all final scenarios. Deterministic PPO became less reliable in the most complex settings, especially in the 3-agent 10x10 random-obstacle environment.

---

## Experiment 27 - Learned-Behavior and Failure Analysis

| Field | Value |
|---|---|
| Analysis Type | Learned-behavior and failure analysis |
| Main Focus | Deterministic vs stochastic PPO behavior |
| Output File | `results/final_analysis/learned_behavior_analysis.md` |
| Related Experiments | Experiments 13, 14, 23, 24, 25, 26 |

### Conclusion

Experiment 27 analyzed what the PPO policy appeared to learn and why stochastic deployment performed better than deterministic deployment.

The analysis emphasized that PPO stochastic deployment is not equivalent to random exploration. Instead, actions are sampled from the learned PPO policy distribution. In complex scenarios, deterministic argmax action selection can repeat locally likely but globally ineffective actions, causing loops or no-move behavior. Stochastic sampling can select alternative high-probability actions and therefore improve robustness.

This analysis also connected the results to the multi-agent setting. As the number of agents increased, the centralized joint action space grew from `4^2 = 16` to `4^3 = 64`, making deterministic joint-action selection more fragile. Stochastic PPO remained highly successful under this increased action-space complexity.

---

## Experiment 28 - Sensitivity Analysis

| Field | Value |
|---|---|
| Analysis Type | Sensitivity and robustness analysis |
| Main Factors | Obstacle configuration, grid size, agent count, PPO deployment mode |
| Plot Script | `evaluation/generate_sensitivity_analysis_plot.py` |
| Plot Output | `plots/sensitivity_to_key_factors.png` |
| Summary Output | `results/final_analysis/sensitivity_analysis_summary.txt` |

### Compared Factors

| Factor | Compared Values |
|---|---|
| Obstacle configuration | Fixed obstacles vs random obstacles |
| Grid size | 5x5 vs 10x10 |
| Agent count | 2 agents vs 3 agents |
| PPO deployment mode | Deterministic vs stochastic |

### Main Findings

The sensitivity analysis shows that randomized obstacles mainly reduce deterministic PPO performance, while stochastic PPO remains highly robust.

Increasing the grid size strongly penalizes the random baseline, confirming that uninformed exploration becomes much harder in larger spatial environments.

Increasing the number of agents from two to three substantially reduces deterministic PPO performance because the centralized joint action space grows from `4^2 = 16` to `4^3 = 64`.

The gain from stochastic PPO deployment becomes larger in more complex scenarios, supporting the conclusion that sampling from the learned PPO policy distribution improves robustness compared to deterministic argmax deployment.

# Summary Table

| Experiment / Task | Training Environment | Evaluation Environment | Timesteps | Success Rate | Avg Reward | Avg Episode Length |
|---|---|---|---:|---:|---:|---:|
| Random Agent | N/A | 5x5 no obstacles | N/A | 63% | 0.63 | 30.55 |
| PPO Baseline | 5x5 no obstacles | 5x5 no obstacles | 10k | 100% | 1.00 | 4.26 |
| Grid Generalization | 5x5 no obstacles | 10x10 no obstacles | 10k | 96% | 0.96 | 12.88 |
| PPO Fixed Obstacles | 5x5 fixed obstacles | 5x5 fixed obstacles | 20k | 79% | 0.79 | 13.34 |
| PPO Fixed Obstacles | 5x5 fixed obstacles | 5x5 fixed obstacles | 50k | 70% | 0.70 | 17.35 |
| PPO Fixed Obstacles + State | 5x5 fixed obstacles | 5x5 fixed obstacles | 50k | 87% | 0.87 | 9.94 |
| Fixed PPO on Random Obstacles | 5x5 fixed obstacles | 5x5 randomized obstacles | 50k | 65.50% | 0.66 | 19.59 |
| PPO Random Obstacles | 5x5 randomized obstacles | 5x5 randomized obstacles | 50k | 70.90% | 0.71 | 17.14 |
| PPO Random Obstacles | 5x5 randomized obstacles | 5x5 randomized obstacles | 100k | 76.30% | 0.76 | 14.74 |
| PPO Random Obstacles | 5x5 randomized obstacles | 5x5 randomized obstacles | 200k | 76.20% | 0.76 | 14.82 |
| PPO Random Obstacles 100k Stochastic | 5x5 randomized obstacles | 5x5 randomized obstacles | 100k | 96.92% | 0.97 | 7.65 |
| PPO Random Obstacles 200k Stochastic | 5x5 randomized obstacles | 5x5 randomized obstacles | 200k | 97.58% | 0.98 | 7.17 |
| BFS Oracle | N/A | 5x5 randomized obstacles | N/A | 100.00% | 1.00 | 4.29 |
| PPO Random Obstacles 200k Stochastic | 5x5 randomized obstacles | 10x10 randomized obstacles | 200k | 98.08% | 0.98 | 12.60 |
| Multi-Agent Random Baseline | N/A | 5x5 no obstacles, 2 agents | N/A | 84.30% | N/A | 21.30 |
| Multi-Agent Greedy Baseline | N/A | 5x5 no obstacles, 2 agents | N/A | 100.00% | N/A | 2.41 |
| Multi-Agent PPO Deterministic | 5x5 no obstacles, 2 agents | 5x5 no obstacles, 2 agents | 100k | 98.90% | N/A | 2.94 |
| Multi-Agent PPO Stochastic | 5x5 no obstacles, 2 agents | 5x5 no obstacles, 2 agents | 100k | 100.00% | N/A | 2.58 |
| Multi-Agent PPO Stochastic Generalization | 5x5 no obstacles, 2 agents | 10x10 no obstacles, 2 agents | 100k | 99.88% | N/A | 5.64 |
| Multi-Agent Fixed-Obstacle Random Baseline | N/A | 5x5 fixed obstacles, 2 agents | N/A | 76.40% | 0.76 | 24.63 |
| Multi-Agent Fixed-Obstacle Greedy Baseline | N/A | 5x5 fixed obstacles, 2 agents | N/A | 93.10% | 0.93 | 5.83 |
| Multi-Agent Fixed-Obstacle PPO Deterministic | 5x5 fixed obstacles, 2 agents | 5x5 fixed obstacles, 2 agents | 100k | 92.80% | 0.93 | 6.08 |
| Multi-Agent Fixed-Obstacle PPO Stochastic | 5x5 fixed obstacles, 2 agents | 5x5 fixed obstacles, 2 agents | 100k | 100.00% | 1.00 | 3.32 |
| Multi-Agent Random-Obstacle PPO Deterministic | 5x5 randomized obstacles, 2 agents | 5x5 randomized obstacles, 2 agents | 100k | 82.40% | 0.82 | 10.77 |
| Multi-Agent Random-Obstacle PPO Stochastic | 5x5 randomized obstacles, 2 agents | 5x5 randomized obstacles, 2 agents | 100k | 99.40% | 0.99 | 3.91 |
| Multi-Agent Random-Obstacle PPO Deterministic Generalization | 5x5 randomized obstacles, 2 agents | 10x10 randomized obstacles, 2 agents | 100k | 86.30% | 0.86 | 11.51 |
| Multi-Agent Random-Obstacle PPO Stochastic Generalization | 5x5 randomized obstacles, 2 agents | 10x10 randomized obstacles, 2 agents | 100k | 99.98% | 1.00 | 7.15 |
| 3-Agent Random-Obstacle Random Baseline | N/A | 10x10 randomized obstacles, 3 agents | N/A | 50.00% | 0.50 | 35.12 |
| 3-Agent Random-Obstacle Greedy Baseline | N/A | 10x10 randomized obstacles, 3 agents | N/A | 96.20% | 0.96 | 5.66 |
| 3-Agent Random-Obstacle PPO Deterministic | 10x10 randomized obstacles, 3 agents | 10x10 randomized obstacles, 3 agents | 200k | 65.00% | 0.65 | 21.72 |
| 3-Agent Random-Obstacle PPO Stochastic | 10x10 randomized obstacles, 3 agents | 10x10 randomized obstacles, 3 agents | 200k | 99.82% | 1.00 | 9.92 |
| Final Comparative Analysis | N/A | Key final multi-agent scenarios | N/A | N/A | N/A | N/A |
| Learned-Behavior Analysis | N/A | Final PPO behavior and failure analysis | N/A | N/A | N/A | N/A |
| Sensitivity Analysis | N/A | Key experimental factors | N/A | N/A | N/A | N/A |

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
10. Stochastic PPO evaluation greatly improved randomized-obstacle performance, reaching 97.58% for the 200k single-agent model.
11. Random-obstacle grid-size generalization was successful under stochastic action sampling.
12. The multi-agent no-obstacle environment worked for both 2 and 3 agents.
13. Centralized PPO successfully learned the 2-agent no-obstacle task.
14. The 2-agent PPO policy generalized well from 5x5 to 10x10 in the no-obstacle setting.
15. Fixed-obstacle support was successfully added to the multi-agent environment without breaking backward compatibility.
16. In the multi-agent fixed-obstacle setting, the random baseline achieved 76.40% success.
17. The greedy obstacle-aware baseline achieved 93.10% success and was much more efficient than random exploration.
18. The centralized PPO controller successfully learned the multi-agent fixed-obstacle task.
19. Stochastic PPO achieved 100.00% success in the multi-agent fixed-obstacle environment and outperformed deterministic PPO in efficiency.
20. Randomized obstacles were successfully added to the multi-agent environment with reachability validation and backward-compatible tests.
21. In the 2-agent randomized-obstacle 5x5 environment, PPO stochastic achieved 99.40% success, while deterministic PPO achieved only 82.40%.
22. The 2-agent randomized-obstacle PPO policy generalized extremely well to the 10x10 grid under stochastic deployment, reaching 99.98% success.
23. In the 3-agent 10x10 randomized-obstacle environment, the joint action space increased from 16 to 64 actions.
24. Deterministic PPO became less robust in the 3-agent 10x10 environment, achieving 65.00% success.
25. Stochastic PPO remained highly robust in the 3-agent 10x10 environment, achieving 99.82% mean success.
26. PPO stochastic deployment is not random exploration; it samples from the learned PPO policy distribution.
27. The project is best framed as swarm-inspired multi-agent foraging with centralized joint-action reinforcement learning, not as a fully decentralized swarm intelligence system.
28. Final comparative analysis consolidated the main multi-agent scenarios into a single set of tables, plots, and summary files.
29. Learned-behavior analysis showed that deterministic PPO can become fragile because argmax action selection may repeat locally likely but ineffective actions.
30. Sensitivity analysis showed that performance depends strongly on obstacle configuration, grid size, number of agents, and PPO deployment mode.
31. The gain from stochastic PPO deployment increases in more complex scenarios, especially in the 3-agent 10x10 random-obstacle setting.

