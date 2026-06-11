# ROADMAP - SWARM RL FORAGING

This roadmap defines the planned development path for the `swarm_rl-foraging` project.

The project is structured as a progressive reinforcement learning study, where the environment complexity is increased step by step.

---

## Overall Scientific Direction

The project investigates how PPO performance changes under increasing environment complexity.

The main research themes are:

* baseline comparison;
* learning efficiency;
* grid-size generalization;
* obstacle navigation;
* state representation;
* randomized environment robustness;
* training-duration sensitivity;
* oracle validation;
* failure-case analysis;
* multi-agent scalability.

The final goal is to present a controlled experimental study of reinforcement learning for foraging behavior, progressing from a single-agent system toward a simple swarm-like multi-agent setup.

---

## Completed Work

### Day 1 - Base Environment

Status: completed

Implemented:

* custom Gymnasium environment;
* 2D grid-world;
* random food spawning;
* agent movement;
* reward and termination logic;
* text rendering.

---

### Day 2 - PPO Training

Status: completed

Implemented:

* PPO training pipeline;
* Stable-Baselines3 integration;
* model saving.

Model:

`models/ppo_foraging.zip`

---

### Day 3 - Evaluation and Random Baseline

Status: completed

Implemented:

* random agent baseline;
* PPO evaluation;
* baseline comparison.

Key result:

| Agent | Success Rate | Average Steps |
|---|---:|---:|
| Random Agent | 63% | 30.55 |
| PPO Agent | 100% | 4.26 |

---

### Day 4 - Grid-Size Generalization

Status: completed

Experiment:

Train PPO on 5x5 and evaluate on 10x10.

Key result:

| Experiment | Success Rate | Average Steps |
|---|---:|---:|
| PPO 5x5 evaluated on 10x10 | 96% | 12.88 |

---

### Day 5 - Results and Visualization

Status: completed

Implemented:

* results folder;
* plots;
* report metrics;
* performance comparison figures.

---

### Day 6 - Fixed Obstacles and State Representation

Status: completed

Implemented:

* fixed obstacles;
* obstacle collision handling;
* obstacle rendering;
* PPO training with obstacles;
* state augmentation.

Key result:

| Experiment | Success Rate | Average Steps |
|---|---:|---:|
| PPO fixed obstacles 50k | 70% | 17.35 |
| PPO fixed obstacles + state augmentation 50k | 87% | 9.94 |

Main finding:

Adding obstacle coordinates to the observation improved performance from 70% to 87%.

---

### Day 7 - Randomized Obstacles

Status: completed

Implemented:

* randomized obstacle generation;
* obstacle overlap prevention;
* food-obstacle overlap prevention;
* reachability validation;
* fixed-obstacle PPO evaluation on randomized environments.

Key validation:

| Test | Result |
|---|---:|
| Reachable randomized environments | 1000 / 1000 |

Key result:

| Model | Evaluation Environment | Success Rate | Average Steps |
|---|---|---:|---:|
| PPO fixed obstacles | Randomized obstacles | 65.50% | 19.59 |

Main finding:

The fixed-obstacle PPO policy does not fully generalize to randomized obstacle layouts.

---

### Day 8 - PPO on Randomized Obstacles

Status: completed

Implemented:

* PPO training on randomized obstacles;
* 50k randomized-obstacle model;
* 100k randomized-obstacle model;
* controlled evaluation on the same 1000 seeded randomized environments.

Key result:

| Model | Success Rate | Average Steps |
|---|---:|---:|
| PPO fixed obstacles on random obstacles | 65.50% | 19.59 |
| PPO random obstacles 50k | 70.90% | 17.14 |
| PPO random obstacles 100k | 76.30% | 14.74 |

Main finding:

Training directly on randomized obstacles improves generalization, and increasing training from 50k to 100k improves robustness and efficiency.

---

### Day 9 - Randomized Obstacle PPO 200k

Status: completed

Implemented:

* 200k PPO training on randomized reachable obstacle environments;
* separate 200k randomized-obstacle model;
* updated seeded comparison script;
* controlled comparison between fixed-obstacle, 50k, 100k, and 200k models on the same 1000 seeded randomized environments.

Model:

`models/ppo_foraging_random_obstacles_200k.zip`

Key result:

| Model | Success Rate | Average Reward | Average Steps |
|---|---:|---:|---:|
| PPO fixed obstacles on random obstacles | 65.50% | 0.66 | 19.59 |
| PPO random obstacles 50k | 70.90% | 0.71 | 17.14 |
| PPO random obstacles 100k | 76.30% | 0.76 | 14.74 |
| PPO random obstacles 200k | 76.20% | 0.76 | 14.82 |

Main finding:

Increasing randomized-obstacle training from 100k to 200k timesteps did not produce a meaningful improvement.

The 100k model achieved a success rate of 76.30%, while the 200k model achieved 76.20%. The average episode length also remained almost unchanged.

This suggests that, under the current sparse reward function and observation design, the randomized-obstacle PPO policy reaches a performance plateau around 100k timesteps.

---

### Post-Day 9 - BFS Oracle and PPO Failure-Case Analysis

Status: completed

Implemented:

* BFS oracle evaluation on 1000 seeded randomized-obstacle environments;
* shortest-path baseline for environment solvability validation;
* PPO failure-case analysis using the 100k randomized-obstacle model;
* saved example failure cases in `results/random_obstacle_failure_cases.txt`.

BFS oracle result:

| Method | Success Rate | Average Shortest Path Length | Average Steps | Failed Seeds |
|---|---:|---:|---:|---|
| BFS oracle | 100.00% | 4.29 | 4.29 | `[]` |

PPO failure-case analysis result:

| Model | Success Rate | Failure Rate | Failures | Avg No-Move Steps per Failed Episode | Avg Repeated-Position Steps per Failed Episode |
|---|---:|---:|---:|---:|---:|
| PPO random obstacles 100k | 76.30% | 23.70% | 237 | 48.45 | 48.45 |

Main finding:

The BFS oracle confirmed that all evaluated randomized-obstacle environments were solvable.

PPO failure-case analysis showed that failed episodes were dominated by repeated no-move behavior. This suggests that PPO failures are not caused by unreachable food positions, but by limitations of the learned deterministic policy under sparse rewards and randomized obstacle layouts.

---

## Next Planned Work

## Day 10 - Random-Obstacle Grid-Size Generalization

Status: next

### Goal

Evaluate whether the best PPO policy trained on 5x5 randomized obstacles generalizes to a larger 10x10 randomized-obstacle environment.

### Selected Model

The current best randomized-obstacle model is:

`models/ppo_foraging_random_obstacles_100k.zip`

Although a 200k model was trained, it did not improve over the 100k model. Therefore, the 100k model remains the best policy for the next generalization experiment.

### Scientific Question

Does a PPO agent trained on randomized 5x5 obstacle layouts generalize to a larger 10x10 grid with randomized obstacles?

### Planned Setup

Training environment:

* 5x5 grid;
* randomized reachable obstacles;
* 3 obstacles;
* sparse reward.

Evaluation environment:

* 10x10 grid;
* randomized reachable obstacles;
* 3 obstacles;
* same observation size.

Important note:

The number of obstacles should remain fixed at 3, because the observation space includes obstacle coordinates. Changing the number of obstacles would change the observation dimension and would not be compatible with the already trained PPO model.

### Planned Tasks

1. Create an evaluation script for 10x10 randomized obstacles.
2. Load the best randomized-obstacle model: `models/ppo_foraging_random_obstacles_100k.zip`.
3. Evaluate over 1000 seeded randomized environments.
4. Compare performance against the 5x5 randomized-obstacle evaluation.
5. Record results in `EXPERIMENT_LOG.md`.
6. Update `PROJECT_CONTEXT.md` and `ROADMAP.md`.
7. Commit and push the completed experiment.

### Expected Outcome

Performance may decrease due to the larger grid and longer required paths.

This experiment extends the earlier grid-size generalization study to the more complex randomized-obstacle setting.

---

## Day 11 - Multi-Agent Environment

Status: planned

### Goal

Create a simple multi-agent version of the foraging environment.

### Scientific Question

Can a centralized PPO policy control multiple agents in a shared foraging environment?

### Initial Setup

* 2 agents;
* 1 food source;
* 5x5 grid;
* no obstacles initially;
* centralized control;
* joint action space.

### Planned Environment File

`env/multi_agent_foraging_env.py`

### Planned Observation Space

Possible observation:

```text
agent1_x, agent1_y,
agent2_x, agent2_y,
food_x, food_y
```

### Planned Action Space

For 2 agents, each with 4 actions:

```text
4 x 4 = 16 joint actions
```

### Planned Tasks

1. Implement the multi-agent environment.
2. Add rendering for two agents.
3. Add tests for reset, step, boundaries, and food collection.
4. Create a random multi-agent baseline.
5. Train PPO using centralized control.
6. Evaluate performance.

---

## Day 12 - Multi-Agent Evaluation and Scalability

Status: planned

### Goal

Evaluate whether adding agents improves foraging efficiency.

### Scientific Questions

* Does adding a second agent improve success rate?
* Does adding a second agent reduce average episode length?
* Does the multi-agent setup create coordination challenges?
* Can the approach scale beyond two agents?

### Planned Experiments

| Experiment | Agents | Policy | Environment |
|---|---:|---|---|
| Random baseline | 2 | Random | 5x5 |
| PPO multi-agent | 2 | PPO | 5x5 |
| Optional scalability | 3 | PPO | 5x5 |

### Planned Metrics

* success rate;
* average reward;
* average episode length;
* possibly collected resources per agent;
* qualitative trajectory examples.

---

## Final Reporting Phase

Status: planned

### Goal

Prepare the final project report and organize all results.

### Planned Tasks

1. Update `README.md`.
2. Update `PROJECT_CONTEXT.md`.
3. Update `EXPERIMENT_LOG.md`.
4. Generate final plots.
5. Create final summary tables.
6. Add trajectory or behavior analysis.
7. Discuss limitations.
8. Prepare final conclusions.

### Important Report Sections

The final report should include:

1. Introduction
2. Environment design
3. Reinforcement learning method
4. Experimental setup
5. Results
6. Sensitivity analysis
7. Oracle baseline and failure-case analysis
8. Multi-agent extension
9. Discussion
10. Limitations
11. Conclusion

---

## Priority List for a High-Quality Final Project

### Essential

* Complete random-obstacle 10x10 generalization.
* Implement a working 2-agent environment.
* Compare multi-agent PPO against a random baseline.
* Generate final plots and summary tables.
* Write clear scientific conclusions.

### Completed Quality Improvements

* Completed 200k randomized-obstacle training sensitivity analysis.
* Added BFS oracle baseline for environment solvability validation.
* Added PPO failure-case analysis for learned-behavior interpretation.

### Strongly Recommended

* Add trajectory examples for successful PPO episodes.
* Analyze success and failure cases in the final report.
* Explain the learned behavior qualitatively.
* Keep all experiments reproducible and documented.

### Optional Extensions

* 3-agent scalability.
* Reward shaping.
* Invalid-action penalty.
* Obstacle density variation.
* Larger-grid training from scratch.

---

## Current Best Models

Best fixed-obstacle model:

`models/ppo_foraging_obstacles_state.zip`

Best randomized-obstacle model:

`models/ppo_foraging_random_obstacles_100k.zip`

Additional randomized-obstacle sensitivity model:

`models/ppo_foraging_random_obstacles_200k.zip`

The 200k model was useful for sensitivity analysis, but it did not improve over the 100k model.

---

## Current Main Conclusion

The current results show that state representation and environment randomization both play important roles in PPO performance.

Adding obstacle coordinates to the observation space improved fixed-obstacle performance from 70% to 87%.

Randomized obstacle training improved generalization from 65.50% to 76.30%, and reduced the average episode length from 19.59 to 14.74 steps.

Increasing randomized-obstacle training from 100k to 200k did not improve performance, suggesting that the current PPO setup reaches a plateau around 100k timesteps under the sparse reward and observation design.

The BFS oracle achieved 100% success on the same randomized-obstacle environments, confirming that the evaluation environments were solvable. Failure-case analysis showed that PPO failures were dominated by repeated no-move behavior, suggesting that the model sometimes becomes trapped in ineffective deterministic actions rather than failing because of impossible task configurations.

Further work will test whether larger-grid evaluation and multi-agent extensions improve robustness and scalability.
