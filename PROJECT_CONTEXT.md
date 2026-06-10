# PROJECT CONTEXT - SWARM RL FORAGING

## Project Overview

This project is being developed for the BIO course.

The objective is to study swarm foraging behavior using Reinforcement Learning.

The current implementation focuses on a single-agent environment as a foundation before extending to multi-agent swarm systems.

The project is intentionally 2D.

The goal is not to build a realistic simulator but to investigate learning, resource collection efficiency, coordination, scalability, and state representation effects in reinforcement learning systems.

---

# Development Status

Current status:

Day 1: COMPLETED
Day 2: COMPLETED
Day 3: COMPLETED
Day 4: COMPLETED
Day 5: COMPLETED
Day 6: COMPLETED

---

# Technologies

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

# Project Structure

swarm_rl/

├── env/
│   ├── **init**.py
│   └── foraging_env.py
│
├── tests/
│   ├── random_agent.py
│   ├── test_large_grid.py
│   └── test_obstacles.py
│
├── train/
│   ├── train_ppo.py
│   └── train_ppo_obstacles.py
│
├── evaluation/
│   ├── test_trained_agent.py
│   ├── test_obstacle_agent.py
│   ├── evaluate_model.py
│   ├── evaluate_random_agent.py
│   ├── evaluate_grid_size.py
│   ├── evaluate_obstacle_model.py
│   ├── compare_agents.py
│   └── generate_plots.py
│
├── models/
│   ├── ppo_foraging.zip
│   ├── ppo_foraging_obstacles.zip
│   ├── ppo_foraging_obstacles_50k.zip
│   └── ppo_foraging_obstacles_state.zip
│
├── results/
│   ├── success_rate_comparison.png
│   ├── average_steps_comparison.png
│   ├── grid_size_success_rate.png
│   ├── grid_size_average_steps.png
│   ├── obstacle_success_rate.png
│   ├── obstacle_average_steps.png
│   └── report_metrics.txt
│
├── README.md
├── requirements.txt
├── PROJECT_CONTEXT.md

---

# Environment Description

Grid world environment.

Current grid size:

5 x 5

Action Space:

0 = up
1 = down
2 = left
3 = right

Current Observation Space:

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

Observation size:

10

Reward:

+1 when the agent reaches the food
0 otherwise

Episode termination:

* food collected
* maximum number of steps reached

Maximum steps:

50

Food position:

Randomly generated every episode.

Obstacles:

Three static obstacles:

[2,1]
[2,2]
[2,3]

Obstacle collisions are blocked.

---

# Implemented Features

## Day 1 - Environment

Implemented:

* Custom Gymnasium environment
* Reset function
* Step function
* Text rendering
* Random food spawning

---

## Day 2 - PPO Training

Implemented:

* PPO training pipeline
* Stable-Baselines3 integration
* Model saving and loading

Training:

10000 timesteps

Model:

models/ppo_foraging.zip

---

## Day 3 - Evaluation & Baseline

Implemented:

* PPO evaluation
* Random baseline evaluation
* Agent comparison

Results:

Random Agent:

Success Rate: 63%
Average Reward: 0.63
Average Steps: 30.55

PPO Agent:

Success Rate: 100%
Average Reward: 1.00
Average Steps: 4.26

Conclusion:

PPO significantly outperforms the random baseline.

---

## Day 4 - Generalization

Experiment:

Train PPO on 5x5 environment.

Evaluate on 10x10 environment.

Results:

Success Rate: 96%
Average Reward: 0.96
Average Episode Length: 12.88

Conclusion:

The PPO policy generalizes reasonably well to larger environments, although efficiency decreases.

---

## Day 5 - Reporting & Visualization

Implemented:

* Results folder
* Automatic plot generation
* Metrics report generation
* Performance comparison figures

Generated files:

* success_rate_comparison.png
* average_steps_comparison.png
* grid_size_success_rate.png
* grid_size_average_steps.png
* report_metrics.txt

---

## Day 6 - Obstacles & State Representation

Implemented:

* Static obstacles
* Obstacle rendering (X)
* Collision handling
* Obstacle testing
* PPO training with obstacles
* State augmentation experiment

New Observation Space:

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

---

# Experimental Results

## Baseline PPO (No Obstacles)

Success Rate: 100%
Average Reward: 1.00
Average Episode Length: 3.86

---

## Random Agent

Success Rate: 63%
Average Reward: 0.63
Average Episode Length: 30.55

---

## Grid Size Generalization (10x10)

Success Rate: 96%
Average Reward: 0.96
Average Episode Length: 12.88

---

## Obstacle Experiment

### PPO + Obstacles (20k)

Success Rate: 79%
Average Reward: 0.79
Average Episode Length: 13.34

### PPO + Obstacles (50k)

Success Rate: 70%
Average Reward: 0.70
Average Episode Length: 17.35

### PPO + Obstacles + State Augmentation (50k)

Success Rate: 87%
Average Reward: 0.87
Average Episode Length: 9.94

---

# Main Scientific Findings

Finding 1:

PPO significantly outperforms a random policy.

Finding 2:

The trained PPO agent generalizes well from a 5x5 environment to a larger 10x10 environment.

Finding 3:

Adding obstacles substantially increases task difficulty.

Finding 4:

Increasing training timesteps alone did not improve obstacle performance.

Finding 5:

Providing obstacle information in the observation space improved performance from 70% to 87%.

Main Conclusion:

State representation was a more important factor than simply increasing training duration.

This is currently the most important scientific result obtained in the project.

---

# GitHub Information

Repository:

swarm_rl-foraging

Owner:

miscigiacomo-sketch

GitHub Desktop is configured.

Workflow:

1. Edit code in VS Code
2. Test code
3. Commit in GitHub Desktop
4. Push to GitHub

Rule:

Whenever a feature works:

Commit + Push

Latest completed commit:

Improve obstacle navigation through state augmentation

---

# Professor Presentation Summary

Current pitch:

This project investigates resource collection behavior in a custom Gymnasium environment using Reinforcement Learning.

A PPO agent is trained to locate and collect resources efficiently under increasing environmental complexity.

Experiments include:

* Random baseline comparison
* Environment scaling
* Obstacle navigation
* State representation analysis

A key finding is that providing obstacle information within the state representation significantly improves PPO performance, demonstrating the importance of observation design in reinforcement learning systems.

Potential applications:

* Swarm robotics
* Warehouse automation
* Drone coordination
* Search and rescue
* Distributed autonomous systems

---

# Roadmap

## Day 7

Randomized Obstacles

Goals:

* Generate obstacle positions randomly
* Prevent obstacle overlap
* Prevent food spawning inside obstacles
* Prevent obstacle spawning on the agent
* Evaluate PPO generalization

Hypothesis:

A PPO agent trained on fixed obstacles may not generalize well to randomized environments.

---

## Day 8

Multi-Agent Extension

Goals:

* Two agents
* Shared environment
* Shared resource collection
* Cooperation experiments
* Scalability analysis

---

# How Future Chats Should Behave

Act as:

* Reinforcement Learning mentor
* Python mentor
* Research project supervisor

Requirements:

* Explain every step before implementation.
* Always explain WHY a change is useful.
* Prefer educational explanations.
* Keep code professional and fully in English.
* Maintain compatibility with the current project structure.
* Use GitHub best practices.
* Focus on producing report-quality results.
* Track experiments and scientific conclusions carefully.

The user is still learning Reinforcement Learning, GitHub, and software engineering practices, so explanations should be detailed, educational, and practical.
