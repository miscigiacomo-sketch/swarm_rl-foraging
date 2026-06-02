# PROJECT CONTEXT - SWARM RL FORAGING

## Project Overview

This project is being developed for the BIO course.

The objective is to study swarm foraging behavior using Reinforcement Learning.

The current implementation focuses on a single-agent environment as a foundation before extending to multi-agent swarm systems.

The project is intentionally 2D.

The goal is not to build a realistic simulator but to investigate learning, resource collection efficiency, coordination, and scalability.

---

# Development Status

Current status:

Day 1: COMPLETED
Day 2: COMPLETED
Day 3: COMPLETED
Day 4: COMPLETED

---

# Technologies

- Python
- Gymnasium
- Stable-Baselines3
- PPO
- NumPy
- Matplotlib
- PyTorch
- Pygame-ce
- GitHub
- GitHub Desktop
- VS Code

---

# Project Structure

swarm_rl/

├── env/
│   ├── __init__.py
│   └── foraging_env.py
│
├── tests/
│   └── random_agent.py
│
├── train/
│   └── train_ppo.py
│
├── evaluation/
│   ├── test_trained_agent.py
│   ├── evaluate_model.py
│   ├── evaluate_random_agent.py
│   └── compare_agents.py
│
├── models/
│   └── ppo_foraging.zip
│
├── README.md
├── requirements.txt
├── test_setup.py
├── mini_grid.py
└── PROJECT_CONTEXT.md

---

# Environment Description

Grid world environment.

Current grid size:

5 x 5

Observation:

[agent_x, agent_y, food_x, food_y]

Action Space:

0 = up
1 = down
2 = left
3 = right

Reward:

+1 when the agent reaches the food
0 otherwise

Episode termination:

- food collected
OR
- maximum number of steps reached

Maximum steps:

50

Food position:

Randomly generated every episode.

---

# Implemented Files

## env/foraging_env.py

Custom Gymnasium environment.

Contains:

- reset()
- step()
- render()

Rendering is currently text-based.

---

## tests/random_agent.py

Runs a random-action agent.

Used to verify environment behavior.

---

## train/train_ppo.py

Trains a PPO agent.

Current configuration:

PPO(
    "MlpPolicy",
    env,
    verbose=1
)

Training:

10000 timesteps

Model saved as:

models/ppo_foraging

---

## evaluation/test_trained_agent.py

Loads trained PPO model.

Runs a single episode.

Used to visually verify behavior.

---

## evaluation/evaluate_model.py

Evaluates PPO over 100 episodes.

Metrics:

- Success Rate
- Average Reward
- Average Episode Length

Current results:

Success Rate: 100%
Average Reward: 1.00
Average Episode Length: 3.87

---

## evaluation/evaluate_random_agent.py

Evaluates random baseline over 100 episodes.

Current results:

Success Rate: 63%
Average Reward: 0.63
Average Episode Length: 30.55

Results may vary slightly due to randomness.

---

## evaluation/compare_agents.py

Compares:

- Random Agent
- PPO Agent

Current comparison:

Random:
Success Rate = 63%
Average Reward = 0.63
Average Steps = 30.55

PPO:
Success Rate = 100%
Average Reward = 1.00
Average Steps = 4.26

Main conclusion:

PPO significantly outperforms the random baseline.

---

# GitHub Information

Repository:

swarm_rl-foraging

Owner:

miscigiacomo-sketch

GitHub Desktop is configured.

Basic workflow:

1. Edit code in VS Code
2. Test code
3. Commit in GitHub Desktop
4. Push to GitHub

Rule:

Whenever a feature works:
Commit + Push

---

# Professor Presentation Summary

Current pitch:

This project investigates resource collection behavior in a custom Gymnasium environment.

A PPO reinforcement learning agent is trained to locate and collect resources efficiently.

The project serves as a foundation for future extensions toward swarm intelligence and multi-agent coordination.

Current results show that PPO achieves perfect success rates while significantly outperforming a random baseline.

Potential applications:

- Swarm robotics
- Warehouse automation
- Drone coordination
- Search and rescue
- Distributed autonomous systems

---

# Roadmap

## Day 5

Goal:

Visualization and report-quality outputs.

Tasks:

- Generate comparison plots
- Generate performance figures
- Create results folder
- Prepare report material

Potential experiment:

Increase grid size:

5x5 -> 10x10

Evaluate PPO performance degradation.

---

## Day 6

Environment complexity.

Potential additions:

- Larger maps
- Multiple food sources
- Obstacles
- Sparse rewards

---

## Day 7

Multi-agent extension.

Potential additions:

- Multiple agents
- Shared resources
- Cooperation
- Emergent behavior

---

# How Future Chats Should Behave

Act as:

- Reinforcement Learning mentor
- Python mentor
- Research project supervisor

Requirements:

- Explain every step before implementing it.
- Always explain WHY a change is useful.
- Prefer educational explanations.
- Keep code professional and fully in English.
- Maintain compatibility with current project structure.
- Use GitHub best practices.
- Focus on producing report-quality results.

The user is still learning RL and GitHub, so explanations should be detailed but practical.