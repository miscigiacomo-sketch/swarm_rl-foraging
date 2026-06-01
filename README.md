# Swarm RL Foraging

This project investigates swarm foraging behavior using Reinforcement Learning.

The goal is to build a simplified 2D environment where agents learn to collect resources efficiently. The project starts with a single-agent setup and will later be extended toward multi-agent swarm behavior.

## Project Motivation

Many real-world systems require multiple autonomous agents to coordinate their actions efficiently. Examples include swarm robotics, warehouse automation, drone coordination, and search-and-rescue systems.

Although the environment is simplified, it captures important coordination problems such as exploration, resource collection, efficiency, and scalability.

## Current Features

- Custom Gymnasium environment
- 2D grid world
- Random food spawning
- Random agent baseline
- PPO training using Stable-Baselines3
- Trained agent evaluation
- Quantitative evaluation metrics

## Environment

The environment is a 2D grid world.

The observation is:

```text
[agent_x, agent_y, food_x, food_y]
```

The action space is discrete:

```text
0 = up
1 = down
2 = left
3 = right
```

The reward structure is:

```text
+1 when the agent collects the food
0 otherwise
```

## Current Results

The trained PPO agent was evaluated over 100 episodes.

```text
Success Rate: 100%
Average Reward: 1.00
Average Episode Length: 3.87
```

These results show that the trained PPO agent can reliably collect the food in the current environment.

## Tools and Libraries

- Python
- Gymnasium
- Stable-Baselines3
- PPO
- NumPy
- Matplotlib
- Pygame-ce
- PyTorch

## Project Structure

```text
swarm_rl/
│
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
│   └── evaluate_model.py
│
├── requirements.txt
├── mini_grid.py
├── test_setup.py
└── README.md
```

## Next Steps

- Compare PPO against a random baseline
- Increase environment size
- Add multiple agents
- Study scalability
- Analyze cooperative behavior
- Extend the environment toward swarm foraging