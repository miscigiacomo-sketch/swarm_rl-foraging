# PROJECT_CONTEXT.md

# Swarm RL Project Context

## Project Goal

Build a Reinforcement Learning environment for swarm foraging using Python.

Final objective:
- multi-agent swarm
- agents collect food in a 2D environment
- PPO training
- sensitivity analysis
- evaluation and report

---

# Current Status

## Day 1 Completed

### Installed Software

- Python 3.14.5
- VS Code
- OpenAI Codex extension

### Python Environment

Virtual environment created:

```text
venv/
```

Activation command (Windows):

```cmd
venv\Scripts\activate.bat
```

### Installed Libraries

- gymnasium
- stable-baselines3
- torch
- matplotlib
- pygame-ce
- numpy

### Important Note

`pygame` had compatibility issues with Python 3.14.

Solution:

```cmd
pip install pygame-ce
```

---

# Current Project Structure

```text
swarm_rl/
│
├── venv/
├── requirements.txt
├── test_setup.py
└── mini_grid.py
```

---

# Reinforcement Learning Concepts Learned

## Environment

The environment is:
- the world where the agent lives
- receives actions
- updates the world state
- gives rewards

## Agent

The agent:
- observes the environment
- chooses actions
- tries to maximize reward

## State Transition

Core RL idea:

```text
state → action → new state
```

---

# Planned Gymnasium Environment

Features:
- 5x5 grid
- one agent
- one food item
- discrete actions
- rewards
- episode truncation
- random-agent testing

Actions:
- 0 = up
- 1 = down
- 2 = left
- 3 = right

Reward:
- +1 when food is collected

Episode termination:
- when food is collected
- or max steps reached

---

# Real-World Applications

Possible applications of the project:

- swarm robotics
- warehouse automation
- drone coordination
- search-and-rescue systems
- distributed autonomous systems

The project focuses on emergent cooperative behavior and swarm intelligence.

---

# Important Design Choice

The environment is intentionally 2D.

Reason:
- faster experimentation
- easier debugging
- clearer visualization
- focus on cooperation and RL dynamics instead of graphics

The goal is not realistic 3D simulation, but studying multi-agent coordination and learning.

---

# DAY2_SUMMARY.md

# Day 2 Summary — Swarm RL Project

## Main Goal

The objective of Day 2 was to:
- understand the basics of Reinforcement Learning environments
- build a simple grid world
- create a Gymnasium-compatible RL environment
- test the environment with a random agent
- add basic rendering

---

# Part 1 — mini_grid.py

Created file:

```text
mini_grid.py
```

Purpose:
Build a simple manual grid world without Gymnasium.

Implemented:
- agent coordinates
- food coordinates
- keyboard movement
- rewards
- episode loop

---

# Part 2 — Gymnasium Environment

Created file:

```text
env/foraging_env.py
```

Purpose:
Convert the manual grid world into a real RL environment compatible with Gymnasium and PPO.

---

# Environment Structure

## Class Definition

```python
class ForagingEnv(gym.Env):
```

This defines a custom RL environment.

---

# Action Space

```python
self.action_space = spaces.Discrete(4)
```

Actions:
- 0 = up
- 1 = down
- 2 = left
- 3 = right

---

# Observation Space

```python
[agent_x, agent_y, food_x, food_y]
```

Example:

```text
[1,0,2,0]
```

means:
- agent at (1,0)
- food at (2,0)

---

# Reward System

```python
reward = 1
```

when:

```python
agent reaches food
```

otherwise:

```python
reward = 0
```

---

# Part 3 — Random Agent Test

Created file:

```text
tests/random_agent.py
```

Purpose:
Test whether the environment works correctly.

The random agent chooses actions using:

```python
action = env.action_space.sample()
```

This is NOT learning yet.

---

# Part 4 — Rendering

Added function:

```python
def render(self):
```

Purpose:
Display the environment visually inside the terminal.

Symbols:
- A = agent
- F = food
- . = empty cell

Example:

```text
A . F . .
. . . . .
. . . . .
```

---

# Current Project Structure

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
├── mini_grid.py
├── test_setup.py
├── requirements.txt
└── venv/
```

---

# Current Project Status

Completed:
- basic RL logic
- custom Gymnasium environment
- random agent testing
- terminal rendering

---

# Next Steps (Day 3)

Planned:
- random food spawning
- improved rendering
- PPO training
- learning agent instead of random agent

---

# Important Understanding

The project is focused on:
- reinforcement learning
- swarm intelligence
- emergent coordination
- cooperative behavior

The simplified 2D environment is intentional and appropriate for experimentation and analysis.

