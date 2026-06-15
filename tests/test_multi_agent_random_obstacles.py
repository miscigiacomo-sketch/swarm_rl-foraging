import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from env.multi_agent_foraging_env import MultiAgentForagingEnv


def parse_observation(obs, num_agents):
    agent_positions = []

    for agent_index in range(num_agents):
        x = int(obs[2 * agent_index])
        y = int(obs[2 * agent_index + 1])
        agent_positions.append((x, y))

    food_start_index = 2 * num_agents

    food_pos = (
        int(obs[food_start_index]),
        int(obs[food_start_index + 1]),
    )

    obstacle_positions = []

    obstacle_start_index = food_start_index + 2

    for index in range(obstacle_start_index, len(obs), 2):
        obstacle_positions.append(
            (
                int(obs[index]),
                int(obs[index + 1]),
            )
        )

    return agent_positions, food_pos, obstacle_positions


def is_food_reachable_from_any_agent(agent_positions, food_pos, obstacles, grid_size):
    obstacle_set = set(obstacles)

    directions = [
        (0, -1),  # up
        (0, 1),   # down
        (-1, 0),  # left
        (1, 0),   # right
    ]

    for start in agent_positions:
        queue = deque([start])
        visited = {start}

        while queue:
            current_x, current_y = queue.popleft()

            if (current_x, current_y) == food_pos:
                return True

            for dx, dy in directions:
                next_x = current_x + dx
                next_y = current_y + dy
                next_position = (next_x, next_y)

                inside_grid = (
                    0 <= next_x < grid_size
                    and 0 <= next_y < grid_size
                )

                if not inside_grid:
                    continue

                if next_position in obstacle_set:
                    continue

                if next_position in visited:
                    continue

                visited.add(next_position)
                queue.append(next_position)

    return False


def test_multi_agent_random_obstacle_env_check():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    check_env(env, warn=True)


def test_random_obstacle_observation_shape():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    obs, info = env.reset(seed=42)

    assert obs.shape == (12,)
    assert env.action_space.n == 16


def test_random_obstacles_do_not_overlap():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    for seed in range(100):
        obs, info = env.reset(seed=seed)

        _, _, obstacle_positions = parse_observation(
            obs=obs,
            num_agents=2,
        )

        assert len(obstacle_positions) == 3
        assert len(set(obstacle_positions)) == len(obstacle_positions)


def test_agents_do_not_spawn_on_random_obstacles():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    for seed in range(100):
        obs, info = env.reset(seed=seed)

        agent_positions, _, obstacle_positions = parse_observation(
            obs=obs,
            num_agents=2,
        )

        obstacle_set = set(obstacle_positions)

        for agent_position in agent_positions:
            assert agent_position not in obstacle_set


def test_food_does_not_spawn_on_random_obstacles():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    for seed in range(100):
        obs, info = env.reset(seed=seed)

        _, food_pos, obstacle_positions = parse_observation(
            obs=obs,
            num_agents=2,
        )

        assert food_pos not in set(obstacle_positions)


def test_food_does_not_spawn_on_agents():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    for seed in range(100):
        obs, info = env.reset(seed=seed)

        agent_positions, food_pos, _ = parse_observation(
            obs=obs,
            num_agents=2,
        )

        assert food_pos not in set(agent_positions)


def test_random_obstacle_food_is_reachable():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    for seed in range(1000):
        obs, info = env.reset(seed=seed)

        agent_positions, food_pos, obstacle_positions = parse_observation(
            obs=obs,
            num_agents=2,
        )

        reachable = is_food_reachable_from_any_agent(
            agent_positions=agent_positions,
            food_pos=food_pos,
            obstacles=obstacle_positions,
            grid_size=env.grid_size,
        )

        assert reachable


def test_fixed_and_random_obstacles_cannot_be_combined():
    with pytest.raises(ValueError):
        MultiAgentForagingEnv(
            grid_size=5,
            num_agents=2,
            max_steps=50,
            obstacles=[(2, 1), (2, 2), (2, 3)],
            random_obstacles=True,
            num_obstacles=3,
        )