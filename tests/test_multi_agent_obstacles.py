import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3.common.env_checker import check_env

from env.multi_agent_foraging_env import MultiAgentForagingEnv


FIXED_OBSTACLES = [
    (2, 1),
    (2, 2),
    (2, 3),
]


def test_multi_agent_obstacle_env_check():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

    check_env(env, warn=True)


def test_multi_agent_obstacle_observation_shape():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

    obs, info = env.reset(seed=42)

    assert obs.shape == (12,)
    assert env.action_space.n == 16


def test_obstacles_are_in_observation():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

    obs, info = env.reset(seed=42)

    obstacle_part = obs[-6:]

    expected_obstacles = np.array(
        [2, 1, 2, 2, 2, 3],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(obstacle_part, expected_obstacles)


def test_agents_do_not_spawn_on_obstacles():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

    obstacle_set = set(FIXED_OBSTACLES)

    for seed in range(100):
        obs, info = env.reset(seed=seed)

        for agent_position in env.agent_positions:
            assert tuple(agent_position) not in obstacle_set


def test_food_does_not_spawn_on_obstacles():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

    obstacle_set = set(FIXED_OBSTACLES)

    for seed in range(100):
        obs, info = env.reset(seed=seed)

        assert tuple(env.food_pos) not in obstacle_set


def test_agent_is_blocked_by_obstacle():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

    env.reset(seed=42)

    env.agent_positions = [
        np.array([2, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
    ]

    env.food_pos = np.array([4, 4], dtype=np.int32)

    # Agent A tries to move down into obstacle (2, 1).
    # Agent B moves right.
    # Encoding: joint_action = action_A + 4 * action_B
    # down = 1, right = 3
    joint_action = 1 + 4 * 3

    obs, reward, terminated, truncated, info = env.step(joint_action)

    assert np.array_equal(
        env.agent_positions[0],
        np.array([2, 0], dtype=np.int32),
    )

    assert not np.array_equal(
        env.agent_positions[0],
        np.array([2, 1], dtype=np.int32),
    )