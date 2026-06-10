import os
import sys
from collections import deque

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.foraging_env import ForagingEnv


def is_reachable(agent_pos, food_pos, obstacles, grid_size):
    """
    Check whether there is at least one valid path from the agent
    to the food position using breadth-first search.
    """
    start = tuple(agent_pos)
    goal = tuple(food_pos)
    obstacle_set = {tuple(obstacle) for obstacle in obstacles}

    queue = deque([start])
    visited = {start}

    directions = [
        (0, -1),  # up
        (0, 1),   # down
        (-1, 0),  # left
        (1, 0),   # right
    ]

    while queue:
        current_x, current_y = queue.popleft()

        if (current_x, current_y) == goal:
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


def test_random_obstacle_reachability():
    env = ForagingEnv(random_obstacles=True)

    episodes = 1000
    reachable_count = 0
    unreachable_count = 0

    for episode in range(episodes):
        obs, info = env.reset()

        agent_pos = obs[0:2]
        food_pos = obs[2:4]
        obstacle_positions = obs[4:].reshape(-1, 2)

        reachable = is_reachable(
            agent_pos=agent_pos,
            food_pos=food_pos,
            obstacles=obstacle_positions,
            grid_size=env.grid_size,
        )

        if reachable:
            reachable_count += 1
        else:
            unreachable_count += 1
            print(f"\nUnreachable environment found at episode {episode + 1}")
            env.render()

    reachability_rate = reachable_count / episodes

    print("\n===== RANDOM OBSTACLE REACHABILITY TEST =====")
    print(f"Episodes tested: {episodes}")
    print(f"Reachable environments: {reachable_count}")
    print(f"Unreachable environments: {unreachable_count}")
    print(f"Reachability Rate: {reachability_rate * 100:.2f}%")

    assert unreachable_count == 0, "Some randomly generated environments are unreachable."


if __name__ == "__main__":
    test_random_obstacle_reachability()