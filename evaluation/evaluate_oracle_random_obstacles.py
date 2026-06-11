import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from env.foraging_env import ForagingEnv


ACTIONS = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
}


def parse_observation(obs):
    agent_pos = tuple(obs[0:2].astype(int))
    food_pos = tuple(obs[2:4].astype(int))

    obstacles = []
    for i in range(4, len(obs), 2):
        obstacles.append(tuple(obs[i:i + 2].astype(int)))

    return agent_pos, food_pos, obstacles


def find_shortest_path(agent_pos, food_pos, obstacles, grid_size):
    obstacle_set = set(obstacles)

    queue = deque()
    queue.append((agent_pos, []))

    visited = set()
    visited.add(agent_pos)

    while queue:
        current_pos, path = queue.popleft()

        if current_pos == food_pos:
            return path

        current_x, current_y = current_pos

        for action, (dx, dy) in ACTIONS.items():
            next_pos = (current_x + dx, current_y + dy)
            next_x, next_y = next_pos

            if next_x < 0 or next_x >= grid_size:
                continue

            if next_y < 0 or next_y >= grid_size:
                continue

            if next_pos in obstacle_set:
                continue

            if next_pos in visited:
                continue

            visited.add(next_pos)
            queue.append((next_pos, path + [action]))

    return None


def evaluate_oracle(episodes=1000, grid_size=5):
    env = ForagingEnv(grid_size=grid_size, random_obstacles=True)

    successes = 0
    path_lengths = []
    episode_lengths = []
    failed_seeds = []

    for seed in range(episodes):
        obs, info = env.reset(seed=seed)

        agent_pos, food_pos, obstacles = parse_observation(obs)

        path = find_shortest_path(
            agent_pos=agent_pos,
            food_pos=food_pos,
            obstacles=obstacles,
            grid_size=grid_size,
        )

        if path is None:
            failed_seeds.append(seed)
            continue

        done = False
        total_reward = 0
        steps = 0

        for action in path:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if done:
                break

        if total_reward > 0:
            successes += 1

        path_lengths.append(len(path))
        episode_lengths.append(steps)

    success_rate = successes / episodes
    average_path_length = np.mean(path_lengths)
    average_episode_length = np.mean(episode_lengths)

    print("===== BFS ORACLE RANDOM OBSTACLE EVALUATION =====")
    print(f"Episodes: {episodes}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Average Shortest Path Length: {average_path_length:.2f}")
    print(f"Average Episode Length: {average_episode_length:.2f}")
    print(f"Failed Seeds: {failed_seeds}")


def main():
    evaluate_oracle(episodes=1000, grid_size=5)


if __name__ == "__main__":
    main()