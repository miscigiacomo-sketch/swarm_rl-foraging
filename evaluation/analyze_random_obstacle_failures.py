import os
import sys
from collections import Counter, deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO

from env.foraging_env import ForagingEnv


ACTIONS = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
}

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
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


def format_grid(grid_size, agent_pos, food_pos, obstacles, trajectory=None):
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    for obs_x, obs_y in obstacles:
        grid[obs_y][obs_x] = "X"

    if trajectory is not None:
        for x, y in trajectory:
            if (x, y) != agent_pos and (x, y) != food_pos and (x, y) not in obstacles:
                grid[y][x] = "*"

    food_x, food_y = food_pos
    grid[food_y][food_x] = "F"

    agent_x, agent_y = agent_pos
    grid[agent_y][agent_x] = "A"

    return "\n".join(" ".join(row) for row in grid)


def analyze_failures(
    model_path="models/ppo_foraging_random_obstacles_100k.zip",
    episodes=1000,
    grid_size=5,
    max_cases_to_save=10,
):
    env = ForagingEnv(grid_size=grid_size, random_obstacles=True)
    model = PPO.load(model_path)

    successes = 0
    failures = 0
    failure_cases = []

    total_no_move_steps = 0
    total_loop_steps = 0

    for seed in range(episodes):
        obs, info = env.reset(seed=seed)

        initial_agent_pos, food_pos, obstacles = parse_observation(obs)
        oracle_path = find_shortest_path(
            agent_pos=initial_agent_pos,
            food_pos=food_pos,
            obstacles=obstacles,
            grid_size=grid_size,
        )

        done = False
        total_reward = 0
        steps = 0

        positions = [initial_agent_pos]
        actions_taken = []
        no_move_steps = 0

        while not done:
            previous_agent_pos, _, _ = parse_observation(obs)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)

            current_agent_pos, _, _ = parse_observation(obs)

            if current_agent_pos == previous_agent_pos:
                no_move_steps += 1

            actions_taken.append(action)
            positions.append(current_agent_pos)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        if total_reward > 0:
            successes += 1
        else:
            failures += 1

            position_counts = Counter(positions)
            repeated_positions = {
                pos: count for pos, count in position_counts.items() if count > 1
            }

            loop_steps = sum(count - 1 for count in position_counts.values() if count > 1)

            total_no_move_steps += no_move_steps
            total_loop_steps += loop_steps

            if len(failure_cases) < max_cases_to_save:
                failure_cases.append(
                    {
                        "seed": seed,
                        "initial_agent_pos": initial_agent_pos,
                        "final_agent_pos": positions[-1],
                        "food_pos": food_pos,
                        "obstacles": obstacles,
                        "steps": steps,
                        "oracle_shortest_path_length": len(oracle_path)
                        if oracle_path is not None
                        else None,
                        "no_move_steps": no_move_steps,
                        "loop_steps": loop_steps,
                        "repeated_positions": repeated_positions,
                        "actions_taken": actions_taken,
                        "positions": positions,
                    }
                )

    success_rate = successes / episodes
    failure_rate = failures / episodes

    os.makedirs("results", exist_ok=True)
    output_path = "results/random_obstacle_failure_cases.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("RANDOM OBSTACLE PPO FAILURE CASE ANALYSIS\n")
        file.write("=========================================\n\n")
        file.write(f"Model: {model_path}\n")
        file.write(f"Episodes: {episodes}\n")
        file.write(f"Grid Size: {grid_size}x{grid_size}\n")
        file.write(f"Successes: {successes}\n")
        file.write(f"Failures: {failures}\n")
        file.write(f"Success Rate: {success_rate * 100:.2f}%\n")
        file.write(f"Failure Rate: {failure_rate * 100:.2f}%\n\n")

        if failures > 0:
            file.write(f"Average no-move steps per failed episode: {total_no_move_steps / failures:.2f}\n")
            file.write(f"Average repeated-position steps per failed episode: {total_loop_steps / failures:.2f}\n\n")

        file.write("Saved Failure Cases\n")
        file.write("===================\n\n")

        for index, case in enumerate(failure_cases, start=1):
            file.write(f"Failure Case {index}\n")
            file.write("--------------------\n")
            file.write(f"Seed: {case['seed']}\n")
            file.write(f"Initial Agent Position: {case['initial_agent_pos']}\n")
            file.write(f"Final Agent Position: {case['final_agent_pos']}\n")
            file.write(f"Food Position: {case['food_pos']}\n")
            file.write(f"Obstacles: {case['obstacles']}\n")
            file.write(f"Episode Steps: {case['steps']}\n")
            file.write(f"Oracle Shortest Path Length: {case['oracle_shortest_path_length']}\n")
            file.write(f"No-Move Steps: {case['no_move_steps']}\n")
            file.write(f"Repeated-Position Steps: {case['loop_steps']}\n")
            file.write(f"Repeated Positions: {case['repeated_positions']}\n\n")

            action_names = [ACTION_NAMES[action] for action in case["actions_taken"]]
            file.write(f"Actions Taken:\n{action_names}\n\n")

            file.write("Initial Grid:\n")
            file.write(
                format_grid(
                    grid_size=grid_size,
                    agent_pos=case["initial_agent_pos"],
                    food_pos=case["food_pos"],
                    obstacles=case["obstacles"],
                )
            )
            file.write("\n\n")

            file.write("Trajectory Grid:\n")
            file.write(
                format_grid(
                    grid_size=grid_size,
                    agent_pos=case["final_agent_pos"],
                    food_pos=case["food_pos"],
                    obstacles=case["obstacles"],
                    trajectory=case["positions"],
                )
            )
            file.write("\n\n")

    print("===== RANDOM OBSTACLE PPO FAILURE CASE ANALYSIS =====")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Failure Rate: {failure_rate * 100:.2f}%")
    print(f"Failures: {failures}")
    print(f"Saved failure cases: {len(failure_cases)}")
    print(f"Output saved to: {output_path}")

    if failures > 0:
        print(f"Average no-move steps per failed episode: {total_no_move_steps / failures:.2f}")
        print(f"Average repeated-position steps per failed episode: {total_loop_steps / failures:.2f}")


def main():
    analyze_failures()


if __name__ == "__main__":
    main()