import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from env.multi_agent_foraging_env import MultiAgentForagingEnv


ACTIONS = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
}

FIXED_OBSTACLES = [
    (2, 1),
    (2, 2),
    (2, 3),
]


def parse_observation(obs, num_agents):
    """
    Parse the multi-agent observation.

    Observation layout:
    [agent1_x, agent1_y, agent2_x, agent2_y, ..., food_x, food_y,
     obstacle1_x, obstacle1_y, obstacle2_x, obstacle2_y, ...]
    """
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


def manhattan_distance(pos_a, pos_b):
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def apply_action(position, action, grid_size):
    dx, dy = ACTIONS[action]

    new_x = position[0] + dx
    new_y = position[1] + dy

    new_x = min(max(new_x, 0), grid_size - 1)
    new_y = min(max(new_y, 0), grid_size - 1)

    return (new_x, new_y)


def encode_joint_action(individual_actions):
    """
    Encode individual actions into the centralized joint action.

    The environment decodes actions using base-4:
    action = a1 + 4*a2 + 16*a3 + ...
    """
    joint_action = 0

    for agent_index, action in enumerate(individual_actions):
        joint_action += int(action) * (4 ** agent_index)

    return joint_action


def causes_swap(current_agent_index, current_position, proposed_position,
                old_positions, proposed_positions):
    """
    Check whether the current proposed move would swap positions
    with one of the agents that has already selected an action.
    """
    for previous_agent_index, previous_proposed_position in enumerate(proposed_positions):
        previous_old_position = old_positions[previous_agent_index]

        if (
            proposed_position == previous_old_position
            and previous_proposed_position == current_position
        ):
            return True

    return False


def greedy_obstacle_aware_action(obs, num_agents, grid_size):
    """
    Simple obstacle-aware decentralized greedy baseline.

    Each agent selects an action that reduces Manhattan distance to the food,
    while avoiding:
    - obstacle cells
    - already proposed cells by other agents
    - direct position swaps when possible
    """
    agent_positions, food_pos, obstacle_positions = parse_observation(
        obs=obs,
        num_agents=num_agents,
    )

    obstacle_set = set(obstacle_positions)

    proposed_positions = []
    individual_actions = []

    for agent_index, agent_position in enumerate(agent_positions):
        candidate_actions = []

        for action in ACTIONS:
            next_position = apply_action(
                position=agent_position,
                action=action,
                grid_size=grid_size,
            )

            if next_position in obstacle_set:
                continue

            if next_position in proposed_positions:
                continue

            if causes_swap(
                current_agent_index=agent_index,
                current_position=agent_position,
                proposed_position=next_position,
                old_positions=agent_positions,
                proposed_positions=proposed_positions,
            ):
                continue

            distance_to_food = manhattan_distance(next_position, food_pos)

            # Penalize no-move actions caused by grid boundaries.
            no_move_penalty = 1 if next_position == agent_position else 0

            candidate_actions.append(
                (
                    distance_to_food,
                    no_move_penalty,
                    action,
                    next_position,
                )
            )

        if not candidate_actions:
            # Fallback: choose the least bad action. The environment itself
            # will still enforce obstacle, collision, and swap constraints.
            for action in ACTIONS:
                next_position = apply_action(
                    position=agent_position,
                    action=action,
                    grid_size=grid_size,
                )

                distance_to_food = manhattan_distance(next_position, food_pos)
                no_move_penalty = 1 if next_position == agent_position else 0

                candidate_actions.append(
                    (
                        distance_to_food,
                        no_move_penalty,
                        action,
                        next_position,
                    )
                )

        candidate_actions.sort(key=lambda item: (item[0], item[1]))

        selected_action = candidate_actions[0][2]
        selected_position = candidate_actions[0][3]

        individual_actions.append(selected_action)
        proposed_positions.append(selected_position)

    return encode_joint_action(individual_actions)


def evaluate_random_baseline(
    grid_size=5,
    num_agents=2,
    episodes=1000,
    action_seed=42,
):
    env = MultiAgentForagingEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

    rng = np.random.default_rng(action_seed)

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = int(rng.integers(env.action_space.n))

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        if total_reward > 0:
            successes += 1

        total_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    return {
        "success_rate": successes / episodes,
        "average_reward": np.mean(total_rewards),
        "average_episode_length": np.mean(episode_lengths),
    }


def evaluate_greedy_baseline(
    grid_size=5,
    num_agents=2,
    episodes=1000,
):
    env = MultiAgentForagingEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = greedy_obstacle_aware_action(
                obs=obs,
                num_agents=num_agents,
                grid_size=grid_size,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        if total_reward > 0:
            successes += 1

        total_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    return {
        "success_rate": successes / episodes,
        "average_reward": np.mean(total_rewards),
        "average_episode_length": np.mean(episode_lengths),
    }


def print_results(title, results):
    print(title)
    print(f"Success Rate: {results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {results['average_reward']:.2f}")
    print(f"Average Episode Length: {results['average_episode_length']:.2f}")
    print()


def save_results_to_file(random_results, greedy_results, output_path):
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MULTI-AGENT FIXED-OBSTACLE BASELINE EVALUATION\n")
        file.write("=============================================\n\n")

        file.write("Environment setup:\n")
        file.write("- Grid size: 5x5\n")
        file.write("- Number of agents: 2\n")
        file.write("- Max steps: 50\n")
        file.write("- Obstacles: [(2, 1), (2, 2), (2, 3)]\n")
        file.write("- Control type: centralized joint action space\n\n")

        file.write("Random baseline:\n")
        file.write(f"Success Rate: {random_results['success_rate'] * 100:.2f}%\n")
        file.write(f"Average Reward: {random_results['average_reward']:.2f}\n")
        file.write(
            f"Average Episode Length: "
            f"{random_results['average_episode_length']:.2f}\n\n"
        )

        file.write("Greedy obstacle-aware baseline:\n")
        file.write(f"Success Rate: {greedy_results['success_rate'] * 100:.2f}%\n")
        file.write(f"Average Reward: {greedy_results['average_reward']:.2f}\n")
        file.write(
            f"Average Episode Length: "
            f"{greedy_results['average_episode_length']:.2f}\n\n"
        )

        file.write("Interpretation:\n")
        file.write(
            "The random baseline measures performance under uninformed joint actions. "
            "The greedy obstacle-aware baseline is a stronger non-learning reference: "
            "each agent moves toward the food using Manhattan distance while avoiding "
            "obstacles, collisions, and direct swaps when possible.\n"
        )


def main():
    episodes = 1000
    grid_size = 5
    num_agents = 2

    print("===== MULTI-AGENT FIXED-OBSTACLE BASELINE EVALUATION =====")
    print(f"Episodes: {episodes}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Number of agents: {num_agents}")
    print(f"Fixed obstacles: {FIXED_OBSTACLES}")
    print()

    random_results = evaluate_random_baseline(
        grid_size=grid_size,
        num_agents=num_agents,
        episodes=episodes,
        action_seed=42,
    )

    greedy_results = evaluate_greedy_baseline(
        grid_size=grid_size,
        num_agents=num_agents,
        episodes=episodes,
    )

    print_results("Random baseline:", random_results)
    print_results("Greedy obstacle-aware baseline:", greedy_results)

    output_path = "results/multi_agent_obstacle_baseline_summary.txt"
    save_results_to_file(
        random_results=random_results,
        greedy_results=greedy_results,
        output_path=output_path,
    )

    print("===== FINAL COMPARISON =====")
    print("| Method | Success Rate | Average Reward | Average Episode Length |")
    print("|---|---:|---:|---:|")
    print(
        f"| Random baseline | "
        f"{random_results['success_rate'] * 100:.2f}% | "
        f"{random_results['average_reward']:.2f} | "
        f"{random_results['average_episode_length']:.2f} |"
    )
    print(
        f"| Greedy obstacle-aware baseline | "
        f"{greedy_results['success_rate'] * 100:.2f}% | "
        f"{greedy_results['average_reward']:.2f} | "
        f"{greedy_results['average_episode_length']:.2f} |"
    )

    print()
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()