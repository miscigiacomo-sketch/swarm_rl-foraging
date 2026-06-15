import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.multi_agent_foraging_env import MultiAgentForagingEnv


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def format_positions(positions):
    return [tuple(position.tolist()) for position in positions]


def main():
    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    env.action_space.seed(42)

    obs, info = env.reset(seed=7)

    print("===== MULTI-AGENT RANDOM-OBSTACLE VISUALIZATION =====")
    print("A = Agent 1")
    print("B = Agent 2")
    print("F = Food")
    print("X = Obstacle")
    print(". = Empty cell")
    print()
    print("Initial grid:")
    env.render()

    input("Press Enter to start the random rollout...")

    done = False
    step = 0
    total_reward = 0.0

    while not done:
        clear_terminal()

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1
        done = terminated or truncated

        print("===== MULTI-AGENT RANDOM-OBSTACLE VISUALIZATION =====")
        print(f"Step: {step}")
        print(f"Joint action: {action}")
        print(f"Reward: {reward}")
        print(f"Total reward: {total_reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print()
        print(f"Agent positions: {format_positions(info['agent_positions'])}")
        print(f"Food position: {tuple(info['food_pos'].tolist())}")
        print(f"Obstacles: {format_positions(info['obstacles'])}")
        print()
        env.render()

        time.sleep(0.5)

    env.close()

    print("Episode finished.")


if __name__ == "__main__":
    main()