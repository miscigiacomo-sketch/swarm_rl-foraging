import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from env.foraging_env import ForagingEnv


def evaluate_grid(grid_size, max_steps, episodes=100):
    env = ForagingEnv(grid_size=grid_size, max_steps=max_steps)
    model = PPO.load("models/ppo_foraging")

    successes = 0
    total_reward = 0
    total_steps = 0

    for _ in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward

            if terminated or truncated:
                break

        successes += episode_reward > 0
        total_reward += episode_reward
        total_steps += step + 1

    success_rate = successes / episodes * 100
    average_reward = total_reward / episodes
    average_steps = total_steps / episodes

    return success_rate, average_reward, average_steps


def main():
    episodes = 100

    experiments = [
        {"grid_size": 5, "max_steps": 50},
        {"grid_size": 10, "max_steps": 100},
    ]

    print("===== GRID SIZE EVALUATION =====")

    for experiment in experiments:
        grid_size = experiment["grid_size"]
        max_steps = experiment["max_steps"]

        success_rate, average_reward, average_steps = evaluate_grid(
            grid_size=grid_size,
            max_steps=max_steps,
            episodes=episodes,
        )

        print()
        print(f"Grid Size: {grid_size}x{grid_size}")
        print(f"Episodes: {episodes}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Average Reward: {average_reward:.2f}")
        print(f"Average Episode Length: {average_steps:.2f}")


if __name__ == "__main__":
    main()