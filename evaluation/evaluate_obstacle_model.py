import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from env.foraging_env import ForagingEnv


def evaluate_model(model_path, episodes=100):
    env = ForagingEnv(grid_size=5, max_steps=50)
    model = PPO.load(model_path)

    successes = 0
    total_reward = 0
    total_steps = 0

    for _ in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0

        for step in range(env.max_steps):
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
    success_rate, average_reward, average_steps = evaluate_model(
        model_path="models/ppo_foraging_obstacles_state",
        episodes=100,
    )

    print("===== OBSTACLE PPO EVALUATION =====")
    print("Episodes: 100")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {average_reward:.2f}")
    print(f"Average Episode Length: {average_steps:.2f}")


if __name__ == "__main__":
    main()