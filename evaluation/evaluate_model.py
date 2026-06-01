import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from env.foraging_env import ForagingEnv

NUM_EPISODES = 100

env = ForagingEnv()

model = PPO.load("models/ppo_foraging")

successes = 0
total_reward = 0
total_steps = 0

for episode in range(NUM_EPISODES):

    observation, info = env.reset()

    episode_reward = 0
    episode_steps = 0

    while True:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_steps += 1

        if terminated:

            successes += 1

            total_reward += episode_reward
            total_steps += episode_steps

            break

        if truncated:

            total_reward += episode_reward
            total_steps += episode_steps

            break

success_rate = successes / NUM_EPISODES * 100
average_reward = total_reward / NUM_EPISODES
average_steps = total_steps / NUM_EPISODES

print("\n===== EVALUATION RESULTS =====")

print(f"Episodes: {NUM_EPISODES}")
print(f"Success Rate: {success_rate:.2f}%")
print(f"Average Reward: {average_reward:.2f}")
print(f"Average Episode Length: {average_steps:.2f}")