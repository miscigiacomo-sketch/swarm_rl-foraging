import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from env.foraging_env import ForagingEnv

NUM_EPISODES = 100


def evaluate_random_agent():
    env = ForagingEnv()

    successes = 0
    total_reward = 0
    total_steps = 0

    for episode in range(NUM_EPISODES):
        observation, info = env.reset()

        episode_reward = 0
        episode_steps = 0

        while True:
            action = env.action_space.sample()

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

    return success_rate, average_reward, average_steps


def evaluate_ppo_agent():
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
            action, _ = model.predict(observation, deterministic=True)

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

    return success_rate, average_reward, average_steps


random_success, random_reward, random_steps = evaluate_random_agent()
ppo_success, ppo_reward, ppo_steps = evaluate_ppo_agent()

print("\n===== AGENT COMPARISON =====")
print(f"{'Agent':<15} {'Success Rate':<15} {'Avg Reward':<12} {'Avg Steps':<10}")
print("-" * 55)
print(f"{'Random':<15} {random_success:<15.2f} {random_reward:<12.2f} {random_steps:<10.2f}")
print(f"{'PPO':<15} {ppo_success:<15.2f} {ppo_reward:<12.2f} {ppo_steps:<10.2f}")