import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from env.foraging_env import ForagingEnv

env = ForagingEnv()

model = PPO.load("models/ppo_foraging")

observation, info = env.reset()

print("Initial observation:", observation)
env.render()

for step in range(50):
    action, _states = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)

    print("\nStep:", step)
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)

    env.render()

    time.sleep(0.3)

    if terminated:
        print("Food collected!")
        break

    if truncated:
        print("Maximum number of steps reached.")
        break

print("\nEvaluation finished.")