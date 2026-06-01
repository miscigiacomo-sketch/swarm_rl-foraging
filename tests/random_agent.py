import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.foraging_env import ForagingEnv

env = ForagingEnv()

observation, info = env.reset()

print("Initial observation:", observation)
env.render()

for step in range(10):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print("\nStep:", step)
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)

    env.render()

    if terminated:
        print("Food collected!")
        break

    if truncated:
        print("Maximum number of steps reached.")
        break

print("\nTest finished.")