import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.foraging_env import ForagingEnv


env = ForagingEnv(grid_size=10, max_steps=100)

observation, _ = env.reset()

print("Initial observation:", observation)
env.render()

for step in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, _ = env.step(action)

    print()
    print("Step:", step)
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)

    env.render()

    if terminated:
        print("Food collected!")
        break

    if truncated:
        print("Maximum steps reached!")
        break

print()
print("Large grid test finished.")