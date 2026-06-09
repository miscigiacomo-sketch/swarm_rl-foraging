import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from env.foraging_env import ForagingEnv


def test_obstacle_collision():
    env = ForagingEnv()

    env.agent_pos = np.array([2, 0], dtype=np.int32)
    env.food_pos = np.array([4, 4], dtype=np.int32)

    print("Initial grid:")
    env.render()

    print()
    print("Trying to move down into obstacle at [2, 1]")

    action = 1  # down
    observation, reward, terminated, truncated, _ = env.step(action)

    print("Observation after action:", observation)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.render()

    if np.array_equal(env.agent_pos, np.array([2, 0], dtype=np.int32)):
        print()
        print("Test passed: agent was blocked by the obstacle.")
    else:
        print()
        print("Test failed: agent moved into the obstacle.")


if __name__ == "__main__":
    test_obstacle_collision()