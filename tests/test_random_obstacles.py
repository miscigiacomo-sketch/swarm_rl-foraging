import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.foraging_env import ForagingEnv
import numpy as np


def test_random_obstacles():
    env = ForagingEnv(random_obstacles=True)

    for episode in range(10):
        obs, info = env.reset()

        agent_pos = obs[0:2]
        food_pos = obs[2:4]
        obstacle_positions = obs[4:].reshape(-1, 2)

        print(f"\nEpisode {episode + 1}")
        env.render()

        # Check that no obstacle is on the agent
        for obstacle in obstacle_positions:
            assert not np.array_equal(obstacle, agent_pos), \
                "Obstacle spawned on the agent."

        # Check that food is not on an obstacle
        for obstacle in obstacle_positions:
            assert not np.array_equal(obstacle, food_pos), \
                "Food spawned on an obstacle."

        # Check that obstacles do not overlap
        unique_obstacles = set(tuple(obstacle) for obstacle in obstacle_positions)
        assert len(unique_obstacles) == len(obstacle_positions), \
            "Two obstacles spawned on the same position."

    print("\nRandom obstacle test passed successfully.")


if __name__ == "__main__":
    test_random_obstacles()