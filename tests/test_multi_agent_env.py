import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3.common.env_checker import check_env

from env.multi_agent_foraging_env import MultiAgentForagingEnv


def test_environment(num_agents):
    print(f"===== TESTING MULTI-AGENT ENVIRONMENT: {num_agents} AGENTS =====")

    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=num_agents,
        max_steps=50,
    )

    check_env(env, warn=True)

    obs, info = env.reset(seed=42)

    print("Initial observation:")
    print(obs)
    print()

    print("Initial grid:")
    env.render()

    for step in range(10):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}")
        print(f"Action: {action}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        env.render()

        if terminated or truncated:
            break

    print(f"{num_agents}-agent environment test completed successfully.")
    print()


def main():
    test_environment(num_agents=2)
    test_environment(num_agents=3)

    print("All multi-agent environment tests completed successfully.")


if __name__ == "__main__":
    main()