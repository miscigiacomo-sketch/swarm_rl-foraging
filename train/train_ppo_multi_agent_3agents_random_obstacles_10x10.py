import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env.multi_agent_foraging_env import MultiAgentForagingEnv


def main():
    os.makedirs("models", exist_ok=True)

    env = MultiAgentForagingEnv(
        grid_size=10,
        num_agents=3,
        max_steps=50,
        random_obstacles=True,
        num_obstacles=3,
    )

    check_env(env, warn=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=42,
    )

    print("===== TRAINING 3-AGENT PPO WITH RANDOM OBSTACLES ON 10x10 =====")
    print("Grid Size: 10x10")
    print("Number of Agents: 3")
    print("Max Steps: 50")
    print("Random Obstacles: True")
    print("Number of Obstacles: 3")
    print("Joint Action Space: 64 actions")
    print("Observation Size: 14")
    print("Training Timesteps: 200000")
    print()

    model.learn(total_timesteps=200000)

    model_path = "models/ppo_multi_agent_3agents_random_obstacles_10x10"
    model.save(model_path)

    env.close()

    print()
    print("Training completed.")
    print(f"Model saved to: {model_path}.zip")


if __name__ == "__main__":
    main()