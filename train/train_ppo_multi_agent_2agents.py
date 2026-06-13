import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env.multi_agent_foraging_env import MultiAgentForagingEnv


def main():
    os.makedirs("models", exist_ok=True)

    env = MultiAgentForagingEnv(
        grid_size=5,
        num_agents=2,
        max_steps=50,
    )

    check_env(env, warn=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=42,
    )

    print("===== TRAINING MULTI-AGENT PPO =====")
    print("Grid Size: 5x5")
    print("Number of Agents: 2")
    print("Joint Action Space: 16 actions")
    print("Training Timesteps: 100000")
    print()

    model.learn(total_timesteps=100000)

    model_path = "models/ppo_multi_agent_2agents"
    model.save(model_path)

    env.close()

    print()
    print("Training completed.")
    print(f"Model saved to: {model_path}.zip")


if __name__ == "__main__":
    main()