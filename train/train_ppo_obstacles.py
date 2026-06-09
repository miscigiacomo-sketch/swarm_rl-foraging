import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from env.foraging_env import ForagingEnv


def main():
    env = ForagingEnv(grid_size=5, max_steps=50)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
    )

    model.learn(total_timesteps=50000)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_foraging_obstacles_state")

    print("Obstacle PPO model saved to models/ppo_foraging_obstacles_state.zip")

if __name__ == "__main__":
    main()