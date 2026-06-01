import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from env.foraging_env import ForagingEnv

env = ForagingEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=10000)

model.save("models/ppo_foraging")

print("Training completed. Model saved in models/ppo_foraging")