import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper

class GaussianNoiseObsWrapper(ObservationWrapper):
    def __init__(self, env, std=20.0):
        super().__init__(env)
        self.std = std

    def observation(self, obs):
        noise = np.random.normal(0, self.std, obs.shape)
        return np.clip(obs + noise, 0, 255).astype(np.uint8)

def make_env(env_id, noise=None, seed=0):
    if env_id == "seaquest":
        env = gym.make("SeaquestNoFrameskip-v4")
    else:
        env = gym.make(env_id)
    env.seed(seed)
    if noise is not None and noise.startswith("gaussian"):
        std = float(noise.replace("gaussian", ""))
        env = GaussianNoiseObsWrapper(env, std=std)
    return env