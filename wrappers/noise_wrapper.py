import gymnasium as gym
import ale_py
import numpy as np

class GaussianNoiseWrapper(gym.ObservationWrapper):
    """Add i.i.d Gaussian noise to pixel observations (Atari frames).

    Args:
        env (gym.Env): Base environment.
        sigma (float): Noise std in *fraction of 255*. Example: 0.2 → ±51 std.
    """

    def __init__(self, env: gym.Env, sigma: float = 0.0):
        super().__init__(env)
        assert sigma >= 0.0, "sigma must be non‑negative"
        self.sigma = float(sigma)

    def observation(self, obs):
        if self.sigma == 0.0:
            return obs
        noise = np.random.normal(0.0, self.sigma * 255.0, size=obs.shape)
        noisy = obs.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)