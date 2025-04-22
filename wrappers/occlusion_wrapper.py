import gymnasium as gym
import ale_py
import numpy as np

class RandomOcclusionWrapper(gym.ObservationWrapper):
    """Randomly occlude a square patch with probability *p* each frame."""

    def __init__(self, env: gym.Env, p: float = 0.1, patch_ratio: float = 0.2):
        super().__init__(env)
        assert 0.0 <= p <= 1.0
        self.p = p
        self.patch_ratio = patch_ratio
        self._h, self._w = self.observation_space.shape[-2:]
        self._patch_size = int(min(self._h, self._w) * self.patch_ratio)

    def observation(self, obs):
        if np.random.rand() < self.p:
            y = np.random.randint(0, self._h - self._patch_size)
            x = np.random.randint(0, self._w - self._patch_size)
            obs[..., y : y + self._patch_size, x : x + self._patch_size] = 0
        return obs