#!/usr/bin/env python3
"""
Main UQ evaluation script for RL models, mirroring training environment setup.
Computes ECE, mean/std of returns, and saves a reliability diagram.
"""
import argparse
import os
import numpy as np
import gymnasium as gym
import rl_zoo3.import_envs  # register custom with ExperimentManager
from stable_baselines3 import DQN
from uq.eval_utils import compute_ece
from uq.plot_utils import plot_reliability
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from gymnasium.vector import SyncVectorEnv


def make_env(env_id: str, noise: str = None, seed: int = 0):
    """Create vectorized Gym env identical to training wrappers."""
    # Atari environments: wrap with frame stacking
    if 'NoFrameskip' in env_id or 'Atari' in env_id:
        env = make_atari_env(env_id, n_envs=1, seed=seed)
        env = VecFrameStack(env, n_stack=4)
    else:
        env = gym.make(env_id)

    # Optional Gaussian noise wrapper
    if noise and noise.startswith('gaussian'):
        std = float(noise.replace('gaussian', ''))
        class GaussianNoiseObs(gym.ObservationWrapper):
            def __init__(self, env, std):
                super().__init__(env)
                self.std = std
            def observation(self, obs):
                noise = np.random.normal(0, self.std, obs.shape)
                return np.clip(obs + noise, 0, 255).astype(np.uint8)
        # Apply noise to each vector env
        env = SyncVectorEnv([lambda: GaussianNoiseObs(env, std)])

    return env


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run UQ evaluation with training-like envs.'
    )
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (without .zip)')
    parser.add_argument('--env-id', type=str, required=True,
                        help="Exact Gym env ID used during training, e.g., 'SeaquestNoFrameskip-v4'")
    parser.add_argument('--noise', type=str, default=None,
                        help="Optional noise setting, e.g., 'gaussian20'")
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base random seed')
    parser.add_argument('--n-bins', type=int, default=10,
                        help='Number of bins for ECE computation')
    parser.add_argument('--output-dir', type=str, default='logs/eval',
                        help='Directory to save outputs')
    return parser.parse_args()


def main():
    args = parse_args()

    # Construct environment
    env = make_env(args.env_id, noise=args.noise, seed=args.seed)

    # Load policy with minimal buffer to prevent large memory allocation
    model = DQN.load(
        args.model_path,
        env=env,
        custom_objects={"buffer_size": 1, "n_envs": 1}
    )

    q_values, returns = [], []

        # Evaluate episodes
    for ep in range(args.episodes):
        # Reset environment and handle return formatting
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out
        done = False
        ep_qs = []
        ep_ret = 0.0

        while not done:
            # Get action
            action, _ = model.predict(obs, deterministic=True)

            # Prepare observation tensor for q_net
            import torch as th
            obs_arr = obs  # could be numpy array of shape (1,84,84,4)
            obs_tensor = th.tensor(obs_arr, device=model.device, dtype=th.float32)
            # Permute to (batch, channels, height, width)
            if obs_tensor.ndim == 4:
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            elif obs_tensor.ndim == 3:
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
            obs_tensor = obs_tensor / 255.0

            # Compute Q value
            q_val = model.q_net(obs_tensor).max().item()
            ep_qs.append(q_val)

            # Step environment
            step_out = env.step(action)
            # VecEnv.step returns (obs, reward, done, info) or tuple
            if len(step_out) == 5:
                obs, reward, done, truncated, info = step_out
            else:
                obs, reward, done, info = step_out
            ep_ret += reward

        q_values.extend(ep_qs)
        returns.extend([ep_ret] * len(ep_qs))

    # Compute metrics
    ece = compute_ece(q_values, returns, n_bins=args.n_bins)
    mean_ret, std_ret = np.mean(returns), np.std(returns)
    print(f"Env: {args.env_id} | Noise: {args.noise or 'clean'} | Seed: {args.seed}")
    print(f"Episodes: {args.episodes} | Return: {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"ECE: {ece:.4f}")

    # Save reliability diagram
    save_dir = os.path.join(args.output_dir, args.env_id, args.noise or 'clean', f'seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'reliability.png')
    plot_reliability(q_values, returns, n_bins=args.n_bins, save_path=save_path)
    print(f"Saved reliability diagram to {save_path}")

if __name__ == '__main__':
    main()
