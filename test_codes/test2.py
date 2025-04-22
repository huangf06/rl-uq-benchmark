# manual_eval_final.py – 正确的手动一局评估（完全复用能跑高分的流水线）

import ale_py               # 确保 Atari 环境注册
import torch
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
# from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv

# ===== 配置 =====
ENV_ID     = "SeaquestNoFrameskip-v4"
MODEL_PATH = "logs/dqn/SeaquestNoFrameskip-v4_1/best_model.zip"
SEED       = 0
# ===============

def main():
    # 1) 构建环境（和 enjoy.py 完全一致的预处理）
    # env = make_atari_env(ENV_ID, n_envs=1, seed=SEED, clip_reward=False)
    env = make_atari_env(ENV_ID, n_envs=1, seed=SEED, wrapper_kwargs={"clip_reward": False, "episode_life": False})
    if isinstance(env.envs[0].env, EpisodicLifeEnv):
        env.envs[0].env = env.envs[0].env.env
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # 2) 加载模型（只加载权重，不构建大 buffer）
    model = DQN.load(
        MODEL_PATH,
        env=env,
        buffer_size=1,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "exploration_schedule": lambda _: 0.0,
        },
    )
    print("✅ Model loaded.")

    # 3) 重置环境
    obs = env.reset()  # 返回 shape (1,4,84,84)

    # 4) 计算初始 Q‑values
    obs_tensor = model.policy.obs_to_tensor(obs)
    # 某些版本返回 (tensor, extras)，取第0项
    if isinstance(obs_tensor, tuple):
        obs_tensor = obs_tensor[0]
    with torch.no_grad():
        q_out = model.policy.q_net(obs_tensor)
    q_values = q_out[0].cpu().numpy()
    print("Q-values:", np.round(q_values,3))
    print("Q-max:", q_values.max())

    # 5) 手动跑完整一局
    done = False
    total_return = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_return += reward[0]
    print("Return:", total_return)

    # 6) 计算 ECE
    from utils.uq_metrics import regression_ece
    ece = regression_ece([q_values.max()], [total_return])
    print("ECE:", ece)

    env.close()

if __name__ == "__main__":
    main()
