# manual_eval_one_episode_fixed2.py
import ale_py               # 注册 Atari
import torch
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy

# —— 配置 —— 
ENV_ID     = "SeaquestNoFrameskip-v4"
MODEL_PATH = "logs/dqn/SeaquestNoFrameskip-v4_1/best_model.zip"
SEED       = 0
# =============

def unpack_obs(obs):
    # Gymnasium reset 可能返回 (obs, info)
    return obs[0] if isinstance(obs, tuple) else obs

def unpack_step(result):
    # 支持 gym (obs, reward, done, info)
    # 也支持 gymnasium (obs, reward, terminated, truncated, info)
    if len(result) == 4:
        obs, reward, done, info = result
    else:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    return obs, reward, done, info

def main():
    # 1) 环境
    env = make_atari_env(ENV_ID, n_envs=1, seed=SEED)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # 2) 模型
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
    print("✅ Model loaded.\n")

    rewards, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=1,
            deterministic=True,
            return_episode_rewards=True,
        )
    black_return = float(rewards[0])
    print(f"[Blackbox] Episode reward = {black_return:.1f}")

    # 3) 重置并拿到初始 obs
    obs = env.reset()
    obs = unpack_obs(obs)  # 确保纯 ndarray

    # 4) 转为 Tensor
    obs_tensor = model.policy.obs_to_tensor(obs)
    # obs_to_tensor 在某些版本会返回 (tensor, dict)，取第 0 项
    if isinstance(obs_tensor, tuple):
        obs_tensor = obs_tensor[0]

    # 5) 计算 Q-values
    with torch.no_grad():
        q_out = model.policy.q_net(obs_tensor)
    # q_out.shape == [1, n_actions]
    q_values = q_out[0].cpu().numpy()
    print("Q-values at initial state:", np.round(q_values, 3))
    print("Q-max:", q_values.max(), "\n")

    # 6) 手动执行一局，累加 return
    done = False
    total_return = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)
        obs, reward, done, info = unpack_step(result)
        obs = unpack_obs(obs)
        # reward 可能是 array/list 长度 1
        r = reward[0] if isinstance(reward, (tuple, list, np.ndarray)) else reward
        total_return += r

    print("Return for this episode:", total_return)
    env.close()

if __name__ == "__main__":
    main()
