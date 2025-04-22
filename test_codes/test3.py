# evaluate_clean_uq_final.py – 最终版：加载 best_model.zip，使用 RL-Zoo 环境对齐，并恢复 VecNormalize
import os
import csv
import numpy as np
import torch
import ale_py  # 注册 Atari 环境

from stable_baselines3 import DQN
from utils.uq_metrics import regression_ece
from rl_zoo3.utils import (
    get_model_path,
    get_saved_hyperparams,
    create_test_env,
)

# === 配置 ===
ENV_ID = "SeaquestNoFrameskip-v4"
ALGO = "dqn"
EXP_ID = 1
LOG_FOLDER = "logs/dqn/SeaquestNoFrameskip-v4_1"
MODEL_FILE = "best_model.zip"
SEED = 0
EVAL_EPISODES = 10
OUT_CSV = "results/clean_eval.csv"
# ============

def main():
    # 确定模型路径
    model_path = os.path.join(LOG_FOLDER, MODEL_FILE)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"✅ Loading model from {model_path}")

    # 恢复超参数和 VecNormalize 状态
    stats_path = os.path.join(LOG_FOLDER, ENV_ID)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path)

    # 创建与训练一致的环境包装
    env = create_test_env(
        env_id=ENV_ID,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=SEED,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
    )

    # 加载模型
    model = DQN.load(
        model_path,
        env=env,
        buffer_size=1,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "exploration_schedule": lambda _: 0.0,
        },
    )
    print("✅ Model loaded.")

    # 逐回合评估
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    results = []
    for ep in range(1, EVAL_EPISODES + 1):
        obs = env.reset()
        done = False
        total_return = 0.0

        # 初始 Q-max 值
        with torch.no_grad():
            qv = model.policy.q_net(model.policy.obs_to_tensor(obs)[0])[0].cpu().numpy()
        qmax = float(np.max(qv))

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_return += reward[0]

        ece = regression_ece([qmax], [total_return])
        print(f"Episode {ep}: return={total_return:.1f}, Q-max={qmax:.2f}, ECE={ece:.4f}")
        results.append([ep, total_return, qmax, ece])

    # 保存结果
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "qmax", "ece"])
        writer.writerows(results)
    print(f"\n🎯 Evaluation done, results saved to {OUT_CSV}")

    env.close()

if __name__ == "__main__":
    main()
