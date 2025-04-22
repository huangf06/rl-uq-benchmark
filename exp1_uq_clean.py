# %%
import ale_py  # Ensure Atari environment is registered
import numpy as np, os, torch, pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3.utils import create_test_env, get_saved_hyperparams

# %%
def get_q_value(model, obs_tensor, action):
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
        return q_values[0, action].item()

# %%
ENV_ID = "SeaquestNoFrameskip-v4"
MODEL_PATH = "logs/dqn/SeaquestNoFrameskip-v4_1/best_model.zip"
STATS_PATH = "logs/dqn/SeaquestNoFrameskip-v4_1/SeaquestNoFrameskip-v4"
SEED = 0
EVAL_EPISODES = 5
hyperparams, stats_path = get_saved_hyperparams(STATS_PATH)
env = create_test_env(
    env_id=ENV_ID,
    n_envs=1,
    stats_path=stats_path,
    seed=SEED,
    log_dir=None,
    should_render=False,
    hyperparams=hyperparams,
)

# %%
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
gamma = model.gamma

# %%
data = []
for _ in range(EVAL_EPISODES):
    obs = env.reset()
    rewards, q_vals, actions = [], [], []
    done = [False]
    while not done[0]:
        obs_tensor = torch.as_tensor(obs).to(model.device).float().permute(0, 3, 1, 2) / 255.0
        with torch.no_grad():
            q_values = model.policy.q_net(obs_tensor)[0]
        action = int(torch.argmax(q_values).item())
        q_vals.append(q_values[action].item())
        actions.append(action)
        obs, reward, done, info = env.step([action])
        rewards.append(reward[0])
        done = [done[0]]

    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    for q, R, a in zip(q_vals, returns, actions):
        data.append((q, R, a))

env.close()
df = pd.DataFrame(data, columns=["Q_value", "MC_return", "action"])
df.to_csv("q_mc_eval.csv", index=False)

# %%



