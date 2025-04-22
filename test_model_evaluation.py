import ale_py  # Ensure Atari environment is registered
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# ===== Configuration =====
ENV_ID      = "SeaquestNoFrameskip-v4"
MODEL_PATH  = "logs/dqn/SeaquestNoFrameskip-v4_1/best_model.zip"
SEED        = 0
EVAL_EPISODES = 5
# ==========================

# 1) Create Atari environment with preprocessing
env = make_atari_env(
    ENV_ID,
    n_envs=1,
    seed=SEED
)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# 4) Load the trained DQN model with minimal buffer
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
print("✅ Model loaded successfully.")

obs = env.reset()
done = False

rewards = []

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done:
        break

env.close()