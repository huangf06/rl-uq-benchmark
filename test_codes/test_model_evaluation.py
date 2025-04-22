import ale_py  # Ensure Atari environment is registered
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# ===== Configuration =====
ENV_ID      = "SeaquestNoFrameskip-v4"
MODEL_PATH  = "logs/dqn/SeaquestNoFrameskip-v4_1/best_model.zip"
SEED        = 0
EVAL_EPISODES = 10
# ==========================

def main():
    # 1) Create Atari environment with preprocessing
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=SEED
    )
    # 2) Stack last 4 frames
    env = VecFrameStack(env, n_stack=4)
    # 3) Transpose channels to C×H×W
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

    # 5) Evaluate policy and collect rewards
    rewards, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        return_episode_rewards=True
    )

    # 6) Print per-episode rewards and summary
    for idx, r in enumerate(rewards, 1):
        print(f"Episode {idx}: reward = {r}")
    mean_reward = sum(rewards) / len(rewards)
    print(f"Mean reward over {EVAL_EPISODES} episodes: {mean_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()