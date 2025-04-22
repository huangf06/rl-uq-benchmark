import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_bar(df, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(4, 3))
    algos = df["algo"].unique()
    sigmas = sorted(df["sigma"].unique())
    width = 0.8 / len(algos)
    for i, algo in enumerate(algos):
        sub = df[df.algo == algo].groupby("sigma").mean().reset_index()
        ax.bar(
            np.arange(len(sigmas)) + i * width,
            sub[metric],
            width=width,
            label=algo,
        )
    ax.set_xticks(np.arange(len(sigmas)) + width * (len(algos) - 1) / 2)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.set_xlabel("Noise σ")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_reliability(csv_path, title, filename):
    df = pd.read_csv(csv_path)
    conf, acc = calibration_curve(df["conf"], df["acc"], n_bins=15, strategy="uniform")
    plt.figure(figsize=(3.5, 3.5))
    plt.plot(conf, acc, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_scatter(df, filename):
    plt.figure(figsize=(4, 3))
    for algo in df["algo"].unique():
        sub = df[df.algo == algo]
        plt.scatter(sub["ece"], sub["return_mean"], label=algo, alpha=0.7)
    plt.xlabel("ECE")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def plot_training_curve(csv_paths, filename):
    plt.figure(figsize=(4, 3))
    for csv in csv_paths:
        df = pd.read_csv(csv)
        label = os.path.basename(csv).replace(".csv", "")
        plt.plot(df["step"], df["episode_reward"].rolling(window=50).mean(), label=label)
    plt.xlabel("Env Step")
    plt.ylabel("Episode Return (smoothed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", default="results/metrics_all.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.metrics_csv)

    plot_bar(df, "return_mean", "Return", "01_Return_vs_Noise.png")
    plot_bar(df, "ece", "ECE", "02_ECE_vs_Noise.png")
    # Reliability diagrams require separate CSV (if available)
    # plot_reliability("results/reliability_clean.csv", "Clean (DQN)", "03_Reliability_Clean.png")
    # plot_reliability("results/reliability_sigma02.csv", "σ=0.2 (DQN)", "04_Reliability_Noise.png")
    plot_scatter(df, "05_Scatter_ECE_vs_Return.png")
    # plot_training_curve(["logs/dqn_clean_progress.csv", "logs/dqn_sigma02_progress.csv"], "06_Training_Curve.png")


if __name__ == "__main__":
    main()