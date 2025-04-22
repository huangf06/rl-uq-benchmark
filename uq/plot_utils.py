import matplotlib.pyplot as plt
import numpy as np
import os

def plot_reliability(q_preds, returns, n_bins=10, save_path=None):
    """
    plot reliability diagram，save to save_path
    """
    q = np.array(q_preds, dtype=np.float64)
    r = np.array(returns, dtype=np.float64)
    q_norm = (q - q.min()) / (q.max() - q.min() + 1e-8)
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    confs, accs = [], []
    for i in range(n_bins):
        idx = (q_norm >= bin_edges[i]) & (q_norm < bin_edges[i+1])
        if not np.any(idx):
            continue
        confs.append(q_norm[idx].mean())
        accs.append(r_norm[idx].mean())

    plt.figure(figsize=(5,5))
    plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
    plt.plot(confs, accs, marker='o', label='Model')
    plt.xlabel('Normalized Q-value (confidence)')
    plt.ylabel('Normalized Return (accuracy)')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
