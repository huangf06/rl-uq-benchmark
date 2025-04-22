import numpy as np

def compute_ece(q_preds, returns, n_bins=10):
    """
    Calculated Expected Calibration Error (ECE)。
    q_preds: list or 1D ndarray，predicted Q series
    returns: list or 1D ndarray，actual episode return
    """
    q = np.array(q_preds, dtype=np.float64)
    r = np.array(returns, dtype=np.float64)

    q_norm = (q - q.min()) / (q.max() - q.min() + 1e-8)
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8)

    total = len(q_norm)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        idx = (q_norm >= bin_edges[i]) & (q_norm < bin_edges[i+1])
        if not np.any(idx):
            continue
        conf = q_norm[idx].mean()
        acc = r_norm[idx].mean()
        ece += np.sum(idx) / total * abs(conf - acc)

    return ece