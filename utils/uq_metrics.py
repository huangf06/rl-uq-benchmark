import numpy as np


def regression_ece(preds, targets, n_bins: int = 15):
    """A simple *regression* ECE: bin |predicted return| and compare mean error.

    This is *not* a perfect calibration metric, but works as a first‑cut.
    """

    preds = np.asarray(preds)
    targets = np.asarray(targets)
    assert preds.shape == targets.shape
    abs_err = np.abs(preds - targets)

    bin_edges = np.linspace(preds.min(), preds.max(), n_bins + 1)
    ece = 0.0
    total = 0
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if not mask.any():
            continue
        bin_err = abs_err[mask].mean()
        proportion = mask.mean()
        ece += bin_err * proportion
        total += proportion
    return ece / max(total, 1e-8)