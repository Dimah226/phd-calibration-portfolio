"""Calibration diagnostics: Expected Calibration Error and reliability diagrams."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """Standard equal-width-binning Expected Calibration Error."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def top_decile_calibration_error(y_true, y_prob) -> float:
    """Calibration error restricted to the top 10% of predicted scores.

    Standard ECE is dominated by the very large number of low-score genuine
    transactions and can look small even when the scores that actually drive
    investigation decisions -- the highest ones -- are poorly calibrated.
    This mirrors the practical concern behind Guilbert, Caelen, Chirita &
    Saerens (2024): in imbalanced classification, calibration should be
    assessed where the rare positive class concentrates, not on average.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    threshold = np.quantile(y_prob, 0.90)
    mask = y_prob >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(abs(y_true[mask].mean() - y_prob[mask].mean()))


def reliability_curve(y_true, y_prob, n_bins: int = 15):
    """Return (bin_confidence, bin_accuracy, bin_weight) arrays for plotting."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    confs, accs, weights = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        confs.append(y_prob[mask].mean())
        accs.append(y_true[mask].mean())
        weights.append(mask.sum())
    return np.array(confs), np.array(accs), np.array(weights)


def plot_reliability(ax, y_true, y_prob, n_bins: int = 15, label: str | None = None):
    confs, accs, _ = reliability_curve(y_true, y_prob, n_bins=n_bins)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(confs, accs, marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    return ax
