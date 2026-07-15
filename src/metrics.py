"""Calibration diagnostics for highly imbalanced binary classification."""
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


def adaptive_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """Equal-mass-binning calibration error.

    Unlike equal-width ECE, each bin contains roughly the same number of
    observations. Reporting both exposes the sensitivity of binned calibration
    metrics to the binning scheme.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)
    n = len(y_true)
    return float(
        sum(
            len(idx) / n * abs(y_true[idx].mean() - y_prob[idx].mean())
            for idx in bins
            if len(idx)
        )
    )


def top_fraction_calibration_error(y_true, y_prob, fraction: float = 0.01) -> float:
    """Absolute calibration gap among the highest-scored ``fraction`` cases.

    The default top 1% better reflects a constrained fraud-investigation queue
    than the top decile. The selected count is deterministic, including when
    predicted probabilities contain ties.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y_true) * fraction)))
    idx = np.argpartition(y_prob, -k)[-k:]
    return float(abs(y_true[idx].mean() - y_prob[idx].mean()))


def reliability_curve(y_true, y_prob, n_bins: int = 15, strategy: str = "uniform"):
    """Return (bin_confidence, bin_accuracy, bin_weight) arrays for plotting."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
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


def plot_reliability(
    ax, y_true, y_prob, n_bins: int = 15, label: str | None = None, strategy: str = "uniform"
):
    confs, accs, counts = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    existing_labels = ax.get_legend_handles_labels()[1]
    if "Perfect calibration" not in existing_labels:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    line = ax.plot(confs, accs, linewidth=1.5, label=label)[0]
    marker_sizes = np.clip(4.0 * np.sqrt(counts), 18, 120)
    ax.scatter(
        confs,
        accs,
        s=marker_sizes,
        color=line.get_color(),
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=3,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    return ax
