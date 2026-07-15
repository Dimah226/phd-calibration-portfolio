"""A small PyTorch classifier trained with and without a differentiable,
soft-binned calibration loss added to the standard cross-entropy loss.

This is a simplified, from-scratch illustration of the idea motivating this
portfolio: instead of calibrating a classifier only after training (Platt
scaling, isotonic regression), fold a calibration objective directly into
training so the network learns to be accurate *and* calibrated at once.
"""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def soft_ece_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    temperature: float = 0.05,
) -> torch.Tensor:
    """A differentiable relaxation of the Expected Calibration Error.

    Standard ECE is non-differentiable because it hard-assigns each example
    to a probability bin. Here, each example is given a *soft* membership in
    every bin via a Gaussian kernel centred on its predicted probability --
    similar in spirit to soft-binning approaches in the differentiable
    calibration literature (e.g. Kumar et al., 2018; Karandikar et al.,
    2021). The result is smooth in the network's parameters and can be
    minimised jointly with the usual classification loss.
    """
    bin_centers = torch.linspace(0.0, 1.0, n_bins, device=probs.device)
    diff = probs.unsqueeze(1) - bin_centers.unsqueeze(0)  # (N, n_bins)
    weights = torch.exp(-(diff ** 2) / (2 * temperature ** 2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

    bin_weight = weights.sum(dim=0)
    bin_conf = (weights * probs.unsqueeze(1)).sum(dim=0) / (bin_weight + 1e-12)
    bin_acc = (weights * labels.unsqueeze(1)).sum(dim=0) / (bin_weight + 1e-12)

    n = probs.shape[0]
    return (bin_weight / n * (bin_acc - bin_conf).abs()).sum()


def train_mlp(
    X_train,
    y_train,
    X_val,
    calibration_weight: float = 0.0,
    n_epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Train the MLP with BCE loss, optionally plus `calibration_weight` times
    the soft-ECE loss. calibration_weight=0.0 reproduces plain cross-entropy
    training; a positive value is the differentiable-calibration variant.

    Returns the trained model and its predicted probabilities on X_val.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    model = MLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    n = X_train_t.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            probs = torch.sigmoid(logits)
            loss = bce(logits, yb)
            if calibration_weight > 0:
                loss = loss + calibration_weight * soft_ece_loss(probs, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
    return model, val_probs
