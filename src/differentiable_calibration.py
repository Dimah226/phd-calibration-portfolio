"""Small MLP trained with plain cross-entropy vs cross-entropy + a soft-ECE
term, to see if pushing calibration during training helps instead of fixing
it after the fact (Platt/isotonic)."""
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
    """Soft version of ECE so it's differentiable.

    Normal ECE hard-assigns each point to one bin, which has zero gradient.
    Here each point gets a soft (Gaussian) weight into every bin instead, so
    we can just add this to the loss and backprop through it. Same idea as
    the soft-binning tricks used in some of the differentiable-calibration
    papers (Kumar et al. 2018, Karandikar et al. 2021).
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
    """Train the MLP with BCE, plus calibration_weight * soft-ECE if > 0.
    Returns the model and its predicted probabilities on X_val."""
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
