"""Baseline model training, post-hoc calibration, and the classic
undersampling probability correction for imbalanced classification.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def make_base_estimator(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight=None,  # deliberately left unbalanced: this is the point
        n_jobs=-1,
        random_state=random_state,
    )


def discrimination_report(y_true, y_prob) -> dict:
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-12, 1 - 1e-12)),
    }


def undersample(X, y, neg_pos_ratio: float = 5.0, random_state: int = 42):
    """Keep all positives, randomly keep `neg_pos_ratio` times as many negatives.

    Returns the undersampled (X, y) and beta, the fraction of the original
    negatives kept -- needed for the probability correction below.
    """
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_keep = min(len(neg_idx), int(len(pos_idx) * neg_pos_ratio))
    keep_neg = rng.choice(neg_idx, size=n_keep, replace=False)
    beta = n_keep / len(neg_idx)
    idx = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(idx)
    return X[idx], y[idx], beta


def correct_undersampled_probabilities(p_sampled: np.ndarray, beta: float) -> np.ndarray:
    """Rescale probabilities from a model trained on undersampled data back
    to the true class prior (Elkan 2001, used for fraud by Dal Pozzolo &
    Caelen 2015). beta = fraction of the majority class kept.
    """
    p_sampled = np.clip(p_sampled, 1e-12, 1 - 1e-12)
    return p_sampled / (p_sampled + (1 - p_sampled) / beta)


def platt_scaling(estimator_factory, X_train, y_train, cv: int = 5):
    calibrated = CalibratedClassifierCV(estimator_factory(), method="sigmoid", cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


def isotonic_scaling(estimator_factory, X_train, y_train, cv: int = 5):
    calibrated = CalibratedClassifierCV(estimator_factory(), method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated
