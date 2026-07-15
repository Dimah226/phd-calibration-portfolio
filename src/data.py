"""Data loading for the credit-card fraud calibration project.

Tries, in order:
  1. A local CSV at data/creditcard.csv (manually downloaded from Kaggle:
     https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
  2. OpenML dataset id 1597, a public mirror of the same dataset that
     requires no credentials.
  3. A synthetic imbalanced dataset shaped like the real one, so the
     pipeline always runs end to end even offline.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split

LOCAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
OPENML_DATA_ID = 1597  # public mirror of mlg-ulb/creditcardfraud


def _load_local(path: str = LOCAL_CSV):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _load_openml():
    try:
        bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    except Exception as exc:  # network unavailable, OpenML down, etc.
        print(f"[data] Could not fetch OpenML dataset {OPENML_DATA_ID}: {exc}")
        return None

    if bunch.frame is not None:
        df = bunch.frame.copy()
    else:
        df = bunch.data.copy()
        df[bunch.target.name] = bunch.target

    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    df = df.rename(columns={target_col: "Class"})
    df["Class"] = df["Class"].astype(str).str.strip().str.strip("'\"").astype(int)
    return df


def _make_synthetic(n_samples: int = 60_000, random_state: int = 42):
    n_features = 30
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=3,
        weights=[0.9983, 0.0017],  # ~0.17% positive rate, matches the real dataset
        flip_y=0.001,
        class_sep=0.9,
        random_state=random_state,
    )
    columns = [f"V{i}" for i in range(1, n_features - 1)] + ["Time", "Amount"]
    df = pd.DataFrame(X, columns=columns)
    df["Amount"] = df["Amount"].abs() * 50
    df["Time"] = np.arange(n_samples)
    df["Class"] = y
    return df


def load_data(test_size: float = 0.3, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test, feature_names, source."""
    df = _load_local()
    source = "local file (data/creditcard.csv)"
    if df is None:
        df = _load_openml()
        source = f"OpenML mirror (data_id={OPENML_DATA_ID})"
    if df is None:
        df = _make_synthetic()
        source = "synthetic fallback (dataset unavailable)"

    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"]).values
    feature_names = [c for c in df.columns if c != "Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"[data] Loaded {len(df):,} rows from {source} "
        f"({y.mean() * 100:.3f}% positive rate)."
    )
    return X_train, X_test, y_train, y_test, feature_names, source
