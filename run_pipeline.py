"""End-to-end pipeline: run with

    python run_pipeline.py

It will (1) load the credit-card fraud dataset, (2) train a baseline
classifier and show it discriminates well but is poorly calibrated,
(3) compare Platt scaling and isotonic regression, (4) demonstrate the
classic undersampling probability-correction, and (5) compare a plain
cross-entropy neural network to one trained with an added differentiable
calibration loss. All plots and a metrics table are written to results/.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.calibrate import (
    correct_undersampled_probabilities,
    discrimination_report,
    isotonic_scaling,
    make_base_estimator,
    platt_scaling,
    undersample,
)
from src.data import load_data
from src.differentiable_calibration import train_mlp
from src.metrics import (
    expected_calibration_error,
    plot_reliability,
    reliability_curve,
    top_decile_calibration_error,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def full_report(name: str, y_true, y_prob, results: dict):
    report = discrimination_report(y_true, y_prob)
    report["ece"] = expected_calibration_error(y_true, y_prob)
    report["top_decile_ce"] = top_decile_calibration_error(y_true, y_prob)
    results[name] = report
    print(
        f"[{name}] AUC-ROC={report['auc_roc']:.4f}  AUC-PR={report['auc_pr']:.4f}  "
        f"Brier={report['brier']:.5f}  ECE={report['ece']:.5f}  "
        f"Top-decile CE={report['top_decile_ce']:.5f}"
    )
    return report


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results: dict = {}

    # ---------------------------------------------------------------
    # 1. Data
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test, _feature_names, source = load_data()

    # ---------------------------------------------------------------
    # 2. Baseline: good discrimination, poor calibration
    # ---------------------------------------------------------------
    baseline = make_base_estimator()
    baseline.fit(X_train, y_train)
    p_baseline = baseline.predict_proba(X_test)[:, 1]
    full_report("baseline_uncalibrated", y_test, p_baseline, results)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_reliability(ax, y_test, p_baseline, label="Baseline (uncalibrated)")
    ax.set_title("Baseline model: reliability diagram")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "01_baseline_reliability.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 3. Post-hoc calibration: Platt scaling vs isotonic regression
    # ---------------------------------------------------------------
    platt = platt_scaling(make_base_estimator, X_train, y_train, cv=3)
    p_platt = platt.predict_proba(X_test)[:, 1]
    full_report("platt_scaling", y_test, p_platt, results)

    isotonic = isotonic_scaling(make_base_estimator, X_train, y_train, cv=3)
    p_isotonic = isotonic.predict_proba(X_test)[:, 1]
    full_report("isotonic_regression", y_test, p_isotonic, results)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_reliability(ax, y_test, p_baseline, label="Uncalibrated")
    ax.plot(
        *_reliability_xy(y_test, p_platt),
        marker="s",
        linewidth=1.5,
        label="Platt scaling",
    )
    ax.plot(
        *_reliability_xy(y_test, p_isotonic),
        marker="^",
        linewidth=1.5,
        label="Isotonic regression",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Post-hoc calibration methods")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "02_calibration_methods_comparison.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 4. Undersampling and the Elkan / Dal Pozzolo-Caelen correction
    # ---------------------------------------------------------------
    X_us, y_us, beta = undersample(X_train, y_train, neg_pos_ratio=5.0)
    us_model = make_base_estimator()
    us_model.fit(X_us, y_us)
    p_us_raw = us_model.predict_proba(X_test)[:, 1]
    full_report("undersampled_raw", y_test, p_us_raw, results)

    p_us_corrected = correct_undersampled_probabilities(p_us_raw, beta)
    full_report("undersampled_corrected", y_test, p_us_corrected, results)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_reliability(ax, y_test, p_us_raw, label="Undersampled (raw)")
    ax.plot(
        *_reliability_xy(y_test, p_us_corrected),
        marker="s",
        linewidth=1.5,
        label="Undersampled (corrected)",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Undersampling probability correction (beta={beta:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "03_undersampling_correction.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 5. Differentiable calibration loss (neural network)
    # ---------------------------------------------------------------
    X_us_nn, y_us_nn, _ = undersample(X_train, y_train, neg_pos_ratio=10.0, random_state=1)
    scaler = StandardScaler().fit(X_us_nn)
    X_us_nn_scaled = scaler.transform(X_us_nn)
    X_test_scaled = scaler.transform(X_test)

    _, p_mlp_plain = train_mlp(
        X_us_nn_scaled, y_us_nn, X_test_scaled, calibration_weight=0.0, n_epochs=25
    )
    full_report("mlp_cross_entropy_only", y_test, p_mlp_plain, results)

    _, p_mlp_calibrated = train_mlp(
        X_us_nn_scaled, y_us_nn, X_test_scaled, calibration_weight=0.5, n_epochs=25
    )
    full_report("mlp_with_soft_ece_loss", y_test, p_mlp_calibrated, results)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_reliability(ax, y_test, p_mlp_plain, label="Cross-entropy only")
    ax.plot(
        *_reliability_xy(y_test, p_mlp_calibrated),
        marker="s",
        linewidth=1.5,
        label="Cross-entropy + soft-ECE loss",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Differentiable calibration loss (neural network)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "04_differentiable_calibration_loss.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w") as fh:
        json.dump({"data_source": source, "metrics": results}, fh, indent=2)

    write_summary_markdown(source, results)
    print(f"\nAll plots and metrics written to {RESULTS_DIR}/")


def _reliability_xy(y_true, y_prob, n_bins: int = 15):
    confs, accs, _ = reliability_curve(y_true, y_prob, n_bins=n_bins)
    return confs, accs


def write_summary_markdown(source: str, results: dict):
    rows = [
        ("Baseline (uncalibrated)", "baseline_uncalibrated"),
        ("Platt scaling", "platt_scaling"),
        ("Isotonic regression", "isotonic_regression"),
        ("Undersampled (raw)", "undersampled_raw"),
        ("Undersampled (Elkan-corrected)", "undersampled_corrected"),
        ("MLP, cross-entropy only", "mlp_cross_entropy_only"),
        ("MLP, cross-entropy + soft-ECE", "mlp_with_soft_ece_loss"),
    ]
    lines = [
        f"# Results\n",
        f"Data source: `{source}`\n",
        "| Method | AUC-ROC | AUC-PR | Brier | ECE | Top-decile CE |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in rows:
        r = results[key]
        lines.append(
            f"| {label} | {r['auc_roc']:.4f} | {r['auc_pr']:.4f} | "
            f"{r['brier']:.5f} | {r['ece']:.5f} | {r['top_decile_ce']:.5f} |"
        )
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
