"""Scientific diagnosis and controlled anomaly detection experiments for Bernoulli RBM.

This module evaluates alternative, training-derived detection rules:
- Approach A: Original Baseline (Upper 3-sigma tail: F > mu + 3*sigma)
- Approach B: Lower 3-sigma tail (F < mu - 3*sigma)
- Approach C: Training Lower 1st Percentile (F < q_0.01)

All decision thresholds are derived strictly from normal training telemetry.
Ground-truth test labels are used solely for diagnostic evaluation, never for tuning.
"""

from typing import Dict, Any, Tuple
import numpy as np


def compute_distribution_statistics(arr: np.ndarray) -> Dict[str, float]:
    """Calculate key descriptive statistics and quantiles for an array.

    Parameters
    ----------
    arr : np.ndarray
        1D array of scores.

    Returns
    -------
    dict
        Dictionary containing mean, std (ddof=0), min, max, and quantiles.
    """
    arr_clean = np.asarray(arr, dtype=np.float64)
    quantiles = [0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9]

    if len(arr_clean) == 0:
        stats: Dict[str, float] = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
        for q in quantiles:
            stats[f"q_{q}"] = 0.0
        return stats

    q_vals = np.percentile(arr_clean, quantiles)

    stats: Dict[str, float] = {
        "mean": float(np.mean(arr_clean)),
        "std": float(np.std(arr_clean, ddof=0)),
        "min": float(np.min(arr_clean)),
        "max": float(np.max(arr_clean)),
    }
    for q, val in zip(quantiles, q_vals):
        stats[f"q_{q}"] = float(val)
    return stats


def calculate_metrics(predictions: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Calculate standard binary classification metrics safely.

    Parameters
    ----------
    predictions : np.ndarray of bool
        Boolean predictions array.
    y_test : np.ndarray of bool
        Ground-truth anomaly mask.

    Returns
    -------
    dict
        Dictionary containing TP, FP, TN, FN, precision, recall, f1_score.
    """
    preds = np.asarray(predictions, dtype=bool)
    labels = np.asarray(y_test, dtype=bool)

    tp = int(np.sum(preds & labels))
    fp = int(np.sum(preds & ~labels))
    tn = int(np.sum(~preds & ~labels))
    fn = int(np.sum(~preds & labels))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    # Window-specific detection rates
    # Window 1: [300, 350) (50 samples)
    # Window 2: [420, 440) (20 samples)
    w1_detected = int(np.sum(preds[300:350]))
    w2_detected = int(np.sum(preds[420:440]))

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "injected_count": int(np.sum(labels)),
        "detected_count": int(np.sum(preds)),
        "w1_detected": w1_detected,
        "w1_total": 50,
        "w2_detected": w2_detected,
        "w2_total": 20,
    }


def run_detection_experiments(
    model: Any,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute controlled comparison across Approaches A, B, and C.

    Parameters
    ----------
    model : RBM
        Fitted RBM model.
    data : dict
        Dataset dictionary with 'X_train', 'X_test', 'y_test'.

    Returns
    -------
    dict
        Structured results containing thresholds, predictions, and metrics.
    """
    x_train = data["X_train"]
    x_test = data["X_test"]
    y_test = data["y_test"]

    train_energy: np.ndarray = np.asarray(
        model.compute_free_energy(x_train), dtype=np.float64
    )
    test_energy: np.ndarray = np.asarray(
        model.compute_free_energy(x_test), dtype=np.float64
    )

    mu_train = float(np.mean(train_energy))
    sigma_train = float(np.std(train_energy, ddof=0))

    # -------------------------------------------------------------
    # Approach A: Baseline Upper 3-Sigma Rule (F > mu + 3*sigma)
    # -------------------------------------------------------------
    threshold_a = mu_train + 3.0 * sigma_train
    preds_a = test_energy > threshold_a
    metrics_a = calculate_metrics(preds_a, y_test)

    # -------------------------------------------------------------
    # Approach B: Lower 3-Sigma Rule (F < mu - 3*sigma)
    # -------------------------------------------------------------
    threshold_b = mu_train - 3.0 * sigma_train
    preds_b = test_energy < threshold_b
    metrics_b = calculate_metrics(preds_b, y_test)

    # -------------------------------------------------------------
    # Approach C: Lower 1st Percentile Rule (F < 1st percentile of train)
    # -------------------------------------------------------------
    threshold_c = float(np.percentile(train_energy, 1.0))
    preds_c = test_energy < threshold_c
    metrics_c = calculate_metrics(preds_c, y_test)

    # Diagnostic distribution statistics (for reporting only)
    train_stats = compute_distribution_statistics(train_energy)
    test_norm_stats = compute_distribution_statistics(test_energy[~y_test])
    test_anom_stats = compute_distribution_statistics(test_energy[y_test])
    w1_stats = compute_distribution_statistics(test_energy[300:350])
    w2_stats = compute_distribution_statistics(test_energy[420:440])

    return {
        "train_energy": train_energy,
        "test_energy": test_energy,
        "train_stats": train_stats,
        "test_norm_stats": test_norm_stats,
        "test_anom_stats": test_anom_stats,
        "w1_stats": w1_stats,
        "w2_stats": w2_stats,
        "approach_a": {
            "name": "Approach A (Baseline Upper 3-Sigma)",
            "rule": "F(v) > mu_train + 3*sigma",
            "threshold": threshold_a,
            "predictions": preds_a,
            "metrics": metrics_a,
        },
        "approach_b": {
            "name": "Approach B (Lower 3-Sigma)",
            "rule": "F(v) < mu_train - 3*sigma",
            "threshold": threshold_b,
            "predictions": preds_b,
            "metrics": metrics_b,
        },
        "approach_c": {
            "name": "Approach C (Lower 1st Percentile)",
            "rule": "F(v) < train_quantile_1%",
            "threshold": threshold_c,
            "predictions": preds_c,
            "metrics": metrics_c,
        },
    }
