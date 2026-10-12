"""Analytical Free-Energy Anomaly Detector for Bernoulli RBMs.

This module evaluates RBM-based industrial fault detection using analytical
free energy. The statistical decision threshold is derived exclusively from
normal training data (mean + 3 * std), and anomalies are flagged when test
sample free energy exceeds this threshold.
"""

from typing import Dict, Any
import numpy as np


def evaluate_anomaly_detector(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate fault detection performance using analytical free energy.

    Parameters
    ----------
    model : RBM
        Trained RBM model providing compute_free_energy(X).
    X_train : np.ndarray, shape (n_train, n_visible)
        Normal training dataset in [0.0, 1.0].
    X_test : np.ndarray, shape (n_test, n_visible)
        Test dataset with injected faults in [0.0, 1.0].
    y_test : np.ndarray of bool, shape (n_test,)
        Ground truth boolean anomaly mask (True indicates fault).

    Returns
    -------
    dict
        Dictionary containing:
        - 'train_energy': np.ndarray, shape (n_train,), free energies on training set.
        - 'test_energy': np.ndarray, shape (n_test,), free energies on test set.
        - 'threshold': float, statistical decision boundary (mean + 3*std, ddof=0).
        - 'predictions': np.ndarray of bool, shape (n_test,), True if test_energy > threshold.
        - 'y_test': np.ndarray of bool, shape (n_test,), ground-truth anomaly mask.
        - 'TP': int, True Positives (injected anomalies correctly flagged).
        - 'FP': int, False Positives (normal test telemetry flagged as anomalies).
        - 'TN': int, True Negatives (normal telemetry correctly flagged normal).
        - 'FN': int, False Negatives (injected anomalies missed).
        - 'precision': float, TP / (TP + FP) or 0.0 if undefined.
        - 'recall': float, TP / (TP + FN) or 0.0 if undefined.
        - 'f1_score': float, harmonic mean of precision and recall or 0.0 if undefined.
        - 'injected_anomaly_count': int, total True entries in y_test.
        - 'detected_anomaly_count': int, total True entries in predictions.
    """
    # 1. Compute free-energy scores for normal training samples
    train_energy: np.ndarray = np.asarray(
        model.compute_free_energy(X_train), dtype=np.float64
    )

    # 2. Derive threshold exclusively from training scores (mean + 3*std, ddof=0)
    train_mean: float = float(np.mean(train_energy))
    train_std: float = float(np.std(train_energy, ddof=0))
    threshold: float = train_mean + 3.0 * train_std

    # 3. Compute free-energy scores for test samples
    test_energy: np.ndarray = np.asarray(
        model.compute_free_energy(X_test), dtype=np.float64
    )

    # 4. Strict anomaly decision rule: test_energy > threshold
    predictions: np.ndarray = test_energy > threshold

    # 5. Format ground truth mask
    y_test_arr: np.ndarray = np.asarray(y_test, dtype=bool)

    # 6. Calculate confusion matrix counts
    tp: int = int(np.sum(predictions & y_test_arr))
    fp: int = int(np.sum(predictions & ~y_test_arr))
    tn: int = int(np.sum(~predictions & ~y_test_arr))
    fn: int = int(np.sum(~predictions & y_test_arr))

    # 7. Safe calculation of precision, recall, and F1-score (avoid division by zero)
    if (tp + fp) > 0:
        precision: float = float(tp / (tp + fp))
    else:
        precision = 0.0

    if (tp + fn) > 0:
        recall: float = float(tp / (tp + fn))
    else:
        recall = 0.0

    if (precision + recall) > 0.0:
        f1_score: float = float(2.0 * (precision * recall) / (precision + recall))
    else:
        f1_score = 0.0

    injected_count: int = int(np.sum(y_test_arr))
    detected_count: int = int(np.sum(predictions))

    return {
        "train_energy": train_energy,
        "test_energy": test_energy,
        "threshold": threshold,
        "predictions": predictions,
        "y_test": y_test_arr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "injected_anomaly_count": injected_count,
        "detected_anomaly_count": detected_count,
    }
