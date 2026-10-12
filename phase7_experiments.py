"""Phase 7 Advanced Anomaly Detection Experiments.

Investigates:
1. Bilateral free-energy thresholding (A1: 3-sigma, A2: [1st, 99th] percentiles).
2. Per-sample reconstruction error (B1: 3-sigma, B2: 99th percentile).
3. Combined multi-score detectors (C1: OR [A1 | B2], C2: AND [A1 & B2]).

All thresholds are derived strictly from normal training data without test label usage.
Historical Phase 6 comparisons are included for reference.
"""

from typing import Dict, Any, Tuple
import numpy as np


def compute_sample_reconstruction_error(model: Any, X: np.ndarray) -> np.ndarray:
    """Calculate deterministic mean squared reconstruction error per sample.

    Parameters
    ----------
    model : RBM
        Trained RBM model providing reconstruct(X).
    X : np.ndarray, shape (N, n_visible)
        Input observations.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-sample MSE: mean((X - reconstruct(X))^2, axis=1).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    v_rec = model.reconstruct(X_arr)
    return np.mean((X_arr - v_rec) ** 2, axis=1)


def compute_detailed_distribution_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute distribution summary: count, mean, std, min, max, q1, q50, q99.

    Parameters
    ----------
    arr : np.ndarray
        1D score array.

    Returns
    -------
    dict
        Descriptive statistics dictionary.
    """
    clean = np.asarray(arr, dtype=np.float64)
    if len(clean) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q99": 0.0,
        }

    return {
        "count": int(len(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=0)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "q1": float(np.percentile(clean, 1.0)),
        "median": float(np.percentile(clean, 50.0)),
        "q99": float(np.percentile(clean, 99.0)),
    }


def evaluate_predictions(predictions: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Calculate confusion matrix, precision, recall, F1, and window breakdowns.

    Parameters
    ----------
    predictions : np.ndarray of bool
        Boolean prediction mask.
    y_test : np.ndarray of bool
        Ground-truth anomaly mask.

    Returns
    -------
    dict
        Dictionary of classification metrics.
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

    w1_det = int(np.sum(preds[300:350]))
    w2_det = int(np.sum(preds[420:440]))

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
        "w1_detected": w1_det,
        "w1_total": 50,
        "w2_detected": w2_det,
        "w2_total": 20,
    }


def run_phase7_experiments(model: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete Phase 7 experimental evaluation across all methods.

    Parameters
    ----------
    model : RBM
        Trained RBM model.
    data : dict
        Dataset containing 'X_train', 'X_test', 'y_test'.

    Returns
    -------
    dict
        Comprehensive results including distributions, thresholds, predictions, and metrics.
    """
    x_train = data["X_train"]
    x_test = data["X_test"]
    y_test = data["y_test"]

    # 1. Compute Free Energy scores
    train_fe = np.asarray(model.compute_free_energy(x_train), dtype=np.float64)
    test_fe = np.asarray(model.compute_free_energy(x_test), dtype=np.float64)

    # 2. Compute Reconstruction Error scores (sample-wise MSE)
    train_rec = compute_sample_reconstruction_error(model, x_train)
    test_rec = compute_sample_reconstruction_error(model, x_test)

    # 3. Free Energy Distribution Diagnostics
    fe_diagnostics = {
        "train": compute_detailed_distribution_stats(train_fe),
        "test_normal": compute_detailed_distribution_stats(test_fe[~y_test]),
        "test_anomalies": compute_detailed_distribution_stats(test_fe[y_test]),
        "w1": compute_detailed_distribution_stats(test_fe[300:350]),
        "w2": compute_detailed_distribution_stats(test_fe[420:440]),
    }

    # 4. Reconstruction Error Distribution Diagnostics
    rec_diagnostics = {
        "train": compute_detailed_distribution_stats(train_rec),
        "test_normal": compute_detailed_distribution_stats(test_rec[~y_test]),
        "test_anomalies": compute_detailed_distribution_stats(test_rec[y_test]),
        "w1": compute_detailed_distribution_stats(test_rec[300:350]),
        "w2": compute_detailed_distribution_stats(test_rec[420:440]),
    }

    # -------------------------------------------------------------
    # Historical Comparisons (Baseline & Phase 6)
    # -------------------------------------------------------------
    mu_fe = float(np.mean(train_fe))
    sigma_fe = float(np.std(train_fe, ddof=0))

    # Baseline Upper 3-Sigma: F > mu + 3*sigma
    tau_base = mu_fe + 3.0 * sigma_fe
    preds_base = test_fe > tau_base
    metrics_base = evaluate_predictions(preds_base, y_test)

    # Phase 6 Lower 3-Sigma: F < mu - 3*sigma
    tau_p6_lower_3s = mu_fe - 3.0 * sigma_fe
    preds_p6_lower_3s = test_fe < tau_p6_lower_3s
    metrics_p6_lower_3s = evaluate_predictions(preds_p6_lower_3s, y_test)

    # Phase 6 Lower 1st Percentile: F < q1
    tau_p6_q1 = float(np.percentile(train_fe, 1.0))
    preds_p6_q1 = test_fe < tau_p6_q1
    metrics_p6_q1 = evaluate_predictions(preds_p6_q1, y_test)

    # -------------------------------------------------------------
    # Experiment A: Bilateral Free-Energy Scoring
    # -------------------------------------------------------------
    # A1 — Bilateral 3-Sigma: abs(F - mu) > 3*sigma
    preds_a1 = np.abs(test_fe - mu_fe) > 3.0 * sigma_fe
    metrics_a1 = evaluate_predictions(preds_a1, y_test)

    # A2 — Bilateral Percentile: F < q1 or F > q99
    tau_a2_low = float(np.percentile(train_fe, 1.0))
    tau_a2_high = float(np.percentile(train_fe, 99.0))
    preds_a2 = (test_fe < tau_a2_low) | (test_fe > tau_a2_high)
    metrics_a2 = evaluate_predictions(preds_a2, y_test)

    # -------------------------------------------------------------
    # Experiment B: Reconstruction-Error Scoring
    # -------------------------------------------------------------
    mu_rec = float(np.mean(train_rec))
    sigma_rec = float(np.std(train_rec, ddof=0))

    # B1 — Reconstruction Error 3-Sigma: RecErr > mu + 3*sigma
    tau_b1 = mu_rec + 3.0 * sigma_rec
    preds_b1 = test_rec > tau_b1
    metrics_b1 = evaluate_predictions(preds_b1, y_test)

    # B2 — Reconstruction Error 99th Percentile: RecErr > q99
    tau_b2 = float(np.percentile(train_rec, 99.0))
    preds_b2 = test_rec > tau_b2
    metrics_b2 = evaluate_predictions(preds_b2, y_test)

    # -------------------------------------------------------------
    # Experiment C: Combined Anomaly Scoring
    # -------------------------------------------------------------
    # C1 — OR Combination: A1 (Bilateral 3-Sigma) OR B2 (Recon 99th percentile)
    preds_c1 = preds_a1 | preds_b2
    metrics_c1 = evaluate_predictions(preds_c1, y_test)

    # C2 — AND Combination: A1 (Bilateral 3-Sigma) AND B2 (Recon 99th percentile)
    preds_c2 = preds_a1 & preds_b2
    metrics_c2 = evaluate_predictions(preds_c2, y_test)

    methods = {
        "baseline": {
            "name": "Historical Baseline (Upper 3-Sigma)",
            "rule": "F(v) > mu + 3*sigma",
            "thresholds": {"tau": tau_base},
            "predictions": preds_base,
            "metrics": metrics_base,
        },
        "phase6_lower_3s": {
            "name": "Historical Phase 6 (Lower 3-Sigma)",
            "rule": "F(v) < mu - 3*sigma",
            "thresholds": {"tau": tau_p6_lower_3s},
            "predictions": preds_p6_lower_3s,
            "metrics": metrics_p6_lower_3s,
        },
        "phase6_lower_q1": {
            "name": "Historical Phase 6 (Lower 1st %ile)",
            "rule": "F(v) < q1",
            "thresholds": {"tau": tau_p6_q1},
            "predictions": preds_p6_q1,
            "metrics": metrics_p6_q1,
        },
        "A1": {
            "name": "A1 - Bilateral 3-Sigma Free Energy",
            "rule": "|F(v) - mu| > 3*sigma",
            "thresholds": {
                "lower": mu_fe - 3.0 * sigma_fe,
                "upper": mu_fe + 3.0 * sigma_fe,
                "mu": mu_fe,
                "sigma": sigma_fe,
            },
            "predictions": preds_a1,
            "metrics": metrics_a1,
        },
        "A2": {
            "name": "A2 - Bilateral Percentile Free Energy",
            "rule": "F(v) < q1 OR F(v) > q99",
            "thresholds": {"lower": tau_a2_low, "upper": tau_a2_high},
            "predictions": preds_a2,
            "metrics": metrics_a2,
        },
        "B1": {
            "name": "B1 - Reconstruction Error 3-Sigma",
            "rule": "RecErr > mu + 3*sigma",
            "thresholds": {"tau": tau_b1, "mu": mu_rec, "sigma": sigma_rec},
            "predictions": preds_b1,
            "metrics": metrics_b1,
        },
        "B2": {
            "name": "B2 - Reconstruction Error 99th %ile",
            "rule": "RecErr > q99",
            "thresholds": {"tau": tau_b2},
            "predictions": preds_b2,
            "metrics": metrics_b2,
        },
        "C1": {
            "name": "C1 - Combined OR (A1 | B2)",
            "rule": "Bilateral 3-Sigma FE OR Recon 99th %ile",
            "thresholds": {"fe_bilateral_3s": (mu_fe - 3.0 * sigma_fe, mu_fe + 3.0 * sigma_fe), "rec_q99": tau_b2},
            "predictions": preds_c1,
            "metrics": metrics_c1,
        },
        "C2": {
            "name": "C2 - Combined AND (A1 & B2)",
            "rule": "Bilateral 3-Sigma FE AND Recon 99th %ile",
            "thresholds": {"fe_bilateral_3s": (mu_fe - 3.0 * sigma_fe, mu_fe + 3.0 * sigma_fe), "rec_q99": tau_b2},
            "predictions": preds_c2,
            "metrics": metrics_c2,
        },
    }

    return {
        "train_fe": train_fe,
        "test_fe": test_fe,
        "train_rec": train_rec,
        "test_rec": test_rec,
        "fe_diagnostics": fe_diagnostics,
        "rec_diagnostics": rec_diagnostics,
        "methods": methods,
    }
