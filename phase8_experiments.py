"""Phase 8 Channel-Wise Reconstruction Residuals for Industrial Fault Detection.

Investigates per-sensor reconstruction residuals:
- R[i, j] = (X[i, j] - X_hat[i, j])^2 across all 8 sensor channels.
- Method A: Per-Channel 3-Sigma (R[:, j] > mu_j + 3*sigma_j, sample flagged if any channel exceeds).
- Method B: Per-Channel 99th Percentile (R[:, j] > q99_j, sample flagged if any channel exceeds).
- Method C: Maximum Normalized Residual Scoring (score[i] = max_j z[i, j], sample flagged if score > q99_train).

All thresholds derived strictly from normal training data.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def compute_channel_residuals(model: Any, X: np.ndarray) -> np.ndarray:
    """Compute per-sensor squared reconstruction residuals: (X - X_hat)^2.

    Parameters
    ----------
    model : RBM
        Trained RBM model providing deterministic reconstruct(X).
    X : np.ndarray, shape (N, n_visible)
        Input observations.

    Returns
    -------
    np.ndarray, shape (N, n_visible)
        Matrix of squared reconstruction residuals.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    v_rec = model.reconstruct(X_arr)
    return (X_arr - v_rec) ** 2


def evaluate_detection_metrics(predictions: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Calculate classification metrics and fault window breakdowns safely.

    Parameters
    ----------
    predictions : np.ndarray of bool
        Predicted anomaly flags.
    y_test : np.ndarray of bool
        Ground-truth anomaly mask.

    Returns
    -------
    dict
        Dictionary of TP, FP, TN, FN, precision, recall, f1_score, and window counts.
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


def compute_sensor_diagnostics(
    data: Dict[str, Any],
    r_train: np.ndarray,
    r_test: np.ndarray,
    flags_a: np.ndarray,
    flags_b: np.ndarray,
) -> List[Dict[str, Any]]:
    """Generate detailed per-sensor diagnostic statistics across fault regimes.

    Parameters
    ----------
    data : dict
        Dataset dictionary with 'sensor_names', 'y_test'.
    r_train : np.ndarray, shape (N_train, 8)
        Training residual matrix.
    r_test : np.ndarray, shape (N_test, 8)
        Test residual matrix.
    flags_a : np.ndarray of bool, shape (N_test, 8)
        Channel-wise flags from Method A (3-sigma).
    flags_b : np.ndarray of bool, shape (N_test, 8)
        Channel-wise flags from Method B (99th %ile).

    Returns
    -------
    list of dict
        Per-sensor statistics list.
    """
    sensor_names = data["sensor_names"]
    y_test = data["y_test"]
    n_sensors = len(sensor_names)

    diagnostics: List[Dict[str, Any]] = []

    for j in range(n_sensors):
        train_col = r_train[:, j]
        test_col = r_test[:, j]

        mu_j = float(np.mean(train_col))
        sigma_j = float(np.std(train_col, ddof=0))
        q99_j = float(np.percentile(train_col, 99.0))

        norm_mean = float(np.mean(test_col[~y_test]))
        w1_mean = float(np.mean(test_col[300:350]))
        w2_mean = float(np.mean(test_col[420:440]))

        # Detections per window for this sensor
        w1_flagged_a = int(np.sum(flags_a[300:350, j]))
        w2_flagged_a = int(np.sum(flags_a[420:440, j]))
        fp_flagged_a = int(np.sum(flags_a[~y_test, j]))

        w1_flagged_b = int(np.sum(flags_b[300:350, j]))
        w2_flagged_b = int(np.sum(flags_b[420:440, j]))
        fp_flagged_b = int(np.sum(flags_b[~y_test, j]))

        diagnostics.append({
            "channel_idx": j,
            "sensor_name": sensor_names[j],
            "train_mean": mu_j,
            "train_std": sigma_j,
            "train_q99": q99_j,
            "test_norm_mean": norm_mean,
            "w1_mean": w1_mean,
            "w2_mean": w2_mean,
            "method_a": {
                "w1_detected": w1_flagged_a,
                "w2_detected": w2_flagged_a,
                "fp": fp_flagged_a,
            },
            "method_b": {
                "w1_detected": w1_flagged_b,
                "w2_detected": w2_flagged_b,
                "fp": fp_flagged_b,
            },
        })

    return diagnostics


def run_phase8_experiments(model: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete Phase 8 channel-wise evaluation suite.

    Parameters
    ----------
    model : RBM
        Trained Bernoulli RBM model.
    data : dict
        Dataset dictionary with 'X_train', 'X_test', 'y_test', 'sensor_names'.

    Returns
    -------
    dict
        Structured results with residuals, channel thresholds, metrics, and diagnostics.
    """
    x_train = data["X_train"]
    x_test = data["X_test"]
    y_test = data["y_test"]

    # 1. Compute per-sensor residual matrices
    r_train = compute_channel_residuals(model, x_train)  # (2000, 8)
    r_test = compute_channel_residuals(model, x_test)    # (500, 8)

    # 2. Derive per-channel training statistics
    mu_j = np.mean(r_train, axis=0)
    sigma_j = np.std(r_train, axis=0, ddof=0)
    q99_j = np.percentile(r_train, 99.0, axis=0)

    # Numerical safeguard to avoid division by zero on constant/near-zero variance channels
    variance_safeguard = 1e-8
    safe_sigma_j = np.where(sigma_j < variance_safeguard, variance_safeguard, sigma_j)

    # -------------------------------------------------------------
    # Method A: Per-Channel 3-Sigma
    # Flag channel j when: R_test[i, j] > mu_j + 3 * sigma_j
    # Flag sample i if: any channel exceeds its threshold
    # -------------------------------------------------------------
    tau_a = mu_j + 3.0 * sigma_j
    channel_flags_a = r_test > tau_a
    preds_a = np.any(channel_flags_a, axis=1)
    metrics_a = evaluate_detection_metrics(preds_a, y_test)

    # -------------------------------------------------------------
    # Method B: Per-Channel 99th Percentile
    # Flag channel j when: R_test[i, j] > q99_j
    # Flag sample i if: any channel exceeds its threshold
    # -------------------------------------------------------------
    tau_b = q99_j
    channel_flags_b = r_test > tau_b
    preds_b = np.any(channel_flags_b, axis=1)
    metrics_b = evaluate_detection_metrics(preds_b, y_test)

    # -------------------------------------------------------------
    # Method C: Maximum Normalized Residual Scoring (Top-1 / Max-Z)
    # score[i] = max_j z[i, j] where z[i, j] = (R[i, j] - mu_j) / sigma_j
    # Threshold = 99th percentile of score_train
    # -------------------------------------------------------------
    z_train = (r_train - mu_j) / safe_sigma_j
    score_train = np.max(z_train, axis=1)
    tau_c = float(np.percentile(score_train, 99.0))

    z_test = (r_test - mu_j) / safe_sigma_j
    score_test = np.max(z_test, axis=1)
    preds_c = score_test > tau_c
    metrics_c = evaluate_detection_metrics(preds_c, y_test)

    # Sensor diagnostics
    sensor_diagnostics = compute_sensor_diagnostics(
        data, r_train, r_test, channel_flags_a, channel_flags_b
    )

    # -------------------------------------------------------------
    # Historical Methods for Comparison (Without modifying their definitions)
    # -------------------------------------------------------------
    # 1. Historical Baseline (Upper 3-Sigma Free Energy)
    train_fe = np.asarray(model.compute_free_energy(x_train), dtype=np.float64)
    test_fe = np.asarray(model.compute_free_energy(x_test), dtype=np.float64)
    mu_fe = float(np.mean(train_fe))
    sigma_fe = float(np.std(train_fe, ddof=0))
    preds_hist_baseline = test_fe > (mu_fe + 3.0 * sigma_fe)
    metrics_hist_baseline = evaluate_detection_metrics(preds_hist_baseline, y_test)

    # 2. Historical Phase 6 (Lower 1st-Percentile Free Energy)
    tau_p6_q1 = float(np.percentile(train_fe, 1.0))
    preds_hist_p6_q1 = test_fe < tau_p6_q1
    metrics_hist_p6_q1 = evaluate_detection_metrics(preds_hist_p6_q1, y_test)

    # 3. Historical Phase 7 (Reconstruction-Error 3-Sigma)
    train_rec_scalar = np.mean(r_train, axis=1)
    test_rec_scalar = np.mean(r_test, axis=1)
    mu_rec_scalar = float(np.mean(train_rec_scalar))
    sigma_rec_scalar = float(np.std(train_rec_scalar, ddof=0))
    tau_p7_rec_3s = mu_rec_scalar + 3.0 * sigma_rec_scalar
    preds_hist_p7_rec = test_rec_scalar > tau_p7_rec_3s
    metrics_hist_p7_rec = evaluate_detection_metrics(preds_hist_p7_rec, y_test)

    # 4. Historical Phase 7 (Bilateral 3-Sigma Free Energy)
    preds_hist_p7_bilateral = np.abs(test_fe - mu_fe) > 3.0 * sigma_fe
    metrics_hist_p7_bilateral = evaluate_detection_metrics(preds_hist_p7_bilateral, y_test)

    methods = {
        "hist_baseline": {
            "name": "Historical Baseline (Upper 3-Sigma FE)",
            "rule": "F(v) > mu + 3*sigma",
            "metrics": metrics_hist_baseline,
        },
        "hist_phase6_q1": {
            "name": "Historical Phase 6 (Lower 1st %ile FE)",
            "rule": "F(v) < q1",
            "metrics": metrics_hist_p6_q1,
        },
        "hist_phase7_bilateral": {
            "name": "Historical Phase 7 (Bilateral 3-Sigma FE)",
            "rule": "|F(v) - mu| > 3*sigma",
            "metrics": metrics_hist_p7_bilateral,
        },
        "hist_phase7_rec_3s": {
            "name": "Historical Phase 7 (Recon-Error 3-Sigma)",
            "rule": "RecErr_mean > mu + 3*sigma",
            "metrics": metrics_hist_p7_rec,
        },
        "phase8_method_a": {
            "name": "Method A: Per-Channel 3-Sigma (Union)",
            "rule": "Any channel R[:, j] > mu_j + 3*sigma_j",
            "thresholds": tau_a,
            "predictions": preds_a,
            "channel_flags": channel_flags_a,
            "metrics": metrics_a,
        },
        "phase8_method_b": {
            "name": "Method B: Per-Channel 99th %ile (Union)",
            "rule": "Any channel R[:, j] > q99_j",
            "thresholds": tau_b,
            "predictions": preds_b,
            "channel_flags": channel_flags_b,
            "metrics": metrics_b,
        },
        "phase8_method_c": {
            "name": "Method C: Max Normalized Residual (99th %ile)",
            "rule": "max_j z_j > train_q99",
            "threshold": tau_c,
            "predictions": preds_c,
            "score_train": score_train,
            "score_test": score_test,
            "metrics": metrics_c,
        },
    }

    return {
        "r_train": r_train,
        "r_test": r_test,
        "mu_j": mu_j,
        "sigma_j": sigma_j,
        "q99_j": q99_j,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "tau_c": tau_c,
        "score_test_c": score_test,
        "sensor_diagnostics": sensor_diagnostics,
        "methods": methods,
    }
