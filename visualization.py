"""Diagnostic visualization module for industrial fault detection report.

Generates a dual-panel visualization:
1. Top panel: Multivariate continuous sensor telemetry time series across all 8 channels
   with shaded ground-truth fault injection windows and predicted anomalies.
2. Bottom panel: Analytical sample free energy timeline with the training-derived 3-sigma
   threshold and highlighted fault windows.
"""

from typing import Dict, Any, Union
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Headless backend for terminal and automated execution
import matplotlib.pyplot as plt
import numpy as np


def save_anomaly_report(
    data: Dict[str, Any],
    results: Dict[str, Any],
    output_path: Union[str, Path] = "anomaly_report.png",
) -> Path:
    """Generate and save dual-panel anomaly diagnostic report.

    Parameters
    ----------
    data : dict
        Dataset dictionary produced by generate_sensor_data, containing:
        - 'X_test': np.ndarray, shape (500, 8)
        - 'sensor_names': list of 8 sensor names
        - 'y_test': np.ndarray of bool, shape (500,)
    results : dict
        Evaluation dictionary produced by evaluate_anomaly_detector, containing:
        - 'test_energy': np.ndarray, shape (500,)
        - 'threshold': float
        - 'predictions': np.ndarray of bool, shape (500,)
    output_path : str or Path, default="anomaly_report.png"
        Destination file path for the rendered figure.

    Returns
    -------
    Path
        Absolute path to the saved report image.
    """
    output_file = Path(output_path).resolve()
    x_test = data["X_test"]
    sensor_names = data["sensor_names"]
    y_test = data["y_test"]
    test_energy = results["test_energy"]
    threshold = results["threshold"]
    predictions = results["predictions"]

    n_samples, n_sensors = x_test.shape
    time_steps = np.arange(n_samples)

    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 8), sharex=True, dpi=150
    )

    # -------------------------------------------------------------
    # 1. Top Panel: Multivariate Sensor Time Series
    # -------------------------------------------------------------
    # Use high-contrast color palette for the 8 sensors
    colors = plt.cm.tab10(np.linspace(0, 1, n_sensors))

    for i in range(n_sensors):
        ax_top.plot(
            time_steps,
            x_test[:, i],
            label=sensor_names[i],
            color=colors[i],
            linewidth=1.0,
            alpha=0.8,
        )

    # Highlight ground-truth anomaly windows [300, 350) and [420, 440)
    ax_top.axvspan(
        300,
        350,
        color="crimson",
        alpha=0.18,
        label="Fault Window 1: Saturation/Drift [300, 350)",
    )
    ax_top.axvspan(
        420,
        440,
        color="darkorange",
        alpha=0.20,
        label="Fault Window 2: Noise Spikes [420, 440)",
    )

    # Mark predicted anomalies if any
    pred_indices = np.where(predictions)[0]
    if len(pred_indices) > 0:
        ax_top.scatter(
            pred_indices,
            np.mean(x_test[pred_indices], axis=1),
            color="black",
            marker="x",
            s=40,
            zorder=5,
            label=f"Predicted Anomalies ({len(pred_indices)})",
        )

    ax_top.set_title(
        "Multivariate Sensor Telemetry & Injected Fault Windows (Test Set)",
        fontsize=12,
        fontweight="bold",
    )
    ax_top.set_ylabel("Normalized Telemetry [0.0, 1.0]", fontsize=10)
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # 2. Bottom Panel: Analytical Free-Energy Timeline
    # -------------------------------------------------------------
    ax_bottom.plot(
        time_steps,
        test_energy,
        color="#1f77b4",
        linewidth=1.2,
        label="Test Sample Free Energy $F(v)$",
    )

    # Threshold horizontal line
    ax_bottom.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Decision Threshold $\\tau = \\mu_{{train}} + 3\\sigma$ ({threshold:.4f})",
    )

    # Re-shade fault windows on bottom panel
    ax_bottom.axvspan(300, 350, color="crimson", alpha=0.18)
    ax_bottom.axvspan(420, 440, color="darkorange", alpha=0.20)

    # Mark any detected anomalies on the free-energy curve
    if len(pred_indices) > 0:
        ax_bottom.scatter(
            pred_indices,
            test_energy[pred_indices],
            color="red",
            marker="o",
            s=30,
            zorder=5,
            label="Detected Flag ($F(v) > \\tau$)",
        )

    ax_bottom.set_title(
        "Analytical Free-Energy Timeline vs. Training Decision Threshold",
        fontsize=12,
        fontweight="bold",
    )
    ax_bottom.set_xlabel("Test Telemetry Index (Time Step)", fontsize=10)
    ax_bottom.set_ylabel("Free Energy $F(v)$", fontsize=10)
    ax_bottom.grid(True, linestyle="--", alpha=0.5)
    ax_bottom.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

    return output_file
