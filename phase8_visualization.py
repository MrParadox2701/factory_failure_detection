"""Phase 8 diagnostic visualization module.

Generates a 4-panel diagnostic figure saved as phase8_anomaly_report.png:
1. Eight sensor continuous telemetry time series with shaded fault windows.
2. Heatmap of per-sensor test squared reconstruction residuals (8 sensors x 500 time steps).
3. Per-sensor threshold vs. mean residual diagnostic bar chart across fault regimes.
4. Comparative classification performance (Precision, Recall, F1-Score) across methods.
"""

from typing import Dict, Any, Union
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np


def save_phase8_anomaly_report(
    data: Dict[str, Any],
    results: Dict[str, Any],
    output_path: Union[str, Path] = "phase8_anomaly_report.png",
) -> Path:
    """Generate and save 4-panel Phase 8 diagnostic artifact.

    Parameters
    ----------
    data : dict
        Dataset containing 'X_test', 'sensor_names', 'y_test'.
    results : dict
        Results dictionary from run_phase8_experiments.
    output_path : str or Path, default="phase8_anomaly_report.png"
        Destination file path.

    Returns
    -------
    Path
        Absolute path to the saved PNG artifact.
    """
    output_file = Path(output_path).resolve()
    x_test = data["X_test"]
    sensor_names = data["sensor_names"]
    y_test = data["y_test"]
    r_test = results["r_test"]  # shape (500, 8)
    diagnostics = results["sensor_diagnostics"]
    methods = results["methods"]

    n_samples, n_sensors = x_test.shape
    time_steps = np.arange(n_samples)

    fig = plt.figure(figsize=(15, 15), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1.1, 1.1, 1.1, 1.0], hspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])

    # -------------------------------------------------------------
    # Panel 1: Multivariate Sensor Time Series
    # -------------------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, n_sensors))
    for i in range(n_sensors):
        ax1.plot(
            time_steps,
            x_test[:, i],
            label=sensor_names[i],
            color=colors[i],
            linewidth=1.0,
            alpha=0.75,
        )

    ax1.axvspan(300, 350, color="crimson", alpha=0.18, label="Fault Window 1: Saturation [300, 350)")
    ax1.axvspan(420, 440, color="darkorange", alpha=0.20, label="Fault Window 2: Spikes [420, 440)")
    ax1.set_title("1. Multivariate Test Telemetry & Injected Fault Windows", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Normalized Value", fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=7.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Heatmap of Per-Sensor Squared Residuals
    # -------------------------------------------------------------
    # r_test shape is (500, 8). Transpose to (8, 500) for sensor rows vs time columns.
    im = ax2.imshow(
        r_test.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=[0, n_samples, -0.5, n_sensors - 0.5],
    )
    cbar = fig.colorbar(im, ax=ax2, pad=0.015, fraction=0.03)
    cbar.set_label("Squared Residual $(X - \\hat{X})^2$", fontsize=8)

    # Mark fault windows on heatmap
    ax2.axvline(300, color="crimson", linestyle="--", linewidth=1.2)
    ax2.axvline(350, color="crimson", linestyle="--", linewidth=1.2)
    ax2.axvline(420, color="darkorange", linestyle="--", linewidth=1.2)
    ax2.axvline(440, color="darkorange", linestyle="--", linewidth=1.2)

    ax2.set_yticks(np.arange(n_sensors))
    ax2.set_yticklabels(sensor_names, fontsize=8)
    ax2.set_title("2. Heatmap of Per-Sensor Squared Reconstruction Residuals Across Test Telemetry", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Sensor Channels", fontsize=9)
    ax2.set_xlabel("Test Telemetry Index (Time Step)", fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Sensor-Level Threshold vs. Mean Residual Diagnostics
    # -------------------------------------------------------------
    x_pos = np.arange(n_sensors)
    bar_w = 0.20

    tau_a_vals = [d["train_mean"] + 3.0 * d["train_std"] for d in diagnostics]
    norm_vals = [d["test_norm_mean"] for d in diagnostics]
    w1_vals = [d["w1_mean"] for d in diagnostics]
    w2_vals = [d["w2_mean"] for d in diagnostics]

    ax3.bar(x_pos - 1.5 * bar_w, tau_a_vals, bar_w, label="Training Threshold ($\\mu + 3\\sigma$)", color="#4e79a7")
    ax3.bar(x_pos - 0.5 * bar_w, norm_vals, bar_w, label="Normal Test Mean", color="#59a14f")
    ax3.bar(x_pos + 0.5 * bar_w, w1_vals, bar_w, label="Window 1 Mean (Saturation)", color="crimson")
    ax3.bar(x_pos + 1.5 * bar_w, w2_vals, bar_w, label="Window 2 Mean (Noise Spikes)", color="darkorange")

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"S{d['channel_idx']}:\n" + d["sensor_name"].split("_")[0] for d in diagnostics], fontsize=8)
    ax3.set_ylabel("Squared Residual MSE", fontsize=9)
    ax3.set_title("3. Per-Sensor Residual Comparison: Normal Baseline vs. Fault Windows 1 & 2", fontsize=11, fontweight="bold")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 4: Comparative Method Performance Bar Chart
    # -------------------------------------------------------------
    method_keys = [
        "hist_baseline",
        "hist_phase6_q1",
        "hist_phase7_bilateral",
        "hist_phase7_rec_3s",
        "phase8_method_a",
        "phase8_method_b",
        "phase8_method_c",
    ]
    labels = [
        "Historical Baseline\n(Upper 3$\\sigma$ FE)",
        "Historical P6\n(Lower 1% FE)",
        "Historical P7\n(Bilateral 3$\\sigma$ FE)",
        "Historical P7\n(Recon 3$\\sigma$)",
        "Phase 8 Method A\n(Per-Chan 3$\\sigma$)",
        "Phase 8 Method B\n(Per-Chan 99%ile)",
        "Phase 8 Method C\n(Max-Z 99%ile)",
    ]

    precisions = [methods[k]["metrics"]["precision"] for k in method_keys]
    recalls = [methods[k]["metrics"]["recall"] for k in method_keys]
    f1_scores = [methods[k]["metrics"]["f1_score"] for k in method_keys]

    x_m = np.arange(len(labels))
    w_m = 0.25

    ax4.bar(x_m - w_m, precisions, w_m, label="Precision", color="#4e79a7")
    ax4.bar(x_m, recalls, w_m, label="Recall", color="#f28e2b")
    rects_f1 = ax4.bar(x_m + w_m, f1_scores, w_m, label="F1-Score", color="#59a14f")

    for rect in rects_f1:
        h = rect.get_height()
        if h > 0.02:
            ax4.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    ax4.set_xticks(x_m)
    ax4.set_xticklabels(labels, fontsize=8)
    ax4.set_ylabel("Score [0.0, 1.0]", fontsize=9)
    ax4.set_ylim(0.0, 1.15)
    ax4.set_title("4. Detection Performance Comparison: Historical vs. Phase 8 Channel-Wise Residuals", fontsize=11, fontweight="bold")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=8, framealpha=0.9)

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

    return output_file
