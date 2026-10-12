"""Phase 7 diagnostic visualization module.

Renders a 4-panel diagnostic figure saved as phase7_anomaly_report.png:
1. Multivariate test sensor measurements with shaded fault windows.
2. Analytical free-energy timeline with training bilateral threshold bounds.
3. Sample reconstruction-error timeline with training 3-sigma and 99th percentile thresholds.
4. Comparative performance metrics (Precision, Recall, F1-Score) across all evaluated methods.
"""

from typing import Dict, Any, Union
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np


def save_phase7_anomaly_report(
    data: Dict[str, Any],
    results: Dict[str, Any],
    output_path: Union[str, Path] = "phase7_anomaly_report.png",
) -> Path:
    """Generate and save 4-panel Phase 7 diagnostic artifact.

    Parameters
    ----------
    data : dict
        Dataset dictionary with 'X_test', 'sensor_names', 'y_test'.
    results : dict
        Results dictionary from run_phase7_experiments.
    output_path : str or Path, default="phase7_anomaly_report.png"
        Destination image path.

    Returns
    -------
    Path
        Absolute path to the saved PNG image.
    """
    output_file = Path(output_path).resolve()
    x_test = data["X_test"]
    sensor_names = data["sensor_names"]
    y_test = data["y_test"]
    test_fe = results["test_fe"]
    test_rec = results["test_rec"]
    methods = results["methods"]

    n_samples, n_sensors = x_test.shape
    time_steps = np.arange(n_samples)

    fig = plt.figure(figsize=(15, 14), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1.2, 1.2, 1.2, 1.0], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0])

    # -------------------------------------------------------------
    # Panel 1: Multivariate Sensor Measurements
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
    ax1.set_title("1. Multivariate Sensor Measurements & Injected Fault Windows (Test Set)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Normalized Telemetry", fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=7.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Free-Energy Scores with Bilateral Thresholds
    # -------------------------------------------------------------
    ax2.plot(time_steps, test_fe, color="#1f77b4", linewidth=1.1, label="Test Free Energy $F(v)$")
    
    # Bilateral 3-sigma lines
    tau_a1_upper = methods["A1"]["thresholds"]["upper"]
    tau_a1_lower = methods["A1"]["thresholds"]["lower"]
    ax2.axhline(tau_a1_upper, color="black", linestyle="--", linewidth=1.2, label=f"Upper 3$\\sigma$ ($\\tau={tau_a1_upper:.3f}$)")
    ax2.axhline(tau_a1_lower, color="purple", linestyle="--", linewidth=1.2, label=f"Lower 3$\\sigma$ ($\\tau={tau_a1_lower:.3f}$)")

    # 1st percentile line
    tau_a2_lower = methods["A2"]["thresholds"]["lower"]
    ax2.axhline(tau_a2_lower, color="darkgreen", linestyle=":", linewidth=1.2, label=f"Lower 1st %ile ($\\tau={tau_a2_lower:.3f}$)")

    ax2.axvspan(300, 350, color="crimson", alpha=0.18)
    ax2.axvspan(420, 440, color="darkorange", alpha=0.20)
    ax2.set_title("2. Analytical Free-Energy Timeline & Training Bilateral Decision Boundaries", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Free Energy $F(v)$", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=7.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 3: Reconstruction Error Scores
    # -------------------------------------------------------------
    ax3.plot(time_steps, test_rec, color="#2ca02c", linewidth=1.1, label="Test Reconstruction MSE")
    tau_b1 = methods["B1"]["thresholds"]["tau"]
    tau_b2 = methods["B2"]["thresholds"]["tau"]
    ax3.axhline(tau_b1, color="red", linestyle="--", linewidth=1.2, label=f"Recon 3$\\sigma$ ($\\tau={tau_b1:.4f}$)")
    ax3.axhline(tau_b2, color="navy", linestyle=":", linewidth=1.2, label=f"Recon 99th %ile ($\\tau={tau_b2:.4f}$)")

    ax3.axvspan(300, 350, color="crimson", alpha=0.18)
    ax3.axvspan(420, 440, color="darkorange", alpha=0.20)
    ax3.set_title("3. Sample-Wise Reconstruction Error MSE & Training Decision Boundaries", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Test Telemetry Index (Time Step)", fontsize=9)
    ax3.set_ylabel("Reconstruction MSE", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=7.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 4: Method Comparison Bar Chart
    # -------------------------------------------------------------
    method_keys = ["baseline", "phase6_lower_3s", "A1", "A2", "B1", "B2", "C1", "C2"]
    labels = [
        "Baseline\n(Upper 3$\\sigma$)",
        "P6 Lower\n(3$\\sigma$)",
        "A1 Bilateral\n(3$\\sigma$)",
        "A2 Bilateral\n(%ile)",
        "B1 Recon\n(3$\\sigma$)",
        "B2 Recon\n(99th %ile)",
        "C1 Combined\n(A1 | B2)",
        "C2 Combined\n(A1 & B2)",
    ]

    precisions = [methods[k]["metrics"]["precision"] for k in method_keys]
    recalls = [methods[k]["metrics"]["recall"] for k in method_keys]
    f1_scores = [methods[k]["metrics"]["f1_score"] for k in method_keys]

    x = np.arange(len(labels))
    width = 0.25

    rects1 = ax4.bar(x - width, precisions, width, label="Precision", color="#4e79a7")
    rects2 = ax4.bar(x, recalls, width, label="Recall", color="#f28e2b")
    rects3 = ax4.bar(x + width, f1_scores, width, label="F1-Score", color="#59a14f")

    # Add numeric labels on F1 bars
    for rect in rects3:
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

    ax4.set_title("4. Comparative Detection Performance Across Evaluated Methods", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=8)
    ax4.set_ylabel("Score [0.0, 1.0]", fontsize=9)
    ax4.set_ylim(0.0, 1.15)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=8, framealpha=0.9)

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

    return output_file
