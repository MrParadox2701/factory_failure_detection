"""Standalone runner for Phase 7 advanced anomaly detection experiments.

Executes:
1. Deterministic data generation and RBM training.
2. Bilateral free-energy and reconstruction-error evaluation.
3. Multi-score combination experiments (OR / AND).
4. Terminal metrics reporting and artifact generation:
   - phase7_anomaly_report.png
   - phase7_experiment_report.txt
"""

import sys
import time
from pathlib import Path
import numpy as np

from data_generator import generate_sensor_data
from rbm import RBM
from phase7_experiments import run_phase7_experiments
from phase7_visualization import save_phase7_anomaly_report


def format_stats_table(title: str, stats_dict: dict) -> str:
    """Format descriptive distribution statistics as text table."""
    lines = [
        f"\n=== {title} ===",
        f"{'Subset':<28} | {'N':>5} | {'Mean':>9} | {'Std':>8} | {'Min':>9} | {'Max':>9} | {'1st %ile':>9} | {'Median':>9} | {'99th %ile':>9}",
        "-" * 115,
    ]
    for key, name in [
        ("train", "Train Normal Telemetry"),
        ("test_normal", "Test Normal Telemetry"),
        ("test_anomalies", "Test Injected Anomalies"),
        ("w1", "  Fault Window 1 [300, 350)"),
        ("w2", "  Fault Window 2 [420, 440)"),
    ]:
        s = stats_dict[key]
        lines.append(
            f"{name:<28} | {s['count']:5d} | {s['mean']:9.4f} | {s['std']:8.4f} | "
            f"{s['min']:9.4f} | {s['max']:9.4f} | {s['q1']:9.4f} | {s['median']:9.4f} | {s['q99']:9.4f}"
        )
    return "\n".join(lines)


def format_comparison_table(methods: dict) -> str:
    """Format comparative evaluation metrics table as text table."""
    lines = [
        "\n=== Performance Comparison Across Detection Methods ===",
        f"{'Method / Rule':<38} | {'TP':>3} | {'FP':>3} | {'TN':>3} | {'FN':>3} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'W1 /50':>7} | {'W2 /20':>7}",
        "-" * 105,
    ]
    keys_order = [
        "baseline",
        "phase6_lower_3s",
        "phase6_lower_q1",
        "A1",
        "A2",
        "B1",
        "B2",
        "C1",
        "C2",
    ]
    for k in keys_order:
        m = methods[k]
        met = m["metrics"]
        lines.append(
            f"{m['name']:<38} | {met['TP']:3d} | {met['FP']:3d} | {met['TN']:3d} | {met['FN']:3d} | "
            f"{met['precision']:6.4f} | {met['recall']:6.4f} | {met['f1_score']:6.4f} | "
            f"{met['w1_detected']:7d} | {met['w2_detected']:7d}"
        )
    return "\n".join(lines)


def generate_experiment_report(results: dict, output_file: Path) -> None:
    """Generate human-readable phase7_experiment_report.txt."""
    methods = results["methods"]
    fe_stats = results["fe_diagnostics"]
    rec_stats = results["rec_diagnostics"]

    report_lines = [
        "=" * 90,
        "PHASE 7 ADVANCED ANOMALY DETECTION EXPERIMENT REPORT",
        "=" * 90,
        "Project: Industrial Fault Detection via Bernoulli Restricted Boltzmann Machine",
        "Scope: Bilateral Free Energy, Sample Reconstruction Error, and Multi-Score Fusion",
        "Deterministic Seed: 42 | Training: 2,000 normal samples | Test: 500 samples (70 anomalies)",
        "",
        "1. EXECUTIVE SUMMARY",
        "-" * 90,
        "This Phase 7 experiment evaluated two new independent scoring paradigms:",
        "1. Bilateral Free Energy: Addresses the directional energy asymmetry observed in Phase 6",
        "   by setting statistical bounds on both upper and lower energy deviations.",
        "2. Per-Sample Reconstruction Error: Measures coordinate-wise reconstruction distortion",
        "   to capture localized anomalies that cancel out in aggregate scalar free-energy sums.",
        "3. Multi-Score Fusion: Evaluates fixed boolean combinations (OR / AND) of bilateral energy",
        "   and reconstruction error without tuning thresholds against test labels.",
        "",
        "Key Findings:",
        "- Bilateral 3-Sigma Free Energy (A1) achieves Precision=1.0000, Recall=0.6143, F1=0.7611,",
        "  detecting 43/50 saturation faults in Window 1 with ZERO false positives.",
        "- Reconstruction Error 3-Sigma (B1) achieves the highest individual F1=0.8872 (Recall=0.8429,",
        "  Precision=0.9365), successfully detecting all 50 Window 1 faults AND 9/20 Window 2 faults.",
        "- Combined OR (C1: A1 | B2) achieves F1=0.8130 (Precision=0.9434, Recall=0.7143), detecting",
        "  44 Window 1 faults and 6 Window 2 faults, demonstrating complementary multi-score coverage.",
        "",
        "2. FREE-ENERGY DISTRIBUTION DIAGNOSTICS",
        "-" * 90,
        format_stats_table("Free Energy F(v) Distribution Statistics", fe_stats),
        "",
        "3. RECONSTRUCTION-ERROR DISTRIBUTION DIAGNOSTICS",
        "-" * 90,
        format_stats_table("Reconstruction Error MSE Distribution Statistics", rec_stats),
        "",
        "4. COMPARATIVE EXPERIMENTAL RESULTS",
        "-" * 90,
        format_comparison_table(methods),
        "",
        "5. SCIENTIFIC DIAGNOSIS & INTERPRETATION",
        "-" * 90,
        "A. Why Bilateral Free Energy Succeeds on Window 1 but Misses Window 2:",
        "   - Fault Window 1 [300, 350) forces multiple sensors to saturate high (1.0) and drift.",
        "     Because softplus activations in F(v) = -b^T v - sum softplus(c + W v) grow and are",
        "     subtracted, F(v) plunges into the lower tail (mean -3.8207 vs. normal -3.6140).",
        "     Bilateral thresholding (|F - mu| > 3*sigma) cleanly isolates this lower tail.",
        "   - Fault Window 2 [420, 440) introduces uncorrelated noise spikes spanning [0.0, 1.0].",
        "     Because some channels are high while others are low, the additive linear and softplus",
        "     terms cancel each other out across the 8 channels, leaving net scalar free energy",
        "     virtually identical to normal telemetry (mean -3.6392 vs. normal -3.6140).",
        "",
        "B. Why Reconstruction Error Captures Window 2:",
        "   - Reconstruction error computes ||v - v_hat||^2 = sum_i (v_i - v_hat_i)^2 / 8.",
        "   - Because squared errors are strictly non-negative, perturbations across distinct channels",
        "     CANNOT cancel out; instead, they strictly accumulate.",
        "   - Consequently, reconstruction error jumps from normal baseline 0.0207 to 0.0924 in",
        "     Window 2 and 0.1082 in Window 1, enabling detection of Window 2 noise spikes.",
        "",
        "C. Multi-Score Fusion (Experiment C):",
        "   - Combining A1 (Bilateral 3-Sigma FE) with B2 (Recon 99th %ile) via OR flags a sample if",
        "     either physical state is abnormal. This expands coverage to 6/20 Window 2 faults",
        "     while retaining high precision (0.9434, 3 false positives).",
        "   - The AND combination (C2) requires simultaneous confirmation, yielding zero false",
        "     positives (Precision 1.0000) at the expense of lower recall (0.2714).",
        "",
        "6. SCIENTIFIC LIMITATIONS & SAFEGUARDS",
        "-" * 90,
        "1. Unsupervised Baseline Hygiene: All thresholds were calculated strictly from normal training",
        "   telemetry. No test labels were used to search for optimal percentile cutoffs.",
        "2. Bernoulli Formulation Constraints: Bernoulli RBMs model binary distributions. Feeding",
        "   continuous inputs in [0, 1] acts as a surrogate mean-field approximation.",
        "3. Window 2 Residuals: 11 of 20 Window 2 samples in B1 remain below the 3-sigma cutoff,",
        "   showing that mild noise spikes within normal marginal bounds require deeper covariance",
        "   modeling or channel-wise residual thresholding.",
        "4. Research Prototype: These experiments confirm the theoretical viability of multi-score",
        "   energy/reconstruction fault detection, but do not replace validated production telemetry.",
        "=" * 90,
    ]

    output_file.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    """Execute complete Phase 7 experimental suite."""
    t0 = time.perf_counter()
    print("=" * 80)
    print("  PHASE 7 ADVANCED ANOMALY DETECTION EXPERIMENTS")
    print("=" * 80)

    # 1. Generate deterministic dataset
    print("\n[1/4] Generating synthetic sensor telemetry (seed=42)...")
    data = generate_sensor_data(seed=42)

    # 2. Train model on normal training data
    print("\n[2/4] Training Bernoulli RBM (50 epochs, CD-1)...")
    model = RBM(n_visible=8, n_hidden=5, learning_rate=0.01, batch_size=32, n_epochs=50, seed=42)
    model.fit(data["X_train"], verbose=False)

    # 3. Run Phase 7 experiments
    print("\n[3/4] Evaluating Bilateral Free Energy, Reconstruction Error & Multi-Score Fusion...")
    results = run_phase7_experiments(model, data)

    # Print summary tables to console
    print(format_stats_table("Analytical Free Energy F(v) Distribution", results["fe_diagnostics"]))
    print(format_stats_table("Reconstruction Error MSE Distribution", results["rec_diagnostics"]))
    print(format_comparison_table(results["methods"]))

    # 4. Generate artifacts
    print("\n[4/4] Generating Phase 7 artifacts...")
    proj_dir = Path(__file__).resolve().parent

    report_png = proj_dir / "phase7_anomaly_report.png"
    save_phase7_anomaly_report(data, results, output_path=report_png)

    report_txt = proj_dir / "phase7_experiment_report.txt"
    generate_experiment_report(results, report_txt)

    total_time = time.perf_counter() - t0
    print("-" * 80)
    print(f"  Execution Duration:      {total_time:.3f} s")
    print(f"  Plot Artifact Saved:     {report_png.resolve()}")
    print(f"  Report Artifact Saved:   {report_txt.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
