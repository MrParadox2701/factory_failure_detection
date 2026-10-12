"""End-to-end industrial fault detection pipeline using Bernoulli RBM.

Executes:
1. Deterministic synthetic sensor telemetry generation and training-only Min-Max scaling.
2. Bernoulli RBM training on normal telemetry via Contrastive Divergence (CD-1).
3. Statistical free-energy thresholding (mean + 3 * std) and test anomaly evaluation.
4. Terminal metrics reporting and diagnostic dual-panel artifact visualization.
"""

import sys
import time
from pathlib import Path
import numpy as np

from data_generator import generate_sensor_data
from rbm import RBM
from anomaly_detector import evaluate_anomaly_detector
from visualization import save_anomaly_report


def main() -> None:
    """Execute complete end-to-end fault detection pipeline."""
    start_total_time = time.perf_counter()
    print("=" * 70)
    print("  BERNOULLI RBM INDUSTRIAL FAULT DETECTION PROTOTYPE")
    print("=" * 70)

    # 1. Generate deterministic synthetic sensor data (seed=42)
    print("\n[1/4] Generating synthetic industrial sensor telemetry...")
    seed = 42
    data = generate_sensor_data(seed=seed)
    x_train = data["X_train"]
    x_test = data["X_test"]
    y_test = data["y_test"]
    sensor_names = data["sensor_names"]
    scaler_min = data["scaler_min"]
    scaler_max = data["scaler_max"]

    # 2. Print data properties
    print(f"      - Training samples: {x_train.shape[0]} | Shape: {x_train.shape}")
    print(f"      - Test samples:     {x_test.shape[0]}  | Shape: {x_test.shape}")
    print(f"      - Sensor channels:  {len(sensor_names)} features")
    print(f"      - Scaler min range: [{np.min(scaler_min):.3f}, {np.max(scaler_min):.3f}]")
    print(f"      - Scaler max range: [{np.min(scaler_max):.3f}, {np.max(scaler_max):.3f}]")
    print(f"      - Scaled telemetry bounds: [{np.min(x_train):.1f}, {np.max(x_train):.1f}]")
    print(f"      - Injected test anomalies: {np.sum(y_test)} samples (Windows: [300, 350) and [420, 440))")

    # 3. Instantiate RBM architecture
    print("\n[2/4] Initializing Bernoulli Restricted Boltzmann Machine...")
    n_visible = 8
    n_hidden = 5
    learning_rate = 0.01
    batch_size = 32
    n_epochs = 50

    model = RBM(
        n_visible=n_visible,
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        seed=seed,
    )
    print(f"      - Architecture: {n_visible} Visible Units <-> {n_hidden} Hidden Units")
    print(f"      - Hyperparameters: LR={learning_rate}, Batch Size={batch_size}, Epochs={n_epochs}, Seed={seed}")

    # 4. Train exclusively on normal training telemetry
    print("\n[3/4] Training RBM via Contrastive Divergence (CD-1) on X_train...")
    train_start = time.perf_counter()
    model.fit(x_train, verbose=True)
    train_duration = time.perf_counter() - train_start
    print(f"      - CD-1 Training complete in {train_duration:.3f} s")

    # 5. Evaluate anomaly detection via analytical free energy
    print("\n[4/4] Evaluating analytical free-energy scoring and thresholding...")
    results = evaluate_anomaly_detector(model, x_train, x_test, y_test)

    # 6. Print evaluation summary & metrics
    threshold = results["threshold"]
    tp, fp, tn, fn = results["TP"], results["FP"], results["TN"], results["FN"]
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1_score"]
    injected_count = results["injected_anomaly_count"]
    detected_count = results["detected_anomaly_count"]

    print("\n" + "-" * 70)
    print("  ANOMALY DETECTION EVALUATION RESULTS")
    print("-" * 70)
    print(f"  Decision Threshold (tau = mu + 3*sigma):  {threshold:.6f}")
    print(f"  True Injected Anomalies:                 {injected_count}")
    print(f"  Model Detected Anomalies:                {detected_count}")
    print(f"  Confusion Matrix:                        TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"  Precision:                               {precision:.4f}")
    print(f"  Recall:                                  {recall:.4f}")
    print(f"  F1-Score:                                {f1:.4f}")

    # Print diagnostic free-energy distribution statistics
    train_energy = results["train_energy"]
    test_energy = results["test_energy"]
    mean_train_fe = np.mean(train_energy)
    mean_norm_test_fe = np.mean(test_energy[~y_test])
    mean_anom_test_fe = np.mean(test_energy[y_test])
    print("\n  Diagnostic Free-Energy Distributions:")
    print(f"    - Mean Training Free Energy:           {mean_train_fe:.6f}")
    print(f"    - Mean Normal Test Free Energy:        {mean_norm_test_fe:.6f}")
    print(f"    - Mean Injected Anomaly Free Energy:   {mean_anom_test_fe:.6f}")

    # 7. Generate and save diagnostic visualization
    output_filename = "anomaly_report.png"
    output_path = Path(__file__).resolve().parent / output_filename
    saved_path = save_anomaly_report(data, results, output_path=output_path)

    total_duration = time.perf_counter() - start_total_time
    print("-" * 70)
    print(f"  Total Pipeline Execution Duration:      {total_duration:.3f} s")
    print(f"  Artifact Saved:                          {saved_path.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
