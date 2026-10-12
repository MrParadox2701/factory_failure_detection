"""Synthetic industrial sensor telemetry generator and preprocessor.

This module generates deterministic continuous sensor telemetry for industrial
fault detection benchmarking. It simulates normal operating states via low-dimensional
Gaussian latent factors mixed through a linear matrix with additive sensor noise,
applies training-only Min-Max scaling with constant-range protection and clipping to [0, 1],
and injects correlation-breaking fault regimes into designated test windows.
"""

from typing import Dict, List, Any
import numpy as np


def generate_sensor_data(seed: int = 42) -> Dict[str, Any]:
    """Generate deterministic synthetic sensor data for training and evaluation.

    Parameters
    ----------
    seed : int, default=42
        Deterministic random seed initializing NumPy's default_rng.

    Returns
    -------
    dict
        Dictionary containing:
        - 'X_train': np.ndarray, shape (2000, 8), scaled training data in [0, 1].
        - 'X_test': np.ndarray, shape (500, 8), scaled test data with injected anomalies.
        - 'X_test_normal': np.ndarray, shape (500, 8), scaled test data before anomaly injection.
        - 'y_test': np.ndarray of bool, shape (500,), True for injected anomaly indices.
        - 'sensor_names': list of 8 descriptive industrial sensor names.
        - 'scaler_min': np.ndarray, shape (8,), per-feature training minima.
        - 'scaler_max': np.ndarray, shape (8,), per-feature training maxima.
    """
    rng = np.random.default_rng(seed)

    # 1. Configuration parameters
    n_train: int = 2000
    n_test: int = 500
    n_sensors: int = 8
    n_latent: int = 3
    noise_std: float = 0.05

    sensor_names: List[str] = [
        "Bearing_Vibration_X",
        "Bearing_Vibration_Y",
        "Motor_Winding_Temp",
        "Coolant_Pressure",
        "Pump_Flow_Rate",
        "Motor_Current_PhaseA",
        "Exhaust_Gas_Temp",
        "Lube_Oil_Viscosity",
    ]

    # 2. Fixed random mixing matrix (8 sensors x 3 latent factors)
    mixing_matrix: np.ndarray = rng.standard_normal((n_sensors, n_latent))

    # 3. Generate raw normal training observations
    # Z_train ~ N(0, I_3), shape (2000, 3)
    z_train: np.ndarray = rng.standard_normal((n_train, n_latent))
    noise_train: np.ndarray = rng.normal(0.0, noise_std, size=(n_train, n_sensors))
    # X_raw = Z @ M^T + noise, shape (2000, 8)
    x_train_raw: np.ndarray = z_train @ mixing_matrix.T + noise_train

    # 4. Generate raw normal test observations (same generative physics)
    # Z_test ~ N(0, I_3), shape (500, 3)
    z_test: np.ndarray = rng.standard_normal((n_test, n_latent))
    noise_test: np.ndarray = rng.normal(0.0, noise_std, size=(n_test, n_sensors))
    x_test_raw: np.ndarray = z_test @ mixing_matrix.T + noise_test

    # 5. Fit Min-Max scaling parameters using training data ONLY
    scaler_min: np.ndarray = np.min(x_train_raw, axis=0)
    scaler_max: np.ndarray = np.max(x_train_raw, axis=0)
    feature_ranges: np.ndarray = scaler_max - scaler_min

    # Safely handle constant / zero-range features to prevent division by zero
    safe_ranges: np.ndarray = np.where(feature_ranges < 1e-8, 1.0, feature_ranges)

    # 6. Transform and clip to [0.0, 1.0] to satisfy Bernoulli RBM visible units
    x_train: np.ndarray = np.clip(
        (x_train_raw - scaler_min) / safe_ranges, 0.0, 1.0
    )
    x_test_normal: np.ndarray = np.clip(
        (x_test_raw - scaler_min) / safe_ranges, 0.0, 1.0
    )

    # 7. Preserve pristine normal test data and prepare anomaly injection target
    x_test: np.ndarray = x_test_normal.copy()
    y_test: np.ndarray = np.zeros(n_test, dtype=bool)

    # 8. Inject anomalies strictly at specified half-open intervals
    # Window 1: [300, 350) -> 50 samples
    # Structural correlation break: Sensor saturation and drift
    y_test[300:350] = True
    # Sensor 0 saturates high (1.0), Sensor 2 saturates low (0.0)
    x_test[300:350, 0] = 1.0
    x_test[300:350, 2] = 0.0
    # Sensor 4 experiences severe positive drift
    x_test[300:350, 4] = np.clip(x_test[300:350, 4] + 0.6, 0.0, 1.0)
    # Add uncorrelated disturbances on sensor 6
    x_test[300:350, 6] = np.clip(
        rng.uniform(0.7, 1.0, size=50), 0.0, 1.0
    )

    # Window 2: [420, 440) -> 20 samples
    # Structural correlation break: Severe uncorrelated noise spikes & erratic oscillations
    y_test[420:440] = True
    # Sensors 1, 3, 5, 7 receive uncorrelated uniform noise spanning full [0, 1] range
    x_test[420:440, 1] = rng.uniform(0.0, 0.3, size=20)
    x_test[420:440, 3] = rng.uniform(0.8, 1.0, size=20)
    x_test[420:440, 5] = rng.choice([0.0, 1.0], size=20)
    x_test[420:440, 7] = np.clip(
        x_test[420:440, 7] + rng.normal(0.0, 0.5, size=20), 0.0, 1.0
    )

    # Final guarantee: all scaled values strictly within [0.0, 1.0]
    x_test = np.clip(x_test, 0.0, 1.0)

    return {
        "X_train": x_train,
        "X_test": x_test,
        "X_test_normal": x_test_normal,
        "y_test": y_test,
        "sensor_names": sensor_names,
        "scaler_min": scaler_min,
        "scaler_max": scaler_max,
    }
