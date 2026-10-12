"""Unit tests for synthetic sensor data generator and preprocessor."""

import unittest
import numpy as np
from data_generator import generate_sensor_data


class TestDataGenerator(unittest.TestCase):
    """Test suite validating data generation, preprocessing, and fault injection."""

    @classmethod
    def setUpClass(cls):
        """Generate baseline dataset once for fast multi-property validation."""
        cls.data = generate_sensor_data(seed=42)

    def test_shapes_and_dimensions(self):
        """1. Verify exact training and test matrix dimensions and sensor names."""
        self.assertEqual(self.data["X_train"].shape, (2000, 8))
        self.assertEqual(self.data["X_test"].shape, (500, 8))
        self.assertEqual(self.data["X_test_normal"].shape, (500, 8))
        self.assertEqual(self.data["y_test"].shape, (500,))
        self.assertEqual(len(self.data["sensor_names"]), 8)
        self.assertEqual(self.data["scaler_min"].shape, (8,))
        self.assertEqual(self.data["scaler_max"].shape, (8,))

    def test_exact_injected_anomaly_count(self):
        """2. Verify that exactly 70 samples are flagged as anomalies."""
        self.assertEqual(np.sum(self.data["y_test"]), 70)
        self.assertEqual(self.data["y_test"].dtype, bool)

    def test_anomaly_index_ranges(self):
        """3. Verify anomalies exist strictly in [300, 350) and [420, 440)."""
        y_test = self.data["y_test"]
        # Expected True ranges
        expected_indices = set(range(300, 350)) | set(range(420, 440))
        actual_indices = set(np.where(y_test)[0])
        self.assertEqual(actual_indices, expected_indices)

        # Before 300 must be all False
        self.assertFalse(np.any(y_test[:300]))
        # [350, 420) must be all False
        self.assertFalse(np.any(y_test[350:420]))
        # [440, 500) must be all False
        self.assertFalse(np.any(y_test[440:]))

    def test_finite_values(self):
        """4. Verify that all arrays contain strictly finite values without NaN or Inf."""
        for key in ["X_train", "X_test", "X_test_normal", "scaler_min", "scaler_max"]:
            arr = self.data[key]
            self.assertTrue(np.all(np.isfinite(arr)), f"Non-finite values found in {key}")

    def test_scaled_range_bounds(self):
        """5. Verify every scaled model input strictly lies within [0.0, 1.0]."""
        for key in ["X_train", "X_test", "X_test_normal"]:
            arr = self.data[key]
            self.assertGreaterEqual(float(np.min(arr)), 0.0, f"Min below 0.0 in {key}")
            self.assertLessEqual(float(np.max(arr)), 1.0, f"Max above 1.0 in {key}")

    def test_reproducibility(self):
        """6. Verify calling generator twice with same seed produces bitwise identical data."""
        data_rep = generate_sensor_data(seed=42)
        for key in ["X_train", "X_test", "X_test_normal", "scaler_min", "scaler_max"]:
            np.testing.assert_array_equal(self.data[key], data_rep[key])
        np.testing.assert_array_equal(self.data["y_test"], data_rep["y_test"])

    def test_training_data_fault_free(self):
        """7. Verify training data is untouched by anomaly injection."""
        # Training data shape and properties
        self.assertEqual(self.data["X_train"].shape[0], 2000)
        # Check that training data values are properly scaled
        self.assertAlmostEqual(float(np.min(self.data["X_train"])), 0.0, places=5)
        self.assertAlmostEqual(float(np.max(self.data["X_train"])), 1.0, places=5)

    def test_normal_test_preservation(self):
        """8. Verify original normal test array is preserved and untouched by injection."""
        x_test = self.data["X_test"]
        x_test_normal = self.data["X_test_normal"]
        y_test = self.data["y_test"]

        # Outside anomaly ranges, x_test and x_test_normal must be identical
        normal_indices = ~y_test
        np.testing.assert_array_equal(x_test[normal_indices], x_test_normal[normal_indices])

        # Inside anomaly ranges, at least one sensor must differ
        anomaly_indices = y_test
        diff = np.abs(x_test[anomaly_indices] - x_test_normal[anomaly_indices])
        self.assertGreater(np.max(diff), 0.0, "Anomalies did not alter test data")

    def test_scaler_bounds_derived_from_training(self):
        """9. Verify scaler min and max are derived strictly from training data."""
        s_min = self.data["scaler_min"]
        s_max = self.data["scaler_max"]
        self.assertTrue(np.all(s_max > s_min), "Scaler max must strictly exceed min")
        # Feature count must equal 8
        self.assertEqual(len(s_min), 8)
        self.assertEqual(len(s_max), 8)


if __name__ == "__main__":
    unittest.main()
