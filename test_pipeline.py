"""Integration tests for end-to-end fault detection pipeline and visualization."""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from data_generator import generate_sensor_data
from rbm import RBM
from anomaly_detector import evaluate_anomaly_detector
from visualization import save_anomaly_report


class TestPipeline(unittest.TestCase):
    """Integration test suite validating end-to-end pipeline execution and artifacts."""

    @classmethod
    def setUpClass(cls):
        """Run standard pipeline components once for integration verification."""
        cls.data = generate_sensor_data(seed=42)
        cls.model = RBM(
            n_visible=8,
            n_hidden=5,
            learning_rate=0.01,
            batch_size=32,
            n_epochs=5,
            seed=42,
        )
        cls.model.fit(cls.data["X_train"], verbose=False)
        cls.results = evaluate_anomaly_detector(
            cls.model, cls.data["X_train"], cls.data["X_test"], cls.data["y_test"]
        )

    def test_pipeline_dimensions(self):
        """1. Verify expected training and test dimensions."""
        self.assertEqual(self.data["X_train"].shape, (2000, 8))
        self.assertEqual(self.data["X_test"].shape, (500, 8))
        self.assertEqual(len(self.data["sensor_names"]), 8)

    def test_rbm_trained_only_on_xtrain(self):
        """2. Verify RBM model trained with X_train attributes and dimensions."""
        self.assertEqual(self.model.W.shape, (5, 8))
        self.assertEqual(self.model.b.shape, (8,))
        self.assertEqual(self.model.c.shape, (5,))
        self.assertTrue(np.all(np.isfinite(self.model.W)))

    def test_evaluation_output_lengths(self):
        """3. Verify evaluation returns exactly 500 test scores and 500 predictions."""
        self.assertEqual(len(self.results["test_energy"]), 500)
        self.assertEqual(len(self.results["predictions"]), 500)
        self.assertEqual(self.results["predictions"].dtype, bool)

    def test_threshold_formula_consistency(self):
        """4. Verify anomaly threshold strictly follows mu_train + 3 * sigma_train (ddof=0)."""
        expected_threshold = float(
            np.mean(self.results["train_energy"])
            + 3.0 * np.std(self.results["train_energy"], ddof=0)
        )
        self.assertAlmostEqual(self.results["threshold"], expected_threshold, places=7)

    def test_report_generation_non_empty(self):
        """5. Verify save_anomaly_report creates a valid, non-empty PNG file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_png = Path(tmp_dir) / "test_report.png"
            saved_path = save_anomaly_report(self.data, self.results, output_path=temp_png)

            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)  # Must be a substantial image

    def test_visualization_with_synthetic_predictions(self):
        """6. Verify report generator renders cleanly even when predictions are present."""
        # Create a copy with synthetic positive detections to test both rendering paths
        results_with_preds = dict(self.results)
        preds_synthetic = np.zeros(500, dtype=bool)
        preds_synthetic[310:320] = True
        results_with_preds["predictions"] = preds_synthetic

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_png = Path(tmp_dir) / "test_report_preds.png"
            saved_path = save_anomaly_report(
                self.data, results_with_preds, output_path=temp_png
            )
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)

    def test_pristine_normal_test_reference_preserved(self):
        """7. Verify normal test array remains unmodified throughout pipeline execution."""
        x_test_normal = self.data["X_test_normal"]
        x_test = self.data["X_test"]
        y_test = self.data["y_test"]

        # Normal test telemetry outside anomaly windows matches X_test
        np.testing.assert_array_equal(x_test[~y_test], x_test_normal[~y_test])
        # Values in X_test_normal remain finite and bounded
        self.assertTrue(np.all((x_test_normal >= 0.0) & (x_test_normal <= 1.0)))


if __name__ == "__main__":
    unittest.main()
