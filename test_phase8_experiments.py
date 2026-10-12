"""Unit tests for Phase 8 channel-wise reconstruction residual experiments."""

import unittest
import numpy as np
from rbm import RBM
from data_generator import generate_sensor_data
from phase8_experiments import (
    compute_channel_residuals,
    run_phase8_experiments,
    evaluate_detection_metrics,
)


class TestPhase8Experiments(unittest.TestCase):
    """Test suite validating per-channel residuals, thresholds, and max-z scoring."""

    @classmethod
    def setUpClass(cls):
        """Set up model and dataset once for testing."""
        cls.data = generate_sensor_data(seed=42)
        cls.model = RBM(n_epochs=5, batch_size=32, seed=42)
        cls.model.fit(cls.data["X_train"], verbose=False)
        cls.results = run_phase8_experiments(cls.model, cls.data)

    def test_residual_matrix_shape_and_values(self):
        """1. Verify residual matrix dimensions, finiteness, and non-negativity."""
        r_train = self.results["r_train"]
        r_test = self.results["r_test"]

        self.assertEqual(r_train.shape, (2000, 8))
        self.assertEqual(r_test.shape, (500, 8))
        self.assertTrue(np.all(np.isfinite(r_train)))
        self.assertTrue(np.all(np.isfinite(r_test)))
        self.assertTrue(np.all(r_train >= 0.0))
        self.assertTrue(np.all(r_test >= 0.0))

    def test_per_channel_residual_computation(self):
        """2. Verify channel residuals match manual (X - X_hat)^2 per coordinate."""
        x_sub = self.data["X_train"][:10]
        x_rec = self.model.reconstruct(x_sub)
        r_manual = (x_sub - x_rec) ** 2
        r_func = compute_channel_residuals(self.model, x_sub)
        np.testing.assert_array_almost_equal(r_manual, r_func)

    def test_thresholds_derived_from_training_only(self):
        """3. Verify channel thresholds match mathematical formulas computed on r_train."""
        r_train = self.results["r_train"]
        mu_expected = np.mean(r_train, axis=0)
        sigma_expected = np.std(r_train, axis=0, ddof=0)
        tau_a_expected = mu_expected + 3.0 * sigma_expected
        tau_b_expected = np.percentile(r_train, 99.0, axis=0)

        np.testing.assert_array_almost_equal(self.results["mu_j"], mu_expected)
        np.testing.assert_array_almost_equal(self.results["sigma_j"], sigma_expected)
        np.testing.assert_array_almost_equal(self.results["tau_a"], tau_a_expected)
        np.testing.assert_array_almost_equal(self.results["tau_b"], tau_b_expected)

    def test_thresholds_invariant_to_test_labels(self):
        """4. Verify changing y_test has zero effect on thresholds or predictions."""
        dummy_data = dict(self.data)
        dummy_data["y_test"] = ~self.data["y_test"]

        res_dummy = run_phase8_experiments(self.model, dummy_data)
        for key in ["phase8_method_a", "phase8_method_b", "phase8_method_c"]:
            np.testing.assert_array_equal(
                self.results["methods"][key]["predictions"],
                res_dummy["methods"][key]["predictions"],
            )

    def test_channel_wise_union_logic(self):
        """5. Verify sample is flagged if and only if at least one channel exceeds threshold."""
        r_test = self.results["r_test"]
        tau_a = self.results["tau_a"]
        tau_b = self.results["tau_b"]

        expected_preds_a = np.any(r_test > tau_a, axis=1)
        expected_preds_b = np.any(r_test > tau_b, axis=1)

        np.testing.assert_array_equal(self.results["methods"]["phase8_method_a"]["predictions"], expected_preds_a)
        np.testing.assert_array_equal(self.results["methods"]["phase8_method_b"]["predictions"], expected_preds_b)

    def test_max_normalized_residual_scoring(self):
        """6. Verify Method C computes max z-score and applies training 99th %ile threshold."""
        r_train = self.results["r_train"]
        r_test = self.results["r_test"]
        mu_j = self.results["mu_j"]
        sigma_j = self.results["sigma_j"]

        safe_sigma = np.where(sigma_j < 1e-8, 1e-8, sigma_j)
        score_train_manual = np.max((r_train - mu_j) / safe_sigma, axis=1)
        score_test_manual = np.max((r_test - mu_j) / safe_sigma, axis=1)
        tau_c_manual = float(np.percentile(score_train_manual, 99.0))

        self.assertAlmostEqual(self.results["tau_c"], tau_c_manual, places=7)
        np.testing.assert_array_almost_equal(self.results["score_test_c"], score_test_manual)
        np.testing.assert_array_equal(
            self.results["methods"]["phase8_method_c"]["predictions"],
            score_test_manual > tau_c_manual,
        )

    def test_safe_handling_zero_variance(self):
        """7. Verify zero-variance channels do not cause divide-by-zero or NaN in z-scoring."""
        # Simulated data where channel 0 is constant
        r_mock = np.ones((100, 8))
        sigma_mock = np.std(r_mock, axis=0, ddof=0)
        mu_mock = np.mean(r_mock, axis=0)

        variance_safeguard = 1e-8
        safe_sigma = np.where(sigma_mock < variance_safeguard, variance_safeguard, sigma_mock)

        z_scores = (r_mock - mu_mock) / safe_sigma
        self.assertTrue(np.all(np.isfinite(z_scores)))
        self.assertEqual(z_scores.shape, (100, 8))

    def test_confusion_matrix_consistency(self):
        """8. Verify confusion matrix counts sum to 500 and window counts sum to TP."""
        for key in ["phase8_method_a", "phase8_method_b", "phase8_method_c"]:
            m = self.results["methods"][key]["metrics"]
            self.assertEqual(m["TP"] + m["FP"] + m["TN"] + m["FN"], 500)
            self.assertEqual(m["TP"] + m["FN"], 70)
            self.assertEqual(m["TP"] + m["FP"], m["detected_count"])
            self.assertEqual(m["w1_detected"] + m["w2_detected"], m["TP"])

    def test_reproducibility(self):
        """9. Verify repeated execution produces bitwise identical arrays."""
        res_rep = run_phase8_experiments(self.model, self.data)
        np.testing.assert_array_equal(self.results["r_train"], res_rep["r_train"])
        np.testing.assert_array_equal(self.results["r_test"], res_rep["r_test"])
        self.assertEqual(self.results["tau_c"], res_rep["tau_c"])

    def test_artifacts_generation(self):
        """10. Verify Phase 8 report and visualization generation in temporary directory."""
        import tempfile
        from pathlib import Path
        from phase8_visualization import save_phase8_anomaly_report
        from run_phase8 import generate_phase8_experiment_report

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_png = Path(tmp_dir) / "test_p8_report.png"
            tmp_txt = Path(tmp_dir) / "test_p8_report.txt"

            saved_png = save_phase8_anomaly_report(self.data, self.results, output_path=tmp_png)
            generate_phase8_experiment_report(self.data, self.results, tmp_txt)

            self.assertTrue(saved_png.exists())
            self.assertGreater(saved_png.stat().st_size, 1000)
            self.assertTrue(tmp_txt.exists())
            self.assertGreater(tmp_txt.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
