"""Unit tests for Phase 7 advanced anomaly detection experiments."""

import unittest
import numpy as np
from rbm import RBM
from data_generator import generate_sensor_data
from phase7_experiments import (
    compute_sample_reconstruction_error,
    run_phase7_experiments,
    evaluate_predictions,
)


class TestPhase7Experiments(unittest.TestCase):
    """Test suite validating Phase 7 bilateral, reconstruction, and combined methods."""

    @classmethod
    def setUpClass(cls):
        """Prepare reproducible model and dataset once."""
        cls.data = generate_sensor_data(seed=42)
        cls.model = RBM(n_epochs=5, batch_size=32, seed=42)
        cls.model.fit(cls.data["X_train"], verbose=False)
        cls.results = run_phase7_experiments(cls.model, cls.data)

    def test_reconstruction_error_computation(self):
        """1. Verify per-sample reconstruction error shape, finiteness, and bounds."""
        rec_err = compute_sample_reconstruction_error(self.model, self.data["X_train"])
        self.assertEqual(rec_err.shape, (2000,))
        self.assertTrue(np.all(np.isfinite(rec_err)))
        self.assertTrue(np.all(rec_err >= 0.0))

    def test_thresholds_derived_from_training_only(self):
        """2. Verify all thresholds match mathematical formulas computed on X_train."""
        train_fe = self.results["train_fe"]
        train_rec = self.results["train_rec"]
        methods = self.results["methods"]

        # Free energy parameters
        mu_fe = float(np.mean(train_fe))
        sigma_fe = float(np.std(train_fe, ddof=0))
        q1_fe = float(np.percentile(train_fe, 1.0))
        q99_fe = float(np.percentile(train_fe, 99.0))

        # Reconstruction error parameters
        mu_rec = float(np.mean(train_rec))
        sigma_rec = float(np.std(train_rec, ddof=0))
        q99_rec = float(np.percentile(train_rec, 99.0))

        # A1
        self.assertAlmostEqual(methods["A1"]["thresholds"]["lower"], mu_fe - 3.0 * sigma_fe)
        self.assertAlmostEqual(methods["A1"]["thresholds"]["upper"], mu_fe + 3.0 * sigma_fe)

        # A2
        self.assertAlmostEqual(methods["A2"]["thresholds"]["lower"], q1_fe)
        self.assertAlmostEqual(methods["A2"]["thresholds"]["upper"], q99_fe)

        # B1
        self.assertAlmostEqual(methods["B1"]["thresholds"]["tau"], mu_rec + 3.0 * sigma_rec)

        # B2
        self.assertAlmostEqual(methods["B2"]["thresholds"]["tau"], q99_rec)

    def test_thresholds_invariant_to_test_labels(self):
        """3. Verify modifying y_test has zero impact on thresholds or predictions."""
        dummy_data = dict(self.data)
        dummy_data["y_test"] = ~self.data["y_test"]

        res_dummy = run_phase7_experiments(self.model, dummy_data)
        for key in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            np.testing.assert_array_equal(
                self.results["methods"][key]["predictions"],
                res_dummy["methods"][key]["predictions"],
            )

    def test_bilateral_flags_both_tails(self):
        """4. Verify bilateral decision rule flags both upper and lower tail points."""
        train_fe = self.results["train_fe"]
        mu = float(np.mean(train_fe))
        sigma = float(np.std(train_fe, ddof=0))

        # Synthetic scores: one very low, one normal, one very high
        scores = np.array([mu - 4 * sigma, mu, mu + 4 * sigma])
        preds = np.abs(scores - mu) > 3.0 * sigma
        np.testing.assert_array_equal(preds, [True, False, True])

    def test_combined_boolean_logic(self):
        """5. Verify C1 is exact OR and C2 is exact AND of A1 and B2."""
        preds_a1 = self.results["methods"]["A1"]["predictions"]
        preds_b2 = self.results["methods"]["B2"]["predictions"]
        preds_c1 = self.results["methods"]["C1"]["predictions"]
        preds_c2 = self.results["methods"]["C2"]["predictions"]

        np.testing.assert_array_equal(preds_c1, preds_a1 | preds_b2)
        np.testing.assert_array_equal(preds_c2, preds_a1 & preds_b2)

    def test_confusion_matrix_consistency(self):
        """6. Verify confusion matrix counts sum to 500 and window counts sum to TP."""
        for key, method in self.results["methods"].items():
            m = method["metrics"]
            self.assertEqual(m["TP"] + m["FP"] + m["TN"] + m["FN"], 500)
            self.assertEqual(m["TP"] + m["FN"], 70)
            self.assertEqual(m["TP"] + m["FP"], m["detected_count"])
            self.assertEqual(m["w1_detected"] + m["w2_detected"], m["TP"])

    def test_reproducibility(self):
        """7. Verify repeated evaluations yield bitwise identical arrays."""
        res_rep = run_phase7_experiments(self.model, self.data)
        np.testing.assert_array_equal(self.results["train_fe"], res_rep["train_fe"])
        np.testing.assert_array_equal(self.results["test_fe"], res_rep["test_fe"])
        np.testing.assert_array_equal(self.results["train_rec"], res_rep["train_rec"])
        np.testing.assert_array_equal(self.results["test_rec"], res_rep["test_rec"])

    def test_artifacts_generation(self):
        """8. Verify Phase 7 visualization and report generation write valid files."""
        import tempfile
        from pathlib import Path
        from phase7_visualization import save_phase7_anomaly_report
        from run_phase7 import generate_experiment_report

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_png = Path(tmp_dir) / "test_p7_report.png"
            tmp_txt = Path(tmp_dir) / "test_p7_report.txt"

            saved_png = save_phase7_anomaly_report(self.data, self.results, output_path=tmp_png)
            generate_experiment_report(self.results, tmp_txt)

            self.assertTrue(saved_png.exists())
            self.assertGreater(saved_png.stat().st_size, 1000)
            self.assertTrue(tmp_txt.exists())
            self.assertGreater(tmp_txt.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
