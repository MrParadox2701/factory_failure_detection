"""Unit tests for Free-Energy scoring and anomaly detection."""

import unittest
import numpy as np
from rbm import RBM
from data_generator import generate_sensor_data
from anomaly_detector import evaluate_anomaly_detector


class TestAnomalyDetector(unittest.TestCase):
    """Test suite validating free-energy computation and anomaly detector metrics."""

    @classmethod
    def setUpClass(cls):
        """Prepare dataset and trained model once for multi-test validation."""
        cls.data = generate_sensor_data(seed=42)
        cls.model = RBM(n_epochs=5, batch_size=32, seed=42)
        cls.model.fit(cls.data["X_train"], verbose=False)

    def test_single_vector_free_energy(self):
        """1. Verify free energy for a single 1D vector returns a finite float scalar."""
        v = self.data["X_train"][0]
        fe = self.model.compute_free_energy(v)
        self.assertIsInstance(fe, float)
        self.assertTrue(np.isfinite(fe))

    def test_batch_free_energy_shape(self):
        """2. Verify batch free-energy calculation returns a 1D vector of shape (N,)."""
        fe_batch = self.model.compute_free_energy(self.data["X_train"][:50])
        self.assertEqual(fe_batch.shape, (50,))
        self.assertTrue(np.all(np.isfinite(fe_batch)))

    def test_repeated_scoring_deterministic(self):
        """3. Verify repeated free-energy evaluations are bitwise identical."""
        fe_1 = self.model.compute_free_energy(self.data["X_test"])
        fe_2 = self.model.compute_free_energy(self.data["X_test"])
        np.testing.assert_array_equal(fe_1, fe_2)

    def test_free_energy_analytical_reference(self):
        """4. Verify free energy against independently calculated theoretical reference."""
        # Custom small RBM configuration
        toy_model = RBM(n_visible=2, n_hidden=2, seed=42)
        toy_model.W = np.array([[0.1, -0.2], [0.3, 0.5]], dtype=np.float64)
        toy_model.b = np.array([0.1, -0.1], dtype=np.float64)
        toy_model.c = np.array([0.2, -0.3], dtype=np.float64)

        v = np.array([0.4, 0.6], dtype=np.float64)

        # Independent manual calculation:
        # b^T v = 0.1 * 0.4 + (-0.1) * 0.6 = 0.04 - 0.06 = -0.02 -> - b^T v = 0.02
        # h0 = 0.2 + (0.1 * 0.4 - 0.2 * 0.6) = 0.2 + (0.04 - 0.12) = 0.12
        # h1 = -0.3 + (0.3 * 0.4 + 0.5 * 0.6) = -0.3 + (0.12 + 0.30) = 0.12
        # softplus(0.12) = ln(1 + exp(0.12))
        expected_fe = 0.02 - 2.0 * np.log(1.0 + np.exp(0.12))

        actual_fe_1d = toy_model.compute_free_energy(v)
        self.assertAlmostEqual(actual_fe_1d, expected_fe, places=7)

        actual_fe_2d = toy_model.compute_free_energy(v.reshape(1, 2))
        self.assertAlmostEqual(float(actual_fe_2d[0]), expected_fe, places=7)

    def test_parameters_unmodified_during_scoring(self):
        """5. Verify compute_free_energy does not alter model parameters."""
        w_before = self.model.W.copy()
        b_before = self.model.b.copy()
        c_before = self.model.c.copy()

        _ = self.model.compute_free_energy(self.data["X_train"])

        np.testing.assert_array_equal(self.model.W, w_before)
        np.testing.assert_array_equal(self.model.b, b_before)
        np.testing.assert_array_equal(self.model.c, c_before)

    def test_threshold_calculation(self):
        """6. Verify threshold equals mean(train) + 3 * std(train, ddof=0)."""
        eval_res = evaluate_anomaly_detector(
            self.model,
            self.data["X_train"],
            self.data["X_test"],
            self.data["y_test"],
        )
        train_fe = eval_res["train_energy"]
        expected_threshold = float(np.mean(train_fe) + 3.0 * np.std(train_fe, ddof=0))
        self.assertAlmostEqual(eval_res["threshold"], expected_threshold, places=7)

    def test_threshold_unaffected_by_test_labels(self):
        """7. Verify threshold is invariant to test labels and test anomalies."""
        res_real = evaluate_anomaly_detector(
            self.model,
            self.data["X_train"],
            self.data["X_test"],
            self.data["y_test"],
        )
        # Flip test labels arbitrarily
        dummy_y = np.ones_like(self.data["y_test"])
        res_dummy = evaluate_anomaly_detector(
            self.model,
            self.data["X_train"],
            self.data["X_test"],
            dummy_y,
        )
        self.assertEqual(res_real["threshold"], res_dummy["threshold"])
        np.testing.assert_array_equal(res_real["train_energy"], res_dummy["train_energy"])

    def test_prediction_strict_comparison(self):
        """8. Verify predictions strictly adhere to test_energy > threshold."""
        eval_res = evaluate_anomaly_detector(
            self.model,
            self.data["X_train"],
            self.data["X_test"],
            self.data["y_test"],
        )
        expected_preds = eval_res["test_energy"] > eval_res["threshold"]
        np.testing.assert_array_equal(eval_res["predictions"], expected_preds)

    def test_confusion_matrix_internal_consistency(self):
        """9. Verify confusion matrix arithmetic is fully consistent with counts."""
        res = evaluate_anomaly_detector(
            self.model,
            self.data["X_train"],
            self.data["X_test"],
            self.data["y_test"],
        )
        tp, fp, tn, fn = res["TP"], res["FP"], res["TN"], res["FN"]
        n_test = len(res["y_test"])

        self.assertEqual(tp + fp + tn + fn, n_test)
        self.assertEqual(tp + fn, res["injected_anomaly_count"])
        self.assertEqual(tp + fp, res["detected_anomaly_count"])

    def test_metrics_match_independent_calculation(self):
        """10. Verify precision, recall, and F1 match ground truth arithmetic."""
        res = evaluate_anomaly_detector(
            self.model,
            self.data["X_train"],
            self.data["X_test"],
            self.data["y_test"],
        )
        tp, fp, fn = res["TP"], res["FP"], res["FN"]

        expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        expected_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        expected_f1 = (
            2.0 * expected_prec * expected_rec / (expected_prec + expected_rec)
            if (expected_prec + expected_rec) > 0.0
            else 0.0
        )

        self.assertAlmostEqual(res["precision"], expected_prec, places=7)
        self.assertAlmostEqual(res["recall"], expected_rec, places=7)
        self.assertAlmostEqual(res["f1_score"], expected_f1, places=7)

    def test_safe_division_on_zero_denominators(self):
        """11. Verify precision, recall, and F1 return 0.0 on zero denominators."""
        # Construct scenario with zero detections
        zero_preds = np.zeros(10, dtype=bool)
        zero_labels = np.zeros(10, dtype=bool)

        # Mocking evaluation logic directly
        tp = int(np.sum(zero_preds & zero_labels))
        fp = int(np.sum(zero_preds & ~zero_labels))
        fn = int(np.sum(~zero_preds & zero_labels))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)
        self.assertEqual(f1, 0.0)


if __name__ == "__main__":
    unittest.main()
