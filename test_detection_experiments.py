"""Unit tests for controlled detection experiments."""

import unittest
import numpy as np
from rbm import RBM
from data_generator import generate_sensor_data
from detection_experiments import run_detection_experiments, calculate_metrics


class TestDetectionExperiments(unittest.TestCase):
    """Test suite validating experimental thresholding logic, metrics, and determinism."""

    @classmethod
    def setUpClass(cls):
        """Set up model and dataset once for experimental testing."""
        cls.data = generate_sensor_data(seed=42)
        cls.model = RBM(n_epochs=5, batch_size=32, seed=42)
        cls.model.fit(cls.data["X_train"], verbose=False)
        cls.results = run_detection_experiments(cls.model, cls.data)

    def test_threshold_derivations_from_train_only(self):
        """1. Verify thresholds A, B, and C strictly match training mathematical formulas."""
        train_energy = self.results["train_energy"]
        mu = float(np.mean(train_energy))
        sigma = float(np.std(train_energy, ddof=0))

        expected_a = mu + 3.0 * sigma
        expected_b = mu - 3.0 * sigma
        expected_c = float(np.percentile(train_energy, 1.0))

        self.assertAlmostEqual(self.results["approach_a"]["threshold"], expected_a, places=7)
        self.assertAlmostEqual(self.results["approach_b"]["threshold"], expected_b, places=7)
        self.assertAlmostEqual(self.results["approach_c"]["threshold"], expected_c, places=7)

    def test_thresholds_unaffected_by_test_labels(self):
        """2. Verify experimental thresholds are invariant to test labels."""
        dummy_data = dict(self.data)
        dummy_data["y_test"] = np.ones_like(self.data["y_test"])

        res_dummy = run_detection_experiments(self.model, dummy_data)

        self.assertEqual(
            self.results["approach_a"]["threshold"],
            res_dummy["approach_a"]["threshold"],
        )
        self.assertEqual(
            self.results["approach_b"]["threshold"],
            res_dummy["approach_b"]["threshold"],
        )
        self.assertEqual(
            self.results["approach_c"]["threshold"],
            res_dummy["approach_c"]["threshold"],
        )

    def test_decision_rules_adherence(self):
        """3. Verify predictions strictly adhere to directional decision rules."""
        test_energy = self.results["test_energy"]

        # Approach A: test_energy > threshold
        expected_a = test_energy > self.results["approach_a"]["threshold"]
        np.testing.assert_array_equal(self.results["approach_a"]["predictions"], expected_a)

        # Approach B: test_energy < threshold
        expected_b = test_energy < self.results["approach_b"]["threshold"]
        np.testing.assert_array_equal(self.results["approach_b"]["predictions"], expected_b)

        # Approach C: test_energy < threshold
        expected_c = test_energy < self.results["approach_c"]["threshold"]
        np.testing.assert_array_equal(self.results["approach_c"]["predictions"], expected_c)

    def test_confusion_matrix_arithmetic_consistency(self):
        """4. Verify internal consistency of confusion matrix counts across approaches."""
        for key in ["approach_a", "approach_b", "approach_c"]:
            app = self.results[key]
            m = app["metrics"]
            self.assertEqual(m["TP"] + m["FP"] + m["TN"] + m["FN"], 500)
            self.assertEqual(m["TP"] + m["FN"], 70)
            self.assertEqual(m["TP"] + m["FP"], m["detected_count"])
            self.assertEqual(m["w1_detected"] + m["w2_detected"], m["TP"])

    def test_precision_recall_f1_arithmetic(self):
        """5. Verify metric values match standard classification definitions."""
        for key in ["approach_a", "approach_b", "approach_c"]:
            m = self.results[key]["metrics"]
            tp, fp, fn = m["TP"], m["FP"], m["FN"]
            exp_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            exp_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            exp_f1 = (
                2.0 * exp_p * exp_r / (exp_p + exp_r)
                if (exp_p + exp_r) > 0
                else 0.0
            )

            self.assertAlmostEqual(m["precision"], exp_p, places=7)
            self.assertAlmostEqual(m["recall"], exp_r, places=7)
            self.assertAlmostEqual(m["f1_score"], exp_f1, places=7)

    def test_reproducibility(self):
        """6. Verify experimental runner is completely deterministic."""
        res_rep = run_detection_experiments(self.model, self.data)
        for key in ["approach_a", "approach_b", "approach_c"]:
            self.assertEqual(
                self.results[key]["threshold"],
                res_rep[key]["threshold"],
            )
            np.testing.assert_array_equal(
                self.results[key]["predictions"],
                res_rep[key]["predictions"],
            )


if __name__ == "__main__":
    unittest.main()
