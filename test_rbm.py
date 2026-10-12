"""Unit tests for Bernoulli Restricted Boltzmann Machine (RBM)."""

import unittest
import numpy as np
from rbm import RBM, sigmoid
from data_generator import generate_sensor_data


class TestRBM(unittest.TestCase):
    """Test suite for RBM initialization, sampling, CD-1 updates, and inference."""

    def setUp(self):
        """Set up reproducible model and sample test data."""
        self.model = RBM(
            n_visible=8,
            n_hidden=5,
            learning_rate=0.01,
            batch_size=32,
            n_epochs=50,
            seed=42,
        )
        self.dataset = generate_sensor_data(seed=42)
        self.x_train = self.dataset["X_train"]

    def test_initial_parameter_shapes(self):
        """1. Verify initial parameter shapes: W (5, 8), b (8,), c (5,)."""
        self.assertEqual(self.model.W.shape, (5, 8))
        self.assertEqual(self.model.b.shape, (8,))
        self.assertEqual(self.model.c.shape, (5,))

    def test_zero_initialized_biases(self):
        """2. Verify biases b and c are strictly initialized to zero."""
        np.testing.assert_array_equal(self.model.b, np.zeros(8))
        np.testing.assert_array_equal(self.model.c, np.zeros(5))

    def test_reproducible_initialization(self):
        """3. Verify identical seeds produce bitwise identical initial parameters."""
        model_clone = RBM(seed=42)
        np.testing.assert_array_equal(self.model.W, model_clone.W)
        np.testing.assert_array_equal(self.model.b, model_clone.b)
        np.testing.assert_array_equal(self.model.c, model_clone.c)

    def test_probability_dimensions(self):
        """4. Verify hidden and visible probability output dimensions."""
        batch_size = 16
        v_dummy = np.full((batch_size, 8), 0.5)
        h_dummy = np.full((batch_size, 5), 0.5)

        h_prob = self.model.hidden_probabilities(v_dummy)
        v_prob = self.model.visible_probabilities(h_dummy)

        self.assertEqual(h_prob.shape, (batch_size, 5))
        self.assertEqual(v_prob.shape, (batch_size, 8))

    def test_probability_bounds_and_finiteness(self):
        """5. Verify probability values are finite and strictly in [0.0, 1.0]."""
        v_dummy = np.array([[0.0] * 8, [1.0] * 8, [0.5] * 8])
        h_prob = self.model.hidden_probabilities(v_dummy)
        self.assertTrue(np.all(np.isfinite(h_prob)))
        self.assertTrue(np.all((h_prob >= 0.0) & (h_prob <= 1.0)))

        h_dummy = np.array([[0.0] * 5, [1.0] * 5, [0.5] * 5])
        v_prob = self.model.visible_probabilities(h_dummy)
        self.assertTrue(np.all(np.isfinite(v_prob)))
        self.assertTrue(np.all((v_prob >= 0.0) & (v_prob <= 1.0)))

    def test_bernoulli_samples(self):
        """6. Verify Bernoulli samples contain only values in {0.0, 1.0}."""
        probs = np.full((100, 5), 0.5)
        samples = self.model.sample_bernoulli(probs)
        self.assertTrue(np.all(np.isin(samples, [0.0, 1.0])))

    def test_cd1_parameter_updates(self):
        """7. Verify a CD-1 step changes parameters on a non-degenerate batch."""
        w_before = self.model.W.copy()
        b_before = self.model.b.copy()
        c_before = self.model.c.copy()

        # Run 1 epoch with small subset
        small_batch = self.x_train[:32]
        m = RBM(n_epochs=1, batch_size=32, seed=42)
        m.fit(small_batch, verbose=False)

        self.assertFalse(np.array_equal(m.W, w_before))
        self.assertFalse(np.array_equal(m.b, b_before))
        self.assertFalse(np.array_equal(m.c, c_before))

    def test_training_completes_50_epochs(self):
        """8. Verify training runs 50 epochs on full 2,000-row X_train."""
        model = RBM(n_epochs=50, batch_size=32, seed=42)
        fitted = model.fit(self.x_train, verbose=False)
        self.assertIs(fitted, model)
        self.assertEqual(len(model.epoch_errors), 50)
        self.assertTrue(all(np.isfinite(err) for err in model.epoch_errors))

    def test_trained_parameters_finite(self):
        """9. Verify all trained weights and biases remain strictly finite."""
        model = RBM(n_epochs=5, batch_size=32, seed=42)
        model.fit(self.x_train, verbose=False)
        self.assertTrue(np.all(np.isfinite(model.W)))
        self.assertTrue(np.all(np.isfinite(model.b)))
        self.assertTrue(np.all(np.isfinite(model.c)))

    def test_inference_dimensions(self):
        """10. Verify transform and reconstruct output dimensions."""
        h_repr = self.model.transform(self.x_train[:50])
        v_recon = self.model.reconstruct(self.x_train[:50])
        err = self.model.reconstruction_error(self.x_train[:50])

        self.assertEqual(h_repr.shape, (50, 5))
        self.assertEqual(v_recon.shape, (50, 8))
        self.assertIsInstance(err, float)
        self.assertTrue(np.isfinite(err))

    def test_invalid_inputs_rejected(self):
        """11. Verify invalid shapes, NaNs, infinities, and out-of-range inputs raise ValueError."""
        # Invalid visible shape (wrong columns)
        with self.assertRaises(ValueError):
            self.model.hidden_probabilities(np.zeros((10, 7)))

        # Invalid 1D shape
        with self.assertRaises(ValueError):
            self.model.hidden_probabilities(np.zeros(8))

        # NaN values
        nan_mat = np.full((10, 8), 0.5)
        nan_mat[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model.hidden_probabilities(nan_mat)

        # Inf values
        inf_mat = np.full((10, 8), 0.5)
        inf_mat[0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.model.hidden_probabilities(inf_mat)

        # Out-of-range values (> 1.0)
        high_mat = np.full((10, 8), 1.5)
        with self.assertRaises(ValueError):
            self.model.hidden_probabilities(high_mat)

        # Out-of-range values (< 0.0)
        neg_mat = np.full((10, 8), -0.5)
        with self.assertRaises(ValueError):
            self.model.hidden_probabilities(neg_mat)


if __name__ == "__main__":
    unittest.main()
