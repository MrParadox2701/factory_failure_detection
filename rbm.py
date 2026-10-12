"""Restricted Boltzmann Machine (RBM) with Contrastive Divergence (CD-1).

This module implements a Bernoulli Restricted Boltzmann Machine trained using
single-step Contrastive Divergence (CD-1). All computations use pure NumPy with
numerically stable sigmoid activations, reproducible dedicated random generators,
and exact mini-batch gradient updates.
"""

from typing import List, Tuple, Union, Optional
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the numerically stable logistic sigmoid function.

    Avoids overflow in exp(-x) for negative and positive inputs by branching:
    sigmoid(x) = 1 / (1 + exp(-x)) for x >= 0
    sigmoid(x) = exp(x) / (1 + exp(x)) for x < 0

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Sigmoid activation values in [0.0, 1.0].
    """
    x_arr = np.asarray(x, dtype=np.float64)
    pos_mask = x_arr >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x_arr)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
    exp_neg = np.exp(x_arr[neg_mask])
    out[neg_mask] = exp_neg / (1.0 + exp_neg)
    return out


class RBM:
    """Bernoulli Restricted Boltzmann Machine trained via CD-1.

    Parameters
    ----------
    n_visible : int, default=8
        Number of visible units.
    n_hidden : int, default=5
        Number of hidden units.
    learning_rate : float, default=0.01
        Learning rate for parameter updates.
    batch_size : int, default=32
        Mini-batch size.
    n_epochs : int, default=50
        Number of full training epochs.
    seed : int, default=42
        Seed for reproducible initialization and sampling.

    Attributes
    ----------
    W : np.ndarray, shape (n_hidden, n_visible)
        Coupling weights between hidden and visible units.
    b : np.ndarray, shape (n_visible,)
        Visible unit biases.
    c : np.ndarray, shape (n_hidden,)
        Hidden unit biases.
    epoch_errors : list of float
        Reconstruction MSE recorded after each epoch.
    """

    def __init__(
        self,
        n_visible: int = 8,
        n_hidden: int = 5,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 50,
        seed: int = 42,
    ) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed

        # Dedicated random generator for reproducibility
        self.rng = np.random.default_rng(seed)

        # Initialize parameters:
        # W shape: (n_hidden, n_visible) ~ N(0, 0.01^2)
        self.W = self.rng.normal(
            loc=0.0, scale=0.01, size=(self.n_hidden, self.n_visible)
        ).astype(np.float64)
        # b shape: (n_visible,), zeros
        self.b = np.zeros(self.n_visible, dtype=np.float64)
        # c shape: (n_hidden,), zeros
        self.c = np.zeros(self.n_hidden, dtype=np.float64)

        self.epoch_errors: List[float] = []

    def _validate_visible(self, X: np.ndarray) -> np.ndarray:
        """Validate input visible matrix dimensions, finiteness, and range [0, 1]."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got ndim={X_arr.ndim} shape={X_arr.shape}")
        if X_arr.shape[1] != self.n_visible:
            raise ValueError(
                f"Expected {self.n_visible} visible features, got {X_arr.shape[1]}"
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input array contains NaN or Inf values")
        if np.any(X_arr < 0.0) or np.any(X_arr > 1.0):
            raise ValueError("Input visible values must be within [0.0, 1.0]")
        return X_arr

    def _validate_hidden(self, H: np.ndarray) -> np.ndarray:
        """Validate hidden matrix dimensions and finiteness."""
        H_arr = np.asarray(H, dtype=np.float64)
        if H_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got ndim={H_arr.ndim} shape={H_arr.shape}")
        if H_arr.shape[1] != self.n_hidden:
            raise ValueError(
                f"Expected {self.n_hidden} hidden units, got {H_arr.shape[1]}"
            )
        if not np.all(np.isfinite(H_arr)):
            raise ValueError("Hidden array contains NaN or Inf values")
        return H_arr

    def hidden_probabilities(self, v: np.ndarray) -> np.ndarray:
        """Compute p(h=1|v) = sigmoid(v @ W.T + c).

        Parameters
        ----------
        v : np.ndarray, shape (batch_size, n_visible)
            Visible input data.

        Returns
        -------
        np.ndarray, shape (batch_size, n_hidden)
            Conditional activation probabilities of hidden units.
        """
        v_arr = self._validate_visible(v)
        return sigmoid(v_arr @ self.W.T + self.c)

    def p_h_given_v(self, v: np.ndarray) -> np.ndarray:
        """Alias for hidden_probabilities."""
        return self.hidden_probabilities(v)

    def visible_probabilities(self, h: np.ndarray) -> np.ndarray:
        """Compute p(v=1|h) = sigmoid(h @ W + b).

        Parameters
        ----------
        h : np.ndarray, shape (batch_size, n_hidden)
            Hidden unit activations.

        Returns
        -------
        np.ndarray, shape (batch_size, n_visible)
            Conditional activation probabilities of visible units.
        """
        h_arr = self._validate_hidden(h)
        return sigmoid(h_arr @ self.W + self.b)

    def p_v_given_h(self, h: np.ndarray) -> np.ndarray:
        """Alias for visible_probabilities."""
        return self.visible_probabilities(h)

    def sample_bernoulli(self, probs: np.ndarray) -> np.ndarray:
        """Draw Bernoulli samples from probability matrix using dedicated RNG.

        Parameters
        ----------
        probs : np.ndarray
            Matrix of activation probabilities in [0, 1].

        Returns
        -------
        np.ndarray
            Binary sample matrix in {0.0, 1.0} with same shape as probs.
        """
        probs_arr = np.asarray(probs, dtype=np.float64)
        if not np.all(np.isfinite(probs_arr)):
            raise ValueError("Probabilities contain NaN or Inf values")
        return (self.rng.uniform(size=probs_arr.shape) < probs_arr).astype(np.float64)

    def fit(self, X: np.ndarray, verbose: bool = True) -> "RBM":
        """Train the RBM using Contrastive Divergence with 1 Gibbs step (CD-1).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_visible)
            Training visible dataset in [0.0, 1.0].
        verbose : bool, default=True
            Whether to print progress at epochs 10, 20, 30, 40, 50.

        Returns
        -------
        self : RBM
            Fitted model instance.
        """
        X_arr = self._validate_visible(X)
        n_samples = X_arr.shape[0]
        self.epoch_errors = []

        for epoch in range(1, self.n_epochs + 1):
            # Deterministically shuffle training rows every epoch
            indices = self.rng.permutation(n_samples)
            X_shuffled = X_arr[indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                v0 = X_shuffled[start_idx:end_idx]
                actual_batch_size = v0.shape[0]

                # Positive Phase
                # 1. Calculate h0_prob = p(h=1|v0)
                h0_prob = sigmoid(v0 @ self.W.T + self.c)
                # 2. Sample binary hidden states h0
                h0 = self.sample_bernoulli(h0_prob)

                # Negative Phase
                # 3. Calculate visible reconstruction probabilities from h0
                v1_prob = sigmoid(h0 @ self.W + self.b)
                # 4. Sample binary reconstructed visible states v1
                v1 = self.sample_bernoulli(v1_prob)
                # 5. Calculate h1_prob = p(h=1|v1)
                h1_prob = sigmoid(v1 @ self.W.T + self.c)

                # CD-1 Gradient Updates:
                # delta_W = learning_rate * (h0.T @ v0 - h1_prob.T @ v1) / actual_batch_size
                delta_W = (
                    self.learning_rate
                    * (h0.T @ v0 - h1_prob.T @ v1)
                    / actual_batch_size
                )
                # delta_b = learning_rate * sum(v0 - v1, axis=0) / actual_batch_size
                delta_b = (
                    self.learning_rate
                    * np.sum(v0 - v1, axis=0)
                    / actual_batch_size
                )
                # delta_c = learning_rate * sum(h0 - h1_prob, axis=0) / actual_batch_size
                delta_c = (
                    self.learning_rate
                    * np.sum(h0 - h1_prob, axis=0)
                    / actual_batch_size
                )

                # Update model parameters
                self.W += delta_W
                self.b += delta_b
                self.c += delta_c

            # Record epoch reconstruction MSE using visible reconstruction probabilities
            # Convention: Evaluated analytically via reconstruct(X) to provide low-variance tracking
            epoch_mse = self.reconstruction_error(X_arr)
            self.epoch_errors.append(epoch_mse)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:2d}/{self.n_epochs:2d} - "
                    f"Reconstruction MSE: {epoch_mse:.6f}"
                )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute hidden representation probabilities p(h=1|X).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_visible)
            Visible input data.

        Returns
        -------
        np.ndarray, shape (n_samples, n_hidden)
            Hidden unit activation probabilities. Does not alter model state.
        """
        return self.hidden_probabilities(X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Compute visible reconstruction probabilities p(v=1|p(h=1|X)).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_visible)
            Visible input data.

        Returns
        -------
        np.ndarray, shape (n_samples, n_visible)
            Reconstructed visible probabilities in [0.0, 1.0].
            Does not consume random samples.
        """
        h_probs = self.transform(X)
        return self.visible_probabilities(h_probs)

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Calculate mean squared reconstruction error.

        Uses visible reconstruction probabilities rather than sampled binary states
        to guarantee deterministic and minimal-variance evaluation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_visible)
            Input data.

        Returns
        -------
        float
            Mean squared error across all samples and features.
        """
        X_arr = self._validate_visible(X)
        v_rec = self.reconstruct(X_arr)
        return float(np.mean((X_arr - v_rec) ** 2))

    def compute_free_energy(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """Compute analytical free energy F(v) = -b^T v - sum_i softplus(c_i + W[i,:] @ v).

        Uses NumPy's logaddexp(0.0, z) for numerically stable softplus computation.
        The calculation is purely analytical and deterministic without Gibbs sampling.

        Parameters
        ----------
        X : np.ndarray, shape (n_visible,) or (n_samples, n_visible)
            Visible input vector or batch of vectors in [0.0, 1.0].

        Returns
        -------
        float or np.ndarray
            Scalar free energy if 1D input; 1D array of shape (n_samples,) if 2D input.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        is_1d = False

        if X_arr.ndim == 1:
            if X_arr.shape[0] != self.n_visible:
                raise ValueError(
                    f"Expected 1D vector of length {self.n_visible}, got shape {X_arr.shape}"
                )
            if not np.all(np.isfinite(X_arr)):
                raise ValueError("Input array contains NaN or Inf values")
            if np.any(X_arr < 0.0) or np.any(X_arr > 1.0):
                raise ValueError("Input visible values must be within [0.0, 1.0]")
            is_1d = True
            X_batch = X_arr.reshape(1, self.n_visible)
        elif X_arr.ndim == 2:
            X_batch = self._validate_visible(X_arr)
        else:
            raise ValueError(
                f"Expected 1D or 2D array, got ndim={X_arr.ndim} shape={X_arr.shape}"
            )

        # Visible linear term: - b^T v for each sample
        visible_term = X_batch @ self.b

        # Hidden activation inputs: c_i + W[i, :] @ v
        hidden_inputs = X_batch @ self.W.T + self.c

        # Numerically stable softplus: log(1 + exp(z)) = logaddexp(0.0, z)
        softplus_vals = np.logaddexp(0.0, hidden_inputs)

        # Sum softplus across all hidden units: shape (N,)
        hidden_term = np.sum(softplus_vals, axis=1)

        # Free energy: - visible_term - hidden_term
        free_energy = -visible_term - hidden_term

        if is_1d:
            return float(free_energy[0])
        return free_energy
