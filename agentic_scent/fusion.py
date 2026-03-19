"""
SensorFusionAgent — aggregates readings from multiple sensor arrays
and applies PCA-like dimensionality reduction to a compact feature vector.

In real e-nose deployments you often have multiple sensor modules (MOX arrays,
optical sensors, acoustic sensors). This agent merges their outputs and
projects them into a lower-dimensional space using an incremental covariance
approach (a pure-numpy incremental PCA).
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional

from .sensor import SensorReading


class SensorFusionAgent:
    """
    Fuses readings from multiple sensor arrays into a single feature vector,
    then performs PCA-based dimensionality reduction.

    Parameters
    ----------
    n_components : int
        Target dimensionality after reduction.
    """

    def __init__(self, n_components: int = 4) -> None:
        self.n_components = n_components
        # PCA state — updated via fit()
        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None  # shape (n_components, n_features)
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(self, readings: List[SensorReading]) -> np.ndarray:
        """
        Concatenate sensor readings from multiple arrays into one feature vector.

        Parameters
        ----------
        readings : list of SensorReading
            One reading per sensor module.

        Returns
        -------
        np.ndarray of shape (n_features,) where n_features = sum of each reading length.
        """
        if not readings:
            raise ValueError("Need at least one reading to fuse.")
        return np.concatenate([r.values for r in readings])

    def fit(self, data: np.ndarray) -> "SensorFusionAgent":
        """
        Fit PCA on a dataset of fused readings.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self (for chaining)
        """
        if data.ndim != 2:
            raise ValueError("data must be 2D (n_samples, n_features)")
        self._mean = data.mean(axis=0)
        centered = data - self._mean
        # Economy SVD — equivalent to PCA decomposition
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self._components = Vt[: self.n_components]
        self._fitted = True
        return self

    def transform(self, fused: np.ndarray) -> np.ndarray:
        """
        Project a fused feature vector into the PCA subspace.

        Parameters
        ----------
        fused : np.ndarray of shape (n_features,) or (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_components,) or (n_samples, n_components)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        centered = fused - self._mean
        if centered.ndim == 1:
            return self._components @ centered
        return centered @ self._components.T

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform the training data."""
        self.fit(data)
        return self.transform(data)

    def process(
        self,
        readings_per_module: List[List[SensorReading]],
    ) -> np.ndarray:
        """
        Full pipeline: fuse → transform for a batch of multi-module readings.

        Parameters
        ----------
        readings_per_module : list of lists
            Outer list = samples; inner list = one reading per sensor module.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
        """
        fused = np.stack([self.fuse(rs) for rs in readings_per_module])
        return self.transform(fused)
