"""
OdorantClassifier — nearest-centroid classifier in sensor (or PCA) space.

Trains on a reference library of labeled sensor readings.
Classification is the Euclidean nearest-centroid in the feature space,
which is interpretable, fast, and well-suited to small e-nose datasets.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .sensor import SensorReading


class OdorantClassifier:
    """
    Nearest-centroid classifier for odorant identification.

    Parameters
    ----------
    metric : str
        Distance metric. "euclidean" (default) or "cosine".
    """

    def __init__(self, metric: str = "euclidean") -> None:
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"Unknown metric {metric!r}")
        self.metric = metric
        self._centroids: Dict[str, np.ndarray] = {}
        self._classes: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        labels: List[str],
    ) -> "OdorantClassifier":
        """
        Compute class centroids from labelled feature vectors.

        Parameters
        ----------
        features : np.ndarray of shape (n_samples, n_features)
        labels : list of str, length n_samples

        Returns
        -------
        self
        """
        unique_classes = sorted(set(labels))
        self._classes = unique_classes
        self._centroids = {}
        for cls in unique_classes:
            mask = np.array([l == cls for l in labels])
            self._centroids[cls] = features[mask].mean(axis=0)
        self._fitted = True
        return self

    def fit_readings(self, readings: List[SensorReading]) -> "OdorantClassifier":
        """Convenience: fit directly from labeled SensorReadings."""
        labeled = [r for r in readings if r.label is not None]
        if not labeled:
            raise ValueError("No labeled readings provided.")
        features = np.stack([r.values for r in labeled])
        labels = [r.label for r in labeled]
        return self.fit(features, labels)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> str:
        """Predict the odorant class for a single feature vector."""
        self._check_fitted()
        distances = self._distances(features)
        return min(distances, key=distances.get)

    def predict_proba(self, features: np.ndarray) -> Dict[str, float]:
        """
        Softmax-normalized inverse distances as pseudo-probabilities.
        Useful for confidence estimation.
        """
        self._check_fitted()
        distances = self._distances(features)
        # Convert distance → score (higher = better)
        scores = {cls: 1.0 / (d + 1e-9) for cls, d in distances.items()}
        total = sum(scores.values())
        return {cls: s / total for cls, s in scores.items()}

    def predict_batch(self, features: np.ndarray) -> List[str]:
        """Predict classes for a batch (n_samples, n_features)."""
        return [self.predict(row) for row in features]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _distances(self, features: np.ndarray) -> Dict[str, float]:
        if self.metric == "euclidean":
            return {
                cls: float(np.linalg.norm(features - centroid))
                for cls, centroid in self._centroids.items()
            }
        else:  # cosine
            def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
                return float(1.0 - np.dot(a, b) / denom)
            return {
                cls: cosine_dist(features, centroid)
                for cls, centroid in self._centroids.items()
            }

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_readings() before predict().")

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    @property
    def centroids(self) -> Dict[str, np.ndarray]:
        return dict(self._centroids)
