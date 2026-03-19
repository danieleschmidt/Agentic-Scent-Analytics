"""
AnomalyDetectionAgent — identifies unusual odorant profiles.

Uses two complementary methods:
1. Reconstruction error: fit a PCA subspace on normal data; anomalies
   have high reconstruction error when projected back.
2. Distance threshold: flag samples whose nearest-centroid distance
   exceeds a learned threshold (mean + k*std from training data).

In practice this catches spoilage events, contamination spikes, and
sensor faults that fall outside the normal operating envelope.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class AnomalyResult:
    """Result from anomaly detection for a single sample."""
    is_anomaly: bool
    score: float             # higher = more anomalous
    method: str              # "distance" | "reconstruction"
    nearest_class: Optional[str] = None
    nearest_dist: Optional[float] = None

    def __repr__(self) -> str:
        flag = "⚠ ANOMALY" if self.is_anomaly else "OK"
        return (
            f"AnomalyResult({flag}, score={self.score:.4f}, "
            f"method={self.method!r}, nearest={self.nearest_class})"
        )


class AnomalyDetectionAgent:
    """
    Detects anomalous sensor readings via distance-based thresholding
    and PCA reconstruction error.

    Parameters
    ----------
    k_sigma : float
        Number of standard deviations above the mean distance to use
        as the anomaly threshold. Higher = less sensitive.
    n_components : int
        Number of PCA components to retain for reconstruction-error method.
    method : str
        "distance" uses nearest-centroid distance thresholding.
        "reconstruction" uses PCA reconstruction error.
        "combined" flags as anomaly if EITHER method triggers.
    """

    def __init__(
        self,
        k_sigma: float = 2.5,
        n_components: int = 4,
        method: str = "combined",
    ) -> None:
        if method not in ("distance", "reconstruction", "combined"):
            raise ValueError(f"Unknown method {method!r}")
        self.k_sigma = k_sigma
        self.n_components = n_components
        self.method = method

        # Fitted state
        self._centroids: dict = {}
        self._dist_threshold: float = 0.0
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._recon_threshold: float = 0.0
        self._fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: List[str],
    ) -> "AnomalyDetectionAgent":
        """
        Learn normal operating envelope from labeled training data.

        Parameters
        ----------
        features : np.ndarray (n_samples, n_features)
        labels : list of str
        """
        # Build centroids
        unique_classes = sorted(set(labels))
        self._centroids = {}
        for cls in unique_classes:
            mask = np.array([l == cls for l in labels])
            self._centroids[cls] = features[mask].mean(axis=0)

        # Distance threshold: distribution of nearest-centroid distances on training data
        dists = self._nearest_centroid_distances(features)
        self._dist_threshold = dists.mean() + self.k_sigma * dists.std()

        # PCA for reconstruction error
        mean = features.mean(axis=0)
        self._pca_mean = mean
        centered = features - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self._pca_components = Vt[: self.n_components]

        recon_errors = self._reconstruction_errors(features)
        self._recon_threshold = recon_errors.mean() + self.k_sigma * recon_errors.std()

        self._fitted = True
        return self

    def detect(self, features: np.ndarray) -> AnomalyResult:
        """Detect anomaly for a single feature vector."""
        self._check_fitted()

        dist, nearest_cls = self._nearest_centroid(features)
        recon_err = float(self._reconstruction_error(features))

        is_dist_anomaly = dist > self._dist_threshold
        is_recon_anomaly = recon_err > self._recon_threshold

        if self.method == "distance":
            is_anomaly = is_dist_anomaly
            score = dist / (self._dist_threshold + 1e-9)
            method_used = "distance"
        elif self.method == "reconstruction":
            is_anomaly = is_recon_anomaly
            score = recon_err / (self._recon_threshold + 1e-9)
            method_used = "reconstruction"
        else:  # combined
            is_anomaly = is_dist_anomaly or is_recon_anomaly
            # Normalized max score
            score = max(
                dist / (self._dist_threshold + 1e-9),
                recon_err / (self._recon_threshold + 1e-9),
            )
            method_used = "combined"

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            method=method_used,
            nearest_class=nearest_cls,
            nearest_dist=dist,
        )

    def detect_batch(self, features: np.ndarray) -> List[AnomalyResult]:
        """Detect anomalies in a batch of feature vectors."""
        return [self.detect(row) for row in features]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_centroid_distances(self, features: np.ndarray) -> np.ndarray:
        dists = np.array([
            min(
                np.linalg.norm(row - c)
                for c in self._centroids.values()
            )
            for row in features
        ])
        return dists

    def _nearest_centroid(self, features: np.ndarray) -> Tuple[float, str]:
        best_cls, best_dist = None, float("inf")
        for cls, centroid in self._centroids.items():
            d = float(np.linalg.norm(features - centroid))
            if d < best_dist:
                best_dist, best_cls = d, cls
        return best_dist, best_cls

    def _reconstruction_error(self, features: np.ndarray) -> float:
        centered = features - self._pca_mean
        projected = self._pca_components @ centered
        reconstructed = self._pca_components.T @ projected
        return float(np.linalg.norm(centered - reconstructed))

    def _reconstruction_errors(self, features: np.ndarray) -> np.ndarray:
        return np.array([self._reconstruction_error(row) for row in features])

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before detect().")
