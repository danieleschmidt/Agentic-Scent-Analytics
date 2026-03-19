"""
ScenarioSimulator — simulates factory and food safety e-nose scenarios.

Models realistic deployment contexts:
- Food spoilage detection line (detects musty/acrid anomalies)
- Fragrance QC station (classifies floral/citrus blends)
- Chemical plant safety monitoring (detects acrid/contamination spikes)

Each scenario runs the full pipeline: OdorSensor → SensorFusionAgent →
OdorantClassifier + AnomalyDetectionAgent → structured report.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .sensor import OdorSensor, SensorReading, ODORANT_PROFILES
from .fusion import SensorFusionAgent
from .classifier import OdorantClassifier
from .anomaly import AnomalyDetectionAgent, AnomalyResult


@dataclass
class SampleResult:
    """Result for a single classified sample."""
    sample_id: int
    true_label: Optional[str]
    predicted_label: Optional[str]
    confidence: float
    anomaly: AnomalyResult
    raw_values: np.ndarray

    @property
    def correct(self) -> Optional[bool]:
        if self.true_label is None:
            return None
        return self.true_label == self.predicted_label

    def __repr__(self) -> str:
        corr = "✓" if self.correct else ("✗" if self.correct is False else "?")
        return (
            f"[{self.sample_id:02d}] {corr} true={self.true_label!r:<10} "
            f"pred={self.predicted_label!r:<10} conf={self.confidence:.2f} "
            f"{'⚠ ANOMALY' if self.anomaly.is_anomaly else ''}"
        )


@dataclass
class ScenarioReport:
    """Aggregated results for a full scenario run."""
    scenario_name: str
    n_samples: int
    results: List[SampleResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        labeled = [r for r in self.results if r.correct is not None]
        if not labeled:
            return 0.0
        return sum(r.correct for r in labeled) / len(labeled)

    @property
    def n_anomalies(self) -> int:
        return sum(1 for r in self.results if r.anomaly.is_anomaly)

    def summary(self) -> str:
        lines = [
            f"=== {self.scenario_name} ===",
            f"Samples   : {self.n_samples}",
            f"Accuracy  : {self.accuracy:.1%}",
            f"Anomalies : {self.n_anomalies}",
            "",
        ]
        for r in self.results:
            lines.append(str(r))
        return "\n".join(lines)


class ScenarioSimulator:
    """
    Orchestrates end-to-end agentic e-nose scenarios.

    Parameters
    ----------
    n_train_per_class : int
        Number of training samples per odorant class.
    n_sensor_modules : int
        Number of independent sensor arrays to fuse.
    noise_std : float
        Sensor noise level.
    rng_seed : int | None
        Reproducibility seed.
    """

    CLASSES = list(ODORANT_PROFILES.keys())  # floral, citrus, musty, acrid, neutral

    def __init__(
        self,
        n_train_per_class: int = 20,
        n_sensor_modules: int = 2,
        noise_std: float = 0.05,
        rng_seed: Optional[int] = 42,
    ) -> None:
        self.n_train_per_class = n_train_per_class
        self.n_sensor_modules = n_sensor_modules
        self.noise_std = noise_std
        self.rng_seed = rng_seed

        # Instantiate agents
        self.sensors = [
            OdorSensor(noise_std=noise_std, rng_seed=(rng_seed or 0) + i)
            for i in range(n_sensor_modules)
        ]
        n_features = 8 * n_sensor_modules  # 8 channels × modules
        n_pca = min(4, n_features)
        self.fusion = SensorFusionAgent(n_components=n_pca)
        self.classifier = OdorantClassifier(metric="euclidean")
        self.anomaly_detector = AnomalyDetectionAgent(k_sigma=2.5, method="combined")
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> "ScenarioSimulator":
        """Generate training data and fit all agents."""
        readings_list: List[List[SensorReading]] = []
        labels: List[str] = []

        for cls in self.CLASSES:
            for _ in range(self.n_train_per_class):
                multi = [s.read(cls) for s in self.sensors]
                readings_list.append(multi)
                labels.append(cls)

        # Fuse and fit PCA
        fused = np.stack([self.fusion.fuse(rs) for rs in readings_list])
        features = self.fusion.fit_transform(fused)

        # Fit classifier and anomaly detector on PCA features
        self.classifier.fit(features, labels)
        self.anomaly_detector.fit(features, labels)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    def run_classification_scenario(
        self,
        n_samples: int = 20,
        scenario_name: str = "Fragrance QC Station",
    ) -> ScenarioReport:
        """
        Classify n_samples drawn uniformly from all odorant classes.
        """
        self._check_fitted()
        rng = np.random.default_rng(self.rng_seed)
        report = ScenarioReport(scenario_name=scenario_name, n_samples=n_samples)

        for i in range(n_samples):
            cls = str(rng.choice(self.CLASSES))
            readings = [s.read(cls) for s in self.sensors]
            fused = self.fusion.fuse(readings)
            features = self.fusion.transform(fused)

            predicted = self.classifier.predict(features)
            proba = self.classifier.predict_proba(features)
            confidence = proba[predicted]
            anomaly = self.anomaly_detector.detect(features)

            report.results.append(
                SampleResult(
                    sample_id=i,
                    true_label=cls,
                    predicted_label=predicted,
                    confidence=confidence,
                    anomaly=anomaly,
                    raw_values=fused,
                )
            )
        return report

    def run_anomaly_scenario(
        self,
        n_normal: int = 17,
        n_anomalies: int = 3,
        scenario_name: str = "Food Safety Spoilage Detection",
    ) -> ScenarioReport:
        """
        Run a scenario with a mix of normal samples and injected anomalies.
        """
        self._check_fitted()
        rng = np.random.default_rng((self.rng_seed or 0) + 99)
        report = ScenarioReport(
            scenario_name=scenario_name,
            n_samples=n_normal + n_anomalies,
        )

        # Normal samples
        for i in range(n_normal):
            cls = str(rng.choice(self.CLASSES))
            readings = [s.read(cls) for s in self.sensors]
            fused = self.fusion.fuse(readings)
            features = self.fusion.transform(fused)
            predicted = self.classifier.predict(features)
            proba = self.classifier.predict_proba(features)
            anomaly = self.anomaly_detector.detect(features)
            report.results.append(
                SampleResult(
                    sample_id=i,
                    true_label=cls,
                    predicted_label=predicted,
                    confidence=proba[predicted],
                    anomaly=anomaly,
                    raw_values=fused,
                )
            )

        # Injected anomalies
        for j in range(n_anomalies):
            readings = [s.anomalous_reading() for s in self.sensors]
            fused = self.fusion.fuse(readings)
            features = self.fusion.transform(fused)
            predicted = self.classifier.predict(features)
            proba = self.classifier.predict_proba(features)
            anomaly = self.anomaly_detector.detect(features)
            report.results.append(
                SampleResult(
                    sample_id=n_normal + j,
                    true_label=None,  # unknown / anomalous
                    predicted_label=predicted,
                    confidence=proba[predicted],
                    anomaly=anomaly,
                    raw_values=fused,
                )
            )

        return report

    def run_mixture_scenario(
        self,
        scenario_name: str = "Chemical Plant Safety Monitor",
    ) -> ScenarioReport:
        """
        Simulate a factory scenario with multi-odorant mixtures.
        """
        self._check_fitted()
        mixtures = [
            {"floral": 0.6, "neutral": 0.3},
            {"citrus": 0.8, "floral": 0.2},
            {"musty": 0.5, "acrid": 0.4},
            {"acrid": 0.9, "neutral": 0.1},  # potential contamination
            {"neutral": 0.7, "musty": 0.3},
        ]
        report = ScenarioReport(scenario_name=scenario_name, n_samples=len(mixtures))

        for i, mix in enumerate(mixtures):
            readings = [s.read_mixture(mix) for s in self.sensors]
            fused = self.fusion.fuse(readings)
            features = self.fusion.transform(fused)
            predicted = self.classifier.predict(features)
            proba = self.classifier.predict_proba(features)
            anomaly = self.anomaly_detector.detect(features)
            mix_desc = "+".join(f"{k}({v:.0%})" for k, v in mix.items())
            report.results.append(
                SampleResult(
                    sample_id=i,
                    true_label=mix_desc,
                    predicted_label=predicted,
                    confidence=proba[predicted],
                    anomaly=anomaly,
                    raw_values=fused,
                )
            )
        return report

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call train() before running scenarios.")
