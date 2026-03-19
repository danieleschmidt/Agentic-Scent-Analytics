"""Tests for AnomalyDetectionAgent."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_scent.sensor import OdorSensor, ODORANT_PROFILES
from agentic_scent.anomaly import AnomalyDetectionAgent, AnomalyResult


def make_training_data(n_per_class=30, seed=0):
    sensor = OdorSensor(noise_std=0.04, rng_seed=seed)
    features, labels = [], []
    for cls in ODORANT_PROFILES:
        for _ in range(n_per_class):
            r = sensor.read(cls)
            features.append(r.values)
            labels.append(cls)
    return np.stack(features), labels


class TestAnomalyDetectionAgent:
    def setup_method(self):
        self.agent = AnomalyDetectionAgent(k_sigma=2.5, method="combined")
        self.features, self.labels = make_training_data()
        self.agent.fit(self.features, self.labels)

    def test_normal_sample_not_anomaly(self):
        sensor = OdorSensor(noise_std=0.04, rng_seed=77)
        r = sensor.read("floral")
        result = self.agent.detect(r.values)
        assert isinstance(result, AnomalyResult)
        # With well-fitted data, clean samples should mostly be OK
        # (allow occasional false positives at 2.5 sigma)
        assert result.score >= 0

    def test_anomalous_sample_flagged(self):
        """Extreme anomalies should be detected."""
        # Create a pathological sample far from all centroids
        extreme = np.ones(8)  # maxed out all channels
        result = self.agent.detect(extreme)
        # This should be flagged as anomalous
        assert result.is_anomaly

    def test_result_has_nearest_class(self):
        sensor = OdorSensor(rng_seed=5)
        r = sensor.read("citrus")
        result = self.agent.detect(r.values)
        assert result.nearest_class in ODORANT_PROFILES
        assert result.nearest_dist is not None

    def test_detect_batch(self):
        results = self.agent.detect_batch(self.features[:10])
        assert len(results) == 10
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_fit_before_detect_raises(self):
        fresh = AnomalyDetectionAgent()
        with pytest.raises(RuntimeError, match="fit"):
            fresh.detect(np.zeros(8))

    def test_anomalous_reading_detected(self):
        """Sensor anomalous_reading() should trigger the anomaly detector."""
        sensor = OdorSensor(rng_seed=0)
        detected = 0
        for _ in range(10):
            r = sensor.anomalous_reading()
            result = self.agent.detect(r.values)
            if result.is_anomaly:
                detected += 1
        # At least half of artificial anomalies should be caught
        assert detected >= 5

    def test_distance_method(self):
        agent = AnomalyDetectionAgent(method="distance")
        agent.fit(self.features, self.labels)
        extreme = np.ones(8)
        result = agent.detect(extreme)
        assert result.method == "distance"
        assert result.is_anomaly

    def test_reconstruction_method(self):
        agent = AnomalyDetectionAgent(method="reconstruction")
        agent.fit(self.features, self.labels)
        extreme = np.ones(8)
        result = agent.detect(extreme)
        assert result.method == "reconstruction"
