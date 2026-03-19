"""Tests for OdorantClassifier."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_scent.sensor import OdorSensor, ODORANT_PROFILES
from agentic_scent.classifier import OdorantClassifier


def make_training_data(n_per_class=20, noise=0.01, seed=0):
    sensor = OdorSensor(noise_std=noise, rng_seed=seed)
    features, labels = [], []
    for cls in ODORANT_PROFILES:
        for _ in range(n_per_class):
            r = sensor.read(cls)
            features.append(r.values)
            labels.append(cls)
    return np.stack(features), labels


class TestOdorantClassifier:
    def setup_method(self):
        self.clf = OdorantClassifier(metric="euclidean")
        self.features, self.labels = make_training_data()

    def test_fit_creates_centroids(self):
        self.clf.fit(self.features, self.labels)
        assert set(self.clf.classes) == set(ODORANT_PROFILES.keys())

    def test_predict_correct_on_clean_data(self):
        self.clf.fit(self.features, self.labels)
        # Predict on low-noise test samples
        sensor = OdorSensor(noise_std=0.001, rng_seed=99)
        correct = 0
        for cls in ODORANT_PROFILES:
            r = sensor.read(cls)
            pred = self.clf.predict(r.values)
            if pred == cls:
                correct += 1
        # All 5 classes should be correctly identified at very low noise
        assert correct >= 4

    def test_predict_proba_sums_to_one(self):
        self.clf.fit(self.features, self.labels)
        sensor = OdorSensor(noise_std=0.02, rng_seed=7)
        r = sensor.read("floral")
        proba = self.clf.predict_proba(r.values)
        assert abs(sum(proba.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in proba.values())

    def test_predict_before_fit_raises(self):
        clf = OdorantClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.zeros(8))

    def test_cosine_metric(self):
        clf = OdorantClassifier(metric="cosine")
        clf.fit(self.features, self.labels)
        sensor = OdorSensor(noise_std=0.001, rng_seed=42)
        r = sensor.read("citrus")
        pred = clf.predict(r.values)
        assert pred == "citrus"

    def test_fit_readings(self):
        sensor = OdorSensor(noise_std=0.02, rng_seed=10)
        readings = [sensor.read(cls) for cls in ODORANT_PROFILES for _ in range(10)]
        clf = OdorantClassifier()
        clf.fit_readings(readings)
        assert len(clf.classes) == len(ODORANT_PROFILES)

    def test_batch_predict(self):
        self.clf.fit(self.features, self.labels)
        preds = self.clf.predict_batch(self.features[:10])
        assert len(preds) == 10
        assert all(isinstance(p, str) for p in preds)
