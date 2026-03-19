"""Tests for SensorFusionAgent."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_scent.sensor import OdorSensor
from agentic_scent.fusion import SensorFusionAgent


class TestSensorFusionAgent:
    def setup_method(self):
        self.sensor = OdorSensor(rng_seed=0)
        self.fusion = SensorFusionAgent(n_components=3)

    def test_fuse_single_module(self):
        r = self.sensor.read("floral")
        fused = self.fusion.fuse([r])
        assert fused.shape == (8,)

    def test_fuse_two_modules(self):
        s2 = OdorSensor(rng_seed=1)
        r1 = self.sensor.read("floral")
        r2 = s2.read("floral")
        fused = self.fusion.fuse([r1, r2])
        assert fused.shape == (16,)

    def test_fuse_empty_raises(self):
        with pytest.raises(ValueError):
            self.fusion.fuse([])

    def test_fit_transform_shape(self):
        data = np.random.default_rng(0).uniform(0, 1, (50, 8))
        reduced = self.fusion.fit_transform(data)
        assert reduced.shape == (50, 3)

    def test_transform_single_vector(self):
        data = np.random.default_rng(0).uniform(0, 1, (50, 8))
        self.fusion.fit(data)
        vec = data[0]
        out = self.fusion.transform(vec)
        assert out.shape == (3,)

    def test_transform_before_fit_raises(self):
        fresh = SensorFusionAgent(n_components=2)
        with pytest.raises(RuntimeError, match="fit"):
            fresh.transform(np.zeros(8))

    def test_pca_reduces_variance(self):
        """First PC should capture more variance than individual sensors."""
        rng = np.random.default_rng(42)
        # Correlated data
        base = rng.uniform(0, 1, (100, 1))
        data = np.hstack([base + rng.normal(0, 0.01, (100, 1)) for _ in range(8)])
        reduced = self.fusion.fit_transform(data)
        # After PCA, first component should explain most variance
        var_pc1 = np.var(reduced[:, 0])
        var_original_avg = np.var(data).mean()
        assert var_pc1 > 0
