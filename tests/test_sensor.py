"""Tests for OdorSensor."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_scent.sensor import OdorSensor, ODORANT_PROFILES, N_SENSORS


class TestOdorSensor:
    def setup_method(self):
        self.sensor = OdorSensor(noise_std=0.01, rng_seed=0)

    def test_read_known_class(self):
        reading = self.sensor.read("floral")
        assert reading.label == "floral"
        assert reading.values.shape == (N_SENSORS,)
        assert np.all(reading.values >= 0) and np.all(reading.values <= 1)

    def test_read_all_classes(self):
        for cls in ODORANT_PROFILES:
            r = self.sensor.read(cls)
            assert r.label == cls
            assert r.values.shape == (N_SENSORS,)

    def test_read_unknown_class_raises(self):
        with pytest.raises(ValueError, match="Unknown odorant class"):
            self.sensor.read("rotten_egg")

    def test_read_mixture(self):
        r = self.sensor.read_mixture({"floral": 0.5, "citrus": 0.5})
        assert r.label is None
        assert r.values.shape == (N_SENSORS,)
        assert "components" in r.metadata

    def test_anomalous_reading(self):
        r = self.sensor.anomalous_reading()
        assert r.label is None
        assert r.metadata.get("anomaly") is True
        assert r.values.shape == (N_SENSORS,)

    def test_reproducibility(self):
        s1 = OdorSensor(rng_seed=7)
        s2 = OdorSensor(rng_seed=7)
        r1 = s1.read("citrus")
        r2 = s2.read("citrus")
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_concentration_scales_signal(self):
        """Higher concentration should generally produce higher sensor response."""
        s = OdorSensor(noise_std=0.001, rng_seed=1)
        low = s.read("musty", concentration=0.1)
        high = s.read("musty", concentration=1.0)
        # At least one channel should be higher in the high-concentration reading
        assert np.any(high.values > low.values)
