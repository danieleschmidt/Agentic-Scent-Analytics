"""Tests for ScenarioSimulator end-to-end pipeline."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_scent.simulator import ScenarioSimulator, ScenarioReport


class TestScenarioSimulator:
    def setup_method(self):
        self.sim = ScenarioSimulator(
            n_train_per_class=20,
            n_sensor_modules=2,
            noise_std=0.05,
            rng_seed=42,
        )
        self.sim.train()

    def test_train_fits_all_agents(self):
        assert self.sim._fitted
        assert self.sim.classifier._fitted
        assert self.sim.anomaly_detector._fitted
        assert self.sim.fusion._fitted

    def test_classification_scenario_returns_report(self):
        report = self.sim.run_classification_scenario(n_samples=10)
        assert isinstance(report, ScenarioReport)
        assert len(report.results) == 10

    def test_classification_accuracy_reasonable(self):
        """With low noise, accuracy should be well above chance (>70%)."""
        sim = ScenarioSimulator(n_train_per_class=30, noise_std=0.03, rng_seed=0)
        sim.train()
        report = sim.run_classification_scenario(n_samples=50)
        assert report.accuracy > 0.70, f"Accuracy too low: {report.accuracy:.1%}"

    def test_anomaly_scenario_detects_injected(self):
        """At least some injected anomalies should be caught."""
        report = self.sim.run_anomaly_scenario(n_normal=10, n_anomalies=5)
        # Injected anomalies are the last 5 results
        injected = report.results[10:]
        detected = sum(1 for r in injected if r.anomaly.is_anomaly)
        assert detected >= 2, f"Only {detected}/5 anomalies detected"

    def test_anomaly_scenario_sample_count(self):
        report = self.sim.run_anomaly_scenario(n_normal=17, n_anomalies=3)
        assert len(report.results) == 20

    def test_mixture_scenario_runs(self):
        report = self.sim.run_mixture_scenario()
        assert len(report.results) == 5
        # All results should have a classification
        for r in report.results:
            assert r.predicted_label is not None

    def test_run_before_train_raises(self):
        fresh = ScenarioSimulator()
        with pytest.raises(RuntimeError, match="train"):
            fresh.run_classification_scenario()

    def test_report_summary_string(self):
        report = self.sim.run_classification_scenario(n_samples=5)
        summary = report.summary()
        assert "Accuracy" in summary
        assert "Anomalies" in summary
