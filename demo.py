#!/usr/bin/env python3
"""
Agentic Scent Analytics — Demo

Runs three industrial olfaction scenarios:
  1. Fragrance QC station: classify 20 samples across 5 odorant classes
  2. Food safety spoilage detection: 17 normal + 3 injected anomalies
  3. Chemical plant safety: classify multi-odorant mixtures

Requires only stdlib + numpy.
"""
from agentic_scent import ScenarioSimulator

def main():
    print("=" * 60)
    print("  AGENTIC SCENT ANALYTICS — Industrial E-Nose Demo")
    print("=" * 60)
    print()

    sim = ScenarioSimulator(
        n_train_per_class=30,
        n_sensor_modules=2,
        noise_std=0.05,
        rng_seed=42,
    )

    print("Training on reference library (5 classes × 30 samples × 2 sensor modules)...")
    sim.train()
    print("✓ All agents fitted.\n")

    # --- Scenario 1: Classification ---
    report1 = sim.run_classification_scenario(n_samples=20)
    print(report1.summary())
    print()

    # --- Scenario 2: Anomaly Detection ---
    report2 = sim.run_anomaly_scenario(n_normal=17, n_anomalies=3)
    print(report2.summary())

    # Show which samples were flagged
    flagged = [r for r in report2.results if r.anomaly.is_anomaly]
    print(f"\nFlagged as anomalous: {len(flagged)} samples")
    for r in flagged:
        print(f"  Sample {r.sample_id:02d}: score={r.anomaly.score:.3f} "
              f"nearest={r.anomaly.nearest_class!r}")
    print()

    # --- Scenario 3: Mixtures ---
    report3 = sim.run_mixture_scenario()
    print(report3.summary())
    print()

    print("=" * 60)
    print(f"  Classification accuracy: {report1.accuracy:.1%}")
    print(f"  Anomalies detected:      {report2.n_anomalies} / {report2.n_samples} flagged")
    print("=" * 60)


if __name__ == "__main__":
    main()
