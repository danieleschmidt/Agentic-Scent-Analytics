# Agentic Scent Analytics

**Agentic framework for chemical sensor (e-nose) data analysis, odorant classification, and anomaly detection in industrial environments.**

Built for food safety, fragrance QC, and chemical plant monitoring — where human noses can't be everywhere.

---

## What It Does

Electronic noses (e-noses) are arrays of chemical sensors that produce a fingerprint response to odors. Interpreting those fingerprints at scale — across production lines, in real time — requires automation. This system provides:

- **Sensor modeling** — synthetic e-nose with 8-channel MOX-style sensor array, configurable noise, concentration, and mixture support
- **Sensor fusion** — aggregates multiple sensor modules into a single feature vector with PCA-based dimensionality reduction
- **Odorant classification** — nearest-centroid classifier trained on a reference library; supports Euclidean and cosine distance
- **Anomaly detection** — dual-method detection (nearest-centroid distance threshold + PCA reconstruction error) catches spoilage events, contamination spikes, and sensor faults
- **Scenario simulation** — end-to-end factory scenarios: fragrance QC, food safety, chemical plant safety

---

## Architecture

```
OdorSensor (sensor.py)
    │  generates synthetic sensor readings per odorant class
    ▼
SensorFusionAgent (fusion.py)
    │  fuses multi-module readings → PCA feature vector
    ▼
    ├── OdorantClassifier (classifier.py)
    │       nearest-centroid classification in PCA space
    │
    └── AnomalyDetectionAgent (anomaly.py)
            distance threshold + reconstruction error
            flags spoilage / contamination / sensor faults
```

`ScenarioSimulator` (`simulator.py`) orchestrates the full pipeline for three industrial scenarios.

---

## Odorant Classes

| Class   | Profile                                   | Example                          |
|---------|-------------------------------------------|----------------------------------|
| floral  | High channel 0, 4 — terpene-like         | Rose oil, lavender               |
| citrus  | High channel 1, 2 — limonene-like        | Lemon, orange peel               |
| musty   | High channel 3, 5, 6 — earthy/fungal     | Mold, damp wood, spoiled grain   |
| acrid   | High channel 5, 6, 7 — sulfur/smoke-like | Burning, chemical contamination  |
| neutral | Flat across all channels                  | Clean air baseline               |

---

## Quickstart

```bash
# Install dependencies (stdlib + numpy only)
pip install numpy

# Run the demo
python demo.py
```

**Demo output:**
```
=== Fragrance QC Station ===
Samples   : 20
Accuracy  : 100.0%
Anomalies : 0

=== Food Safety Spoilage Detection ===
Samples   : 20
Accuracy  : 100.0%
Anomalies : 3

Flagged as anomalous: 3 samples
  Sample 17: score=6.686 nearest='neutral'
  Sample 18: score=5.998 nearest='acrid'
  Sample 19: score=7.591 nearest='acrid'
```

---

## Usage

```python
from agentic_scent import ScenarioSimulator

sim = ScenarioSimulator(
    n_train_per_class=30,   # reference library size per class
    n_sensor_modules=2,     # number of sensor arrays to fuse
    noise_std=0.05,         # sensor noise level
    rng_seed=42,
)

sim.train()

# Classify known odorants
report = sim.run_classification_scenario(n_samples=20)
print(report.summary())

# Detect anomalies (spoilage, contamination)
report = sim.run_anomaly_scenario(n_normal=17, n_anomalies=3)
for r in report.results:
    if r.anomaly.is_anomaly:
        print(f"⚠ Anomaly detected: sample {r.sample_id}, score={r.anomaly.score:.3f}")

# Classify mixtures
report = sim.run_mixture_scenario()
```

### Direct agent usage

```python
from agentic_scent import OdorSensor, SensorFusionAgent, OdorantClassifier, AnomalyDetectionAgent

sensor = OdorSensor(noise_std=0.05, rng_seed=0)

# Generate a reading
reading = sensor.read("floral", concentration=0.8)
print(reading)  # SensorReading(label='floral', values=[...])

# Mixture
mix = sensor.read_mixture({"musty": 0.6, "acrid": 0.3})
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

37 tests covering sensor generation, fusion, classification, anomaly detection, and end-to-end scenarios.

---

## Industrial Applications

| Sector               | Scenario                         | Target anomalies                  |
|----------------------|----------------------------------|-----------------------------------|
| Food & Beverage      | Spoilage detection on intake line | Musty/acrid spikes (mold, decay)  |
| Fragrance / Cosmetics| Batch QC for perfume blends      | Off-ratio formulations            |
| Chemical Plants      | Leak detection, air quality       | Acrid spikes (solvents, gases)    |
| Grain storage        | Mycotoxin early warning          | Musty drift over time             |

---

## Dependencies

- Python 3.10+
- `numpy` ≥ 1.24

No LLM APIs, no cloud services, no Kubernetes. Just math and sensors.

---

## License

MIT
