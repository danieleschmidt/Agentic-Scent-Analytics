"""
OdorSensor — models a chemical sensor array (electronic nose).

Each sensor has a characteristic sensitivity profile across odorant classes.
Sensor responses are generated as a weighted combination of odorant concentrations
plus Gaussian noise, mimicking real metal-oxide or optical sensor arrays.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Canonical odorant classes and their chemical fingerprint bases.
# Each value is a sensitivity vector (one entry per sensor channel).
ODORANT_PROFILES: Dict[str, np.ndarray] = {
    "floral":  np.array([0.9, 0.1, 0.05, 0.2, 0.6, 0.3, 0.05, 0.1]),
    "citrus":  np.array([0.1, 0.8, 0.7, 0.05, 0.2, 0.1, 0.3, 0.05]),
    "musty":   np.array([0.2, 0.1, 0.05, 0.9, 0.1, 0.6, 0.7, 0.4]),
    "acrid":   np.array([0.05, 0.3, 0.4, 0.3, 0.05, 0.8, 0.9, 0.8]),
    "neutral": np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
}

N_SENSORS = 8  # number of sensor channels in the array


@dataclass
class SensorReading:
    """A single snapshot from the sensor array."""
    values: np.ndarray          # raw voltage-like response, shape (N_SENSORS,)
    label: Optional[str] = None # ground-truth class if known
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        vals = ", ".join(f"{v:.3f}" for v in self.values)
        return f"SensorReading(label={self.label!r}, values=[{vals}])"


class OdorSensor:
    """
    Simulates a multi-channel chemical sensor array (e-nose).

    Parameters
    ----------
    noise_std : float
        Standard deviation of Gaussian noise added to each reading.
    n_sensors : int
        Number of sensor channels.
    rng_seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        n_sensors: int = N_SENSORS,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.noise_std = noise_std
        self.n_sensors = n_sensors
        self.rng = np.random.default_rng(rng_seed)

    def read(
        self,
        odorant_class: str,
        concentration: float = 1.0,
    ) -> SensorReading:
        """
        Generate a synthetic sensor reading for a given odorant class.

        Parameters
        ----------
        odorant_class : str
            One of the keys in ODORANT_PROFILES.
        concentration : float
            Scaling factor in [0, 1] representing odorant concentration.

        Returns
        -------
        SensorReading
        """
        if odorant_class not in ODORANT_PROFILES:
            raise ValueError(
                f"Unknown odorant class {odorant_class!r}. "
                f"Valid: {list(ODORANT_PROFILES)}"
            )
        profile = ODORANT_PROFILES[odorant_class]
        signal = profile * concentration
        noise = self.rng.normal(0, self.noise_std, size=self.n_sensors)
        values = np.clip(signal + noise, 0.0, 1.0)
        return SensorReading(values=values, label=odorant_class)

    def read_mixture(
        self,
        components: Dict[str, float],
    ) -> SensorReading:
        """
        Generate a sensor reading for a mixture of odorants.

        Parameters
        ----------
        components : dict
            Mapping of odorant_class -> concentration (0..1).
            Concentrations need not sum to 1.

        Returns
        -------
        SensorReading with label=None (mixture has no single class).
        """
        signal = np.zeros(self.n_sensors)
        for cls, conc in components.items():
            if cls not in ODORANT_PROFILES:
                raise ValueError(f"Unknown odorant class {cls!r}")
            signal += ODORANT_PROFILES[cls] * conc
        signal = np.clip(signal, 0.0, 1.0)
        noise = self.rng.normal(0, self.noise_std, size=self.n_sensors)
        values = np.clip(signal + noise, 0.0, 1.0)
        return SensorReading(values=values, metadata={"components": components})

    def anomalous_reading(self) -> SensorReading:
        """
        Generate a synthetic anomalous reading (e.g., spoilage, contamination).

        Produces a random spike pattern that doesn't match any known class.
        """
        values = self.rng.uniform(0.4, 1.0, size=self.n_sensors)
        # Random spikes on a subset of channels to look suspicious
        spike_idx = self.rng.choice(self.n_sensors, size=3, replace=False)
        values[spike_idx] = self.rng.uniform(0.85, 1.0, size=3)
        return SensorReading(values=values, label=None, metadata={"anomaly": True})
