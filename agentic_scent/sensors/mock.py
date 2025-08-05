"""
Mock sensor implementations for testing and development.
"""

import asyncio
import random
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta

from .base import SensorInterface, SensorReading, SensorType, SensorCalibration, SensorError


class MockENoseSensor(SensorInterface):
    """
    Mock electronic nose sensor for testing.
    Simulates realistic e-nose data with configurable noise and drift.
    """
    
    def __init__(self, sensor_id: str, channels: int = 32, sampling_rate: float = 10.0):
        super().__init__(sensor_id, SensorType.E_NOSE)
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.baseline_values = [random.uniform(100, 1000) for _ in range(channels)]
        self.noise_level = 0.05  # 5% noise
        self.drift_rate = 0.001  # 0.1% per hour
        self.start_time = datetime.now()
        
        # Simulation parameters
        self.contamination_probability = 0.02  # 2% chance per reading
        self.fault_probability = 0.001  # 0.1% chance per reading
        
    async def initialize(self) -> bool:
        """Initialize mock sensor."""
        await asyncio.sleep(0.1)  # Simulate initialization delay
        
        # Create mock calibration
        self.calibration = SensorCalibration(
            sensor_id=self.sensor_id,
            calibration_date=datetime.now(),
            baseline_values=self.baseline_values.copy(),
            scale_factors=[1.0] * self.channels,
            offset_values=[0.0] * self.channels,
            valid_until=datetime.now() + timedelta(days=30),
            calibration_method="mock_factory_default"
        )
        
        return True
    
    async def read(self) -> SensorReading:
        """Generate mock sensor reading."""
        if not self.is_online():
            raise SensorError(self.sensor_id, "Sensor is not online")
        
        # Simulate sensor fault
        if random.random() < self.fault_probability:
            raise SensorError(self.sensor_id, "Sensor communication timeout", error_code=1001)
        
        # Calculate time-based drift
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        drift_factor = 1.0 + (self.drift_rate * elapsed_hours)
        
        # Generate base values with drift
        values = [baseline * drift_factor for baseline in self.baseline_values]
        
        # Add random noise
        for i in range(len(values)):
            noise = random.gauss(0, values[i] * self.noise_level)
            values[i] += noise
        
        # Simulate contamination event
        contamination_detected = random.random() < self.contamination_probability
        if contamination_detected:
            # Increase certain channels to simulate contamination signature
            contamination_channels = random.sample(range(self.channels), k=min(6, self.channels))
            for channel in contamination_channels:
                values[channel] *= random.uniform(1.5, 3.0)
        
        # Apply calibration
        calibrated_values = self.apply_calibration(values)
        
        # Ensure positive values
        calibrated_values = [max(0.1, value) for value in calibrated_values]
        
        # Calculate quality score based on signal stability
        quality_score = 1.0 - min(0.5, np.std(calibrated_values) / np.mean(calibrated_values))
        
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            values=calibrated_values,
            metadata={
                "contamination_simulated": contamination_detected,
                "drift_factor": drift_factor,
                "channels": self.channels,
                "sampling_rate": self.sampling_rate
            },
            quality_score=quality_score
        )
        
        self.last_reading = reading
        return reading
    
    async def calibrate(self, reference_values: List[float] = None) -> bool:
        """Perform mock calibration."""
        await asyncio.sleep(2.0)  # Simulate calibration time
        
        if reference_values and len(reference_values) == self.channels:
            # Use provided reference values
            self.baseline_values = reference_values.copy()
        else:
            # Generate new baseline
            self.baseline_values = [random.uniform(100, 1000) for _ in range(self.channels)]
        
        # Update calibration
        self.calibration = SensorCalibration(
            sensor_id=self.sensor_id,
            calibration_date=datetime.now(),
            baseline_values=self.baseline_values.copy(),
            scale_factors=[1.0] * self.channels,
            offset_values=[0.0] * self.channels,
            valid_until=datetime.now() + timedelta(days=30),
            calibration_method="user_calibration"
        )
        
        return True
    
    def set_contamination_probability(self, probability: float):
        """Set contamination simulation probability."""
        self.contamination_probability = max(0.0, min(1.0, probability))
    
    def set_noise_level(self, noise_level: float):
        """Set noise level for simulation."""
        self.noise_level = max(0.0, noise_level)


class MockTemperatureSensor(SensorInterface):
    """Mock temperature sensor."""
    
    def __init__(self, sensor_id: str, target_temp: float = 25.0):
        super().__init__(sensor_id, SensorType.TEMPERATURE)
        self.target_temp = target_temp
        self.current_temp = target_temp
    
    async def initialize(self) -> bool:
        """Initialize mock temperature sensor."""
        await asyncio.sleep(0.05)
        return True
    
    async def read(self) -> SensorReading:
        """Read mock temperature."""
        # Simulate temperature variation
        variation = random.gauss(0, 0.5)  # ±0.5°C variation
        self.current_temp = self.target_temp + variation
        
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            values=[self.current_temp],
            metadata={"target_temp": self.target_temp}
        )
        
        self.last_reading = reading
        return reading
    
    async def calibrate(self, reference_values: List[float] = None) -> bool:
        """Mock temperature calibration."""
        await asyncio.sleep(0.5)
        return True


class MockHumiditySensor(SensorInterface):
    """Mock humidity sensor."""
    
    def __init__(self, sensor_id: str, target_humidity: float = 45.0):
        super().__init__(sensor_id, SensorType.HUMIDITY)
        self.target_humidity = target_humidity
        self.current_humidity = target_humidity
    
    async def initialize(self) -> bool:
        """Initialize mock humidity sensor."""
        await asyncio.sleep(0.05)
        return True
    
    async def read(self) -> SensorReading:
        """Read mock humidity."""
        # Simulate humidity variation
        variation = random.gauss(0, 2.0)  # ±2% RH variation
        self.current_humidity = max(0, min(100, self.target_humidity + variation))
        
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            values=[self.current_humidity],
            metadata={"target_humidity": self.target_humidity}
        )
        
        self.last_reading = reading
        return reading
    
    async def calibrate(self, reference_values: List[float] = None) -> bool:
        """Mock humidity calibration."""
        await asyncio.sleep(0.5)
        return True


class MockPressureSensor(SensorInterface):
    """Mock pressure sensor."""
    
    def __init__(self, sensor_id: str, target_pressure: float = 1013.25):
        super().__init__(sensor_id, SensorType.PRESSURE)
        self.target_pressure = target_pressure
        self.current_pressure = target_pressure
    
    async def initialize(self) -> bool:
        """Initialize mock pressure sensor."""
        await asyncio.sleep(0.05)
        return True
    
    async def read(self) -> SensorReading:
        """Read mock pressure."""
        # Simulate pressure variation
        variation = random.gauss(0, 1.0)  # ±1 hPa variation
        self.current_pressure = self.target_pressure + variation
        
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            values=[self.current_pressure],
            metadata={"target_pressure": self.target_pressure}
        )
        
        self.last_reading = reading
        return reading
    
    async def calibrate(self, reference_values: List[float] = None) -> bool:
        """Mock pressure calibration."""
        await asyncio.sleep(0.5)
        return True