"""
Base sensor interface and data structures.
"""

import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SensorType(Enum):
    """Types of sensors supported."""
    E_NOSE = "electronic_nose"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    PH = "ph"
    FLOW_RATE = "flow_rate"
    VISION = "vision"


@dataclass
class SensorReading:
    """Raw sensor reading data."""
    sensor_id: str
    sensor_type: SensorType
    values: List[float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0  # 0-1, indicates data quality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "values": self.values,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "quality_score": self.quality_score
        }


@dataclass
class SensorCalibration:
    """Sensor calibration parameters."""
    sensor_id: str
    calibration_date: datetime
    baseline_values: List[float]
    scale_factors: List[float]
    offset_values: List[float]
    valid_until: datetime
    calibration_method: str = "factory_default"


class SensorStatus(Enum):
    """Sensor operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"


class SensorInterface(ABC):
    """
    Abstract base class for all sensor interfaces.
    """
    
    def __init__(self, sensor_id: str, sensor_type: SensorType):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.status = SensorStatus.OFFLINE
        self.calibration: Optional[SensorCalibration] = None
        self.last_reading: Optional[SensorReading] = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize sensor hardware and communication.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def read(self) -> SensorReading:
        """
        Read current sensor values.
        
        Returns:
            SensorReading with current values
            
        Raises:
            SensorError if reading fails
        """
        pass
    
    @abstractmethod
    async def calibrate(self, reference_values: List[float] = None) -> bool:
        """
        Perform sensor calibration.
        
        Args:
            reference_values: Known reference values for calibration
            
        Returns:
            True if calibration successful, False otherwise
        """
        pass
    
    async def start(self):
        """Start sensor operations."""
        if await self.initialize():
            self.status = SensorStatus.ONLINE
        else:
            self.status = SensorStatus.ERROR
    
    async def stop(self):
        """Stop sensor operations."""
        self.status = SensorStatus.OFFLINE
    
    def get_status(self) -> SensorStatus:
        """Get current sensor status."""
        return self.status
    
    def is_online(self) -> bool:
        """Check if sensor is online and functional."""
        return self.status == SensorStatus.ONLINE
    
    def needs_calibration(self) -> bool:
        """Check if sensor needs calibration."""
        if not self.calibration:
            return True
        return datetime.now() > self.calibration.valid_until
    
    def apply_calibration(self, raw_values: List[float]) -> List[float]:
        """Apply calibration to raw values."""
        if not self.calibration:
            return raw_values
        
        calibrated = []
        for i, value in enumerate(raw_values):
            if i < len(self.calibration.scale_factors) and i < len(self.calibration.offset_values):
                calibrated_value = (value - self.calibration.offset_values[i]) * self.calibration.scale_factors[i]
                calibrated.append(calibrated_value)
            else:
                calibrated.append(value)
        
        return calibrated


class SensorError(Exception):
    """Base exception for sensor-related errors."""
    
    def __init__(self, sensor_id: str, message: str, error_code: int = None):
        self.sensor_id = sensor_id
        self.error_code = error_code
        super().__init__(f"Sensor {sensor_id}: {message}")


class SensorTimeoutError(SensorError):
    """Sensor communication timeout error."""
    pass


class SensorCalibrationError(SensorError):
    """Sensor calibration error."""
    pass


class SensorManager:
    """
    Manages multiple sensors and provides unified interface.
    """
    
    def __init__(self):
        self.sensors: Dict[str, SensorInterface] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    def register_sensor(self, sensor: SensorInterface):
        """Register a sensor with the manager."""
        self.sensors[sensor.sensor_id] = sensor
    
    def unregister_sensor(self, sensor_id: str):
        """Unregister a sensor."""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
        if sensor_id in self._monitoring_tasks:
            self._monitoring_tasks[sensor_id].cancel()
            del self._monitoring_tasks[sensor_id]
    
    async def start_all_sensors(self):
        """Start all registered sensors."""
        for sensor in self.sensors.values():
            await sensor.start()
    
    async def stop_all_sensors(self):
        """Stop all registered sensors."""
        # Cancel monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        self._monitoring_tasks.clear()
        
        # Stop sensors
        for sensor in self.sensors.values():
            await sensor.stop()
    
    def get_sensor(self, sensor_id: str) -> Optional[SensorInterface]:
        """Get sensor by ID."""
        return self.sensors.get(sensor_id)
    
    def get_online_sensors(self) -> List[SensorInterface]:
        """Get all online sensors."""
        return [sensor for sensor in self.sensors.values() if sensor.is_online()]
    
    async def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read from all online sensors."""
        readings = {}
        
        for sensor_id, sensor in self.sensors.items():
            if sensor.is_online():
                try:
                    reading = await sensor.read()
                    readings[sensor_id] = reading
                except SensorError as e:
                    # Log error but continue with other sensors
                    print(f"Error reading from sensor {sensor_id}: {e}")
        
        return readings
    
    async def start_continuous_monitoring(self, callback=None, interval: float = 1.0):
        """Start continuous monitoring of all sensors."""
        async def monitor_sensor(sensor: SensorInterface):
            while sensor.is_online():
                try:
                    reading = await sensor.read()
                    if callback:
                        await callback(sensor.sensor_id, reading)
                    await asyncio.sleep(interval)
                except SensorError:
                    # Sensor error, wait longer before retry
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    break
        
        # Start monitoring task for each online sensor
        for sensor in self.get_online_sensors():
            if sensor.sensor_id not in self._monitoring_tasks:
                task = asyncio.create_task(monitor_sensor(sensor))
                self._monitoring_tasks[sensor.sensor_id] = task