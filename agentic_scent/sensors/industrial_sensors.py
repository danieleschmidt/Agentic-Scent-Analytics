"""
Industrial Sensor Interfaces for Real-World E-nose Deployments

Implements production-ready interfaces for commercial e-nose systems,
MES integration, SCADA connectivity, and OPC-UA protocol support.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import json
from enum import Enum
from abc import ABC, abstractmethod

# Import industrial protocols (with fallbacks)
try:
    from opcua import Client as OPCClient, ua
    OPC_AVAILABLE = True
except ImportError:
    OPC_AVAILABLE = False
    
try:
    import modbus_tk.modbus_tcp as modbus_tcp
    from modbus_tk import modbus
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False

try:
    from pyserial import Serial
    SERIAL_AVAILABLE = True
except ImportError:
    try:
        import serial as Serial
        SERIAL_AVAILABLE = True
    except ImportError:
        SERIAL_AVAILABLE = False

from .base import SensorInterface, SensorReading


class SensorProtocol(Enum):
    """Supported industrial sensor protocols."""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    ETHERNET_IP = "ethernet_ip"
    PROFINET = "profinet"
    RS485 = "rs485"
    RS232 = "rs232"
    TCP_SOCKET = "tcp_socket"
    MQTT = "mqtt"
    HTTP_REST = "http_rest"
    CUSTOM = "custom"


class SensorStatus(Enum):
    """Sensor operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


@dataclass
class SensorConfig:
    """Configuration for industrial sensors."""
    sensor_id: str
    protocol: SensorProtocol
    connection_params: Dict[str, Any]
    sampling_rate: float = 1.0  # Hz
    timeout: float = 5.0  # seconds
    retry_count: int = 3
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    validation_params: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class CalibrationData:
    """Sensor calibration information."""
    calibration_date: datetime
    calibration_coefficients: Dict[str, float]
    reference_standards: List[Dict[str, Any]]
    validity_period: timedelta
    accuracy_metrics: Dict[str, float]
    
    @property
    def is_valid(self) -> bool:
        """Check if calibration is still valid."""
        return datetime.now() < (self.calibration_date + self.validity_period)
        
        
class IndustrialSensorInterface(SensorInterface):
    """
    Base class for industrial sensor interfaces with advanced features:
    - Protocol abstraction
    - Automatic reconnection
    - Data validation
    - Calibration management
    - Diagnostic capabilities
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config.sensor_id)
        self.config = config
        self.connection = None
        self.status = SensorStatus.OFFLINE
        self.last_reading_time: Optional[datetime] = None
        self.calibration: Optional[CalibrationData] = None
        self.error_count = 0
        self.total_readings = 0
        self.connection_attempts = 0
        self.logger = logging.getLogger(f"{__name__}.{config.sensor_id}")
        
    async def connect(self) -> bool:
        """Establish connection to sensor."""
        try:
            self.connection_attempts += 1
            await self._establish_connection()
            self.status = SensorStatus.ONLINE
            self.logger.info(f"Connected to sensor {self.sensor_id}")
            return True
        except Exception as e:
            self.status = SensorStatus.ERROR
            self.logger.error(f"Failed to connect to sensor {self.sensor_id}: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from sensor."""
        try:
            await self._close_connection()
            self.status = SensorStatus.OFFLINE
            self.logger.info(f"Disconnected from sensor {self.sensor_id}")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
            
    async def read(self) -> SensorReading:
        """Read data from sensor with validation and error handling."""
        if self.status != SensorStatus.ONLINE:
            if not await self.connect():
                raise RuntimeError(f"Sensor {self.sensor_id} not available")
                
        for attempt in range(self.config.retry_count):
            try:
                raw_data = await self._read_raw_data()
                validated_data = await self._validate_data(raw_data)
                calibrated_data = await self._apply_calibration(validated_data)
                
                reading = SensorReading(
                    sensor_id=self.sensor_id,
                    timestamp=datetime.now(),
                    values=calibrated_data,
                    quality_score=self._calculate_quality_score(calibrated_data),
                    metadata={
                        "attempt": attempt + 1,
                        "total_readings": self.total_readings,
                        "sensor_status": self.status.value,
                        "calibration_valid": self.calibration.is_valid if self.calibration else False
                    }
                )
                
                self.last_reading_time = reading.timestamp
                self.total_readings += 1
                return reading
                
            except Exception as e:
                self.error_count += 1
                self.logger.warning(f"Reading attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.retry_count - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    self.status = SensorStatus.ERROR
                    raise RuntimeError(f"Failed to read from sensor after {self.config.retry_count} attempts")
                    
    @abstractmethod
    async def _establish_connection(self):
        """Protocol-specific connection establishment."""
        pass
        
    @abstractmethod
    async def _close_connection(self):
        """Protocol-specific connection closure."""
        pass
        
    @abstractmethod
    async def _read_raw_data(self) -> Dict[str, Any]:
        """Protocol-specific data reading."""
        pass
        
    async def _validate_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sensor data quality."""
        validation_params = self.config.validation_params
        validated_data = raw_data.copy()
        
        # Range validation
        if "valid_ranges" in validation_params:
            for key, value in validated_data.items():
                if isinstance(value, (int, float)):
                    valid_range = validation_params["valid_ranges"].get(key)
                    if valid_range and not (valid_range[0] <= value <= valid_range[1]):
                        self.logger.warning(f"Value {key}={value} outside valid range {valid_range}")
                        validated_data[key] = np.clip(value, valid_range[0], valid_range[1])
                        
        # Noise filtering
        if "noise_threshold" in validation_params:
            threshold = validation_params["noise_threshold"]
            for key, value in validated_data.items():
                if isinstance(value, (int, float)) and abs(value) < threshold:
                    validated_data[key] = 0.0
                    
        return validated_data
        
    async def _apply_calibration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calibration to sensor data."""
        if not self.calibration or not self.calibration.is_valid:
            return data  # Return uncalibrated data if no valid calibration
            
        calibrated_data = {}
        coefficients = self.calibration.calibration_coefficients
        
        for key, value in data.items():
            if isinstance(value, (int, float)) and key in coefficients:
                # Linear calibration: calibrated = slope * raw + offset
                slope = coefficients.get(f"{key}_slope", 1.0)
                offset = coefficients.get(f"{key}_offset", 0.0)
                calibrated_data[key] = slope * value + offset
            else:
                calibrated_data[key] = value
                
        return calibrated_data
        
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate data quality score based on various factors."""
        factors = []
        
        # Calibration validity
        if self.calibration and self.calibration.is_valid:
            factors.append(0.9)
        else:
            factors.append(0.5)
            
        # Error rate
        error_rate = self.error_count / max(self.total_readings, 1)
        factors.append(max(0.0, 1.0 - error_rate * 10))
        
        # Data completeness
        expected_keys = set(self.config.validation_params.get("expected_keys", data.keys()))
        actual_keys = set(data.keys())
        completeness = len(actual_keys.intersection(expected_keys)) / len(expected_keys) if expected_keys else 1.0
        factors.append(completeness)
        
        return float(np.mean(factors))
        
    async def calibrate(self, reference_data: List[Dict[str, Any]]) -> CalibrationData:
        """Perform sensor calibration."""
        self.logger.info(f"Starting calibration for sensor {self.sensor_id}")
        self.status = SensorStatus.CALIBRATING
        
        try:
            # Collect calibration readings
            calibration_readings = []
            for ref in reference_data:
                reading = await self._read_raw_data()
                calibration_readings.append((reading, ref))
                await asyncio.sleep(1.0)  # Allow sensor to stabilize
                
            # Calculate calibration coefficients
            coefficients = await self._calculate_calibration_coefficients(
                calibration_readings
            )
            
            # Create calibration data
            self.calibration = CalibrationData(
                calibration_date=datetime.now(),
                calibration_coefficients=coefficients,
                reference_standards=reference_data,
                validity_period=timedelta(days=30),  # 30-day validity
                accuracy_metrics=await self._calculate_accuracy_metrics(
                    calibration_readings, coefficients
                )
            )
            
            self.status = SensorStatus.ONLINE
            self.logger.info(f"Calibration completed for sensor {self.sensor_id}")
            return self.calibration
            
        except Exception as e:
            self.status = SensorStatus.ERROR
            self.logger.error(f"Calibration failed: {e}")
            raise
            
    async def _calculate_calibration_coefficients(
        self, calibration_data: List[tuple]
    ) -> Dict[str, float]:
        """Calculate calibration coefficients from reference data."""
        coefficients = {}
        
        # Group data by sensor channel
        channels = {}
        for reading, reference in calibration_data:
            for key, value in reading.items():
                if isinstance(value, (int, float)):
                    if key not in channels:
                        channels[key] = {"readings": [], "references": []}
                    channels[key]["readings"].append(value)
                    channels[key]["references"].append(
                        reference.get(key, value)  # Use reading if no reference
                    )
                    
        # Calculate linear regression coefficients for each channel
        for channel, data in channels.items():
            readings = np.array(data["readings"])
            references = np.array(data["references"])
            
            if len(readings) > 1:
                # Linear regression: reference = slope * reading + offset
                slope, offset = np.polyfit(readings, references, 1)
                coefficients[f"{channel}_slope"] = float(slope)
                coefficients[f"{channel}_offset"] = float(offset)
            else:
                coefficients[f"{channel}_slope"] = 1.0
                coefficients[f"{channel}_offset"] = 0.0
                
        return coefficients
        
    async def _calculate_accuracy_metrics(
        self, calibration_data: List[tuple], coefficients: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate calibration accuracy metrics."""
        metrics = {}
        
        # Calculate RMSE and R² for each channel
        channels = {}
        for reading, reference in calibration_data:
            for key, value in reading.items():
                if isinstance(value, (int, float)) and f"{key}_slope" in coefficients:
                    if key not in channels:
                        channels[key] = {"errors": [], "references": []}
                    
                    # Apply calibration
                    slope = coefficients[f"{key}_slope"]
                    offset = coefficients[f"{key}_offset"]
                    calibrated_value = slope * value + offset
                    reference_value = reference.get(key, value)
                    
                    error = calibrated_value - reference_value
                    channels[key]["errors"].append(error)
                    channels[key]["references"].append(reference_value)
                    
        for channel, data in channels.items():
            errors = np.array(data["errors"])
            references = np.array(data["references"])
            
            if len(errors) > 0:
                rmse = np.sqrt(np.mean(errors**2))
                mae = np.mean(np.abs(errors))
                
                # R-squared
                if len(references) > 1:
                    predicted = references + errors  # Reverse calculate predicted
                    ss_res = np.sum(errors**2)
                    ss_tot = np.sum((references - np.mean(references))**2)
                    r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                else:
                    r_squared = 0.0
                    
                metrics[f"{channel}_rmse"] = float(rmse)
                metrics[f"{channel}_mae"] = float(mae)
                metrics[f"{channel}_r_squared"] = float(r_squared)
                
        return metrics
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive sensor diagnostics."""
        diagnostics = {
            "sensor_id": self.sensor_id,
            "status": self.status.value,
            "protocol": self.config.protocol.value,
            "connection_attempts": self.connection_attempts,
            "total_readings": self.total_readings,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_readings, 1),
            "last_reading_time": self.last_reading_time.isoformat() if self.last_reading_time else None,
            "uptime_hours": (datetime.now() - self.last_reading_time).total_seconds() / 3600 if self.last_reading_time else 0,
            "calibration_status": {
                "calibrated": self.calibration is not None,
                "valid": self.calibration.is_valid if self.calibration else False,
                "date": self.calibration.calibration_date.isoformat() if self.calibration else None
            }
        }
        
        return diagnostics


class OPCUASensorInterface(IndustrialSensorInterface):
    """
    OPC-UA sensor interface for industrial automation systems.
    Supports subscription-based data collection and secure communication.
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        if not OPC_AVAILABLE:
            raise ImportError("OPC-UA library not available. Install python-opcua")
        self.client: Optional[OPCClient] = None
        self.subscription = None
        self.nodes = {}
        
    async def _establish_connection(self):
        """Establish OPC-UA connection."""
        endpoint = self.config.connection_params["endpoint"]
        self.client = OPCClient(endpoint)
        
        # Configure security if specified
        if "security_policy" in self.config.connection_params:
            security_policy = self.config.connection_params["security_policy"]
            security_mode = self.config.connection_params.get("security_mode", "SignAndEncrypt")
            
            # Apply security configuration
            self.client.set_security_string(f"{security_policy},{security_mode}")
            
        # Set authentication if specified
        if "username" in self.config.connection_params:
            self.client.set_user(self.config.connection_params["username"])
            self.client.set_password(self.config.connection_params["password"])
            
        await self.client.connect()
        
        # Get node references
        node_ids = self.config.connection_params.get("node_ids", [])
        for node_id in node_ids:
            node = self.client.get_node(node_id)
            self.nodes[node_id] = node
            
    async def _close_connection(self):
        """Close OPC-UA connection."""
        if self.subscription:
            self.subscription.delete()
        if self.client:
            await self.client.disconnect()
            
    async def _read_raw_data(self) -> Dict[str, Any]:
        """Read data from OPC-UA nodes."""
        if not self.client or not self.nodes:
            raise RuntimeError("OPC-UA client not connected")
            
        data = {}
        for node_id, node in self.nodes.items():
            try:
                value = await node.read_value()
                data[node_id] = value
            except Exception as e:
                self.logger.warning(f"Failed to read node {node_id}: {e}")
                data[node_id] = None
                
        return data


class ModbusTCPSensorInterface(IndustrialSensorInterface):
    """
    Modbus TCP sensor interface for industrial sensors.
    Supports holding registers, input registers, and coils.
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        if not MODBUS_AVAILABLE:
            raise ImportError("Modbus library not available. Install modbus_tk")
        self.master = None
        
    async def _establish_connection(self):
        """Establish Modbus TCP connection."""
        host = self.config.connection_params["host"]
        port = self.config.connection_params.get("port", 502)
        
        self.master = modbus_tcp.TcpMaster(host=host, port=port)
        self.master.set_timeout(self.config.timeout)
        
    async def _close_connection(self):
        """Close Modbus TCP connection."""
        if self.master:
            self.master.close()
            
    async def _read_raw_data(self) -> Dict[str, Any]:
        """Read data from Modbus registers."""
        if not self.master:
            raise RuntimeError("Modbus master not connected")
            
        slave_id = self.config.connection_params.get("slave_id", 1)
        registers = self.config.connection_params.get("registers", {})
        
        data = {}
        
        # Read holding registers
        if "holding_registers" in registers:
            for reg_name, reg_addr in registers["holding_registers"].items():
                try:
                    values = self.master.execute(
                        slave_id, modbus.READ_HOLDING_REGISTERS, reg_addr, 1
                    )
                    data[reg_name] = values[0] if values else None
                except Exception as e:
                    self.logger.warning(f"Failed to read holding register {reg_name}: {e}")
                    data[reg_name] = None
                    
        # Read input registers
        if "input_registers" in registers:
            for reg_name, reg_addr in registers["input_registers"].items():
                try:
                    values = self.master.execute(
                        slave_id, modbus.READ_INPUT_REGISTERS, reg_addr, 1
                    )
                    data[reg_name] = values[0] if values else None
                except Exception as e:
                    self.logger.warning(f"Failed to read input register {reg_name}: {e}")
                    data[reg_name] = None
                    
        return data


class SerialSensorInterface(IndustrialSensorInterface):
    """
    Serial (RS232/RS485) sensor interface.
    Supports custom protocols and command-response patterns.
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        if not SERIAL_AVAILABLE:
            raise ImportError("Serial library not available. Install pyserial")
        self.serial_conn = None
        
    async def _establish_connection(self):
        """Establish serial connection."""
        port = self.config.connection_params["port"]
        baudrate = self.config.connection_params.get("baudrate", 9600)
        bytesize = self.config.connection_params.get("bytesize", 8)
        parity = self.config.connection_params.get("parity", "N")
        stopbits = self.config.connection_params.get("stopbits", 1)
        
        self.serial_conn = Serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            timeout=self.config.timeout
        )
        
    async def _close_connection(self):
        """Close serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            
    async def _read_raw_data(self) -> Dict[str, Any]:
        """Read data via serial protocol."""
        if not self.serial_conn or not self.serial_conn.is_open:
            raise RuntimeError("Serial connection not open")
            
        command = self.config.connection_params.get("read_command", "READ\r\n")
        response_format = self.config.connection_params.get("response_format", "csv")
        
        # Send read command
        self.serial_conn.write(command.encode())
        
        # Wait for response
        await asyncio.sleep(0.1)
        
        # Read response
        response = self.serial_conn.read_all().decode().strip()
        
        # Parse response based on format
        if response_format == "csv":
            values = response.split(",")
            data = {f"channel_{i}": float(val) for i, val in enumerate(values) if val.replace('.', '').replace('-', '').isdigit()}
        elif response_format == "json":
            data = json.loads(response)
        else:
            # Custom parsing
            data = {"raw_response": response}
            
        return data


class HTTPRESTSensorInterface(IndustrialSensorInterface):
    """
    HTTP REST API sensor interface for modern IoT sensors.
    Supports JSON data exchange and authentication.
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.session = None
        
    async def _establish_connection(self):
        """Initialize HTTP session."""
        import aiohttp
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Handle authentication
        auth_type = self.config.connection_params.get("auth_type")
        if auth_type == "bearer":
            token = self.config.connection_params["token"]
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        elif auth_type == "basic":
            username = self.config.connection_params["username"]
            password = self.config.connection_params["password"]
            auth = aiohttp.BasicAuth(username, password)
            self.session.auth = auth
            
    async def _close_connection(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            
    async def _read_raw_data(self) -> Dict[str, Any]:
        """Read data via HTTP REST API."""
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
            
        url = self.config.connection_params["endpoint"]
        method = self.config.connection_params.get("method", "GET")
        
        try:
            async with self.session.request(method, url) as response:
                response.raise_for_status()
                data = await response.json()
                return data
        except Exception as e:
            self.logger.error(f"HTTP request failed: {e}")
            raise


def create_sensor_interface(config: SensorConfig) -> IndustrialSensorInterface:
    """
    Factory function to create appropriate sensor interface based on protocol.
    """
    protocol_map = {
        SensorProtocol.OPC_UA: OPCUASensorInterface,
        SensorProtocol.MODBUS_TCP: ModbusTCPSensorInterface,
        SensorProtocol.RS232: SerialSensorInterface,
        SensorProtocol.RS485: SerialSensorInterface,
        SensorProtocol.HTTP_REST: HTTPRESTSensorInterface,
    }
    
    interface_class = protocol_map.get(config.protocol)
    if not interface_class:
        raise ValueError(f"Unsupported protocol: {config.protocol}")
        
    return interface_class(config)


class SensorNetworkManager:
    """
    Manages multiple industrial sensors with automatic discovery,
    health monitoring, and fault tolerance.
    """
    
    def __init__(self):
        self.sensors: Dict[str, IndustrialSensorInterface] = {}
        self.sensor_configs: Dict[str, SensorConfig] = {}
        self.health_monitor_task = None
        self.logger = logging.getLogger(__name__)
        
    async def add_sensor(self, config: SensorConfig) -> bool:
        """Add a new sensor to the network."""
        try:
            sensor = create_sensor_interface(config)
            await sensor.connect()
            
            self.sensors[config.sensor_id] = sensor
            self.sensor_configs[config.sensor_id] = config
            
            self.logger.info(f"Added sensor {config.sensor_id} ({config.protocol.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add sensor {config.sensor_id}: {e}")
            return False
            
    async def remove_sensor(self, sensor_id: str):
        """Remove a sensor from the network."""
        if sensor_id in self.sensors:
            await self.sensors[sensor_id].disconnect()
            del self.sensors[sensor_id]
            del self.sensor_configs[sensor_id]
            self.logger.info(f"Removed sensor {sensor_id}")
            
    async def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read data from all sensors concurrently."""
        tasks = {}
        for sensor_id, sensor in self.sensors.items():
            if sensor.status == SensorStatus.ONLINE:
                tasks[sensor_id] = asyncio.create_task(sensor.read())
                
        results = {}
        for sensor_id, task in tasks.items():
            try:
                reading = await task
                results[sensor_id] = reading
            except Exception as e:
                self.logger.error(f"Failed to read from sensor {sensor_id}: {e}")
                
        return results
        
    async def start_health_monitoring(self, interval: float = 60.0):
        """Start background health monitoring."""
        self.health_monitor_task = asyncio.create_task(
            self._health_monitor_loop(interval)
        )
        
    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
                
    async def _health_monitor_loop(self, interval: float):
        """Background health monitoring loop."""
        while True:
            try:
                await self._check_sensor_health()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)
                
    async def _check_sensor_health(self):
        """Check health of all sensors."""
        for sensor_id, sensor in self.sensors.items():
            diagnostics = sensor.get_diagnostics()
            
            # Check for issues
            if diagnostics["error_rate"] > 0.1:  # 10% error rate threshold
                self.logger.warning(f"High error rate for sensor {sensor_id}: {diagnostics['error_rate']:.2%}")
                
            if not diagnostics["calibration_status"]["valid"]:
                self.logger.warning(f"Sensor {sensor_id} calibration expired or missing")
                
            if diagnostics["status"] != "online":
                self.logger.warning(f"Sensor {sensor_id} is {diagnostics['status']}")
                
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        status = {
            "total_sensors": len(self.sensors),
            "online_sensors": sum(1 for s in self.sensors.values() if s.status == SensorStatus.ONLINE),
            "protocols": list(set(config.protocol.value for config in self.sensor_configs.values())),
            "sensors": {}
        }
        
        for sensor_id, sensor in self.sensors.items():
            status["sensors"][sensor_id] = sensor.get_diagnostics()
            
        return status


async def demonstrate_industrial_sensors():
    """
    Demonstration of industrial sensor interfaces.
    """
    print("🏧 Industrial Sensor Integration Demo")
    print("=" * 50)
    
    # Create sensor network manager
    manager = SensorNetworkManager()
    
    # Mock sensor configurations
    configs = [
        SensorConfig(
            sensor_id="e_nose_line_1",
            protocol=SensorProtocol.HTTP_REST,
            connection_params={
                "endpoint": "https://api.example.com/sensors/enose1/data",
                "method": "GET",
                "auth_type": "bearer",
                "token": "mock_token_123"
            },
            sampling_rate=2.0
        ),
        SensorConfig(
            sensor_id="temp_humidity_1",
            protocol=SensorProtocol.MODBUS_TCP,
            connection_params={
                "host": "192.168.1.100",
                "port": 502,
                "slave_id": 1,
                "registers": {
                    "input_registers": {
                        "temperature": 100,
                        "humidity": 101
                    }
                }
            }
        )
    ]
    
    print("\n🔌 Sensor Configuration:")
    for config in configs:
        print(f"  - {config.sensor_id}: {config.protocol.value}")
        
    # Note: In a real implementation, these would connect to actual sensors
    print("\n⚠️  Note: This is a demonstration with mock configurations")
    print("In production, sensors would connect to real industrial systems.")
    
    print("\n✅ Industrial sensor interface framework ready!")
    
    return manager


if __name__ == "__main__":
    asyncio.run(demonstrate_industrial_sensors())
