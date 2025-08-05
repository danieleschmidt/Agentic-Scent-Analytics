"""
Test configuration and fixtures.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from agentic_scent import ScentAnalyticsFactory, QualityControlAgent
from agentic_scent.sensors.base import SensorReading, SensorType
from agentic_scent.sensors.mock import MockENoseSensor
from agentic_scent.core.config import AgenticScentConfig


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return AgenticScentConfig(
        site_id="test_site",
        environment="testing",
        production_lines=["test_line_1", "test_line_2"]
    )


@pytest.fixture
def mock_sensor():
    """Mock sensor for testing."""
    sensor = MockENoseSensor(
        sensor_id="test_sensor",
        channels=16,
        sampling_rate=5.0
    )
    return sensor


@pytest.fixture
def sample_sensor_reading():
    """Sample sensor reading for testing."""
    return SensorReading(
        sensor_id="test_sensor",
        sensor_type=SensorType.E_NOSE,
        values=[100.0, 150.0, 200.0, 125.0, 175.0, 110.0, 190.0, 140.0],
        timestamp=datetime.now(),
        metadata={"test": True},
        quality_score=0.9
    )


@pytest.fixture
def contaminated_sensor_reading():
    """Contaminated sensor reading for testing."""
    return SensorReading(
        sensor_id="test_sensor",
        sensor_type=SensorType.E_NOSE,
        values=[500.0, 750.0, 1000.0, 625.0, 875.0, 550.0, 950.0, 700.0],  # 5x higher values
        timestamp=datetime.now(),
        metadata={"contaminated": True},
        quality_score=0.3
    )


@pytest.fixture
async def quality_control_agent():
    """Quality control agent for testing."""
    agent = QualityControlAgent(agent_id="test_qc_agent")
    await agent.start()
    yield agent
    await agent.stop()


@pytest.fixture
async def analytics_factory():
    """Analytics factory for testing."""
    factory = ScentAnalyticsFactory(
        production_line="test_line",
        e_nose_config={
            "sensors": ["MOS", "PID"],
            "channels": 16,
            "sampling_rate": 5.0
        },
        site_id="test_factory"
    )
    yield factory


@pytest.fixture
def training_data():
    """Training data for fingerprinting tests."""
    training_readings = []
    base_values = [100.0, 150.0, 200.0, 125.0, 175.0, 110.0, 190.0, 140.0]
    
    for i in range(50):
        # Add small variations to base values
        import random
        varied_values = [val * random.uniform(0.9, 1.1) for val in base_values]
        
        reading = SensorReading(
            sensor_id=f"training_sensor_{i}",
            sensor_type=SensorType.E_NOSE,
            values=varied_values,
            timestamp=datetime.now() - timedelta(hours=i),
            metadata={"batch_id": f"good_batch_{i}"},
            quality_score=0.9
        )
        training_readings.append(reading)
    
    return training_readings


@pytest.fixture
def historical_data():
    """Historical data for predictive analytics tests."""
    import random
    
    historical_records = []
    for i in range(100):
        # Generate mock historical data
        record = {
            "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
            "sensor_values": [random.uniform(50, 200) for _ in range(8)],
            "temperature": random.uniform(20, 30),
            "humidity": random.uniform(40, 60),
            "pressure": random.uniform(1000, 1020),
            "potency": random.uniform(0.85, 1.0),
            "dissolution": random.uniform(0.8, 1.0),
            "stability": random.uniform(0.9, 1.0),
            "uniformity": random.uniform(0.85, 1.0),
            "contamination_risk": random.uniform(0.0, 0.2)
        }
        historical_records.append(record)
    
    return historical_records


@pytest.mark.asyncio
async def async_test_helper(coro):
    """Helper for running async tests."""
    return await coro