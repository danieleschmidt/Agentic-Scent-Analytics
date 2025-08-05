"""
Core functionality tests.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from agentic_scent import ScentAnalyticsFactory, QualityControlAgent
from agentic_scent.sensors.base import SensorReading, SensorType
from agentic_scent.agents.base import AnalysisResult, AgentCapability


class TestScentAnalyticsFactory:
    """Test the main factory class."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = ScentAnalyticsFactory(
            production_line="test_line",
            e_nose_config={
                "sensors": ["MOS", "PID"],
                "channels": 16
            }
        )
        
        assert factory.config.production_line == "test_line"
        assert factory.config.e_nose_config["channels"] == 16
        assert len(factory.sensors) > 0
        assert "e_nose_main" in factory.sensors
    
    def test_agent_registration(self, analytics_factory, quality_control_agent):
        """Test agent registration."""
        initial_count = len(analytics_factory.agents)
        analytics_factory.register_agent(quality_control_agent)
        
        assert len(analytics_factory.agents) == initial_count + 1
        assert quality_control_agent in analytics_factory.agents
    
    def test_process_parameters(self, analytics_factory):
        """Test process parameters retrieval."""
        params = analytics_factory.get_process_parameters()
        
        assert hasattr(params, 'temperature')
        assert hasattr(params, 'humidity')
        assert hasattr(params, 'pressure')
        assert hasattr(params, 'flow_rate')
        assert isinstance(params.temperature, float)
    
    def test_current_state(self, analytics_factory):
        """Test current state retrieval."""
        state = analytics_factory.get_current_state()
        
        assert "batch_id" in state
        assert "is_monitoring" in state
        assert "active_sensors" in state
        assert "active_agents" in state
        assert "process_parameters" in state
        assert "timestamp" in state
    
    @pytest.mark.asyncio
    async def test_sensor_stream(self, analytics_factory):
        """Test sensor data streaming."""
        # Start monitoring briefly
        analytics_factory.is_monitoring = True
        
        reading_count = 0
        async for reading in analytics_factory.sensor_stream():
            assert isinstance(reading, SensorReading)
            assert reading.sensor_id == "e_nose_main"
            assert len(reading.values) > 0
            
            reading_count += 1
            if reading_count >= 3:  # Test with 3 readings
                break
        
        analytics_factory.is_monitoring = False
        assert reading_count == 3


class TestQualityControlAgent:
    """Test quality control agent functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        agent = QualityControlAgent(agent_id="test_agent")
        
        assert agent.agent_id == "test_agent"
        assert AgentCapability.ANOMALY_DETECTION in agent.get_capabilities()
        assert not agent.is_active
        
        await agent.start()
        assert agent.is_active
        
        await agent.stop()
        assert not agent.is_active
    
    @pytest.mark.asyncio
    async def test_normal_reading_analysis(self, quality_control_agent, sample_sensor_reading):
        """Test analysis of normal sensor reading."""
        result = await quality_control_agent.analyze(sample_sensor_reading)
        
        assert isinstance(result, AnalysisResult)
        assert result.agent_id == quality_control_agent.agent_id
        assert 0.0 <= result.confidence <= 1.0
        assert hasattr(result, 'anomaly_detected')
        assert hasattr(result, 'sensor_id')
    
    @pytest.mark.asyncio
    async def test_contaminated_reading_analysis(self, quality_control_agent, contaminated_sensor_reading):
        """Test analysis of contaminated sensor reading."""
        result = await quality_control_agent.analyze(contaminated_sensor_reading)
        
        assert isinstance(result, AnalysisResult)
        assert result.anomaly_detected  # Should detect anomaly
        assert hasattr(result, 'contamination_analysis')
        assert hasattr(result, 'recommended_action')
    
    @pytest.mark.asyncio
    async def test_batch_evaluation(self, quality_control_agent, training_data):
        """Test batch quality evaluation."""
        batch_id = "test_batch_001"
        assessment = await quality_control_agent.evaluate_batch(batch_id, training_data)
        
        assert assessment.batch_id == batch_id
        assert 0.0 <= assessment.overall_quality <= 1.0
        assert 0.0 <= assessment.confidence <= 1.0
        assert isinstance(assessment.passed_checks, list)
        assert isinstance(assessment.failed_checks, list)
        assert assessment.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    @pytest.mark.asyncio
    async def test_analysis_history(self, quality_control_agent, sample_sensor_reading):
        """Test analysis history tracking."""
        # Perform several analyses
        for _ in range(5):
            await quality_control_agent.analyze(sample_sensor_reading)
        
        history = quality_control_agent.get_analysis_history()
        assert len(history) >= 5
        
        # Test limited history
        limited_history = quality_control_agent.get_analysis_history(limit=3)
        assert len(limited_history) == 3


class TestSensorReadings:
    """Test sensor reading functionality."""
    
    def test_sensor_reading_creation(self, sample_sensor_reading):
        """Test sensor reading object creation."""
        assert sample_sensor_reading.sensor_id == "test_sensor"
        assert sample_sensor_reading.sensor_type == SensorType.E_NOSE
        assert len(sample_sensor_reading.values) == 8
        assert isinstance(sample_sensor_reading.timestamp, datetime)
        assert 0.0 <= sample_sensor_reading.quality_score <= 1.0
    
    def test_sensor_reading_serialization(self, sample_sensor_reading):
        """Test sensor reading to dict conversion."""
        reading_dict = sample_sensor_reading.to_dict()
        
        assert "sensor_id" in reading_dict
        assert "sensor_type" in reading_dict
        assert "values" in reading_dict
        assert "timestamp" in reading_dict
        assert "metadata" in reading_dict
        assert "quality_score" in reading_dict
        
        assert reading_dict["sensor_type"] == "electronic_nose"
    
    @pytest.mark.asyncio
    async def test_mock_sensor_functionality(self, mock_sensor):
        """Test mock sensor operations."""
        # Initialize sensor
        await mock_sensor.start()
        assert mock_sensor.is_online()
        
        # Read from sensor
        reading = await mock_sensor.read()
        assert isinstance(reading, SensorReading)
        assert len(reading.values) == 16  # Configured channels
        assert all(isinstance(val, (int, float)) for val in reading.values)
        
        # Test calibration
        calibration_result = await mock_sensor.calibrate()
        assert calibration_result is True
        assert mock_sensor.calibration is not None
        
        await mock_sensor.stop()
        assert not mock_sensor.is_online()


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self, analytics_factory, quality_control_agent):
        """Test complete monitoring workflow."""
        # Register agent
        analytics_factory.register_agent(quality_control_agent)
        
        # Start monitoring
        await quality_control_agent.start()
        
        # Simulate monitoring for a few readings
        reading_count = 0
        analysis_count = 0
        
        analytics_factory.is_monitoring = True
        
        async for reading in analytics_factory.sensor_stream():
            # Analyze reading
            analysis = await quality_control_agent.analyze(reading)
            if analysis:
                analysis_count += 1
            
            reading_count += 1
            if reading_count >= 5:
                break
        
        analytics_factory.is_monitoring = False
        await quality_control_agent.stop()
        
        # Verify workflow
        assert reading_count == 5
        assert analysis_count >= 4  # Allow for some failed analyses
        
        # Check history
        history = quality_control_agent.get_analysis_history()
        assert len(history) >= 4
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_workflow(self, analytics_factory, quality_control_agent):
        """Test anomaly detection workflow."""
        analytics_factory.register_agent(quality_control_agent)
        await quality_control_agent.start()
        
        # Create contaminated reading
        contaminated_reading = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=[1000.0] * 16,  # Very high values
            timestamp=datetime.now(),
            quality_score=0.2
        )
        
        # Analyze contaminated reading
        analysis = await quality_control_agent.analyze(contaminated_reading)
        
        assert analysis is not None
        assert analysis.anomaly_detected
        assert hasattr(analysis, 'recommended_action')
        
        await quality_control_agent.stop()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, analytics_factory, quality_control_agent):
        """Test system performance under load."""
        analytics_factory.register_agent(quality_control_agent)
        await quality_control_agent.start()
        
        # Generate multiple readings rapidly
        readings = []
        for i in range(20):
            reading = SensorReading(
                sensor_id=f"sensor_{i%4}",  # 4 different sensors
                sensor_type=SensorType.E_NOSE,
                values=[100 + i] * 8,
                timestamp=datetime.now()
            )
            readings.append(reading)
        
        # Analyze all readings concurrently
        import time
        start_time = time.time()
        
        tasks = [quality_control_agent.analyze(reading) for reading in readings]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance
        successful_analyses = [r for r in results if isinstance(r, AnalysisResult)]
        assert len(successful_analyses) >= 15  # At least 75% success rate
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        await quality_control_agent.stop()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_sensor_reading(self, quality_control_agent):
        """Test handling of invalid sensor readings."""
        # Test with empty values
        invalid_reading = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=[],
            timestamp=datetime.now()
        )
        
        result = await quality_control_agent.analyze(invalid_reading)
        # Should handle gracefully, may return None or minimal result
        if result:
            assert hasattr(result, 'confidence')
    
    @pytest.mark.asyncio 
    async def test_missing_timestamp(self, quality_control_agent):
        """Test handling of readings with missing timestamp."""
        reading_no_timestamp = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=[100.0] * 8,
            timestamp=None
        )
        
        # Should handle gracefully
        result = await quality_control_agent.analyze(reading_no_timestamp)
        if result:
            assert hasattr(result, 'agent_id')
    
    @pytest.mark.asyncio
    async def test_agent_inactive_state(self, sample_sensor_reading):
        """Test analysis when agent is not active."""
        inactive_agent = QualityControlAgent(agent_id="inactive_test")
        # Don't start the agent
        
        result = await inactive_agent.analyze(sample_sensor_reading)
        assert result is None  # Should return None when inactive
    
    def test_factory_without_config(self):
        """Test factory with minimal configuration."""
        factory = ScentAnalyticsFactory(
            production_line="minimal_line",
            e_nose_config={}
        )
        
        assert factory.config.production_line == "minimal_line"
        assert isinstance(factory.config.e_nose_config, dict)