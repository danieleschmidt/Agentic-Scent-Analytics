"""
Core factory management system for industrial scent analytics.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..sensors.base import SensorInterface, SensorReading
from ..agents.base import BaseAgent


@dataclass
class FactoryConfig:
    """Configuration for factory monitoring system."""
    site_id: str
    production_line: str
    e_nose_config: Dict[str, Any]
    quality_targets: Dict[str, float] = field(default_factory=dict)
    sampling_rate: float = 10.0  # Hz
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProcessParameters:
    """Current process parameters."""
    temperature: float
    humidity: float
    pressure: float
    flow_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


class ScentAnalyticsFactory:
    """
    Main factory analytics system that coordinates sensors, agents, and monitoring.
    """
    
    def __init__(self, production_line: str, e_nose_config: Dict[str, Any], 
                 site_id: str = "default"):
        self.config = FactoryConfig(
            site_id=site_id,
            production_line=production_line,
            e_nose_config=e_nose_config
        )
        self.sensors: Dict[str, SensorInterface] = {}
        self.agents: List[BaseAgent] = []
        self.is_monitoring = False
        self.current_batch_id: Optional[str] = None
        
        # Setup logging
        self.logger = logging.getLogger(f"factory.{site_id}.{production_line}")
        
        # Initialize sensor interfaces based on config
        self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize sensor interfaces based on configuration."""
        # This would be implemented with actual sensor drivers
        # For now, create mock sensors
        from ..sensors.mock import MockENoseSensor
        
        self.sensors["e_nose_main"] = MockENoseSensor(
            sensor_id="e_nose_main",
            channels=self.config.e_nose_config.get("channels", 32),
            sampling_rate=self.config.sampling_rate
        )
    
    def register_agent(self, agent: BaseAgent):
        """Register an AI agent with the factory system."""
        self.agents.append(agent)
        self.logger.info(f"Registered agent: {agent.agent_id}")
    
    async def start_monitoring(self, batch_id: Optional[str] = None):
        """Start continuous monitoring of the production line."""
        self.current_batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_monitoring = True
        
        self.logger.info(f"Starting monitoring for batch {self.current_batch_id}")
        
        # Start sensor data streams
        tasks = []
        for agent in self.agents:
            task = asyncio.create_task(self._monitor_with_agent(agent))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """Stop monitoring and cleanup resources."""
        self.is_monitoring = False
        self.logger.info(f"Stopped monitoring for batch {self.current_batch_id}")
    
    async def sensor_stream(self, sensor_id: str = "e_nose_main") -> AsyncGenerator[SensorReading, None]:
        """Stream sensor readings."""
        if sensor_id not in self.sensors:
            raise ValueError(f"Sensor {sensor_id} not found")
        
        sensor = self.sensors[sensor_id]
        
        while self.is_monitoring:
            try:
                reading = await sensor.read()
                yield reading
                await asyncio.sleep(1.0 / self.config.sampling_rate)
            except Exception as e:
                self.logger.error(f"Error reading from sensor {sensor_id}: {e}")
                await asyncio.sleep(1.0)
    
    async def _monitor_with_agent(self, agent: BaseAgent):
        """Monitor production line with a specific agent."""
        async for reading in self.sensor_stream():
            try:
                # Agent analyzes the reading
                analysis = await agent.analyze(reading)
                
                if analysis and hasattr(analysis, 'anomaly_detected') and analysis.anomaly_detected:
                    await self._handle_anomaly(agent, analysis)
                    
            except Exception as e:
                self.logger.error(f"Error in agent {agent.agent_id}: {e}")
    
    async def _handle_anomaly(self, agent: BaseAgent, analysis):
        """Handle detected anomalies."""
        self.logger.warning(
            f"Anomaly detected by {agent.agent_id}: "
            f"confidence={getattr(analysis, 'confidence', 'unknown')}"
        )
        
        # This would trigger corrective actions
        # For now, just log the event
        
    def get_process_parameters(self) -> ProcessParameters:
        """Get current process parameters."""
        # Mock implementation - would read from actual process control system
        return ProcessParameters(
            temperature=25.0,
            humidity=45.0,
            pressure=1013.25,
            flow_rate=100.0
        )
    
    def adjust_parameters(self, adjustments: Dict[str, float], 
                         reason: str, authorized_by: str):
        """Adjust process parameters."""
        self.logger.info(
            f"Parameter adjustment requested by {authorized_by}: "
            f"{adjustments} - Reason: {reason}"
        )
        # Implementation would interface with process control system
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current factory state."""
        return {
            "batch_id": self.current_batch_id,
            "is_monitoring": self.is_monitoring,
            "active_sensors": list(self.sensors.keys()),
            "active_agents": len(self.agents),
            "process_parameters": self.get_process_parameters().__dict__,
            "timestamp": datetime.now().isoformat()
        }
    
    def execute_corrective_action(self, action_plan: Dict[str, Any]):
        """Execute corrective actions based on agent recommendations."""
        self.logger.info(f"Executing corrective action: {action_plan}")
        # Implementation would execute actual corrective actions