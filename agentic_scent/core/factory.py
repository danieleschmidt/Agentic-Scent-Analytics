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
from .caching import create_optimized_cache, MultiLevelCache, cached
from .task_pool import create_optimized_task_pool, AutoScalingTaskPool, TaskPriority
from .metrics import create_metrics_system, PrometheusMetrics, PerformanceProfiler
from .logging_config import get_contextual_logger, LoggingMixin


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


class ScentAnalyticsFactory(LoggingMixin):
    """
    High-performance factory analytics system with auto-scaling and caching.
    """
    
    def __init__(self, production_line: str, e_nose_config: Dict[str, Any], 
                 site_id: str = "default", enable_scaling: bool = True):
        super().__init__()
        
        self.config = FactoryConfig(
            site_id=site_id,
            production_line=production_line,
            e_nose_config=e_nose_config
        )
        self.sensors: Dict[str, SensorInterface] = {}
        self.agents: List[BaseAgent] = []
        self.is_monitoring = False
        self.current_batch_id: Optional[str] = None
        
        # Performance optimization components
        self.enable_scaling = enable_scaling
        self.cache: Optional[MultiLevelCache] = None
        self.task_pool: Optional[AutoScalingTaskPool] = None
        self.metrics: Optional[PrometheusMetrics] = None
        self.profiler: Optional[PerformanceProfiler] = None
        
        # Set logging context
        self.logger.set_context(
            site_id=site_id,
            production_line=production_line
        )
        
        # Initialize performance systems
        self._initialize_performance_systems()
        
        # Initialize sensor interfaces based on config
        self._initialize_sensors()
    
    def _initialize_performance_systems(self):
        """Initialize performance optimization systems."""
        if not self.enable_scaling:
            self.logger.info("Performance scaling disabled")
            return
        
        try:
            # Initialize metrics system
            self.metrics, self.profiler = create_metrics_system(enable_prometheus=True)
            self.logger.info("Metrics system initialized")
        
        except Exception as e:
            self.logger.warning(f"Failed to initialize metrics: {e}")
            self.metrics, self.profiler = None, None
    
    async def _initialize_async_systems(self):
        """Initialize async performance systems."""
        if not self.enable_scaling:
            return
        
        try:
            # Initialize cache system
            self.cache = await create_optimized_cache(
                memory_size=1000,
                memory_ttl=300,
                redis_url=None,  # Will use mock Redis
                redis_ttl=900
            )
            self.logger.info("Cache system initialized")
            
            # Initialize task pool
            self.task_pool = await create_optimized_task_pool(
                min_workers=2,
                max_workers=8
            )
            self.logger.info("Task pool initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize async systems: {e}")
    
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
        """Start high-performance continuous monitoring."""
        self.current_batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_monitoring = True
        
        # Initialize async performance systems
        await self._initialize_async_systems()
        
        self.logger.info(f"Starting optimized monitoring for batch {self.current_batch_id}")
        
        # Update metrics
        if self.metrics:
            self.metrics.record_sensor_reading("factory_start", "system_event")
        
        # Start sensor data streams with task pool
        if self.task_pool and len(self.agents) > 0:
            # Use task pool for concurrent agent processing
            tasks = []
            for i, agent in enumerate(self.agents):
                task_id = f"monitor_agent_{agent.agent_id}_{i}"
                await self.task_pool.submit_task(
                    task_id,
                    self._monitor_with_agent,
                    agent,
                    priority=TaskPriority.HIGH
                )
                tasks.append(task_id)
            
            self.logger.info(f"Started {len(tasks)} monitoring tasks in task pool")
        
        else:
            # Fallback to direct execution
            tasks = []
            for agent in self.agents:
                task = asyncio.create_task(self._monitor_with_agent(agent))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """Stop monitoring and cleanup resources."""
        self.is_monitoring = False
        
        # Stop task pool
        if self.task_pool:
            await self.task_pool.stop()
            self.logger.info("Task pool stopped")
        
        # Update metrics
        if self.metrics:
            self.metrics.record_sensor_reading("factory_stop", "system_event")
        
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
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "factory_status": {
                "is_monitoring": self.is_monitoring,
                "current_batch": self.current_batch_id,
                "active_agents": len(self.agents),
                "registered_sensors": len(self.sensors)
            }
        }
        
        # Task pool metrics
        if self.task_pool:
            metrics["task_pool"] = self.task_pool.get_status()
        
        # Cache metrics
        if self.cache:
            metrics["cache"] = self.cache.get_stats()
        
        # Prometheus metrics
        if self.metrics:
            metrics["system_metrics"] = self.metrics.get_metrics_summary()
        
        # Performance profiles
        if self.profiler:
            metrics["performance_profiles"] = self.profiler.get_all_profiles()
        
        return metrics
    
    async def export_metrics(self, format: str = "prometheus") -> Optional[str]:
        """Export metrics in specified format."""
        if format == "prometheus" and self.metrics:
            return self.metrics.export_prometheus_metrics()
        elif format == "json":
            import json
            return json.dumps(self.get_performance_metrics(), indent=2, default=str)
        else:
            return None
    
    async def cached_analysis(self, sensor_data: Dict[str, Any], cache_key: str) -> Any:
        """Perform cached analysis to avoid redundant computations."""
        # This would perform expensive analysis that can be cached
        # For now, return mock analysis result
        await asyncio.sleep(0.1)  # Simulate computation
        return {
            "analysis_result": "cached_analysis_complete",
            "timestamp": datetime.now().isoformat(),
            "cache_key": cache_key
        }
    
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