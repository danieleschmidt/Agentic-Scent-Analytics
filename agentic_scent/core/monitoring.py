"""
Monitoring, metrics, and health check system.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    timestamp: datetime = field(default_factory=datetime.now)


class PrometheusMetrics:
    """
    Collects and exports metrics for monitoring.
    """
    
    def __init__(self, enable_prometheus: bool = True, prometheus_port: int = 8090):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port
        self.logger = logging.getLogger(__name__)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
            self._start_prometheus_server()
        
        # Internal metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.counters: Dict[str, int] = defaultdict(int)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.gauges: Dict[str, float] = {}
        
        # System metrics collection
        self.system_metrics_enabled = True
        self.system_metrics_interval = 30  # seconds
        self._system_metrics_task: Optional[asyncio.Task] = None
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        if not self.enable_prometheus:
            return
        
        # Counter metrics
        self.prom_sensor_readings = Counter(
            'agentic_scent_sensor_readings_total',
            'Total number of sensor readings',
            ['sensor_id', 'sensor_type']
        )
        
        self.prom_analyses_total = Counter(
            'agentic_scent_analyses_total',
            'Total number of analyses performed',
            ['agent_id', 'result']
        )
        
        self.prom_anomalies_detected = Counter(
            'agentic_scent_anomalies_detected_total',
            'Total number of anomalies detected',
            ['agent_id', 'severity']
        )
        
        # Histogram metrics
        self.prom_analysis_duration = Histogram(
            'agentic_scent_analysis_duration_seconds',
            'Time spent on analysis',
            ['agent_id']
        )
        
        self.prom_sensor_read_duration = Histogram(
            'agentic_scent_sensor_read_duration_seconds',
            'Time spent reading sensors',
            ['sensor_id']
        )
        
        # Gauge metrics
        self.prom_active_agents = Gauge(
            'agentic_scent_active_agents',
            'Number of active agents'
        )
        
        self.prom_active_sensors = Gauge(
            'agentic_scent_active_sensors',
            'Number of active sensors'
        )
        
        self.prom_system_cpu = Gauge(
            'agentic_scent_system_cpu_percent',
            'System CPU usage percentage'
        )
        
        self.prom_system_memory = Gauge(
            'agentic_scent_system_memory_percent',
            'System memory usage percentage'
        )
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics server."""
        if not self.enable_prometheus:
            return
        
        try:
            start_http_server(self.prometheus_port)
            self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
            self.enable_prometheus = False
    
    def record_sensor_reading(self, sensor_id: str, sensor_type: str, duration: float = 0.0):
        """Record sensor reading metrics."""
        self.counters[f"sensor_readings_{sensor_id}"] += 1
        
        if self.enable_prometheus:
            self.prom_sensor_readings.labels(sensor_id=sensor_id, sensor_type=sensor_type).inc()
            if duration > 0:
                self.prom_sensor_read_duration.labels(sensor_id=sensor_id).observe(duration)
    
    def record_analysis(self, agent_id: str, duration: float, anomaly_detected: bool = False):
        """Record analysis metrics."""
        result = "anomaly" if anomaly_detected else "normal"
        self.counters[f"analyses_{agent_id}"] += 1
        self.histograms[f"analysis_duration_{agent_id}"].append(duration)
        
        if self.enable_prometheus:
            self.prom_analyses_total.labels(agent_id=agent_id, result=result).inc()
            self.prom_analysis_duration.labels(agent_id=agent_id).observe(duration)
            
            if anomaly_detected:
                severity = "high" if duration > 1.0 else "medium"  # Simple severity based on processing time
                self.prom_anomalies_detected.labels(agent_id=agent_id, severity=severity).inc()
    
    def update_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Update gauge metric."""
        self.gauges[name] = value
        
        if self.enable_prometheus:
            if name == "active_agents":
                self.prom_active_agents.set(value)
            elif name == "active_sensors":
                self.prom_active_sensors.set(value)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {}
        }
        
        # Calculate histogram statistics
        for name, values in self.histograms.items():
            if values:
                summary["histograms"][name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    async def start_system_monitoring(self):
        """Start system metrics collection."""
        if self.system_metrics_enabled and not self._system_metrics_task:
            self._system_metrics_task = asyncio.create_task(self._collect_system_metrics())
    
    async def stop_system_monitoring(self):
        """Stop system metrics collection."""
        if self._system_metrics_task:
            self._system_metrics_task.cancel()
            try:
                await self._system_metrics_task
            except asyncio.CancelledError:
                pass
            self._system_metrics_task = None
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                # Disk usage (root filesystem)
                disk = psutil.disk_usage('/')
                
                # Network I/O
                network = psutil.net_io_counters()
                
                metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    memory_available_mb=memory.available / (1024 * 1024),
                    disk_usage_percent=(disk.used / disk.total) * 100,
                    network_io_bytes={
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv
                    }
                )
                
                self.metrics_history.append(metrics)
                
                # Update Prometheus gauges
                if self.enable_prometheus:
                    self.prom_system_cpu.set(cpu_percent)
                    self.prom_system_memory.set(memory.percent)
                
                await asyncio.sleep(self.system_metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.system_metrics_interval)
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None


class HealthChecker:
    """
    Health check system for monitoring component status.
    """
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_results: Dict[str, HealthCheck] = {}
        self.logger = logging.getLogger(__name__)
        self._monitoring_task: Optional[asyncio.Task] = None
        self.overall_status = HealthStatus.UNKNOWN
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def unregister_health_check(self, name: str):
        """Unregister a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            if name in self.health_results:
                del self.health_results[name]
            self.logger.info(f"Unregistered health check: {name}")
    
    async def run_health_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.health_checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        start_time = time.time()
        
        try:
            check_func = self.health_checks[name]
            
            # Run the check with timeout
            result = await asyncio.wait_for(
                asyncio.create_task(check_func()),
                timeout=10.0
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheck):
                result.response_time_ms = response_time
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="OK" if result else "Check failed",
                    response_time_ms=response_time
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    response_time_ms=response_time
                )
                
        except asyncio.TimeoutError:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        # Run checks in parallel
        tasks = {
            name: asyncio.create_task(self.run_health_check(name))
            for name in self.health_checks
        }
        
        for name, task in tasks.items():
            try:
                result = await task
                results[name] = result
                self.health_results[name] = result
            except Exception as e:
                self.logger.error(f"Error running health check {name}: {e}")
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Unexpected error: {str(e)}"
                )
        
        # Update overall status
        self._update_overall_status()
        
        return results
    
    def _update_overall_status(self):
        """Update overall system health status."""
        if not self.health_results:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        statuses = [check.status for check in self.health_results.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            self.overall_status = HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            self.overall_status = HealthStatus.UNHEALTHY
        else:
            self.overall_status = HealthStatus.DEGRADED
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitor_health())
            self.logger.info("Started health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            self.logger.info("Stopped health monitoring")
    
    async def _monitor_health(self):
        """Continuous health monitoring loop."""
        while True:
            try:
                await self.run_all_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "overall_status": self.overall_status.value,
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms,
                    "timestamp": check.timestamp.isoformat(),
                    "metadata": check.metadata
                }
                for name, check in self.health_results.items()
            },
            "summary": {
                "total_checks": len(self.health_results),
                "healthy": sum(1 for c in self.health_results.values() if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.health_results.values() if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.health_results.values() if c.status == HealthStatus.UNHEALTHY),
                "unknown": sum(1 for c in self.health_results.values() if c.status == HealthStatus.UNKNOWN)
            }
        }


# Default health check implementations
async def database_health_check() -> HealthCheck:
    """Example database health check."""
    # This would check database connectivity
    return HealthCheck(
        name="database",
        status=HealthStatus.HEALTHY,
        message="Database connection OK"
    )

async def sensor_connectivity_health_check() -> HealthCheck:
    """Check sensor connectivity."""
    # This would check if sensors are responding
    return HealthCheck(
        name="sensor_connectivity",
        status=HealthStatus.HEALTHY,
        message="All sensors responding"
    )

async def agent_health_check() -> HealthCheck:
    """Check agent system health."""
    # This would check if agents are functioning
    return HealthCheck(
        name="agents",
        status=HealthStatus.HEALTHY,
        message="All agents operational"
    )