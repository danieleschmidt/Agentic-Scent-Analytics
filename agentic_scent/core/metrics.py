"""
Prometheus metrics integration for performance monitoring.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = None


class PrometheusMetrics:
    """Prometheus metrics collection and export."""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        
        if enable_prometheus:
            try:
                from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
                self.prometheus_available = True
                self.registry = CollectorRegistry()
                self._setup_prometheus_metrics()
            except ImportError:
                logger.warning("prometheus_client not installed, using mock metrics")
                self.prometheus_available = False
        else:
            self.prometheus_available = False
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metric collectors."""
        if not self.prometheus_available:
            return
        
        from prometheus_client import Counter, Histogram, Gauge
        
        # System metrics
        self.sensor_readings_total = Counter(
            'agentic_scent_sensor_readings_total',
            'Total sensor readings processed',
            ['sensor_id', 'sensor_type'],
            registry=self.registry
        )
        
        self.analysis_duration_seconds = Histogram(
            'agentic_scent_analysis_duration_seconds',
            'Time spent on analysis',
            ['agent_id', 'analysis_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.anomalies_detected_total = Counter(
            'agentic_scent_anomalies_detected_total',
            'Total anomalies detected',
            ['agent_id', 'severity'],
            registry=self.registry
        )
        
        self.llm_requests_total = Counter(
            'agentic_scent_llm_requests_total',
            'Total LLM API requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_tokens_used_total = Counter(
            'agentic_scent_llm_tokens_used_total',
            'Total LLM tokens consumed',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.cache_operations_total = Counter(
            'agentic_scent_cache_operations_total',
            'Cache operations',
            ['operation', 'level', 'status'],
            registry=self.registry
        )
        
        self.task_pool_workers = Gauge(
            'agentic_scent_task_pool_workers',
            'Active task pool workers',
            ['pool_id'],
            registry=self.registry
        )
        
        self.task_execution_duration_seconds = Histogram(
            'agentic_scent_task_execution_duration_seconds',
            'Task execution time',
            ['priority', 'worker_id'],
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'agentic_scent_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'agentic_scent_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_sensor_reading(self, sensor_id: str, sensor_type: str):
        """Record sensor reading."""
        if self.prometheus_available:
            self.sensor_readings_total.labels(
                sensor_id=sensor_id,
                sensor_type=sensor_type
            ).inc()
        
        # Internal tracking
        self._counters[f"sensor_readings_{sensor_id}"] += 1
    
    def record_analysis_duration(self, agent_id: str, analysis_type: str, duration: float):
        """Record analysis execution time."""
        if self.prometheus_available:
            self.analysis_duration_seconds.labels(
                agent_id=agent_id,
                analysis_type=analysis_type
            ).observe(duration)
        
        # Internal tracking
        key = f"analysis_duration_{agent_id}_{analysis_type}"
        self._histograms[key].append(duration)
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-500:]
    
    def record_anomaly_detection(self, agent_id: str, severity: str):
        """Record anomaly detection."""
        if self.prometheus_available:
            self.anomalies_detected_total.labels(
                agent_id=agent_id,
                severity=severity
            ).inc()
        
        self._counters[f"anomalies_{agent_id}_{severity}"] += 1
    
    def record_llm_request(self, provider: str, model: str, status: str, tokens_used: Optional[int] = None):
        """Record LLM API request."""
        if self.prometheus_available:
            self.llm_requests_total.labels(
                provider=provider,
                model=model,
                status=status
            ).inc()
            
            if tokens_used:
                self.llm_tokens_used_total.labels(
                    provider=provider,
                    model=model
                ).inc(tokens_used)
        
        self._counters[f"llm_requests_{provider}_{model}_{status}"] += 1
        if tokens_used:
            self._counters[f"llm_tokens_{provider}_{model}"] += tokens_used
    
    def record_cache_operation(self, operation: str, level: str, status: str):
        """Record cache operation."""
        if self.prometheus_available:
            self.cache_operations_total.labels(
                operation=operation,
                level=level,
                status=status
            ).inc()
        
        self._counters[f"cache_{operation}_{level}_{status}"] += 1
    
    def set_task_pool_workers(self, pool_id: str, count: int):
        """Set task pool worker count."""
        if self.prometheus_available:
            self.task_pool_workers.labels(pool_id=pool_id).set(count)
        
        self._metrics[f"workers_{pool_id}"].append(MetricPoint(time.time(), count))
    
    def record_task_execution(self, priority: str, worker_id: str, duration: float):
        """Record task execution time."""
        if self.prometheus_available:
            self.task_execution_duration_seconds.labels(
                priority=priority,
                worker_id=worker_id
            ).observe(duration)
        
        key = f"task_execution_{priority}"
        self._histograms[key].append(duration)
    
    def set_system_metrics(self, cpu_percent: float, memory_percent: float):
        """Set system resource metrics."""
        if self.prometheus_available:
            self.system_cpu_usage.set(cpu_percent)
            self.system_memory_usage.set(memory_percent)
        
        timestamp = time.time()
        self._metrics["system_cpu"].append(MetricPoint(timestamp, cpu_percent))
        self._metrics["system_memory"].append(MetricPoint(timestamp, memory_percent))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "counters": dict(self._counters),
            "histograms": {},
            "recent_metrics": {}
        }
        
        # Histogram summaries
        for key, values in self._histograms.items():
            if values:
                summary["histograms"][key] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": sorted(values)[len(values)//2] if values else 0,
                    "p95": sorted(values)[int(len(values)*0.95)] if len(values) > 1 else 0
                }
        
        # Recent metrics
        for key, points in self._metrics.items():
            if points:
                recent_points = list(points)[-10:]  # Last 10 points
                summary["recent_metrics"][key] = [
                    {"timestamp": p.timestamp, "value": p.value}
                    for p in recent_points
                ]
        
        return summary
    
    def export_prometheus_metrics(self) -> Optional[str]:
        """Export metrics in Prometheus format."""
        if not self.prometheus_available:
            return None
        
        try:
            from prometheus_client import generate_latest
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}")
            return None


class PerformanceProfiler:
    """Performance profiling and analysis."""
    
    def __init__(self, metrics: PrometheusMetrics):
        self.metrics = metrics
        self._profiles: Dict[str, List[float]] = defaultdict(list)
        self._active_profiles: Dict[str, float] = {}
    
    def start_profile(self, profile_name: str) -> str:
        """Start profiling a code section."""
        self._active_profiles[profile_name] = time.time()
        return profile_name
    
    def end_profile(self, profile_name: str) -> float:
        """End profiling and record duration."""
        if profile_name not in self._active_profiles:
            logger.warning(f"Profile {profile_name} not found in active profiles")
            return 0.0
        
        duration = time.time() - self._active_profiles.pop(profile_name)
        self._profiles[profile_name].append(duration)
        
        # Keep only recent profiles
        if len(self._profiles[profile_name]) > 1000:
            self._profiles[profile_name] = self._profiles[profile_name][-500:]
        
        return duration
    
    def profile_context(self, profile_name: str):
        """Context manager for profiling."""
        return ProfileContext(self, profile_name)
    
    def get_profile_summary(self, profile_name: str) -> Dict[str, Any]:
        """Get profile summary statistics."""
        durations = self._profiles.get(profile_name, [])
        
        if not durations:
            return {"profile_name": profile_name, "count": 0}
        
        sorted_durations = sorted(durations)
        
        return {
            "profile_name": profile_name,
            "count": len(durations),
            "total_time": sum(durations),
            "avg_time": sum(durations) / len(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "p50": sorted_durations[len(durations)//2],
            "p90": sorted_durations[int(len(durations)*0.9)],
            "p95": sorted_durations[int(len(durations)*0.95)],
            "p99": sorted_durations[int(len(durations)*0.99)] if len(durations) > 10 else max(durations)
        }
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all profile summaries."""
        return {
            name: self.get_profile_summary(name)
            for name in self._profiles.keys()
        }


class ProfileContext:
    """Context manager for performance profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, profile_name: str):
        self.profiler = profiler
        self.profile_name = profile_name
    
    def __enter__(self):
        self.profiler.start_profile(self.profile_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.profiler.end_profile(self.profile_name)
        if exc_type is None:
            logger.debug(f"Profile {self.profile_name}: {duration:.4f}s")


def create_metrics_system(enable_prometheus: bool = True) -> tuple[PrometheusMetrics, PerformanceProfiler]:
    """Create integrated metrics and profiling system."""
    
    metrics = PrometheusMetrics(enable_prometheus=enable_prometheus)
    profiler = PerformanceProfiler(metrics)
    
    logger.info("Metrics system initialized")
    return metrics, profiler


# Decorator for automatic function profiling
def profile(metrics_system: Optional[tuple] = None, profile_name: Optional[str] = None):
    """Decorator to automatically profile function execution."""
    
    def decorator(func):
        nonlocal profile_name
        if profile_name is None:
            profile_name = f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                if metrics_system:
                    _, profiler = metrics_system
                    with profiler.profile_context(profile_name):
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                if metrics_system:
                    _, profiler = metrics_system
                    with profiler.profile_context(profile_name):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator