"""Real-time performance monitoring system."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import statistics
import numpy as np
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class PerformanceAlert:
    metric_name: str
    threshold: float
    current_value: float
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Real-time performance monitoring with quantum-inspired analytics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: List[PerformanceAlert] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = self.config.get("monitoring_interval", 1.0)  # seconds
        self.quantum_coherence_score = 1.0
        
        # Performance thresholds
        self.thresholds = {
            "task_completion_rate": {"min": 0.8, "max": 1.0},
            "average_execution_time": {"min": 0.0, "max": 300.0},  # 5 minutes
            "resource_utilization": {"min": 0.0, "max": 0.9},
            "quantum_efficiency": {"min": 0.5, "max": 1.0},
            "agent_availability": {"min": 0.7, "max": 1.0}
        }
        
        logger.info("PerformanceMonitor initialized")
    
    async def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        logger.info("Starting performance monitoring")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        logger.info("Stopping performance monitoring")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            tags=tags or {},
            unit=unit
        )
        
        self.metrics_history[name].append(metric)
        
        # Trigger callbacks
        for callback in self.metric_callbacks[name]:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Error in metric callback for {name}: {e}")
        
        # Check alert rules
        self._check_alert_rules(metric)
    
    def add_metric_callback(self, metric_name: str, callback: Callable[[PerformanceMetric], None]):
        """Add callback for metric updates."""
        self.metric_callbacks[metric_name].append(callback)
    
    def set_alert_rule(
        self, 
        metric_name: str, 
        threshold: float, 
        condition: str, 
        severity: str = "warning"
    ):
        """Set alert rule for a metric."""
        self.alert_rules[metric_name] = {
            "threshold": threshold,
            "condition": condition,  # "greater_than", "less_than", "equals"
            "severity": severity
        }
        logger.info(f"Set alert rule for {metric_name}: {condition} {threshold}")
    
    def get_metric_history(self, metric_name: str, limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Get historical data for a metric."""
        history = list(self.metrics_history[metric_name])
        if limit:
            history = history[-limit:]
        return history
    
    def get_metric_statistics(self, metric_name: str, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get statistical summary of a metric."""
        history = self.get_metric_history(metric_name)
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            history = [m for m in history if m.timestamp >= cutoff_time]
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "percentile_95": np.percentile(values, 95),
            "percentile_99": np.percentile(values, 99),
            "latest": values[-1] if values else None,
            "trend": self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_quantum_coherence_score(self) -> float:
        """Calculate quantum coherence score based on system performance."""
        # Collect recent performance metrics
        metrics_to_check = [
            "task_completion_rate",
            "quantum_efficiency",
            "agent_availability",
            "load_balance_efficiency"
        ]
        
        coherence_factors = []
        
        for metric_name in metrics_to_check:
            stats = self.get_metric_statistics(metric_name, timedelta(minutes=5))
            if stats and "mean" in stats:
                # Normalize to 0-1 range and add to coherence calculation
                mean_value = stats["mean"]
                threshold = self.thresholds.get(metric_name, {"max": 1.0})
                max_value = threshold["max"]
                
                normalized_value = min(1.0, mean_value / max_value) if max_value > 0 else 0
                coherence_factors.append(normalized_value)
        
        if coherence_factors:
            # Calculate geometric mean for quantum coherence
            self.quantum_coherence_score = np.prod(coherence_factors) ** (1.0 / len(coherence_factors))
        
        return self.quantum_coherence_score
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "quantum_coherence_score": self.get_quantum_coherence_score(),
            "active_alerts": len(self.active_alerts),
            "metrics_summary": {},
            "system_health": "healthy"
        }
        
        # Collect summary for key metrics
        key_metrics = [
            "task_completion_rate",
            "average_execution_time", 
            "resource_utilization",
            "quantum_efficiency",
            "agent_availability"
        ]
        
        for metric_name in key_metrics:
            stats = self.get_metric_statistics(metric_name, timedelta(minutes=15))
            if stats:
                dashboard["metrics_summary"][metric_name] = {
                    "current": stats["latest"],
                    "average": stats["mean"],
                    "trend": stats["trend"],
                    "status": self._get_metric_status(metric_name, stats["latest"])
                }
        
        # Determine overall system health
        if len(self.active_alerts) > 0:
            critical_alerts = [a for a in self.active_alerts if a.severity == "critical"]
            if critical_alerts:
                dashboard["system_health"] = "critical"
            else:
                dashboard["system_health"] = "warning"
        
        if self.quantum_coherence_score < 0.5:
            dashboard["system_health"] = "degraded"
        
        return dashboard
    
    def _get_metric_status(self, metric_name: str, value: Optional[float]) -> str:
        """Get status of a metric based on thresholds."""
        if value is None:
            return "unknown"
        
        thresholds = self.thresholds.get(metric_name)
        if not thresholds:
            return "ok"
        
        min_threshold = thresholds.get("min", float("-inf"))
        max_threshold = thresholds.get("max", float("inf"))
        
        if value < min_threshold or value > max_threshold:
            return "critical"
        
        # Warning thresholds (80% of range)
        range_size = max_threshold - min_threshold
        warning_buffer = range_size * 0.1
        
        if value < min_threshold + warning_buffer or value > max_threshold - warning_buffer:
            return "warning"
        
        return "ok"
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Recovery delay
    
    async def _collect_system_metrics(self):
        """Collect system-wide performance metrics."""
        # System resource metrics
        try:
            import psutil
            
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.record_metric("system_cpu_usage", cpu_percent, unit="percent")
            
            # Memory utilization
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_usage", memory.percent, unit="percent")
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.record_metric("disk_read_bytes", disk_io.read_bytes, unit="bytes")
                self.record_metric("disk_write_bytes", disk_io.write_bytes, unit="bytes")
            
        except ImportError:
            # psutil not available, use placeholder metrics
            self.record_metric("system_cpu_usage", 25.0, unit="percent")
            self.record_metric("system_memory_usage", 45.0, unit="percent")
        
        # Application-specific metrics
        current_time = time.time()
        self.record_metric("monitoring_heartbeat", current_time, unit="timestamp")
        
        # Quantum coherence metric
        coherence = self.get_quantum_coherence_score()
        self.record_metric("quantum_coherence", coherence, unit="score")
    
    def _check_alert_rules(self, metric: PerformanceMetric):
        """Check if metric triggers any alert rules."""
        rule = self.alert_rules.get(metric.name)
        if not rule:
            return
        
        threshold = rule["threshold"]
        condition = rule["condition"]
        severity = rule["severity"]
        
        should_alert = False
        
        if condition == "greater_than" and metric.value > threshold:
            should_alert = True
        elif condition == "less_than" and metric.value < threshold:
            should_alert = True
        elif condition == "equals" and abs(metric.value - threshold) < 0.001:
            should_alert = True
        
        if should_alert:
            alert = PerformanceAlert(
                metric_name=metric.name,
                threshold=threshold,
                current_value=metric.value,
                severity=severity,
                message=f"{metric.name} {condition} {threshold}: current value {metric.value}"
            )
            
            self.active_alerts.append(alert)
            
            # Limit active alerts
            if len(self.active_alerts) > 100:
                self.active_alerts.pop(0)
            
            logger.warning(f"Performance alert: {alert.message}")
    
    def clear_alerts(self, metric_name: Optional[str] = None):
        """Clear active alerts."""
        if metric_name:
            self.active_alerts = [a for a in self.active_alerts if a.metric_name != metric_name]
        else:
            self.active_alerts.clear()
        
        logger.info(f"Cleared alerts for {metric_name or 'all metrics'}")
    
    def get_alerts(self, severity: Optional[str] = None) -> List[PerformanceAlert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = self.active_alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts
    
    def export_metrics(self, format: str = "dict", time_window: Optional[timedelta] = None) -> Any:
        """Export metrics data in specified format."""
        cutoff_time = datetime.now() - time_window if time_window else None
        
        exported_data = {}
        
        for metric_name, history in self.metrics_history.items():
            filtered_history = list(history)
            
            if cutoff_time:
                filtered_history = [m for m in filtered_history if m.timestamp >= cutoff_time]
            
            if format == "dict":
                exported_data[metric_name] = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "value": m.value,
                        "tags": m.tags,
                        "unit": m.unit
                    }
                    for m in filtered_history
                ]
            elif format == "values":
                exported_data[metric_name] = [m.value for m in filtered_history]
        
        return exported_data