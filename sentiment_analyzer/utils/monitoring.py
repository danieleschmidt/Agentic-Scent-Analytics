"""
Monitoring and health check utilities
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricData:
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags or {}
        }


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self._start_time = time.time()
        self.metrics_history = []
        self.max_history_size = 1000
    
    def get_system_metrics(self) -> Dict[str, MetricData]:
        """Get current system metrics"""
        now = datetime.now(timezone.utc)
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network metrics (basic)
            network = psutil.net_io_counters()
            network_sent_gb = network.bytes_sent / (1024**3)
            network_recv_gb = network.bytes_recv / (1024**3)
            
            # Process metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024**2)
            process_cpu_percent = process.cpu_percent()
            
            # Uptime
            uptime_seconds = time.time() - self._start_time
            
            metrics = {
                "cpu_percent": MetricData("cpu_percent", cpu_percent, "%", now),
                "cpu_count": MetricData("cpu_count", cpu_count, "cores", now),
                "memory_percent": MetricData("memory_percent", memory_percent, "%", now),
                "memory_used_gb": MetricData("memory_used_gb", memory_used_gb, "GB", now),
                "memory_total_gb": MetricData("memory_total_gb", memory_total_gb, "GB", now),
                "disk_percent": MetricData("disk_percent", disk_percent, "%", now),
                "disk_used_gb": MetricData("disk_used_gb", disk_used_gb, "GB", now),
                "disk_total_gb": MetricData("disk_total_gb", disk_total_gb, "GB", now),
                "network_sent_gb": MetricData("network_sent_gb", network_sent_gb, "GB", now),
                "network_recv_gb": MetricData("network_recv_gb", network_recv_gb, "GB", now),
                "process_memory_mb": MetricData("process_memory_mb", process_memory_mb, "MB", now),
                "process_cpu_percent": MetricData("process_cpu_percent", process_cpu_percent, "%", now),
                "uptime_seconds": MetricData("uptime_seconds", uptime_seconds, "s", now)
            }
            
            # Store in history
            self._add_to_history(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def _add_to_history(self, metrics: Dict[str, MetricData]):
        """Add metrics to history with size limit"""
        self.metrics_history.append({
            "timestamp": datetime.now(timezone.utc),
            "metrics": {k: v.value for k, v in metrics.items()}
        })
        
        # Limit history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_metrics_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metrics history for specified minutes"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (minutes * 60)
        
        return [
            entry for entry in self.metrics_history
            if entry["timestamp"].timestamp() > cutoff_time
        ]
    
    def get_health_status(self) -> HealthStatus:
        """Determine overall health status based on metrics"""
        try:
            metrics = self.get_system_metrics()
            
            # Health checks
            issues = 0
            
            # CPU check
            cpu_percent = metrics.get("cpu_percent")
            if cpu_percent and cpu_percent.value > 90:
                issues += 1
            elif cpu_percent and cpu_percent.value > 70:
                issues += 0.5
            
            # Memory check
            memory_percent = metrics.get("memory_percent")
            if memory_percent and memory_percent.value > 90:
                issues += 1
            elif memory_percent and memory_percent.value > 80:
                issues += 0.5
            
            # Disk check
            disk_percent = metrics.get("disk_percent")
            if disk_percent and disk_percent.value > 95:
                issues += 1
            elif disk_percent and disk_percent.value > 85:
                issues += 0.5
            
            # Determine status
            if issues >= 2:
                return HealthStatus.UNHEALTHY
            elif issues >= 1:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY
                
        except Exception as e:
            logger.error(f"Failed to determine health status: {e}")
            return HealthStatus.UNKNOWN


class PerformanceMonitor:
    """Application performance monitoring"""
    
    def __init__(self):
        self.request_times = []
        self.request_counts = {"success": 0, "error": 0}
        self.error_details = []
        self.max_history_size = 10000
    
    def record_request(self, duration_ms: float, success: bool = True, 
                      endpoint: str = None, error: str = None):
        """Record request performance data"""
        timestamp = datetime.now(timezone.utc)
        
        self.request_times.append({
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "success": success,
            "endpoint": endpoint
        })
        
        # Update counters
        if success:
            self.request_counts["success"] += 1
        else:
            self.request_counts["error"] += 1
            if error:
                self.error_details.append({
                    "timestamp": timestamp,
                    "error": error,
                    "endpoint": endpoint
                })
        
        # Limit history size
        if len(self.request_times) > self.max_history_size:
            self.request_times = self.request_times[-self.max_history_size:]
        
        if len(self.error_details) > 1000:
            self.error_details = self.error_details[-1000:]
    
    def get_performance_stats(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance statistics for specified time window"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (minutes * 60)
        
        # Filter recent requests
        recent_requests = [
            req for req in self.request_times
            if req["timestamp"].timestamp() > cutoff_time
        ]
        
        if not recent_requests:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "error_requests": 0,
                "success_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "median_response_time_ms": 0.0,
                "p95_response_time_ms": 0.0,
                "p99_response_time_ms": 0.0
            }
        
        # Calculate stats
        successful = [req for req in recent_requests if req["success"]]
        errors = [req for req in recent_requests if not req["success"]]
        
        response_times = [req["duration_ms"] for req in recent_requests]
        response_times.sort()
        
        total_requests = len(recent_requests)
        success_rate = len(successful) / total_requests if total_requests > 0 else 0
        
        # Percentiles
        def percentile(data, p):
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = k - f
            if f == len(data) - 1:
                return data[f]
            return data[f] * (1 - c) + data[f + 1] * c
        
        return {
            "total_requests": total_requests,
            "successful_requests": len(successful),
            "error_requests": len(errors),
            "success_rate": round(success_rate, 4),
            "avg_response_time_ms": round(sum(response_times) / len(response_times), 2) if response_times else 0.0,
            "median_response_time_ms": round(percentile(response_times, 0.5), 2),
            "p95_response_time_ms": round(percentile(response_times, 0.95), 2),
            "p99_response_time_ms": round(percentile(response_times, 0.99), 2)
        }
    
    def get_error_summary(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get summary of recent errors"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (minutes * 60)
        
        recent_errors = [
            error for error in self.error_details
            if error["timestamp"].timestamp() > cutoff_time
        ]
        
        # Group errors by type
        error_counts = {}
        for error in recent_errors:
            error_type = error["error"]
            if error_type not in error_counts:
                error_counts[error_type] = {
                    "count": 0,
                    "latest_timestamp": None,
                    "endpoints": set()
                }
            
            error_counts[error_type]["count"] += 1
            error_counts[error_type]["latest_timestamp"] = error["timestamp"]
            if error["endpoint"]:
                error_counts[error_type]["endpoints"].add(error["endpoint"])
        
        # Convert to list and sort by count
        error_summary = []
        for error_type, data in error_counts.items():
            error_summary.append({
                "error": error_type,
                "count": data["count"],
                "latest_timestamp": data["latest_timestamp"].isoformat(),
                "affected_endpoints": list(data["endpoints"])
            })
        
        return sorted(error_summary, key=lambda x: x["count"], reverse=True)


class HealthChecker:
    """Comprehensive health checking"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.performance_monitor = PerformanceMonitor()
        self._custom_checks = {}
    
    def add_custom_check(self, name: str, check_func):
        """Add custom health check function"""
        self._custom_checks[name] = check_func
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()
        
        health_data = {
            "overall_status": HealthStatus.HEALTHY.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "metrics": {},
            "performance": {},
            "check_duration_ms": 0
        }
        
        try:
            # System health
            system_status = self.system_monitor.get_health_status()
            system_metrics = self.system_monitor.get_system_metrics()
            
            health_data["checks"]["system"] = {
                "status": system_status.value,
                "description": "System resource utilization"
            }
            
            health_data["metrics"]["system"] = {
                name: metric.to_dict() for name, metric in system_metrics.items()
            }
            
            # Performance health
            perf_stats = self.performance_monitor.get_performance_stats()
            health_data["performance"] = perf_stats
            
            # Determine performance status
            if perf_stats["success_rate"] < 0.9:
                perf_status = HealthStatus.UNHEALTHY
            elif perf_stats["success_rate"] < 0.95 or perf_stats["avg_response_time_ms"] > 5000:
                perf_status = HealthStatus.DEGRADED
            else:
                perf_status = HealthStatus.HEALTHY
            
            health_data["checks"]["performance"] = {
                "status": perf_status.value,
                "description": "Application performance metrics"
            }
            
            # Custom checks
            for check_name, check_func in self._custom_checks.items():
                try:
                    if asyncio.iscoroutinefunction(check_func):
                        result = await check_func()
                    else:
                        result = check_func()
                    
                    health_data["checks"][check_name] = {
                        "status": result.get("status", "unknown"),
                        "description": result.get("description", "Custom health check"),
                        "details": result.get("details")
                    }
                except Exception as e:
                    health_data["checks"][check_name] = {
                        "status": HealthStatus.UNHEALTHY.value,
                        "description": "Custom health check failed",
                        "error": str(e)
                    }
            
            # Determine overall status
            check_statuses = [check["status"] for check in health_data["checks"].values()]
            
            if HealthStatus.UNHEALTHY.value in check_statuses:
                health_data["overall_status"] = HealthStatus.UNHEALTHY.value
            elif HealthStatus.DEGRADED.value in check_statuses:
                health_data["overall_status"] = HealthStatus.DEGRADED.value
            else:
                health_data["overall_status"] = HealthStatus.HEALTHY.value
            
        except Exception as e:
            health_data["overall_status"] = HealthStatus.UNKNOWN.value
            health_data["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        # Record check duration
        health_data["check_duration_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return health_data
    
    def get_metrics_export(self, format_type: str = "json") -> str:
        """Export metrics in specified format"""
        metrics_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_metrics": {
                name: metric.to_dict() 
                for name, metric in self.system_monitor.get_system_metrics().items()
            },
            "performance_stats": self.performance_monitor.get_performance_stats(),
            "error_summary": self.performance_monitor.get_error_summary()
        }
        
        if format_type.lower() == "json":
            return json.dumps(metrics_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Global instances
system_monitor = SystemMonitor()
performance_monitor = PerformanceMonitor()
health_checker = HealthChecker()