"""
Comprehensive health checks and system monitoring.
"""

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .logging_config import get_contextual_logger
from .exceptions import AgenticScentError, create_error_with_code, FactoryError


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": round(self.response_time * 1000, 2),
            "metadata": self.metadata or {}
        }


class HealthChecker:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        self.logger = get_contextual_logger("health_checks")
        self.health_checks: Dict[str, Callable] = {}
        self.check_results: Dict[str, HealthCheckResult] = {}
        self.monitoring_active = False
        self._monitoring_task = None


class HealthCheckManager(HealthChecker):
    """Main health check management system."""
    
    def __init__(self):
        super().__init__()
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default system health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("database_connection", self._check_database)
        self.register_check("cache_connection", self._check_cache)
        self.register_check("llm_services", self._check_llm_services)
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            status = HealthStatus.HEALTHY
            if cpu_percent > 90 or memory.percent > 90:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 70 or memory.percent > 70:
                status = HealthStatus.WARNING
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=f"CPU: {cpu_percent}%, Memory: {memory.percent}%",
                timestamp=datetime.now(),
                response_time=0.0,
                metadata={"cpu": cpu_percent, "memory": memory.percent}
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Error checking resources: {e}",
                timestamp=datetime.now(),
                response_time=0.0
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity."""
        await asyncio.sleep(0.1)  # Simulate DB check
        return HealthCheckResult(
            name="database_connection",
            status=HealthStatus.HEALTHY,
            message="Database connection OK",
            timestamp=datetime.now(),
            response_time=0.1
        )
    
    async def _check_cache(self) -> HealthCheckResult:
        """Check cache connectivity."""
        await asyncio.sleep(0.05)  # Simulate cache check
        return HealthCheckResult(
            name="cache_connection", 
            status=HealthStatus.HEALTHY,
            message="Cache connection OK",
            timestamp=datetime.now(),
            response_time=0.05
        )
    
    async def _check_llm_services(self) -> HealthCheckResult:
        """Check LLM service availability."""
        await asyncio.sleep(0.2)  # Simulate LLM check
        return HealthCheckResult(
            name="llm_services",
            status=HealthStatus.HEALTHY,
            message="LLM services operational",
            timestamp=datetime.now(),
            response_time=0.2
        )
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health status."""
        results = await self.run_all_checks()
        overall_status = self.get_overall_status()
        
        return {
            "overall_status": overall_status.value,
            "checks": {name: result.to_dict() for name, result in results.items()},
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(results)
        }
        
    def register_check(self, name: str, check_func: Callable, 
                      interval: int = 60, critical: bool = False):
        """
        Register a health check function.
        
        Args:
            name: Unique name for the health check
            check_func: Async function that returns HealthCheckResult
            interval: Check interval in seconds
            critical: Whether this is a critical system check
        """
        self.health_checks[name] = {
            "func": check_func,
            "interval": interval,
            "critical": critical,
            "last_run": 0
        }
        self.logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                timestamp=datetime.now(),
                response_time=0.0
            )
        
        check_config = self.health_checks[name]
        start_time = time.time()
        
        try:
            result = await check_config["func"]()
            if not isinstance(result, HealthCheckResult):
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check completed",
                    timestamp=datetime.now(),
                    response_time=time.time() - start_time
                )
        
        except Exception as e:
            self.logger.error(f"Health check '{name}' failed: {e}")
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time=time.time() - start_time
            )
        
        self.check_results[name] = result
        check_config["last_run"] = time.time()
        
        return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        current_time = time.time()
        checks_to_run = []
        
        for name, config in self.health_checks.items():
            if current_time - config["last_run"] >= config["interval"]:
                checks_to_run.append(name)
        
        if not checks_to_run:
            return self.check_results
        
        # Run checks concurrently
        tasks = [self.run_check(name) for name in checks_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for name, result in zip(checks_to_run, results):
            if isinstance(result, Exception):
                self.logger.error(f"Health check '{name}' exception: {result}")
            else:
                self.check_results[name] = result
        
        return self.check_results
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval)
        )
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                await self.run_all_checks()
                
                # Check for critical failures
                critical_failures = [
                    name for name, result in self.check_results.items()
                    if result.status == HealthStatus.CRITICAL and 
                    self.health_checks[name]["critical"]
                ]
                
                if critical_failures:
                    self.logger.critical(
                        f"Critical health check failures: {', '.join(critical_failures)}"
                    )
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(interval)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.check_results:
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "message": "No health checks run yet",
                "checks": {}
            }
        
        # Determine overall status
        statuses = [result.status for result in self.check_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Count by status
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(
                1 for s in statuses if s == status
            )
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self.monitoring_active,
            "total_checks": len(self.check_results),
            "status_counts": status_counts,
            "checks": {
                name: result.to_dict() 
                for name, result in self.check_results.items()
            }
        }


# Default health check functions
async def check_system_resources() -> HealthCheckResult:
    """Check system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine status based on resource usage
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = HealthStatus.CRITICAL
            message = f"High resource usage: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
        elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
            status = HealthStatus.WARNING
            message = f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Resource usage normal: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
        
        return HealthCheckResult(
            name="system_resources",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time=0.0,
            metadata={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / 1024**3, 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / 1024**3, 2)
            }
        )
    
    except Exception as e:
        return HealthCheckResult(
            name="system_resources",
            status=HealthStatus.CRITICAL,
            message=f"Failed to check system resources: {e}",
            timestamp=datetime.now(),
            response_time=0.0
        )


async def check_database_connection(connection_string: Optional[str] = None) -> HealthCheckResult:
    """Check database connectivity."""
    start_time = time.time()
    
    try:
        # Mock database check - in production this would test actual DB connection
        await asyncio.sleep(0.1)  # Simulate DB query
        
        response_time = time.time() - start_time
        
        return HealthCheckResult(
            name="database_connection",
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            timestamp=datetime.now(),
            response_time=response_time,
            metadata={
                "connection_string": connection_string or "mock://localhost",
                "query_time_ms": round(response_time * 1000, 2)
            }
        )
    
    except Exception as e:
        return HealthCheckResult(
            name="database_connection",
            status=HealthStatus.CRITICAL,
            message=f"Database connection failed: {e}",
            timestamp=datetime.now(),
            response_time=time.time() - start_time
        )


async def check_cache_connection(cache_url: Optional[str] = None) -> HealthCheckResult:
    """Check cache service connectivity."""
    start_time = time.time()
    
    try:
        # Mock cache check - in production this would test Redis/Memcached
        await asyncio.sleep(0.05)  # Simulate cache ping
        
        response_time = time.time() - start_time
        
        return HealthCheckResult(
            name="cache_connection",
            status=HealthStatus.HEALTHY,
            message="Cache service responsive",
            timestamp=datetime.now(),
            response_time=response_time,
            metadata={
                "cache_url": cache_url or "redis://localhost:6379",
                "ping_time_ms": round(response_time * 1000, 2)
            }
        )
    
    except Exception as e:
        return HealthCheckResult(
            name="cache_connection",
            status=HealthStatus.CRITICAL,
            message=f"Cache connection failed: {e}",
            timestamp=datetime.now(),
            response_time=time.time() - start_time
        )


async def check_llm_services() -> HealthCheckResult:
    """Check LLM service availability."""
    start_time = time.time()
    
    try:
        # Mock LLM service check
        await asyncio.sleep(0.2)  # Simulate API call
        
        response_time = time.time() - start_time
        
        # In production, this would test actual API endpoints
        api_status = "healthy"  # Mock status
        
        if api_status == "healthy":
            status = HealthStatus.HEALTHY
            message = "LLM services operational"
        else:
            status = HealthStatus.WARNING
            message = "LLM services degraded"
        
        return HealthCheckResult(
            name="llm_services",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time=response_time,
            metadata={
                "api_response_time_ms": round(response_time * 1000, 2),
                "services_tested": ["openai", "anthropic"]
            }
        )
    
    except Exception as e:
        return HealthCheckResult(
            name="llm_services",
            status=HealthStatus.CRITICAL,
            message=f"LLM service check failed: {e}",
            timestamp=datetime.now(),
            response_time=time.time() - start_time
        )


def create_default_health_checker() -> HealthChecker:
    """Create health checker with default checks registered."""
    health_checker = HealthChecker()
    
    # Register default health checks
    health_checker.register_check(
        "system_resources", 
        check_system_resources, 
        interval=30, 
        critical=True
    )
    
    health_checker.register_check(
        "database_connection", 
        check_database_connection, 
        interval=60, 
        critical=True
    )
    
    health_checker.register_check(
        "cache_connection", 
        check_cache_connection, 
        interval=60, 
        critical=False
    )
    
    health_checker.register_check(
        "llm_services", 
        check_llm_services, 
        interval=120, 
        critical=False
    )
    
    return health_checker