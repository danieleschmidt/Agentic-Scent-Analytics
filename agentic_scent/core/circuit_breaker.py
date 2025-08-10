"""
Circuit breaker pattern for fault tolerance and system resilience.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import threading


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    name: str = "default"


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    open_timestamp: Optional[float] = None
    last_failure_timestamp: Optional[float] = None


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.metrics = CircuitBreakerMetrics()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"circuit_breaker.{config.name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface."""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
    
    def _sync_wrapper(self, func: Callable) -> Callable:
        """Synchronous wrapper."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def _async_wrapper(self, func: Callable) -> Callable:
        """Asynchronous wrapper."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            self.metrics.total_requests += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.config.name} transitioning to HALF_OPEN")
                else:
                    self.logger.warning(f"Circuit breaker {self.config.name} is OPEN, rejecting call")
                    raise CircuitBreakerError(f"Circuit breaker {self.config.name} is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure(e)
                raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        with self.lock:
            self.metrics.total_requests += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.config.name} transitioning to HALF_OPEN")
                else:
                    self.logger.warning(f"Circuit breaker {self.config.name} is OPEN, rejecting call")
                    raise CircuitBreakerError(f"Circuit breaker {self.config.name} is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (time.time() - self.metrics.open_timestamp) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        with self.lock:
            self.metrics.successful_requests += 1
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.logger.info(f"Circuit breaker {self.config.name} reset to CLOSED")
    
    def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        with self.lock:
            self.metrics.failed_requests += 1
            self.failure_count += 1
            self.metrics.last_failure_timestamp = time.time()
            
            self.logger.warning(
                f"Circuit breaker {self.config.name} recorded failure "
                f"({self.failure_count}/{self.config.failure_threshold}): {exception}"
            )
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.metrics.open_timestamp = time.time()
                self.logger.error(f"Circuit breaker {self.config.name} opened due to failures")
    
    def get_state(self) -> CircuitState:
        """Get current state of circuit breaker."""
        return self.state
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.metrics
    
    def force_open(self):
        """Force circuit breaker to open state."""
        with self.lock:
            self.state = CircuitState.OPEN
            self.metrics.open_timestamp = time.time()
            self.logger.warning(f"Circuit breaker {self.config.name} forced OPEN")
    
    def force_close(self):
        """Force circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.metrics.open_timestamp = None
            self.logger.info(f"Circuit breaker {self.config.name} forced CLOSED")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        if name in self.breakers:
            return self.breakers[name]
        
        if config is None:
            config = CircuitBreakerConfig(name=name)
        else:
            config.name = name
        
        breaker = CircuitBreaker(config)
        self.breakers[name] = breaker
        self.logger.info(f"Created circuit breaker: {name}")
        return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers."""
        return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}
    
    def get_all_states(self) -> Dict[str, CircuitState]:
        """Get states for all circuit breakers."""
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}


# Global registry instance
_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, failure_threshold: int = 5, 
                   recovery_timeout: float = 60.0,
                   expected_exception: type = Exception) -> Callable:
    """
    Decorator for applying circuit breaker pattern.
    
    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening
        recovery_timeout: Time to wait before attempting reset
        expected_exception: Exception type to catch
    
    Returns:
        Decorated function with circuit breaker protection
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        name=name
    )
    
    breaker = _registry.create_breaker(name, config)
    return breaker


def get_circuit_breaker_metrics() -> Dict[str, CircuitBreakerMetrics]:
    """Get metrics for all registered circuit breakers."""
    return _registry.get_all_metrics()


def get_circuit_breaker_states() -> Dict[str, CircuitState]:
    """Get states for all registered circuit breakers."""
    return _registry.get_all_states()


def force_open_breaker(name: str):
    """Force a circuit breaker to open state."""
    breaker = _registry.get_breaker(name)
    if breaker:
        breaker.force_open()


def force_close_breaker(name: str):
    """Force a circuit breaker to closed state."""
    breaker = _registry.get_breaker(name)
    if breaker:
        breaker.force_close()