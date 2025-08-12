"""
Advanced resilience patterns for industrial-grade reliability.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from contextlib import asynccontextmanager
import random
import threading
from datetime import datetime, timedelta


class FailureMode(Enum):
    """Types of failures that can occur."""
    TRANSIENT = "transient"  # Temporary failures that might succeed on retry
    PERMANENT = "permanent"  # Persistent failures unlikely to recover
    TIMEOUT = "timeout"      # Operation timed out
    RESOURCE = "resource"    # Resource exhaustion
    NETWORK = "network"      # Network connectivity issues
    AUTHENTICATION = "auth"  # Authentication/authorization failures


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing fast, not attempting calls
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retryable_exceptions: Tuple[type, ...] = (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given retry attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay
            
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add ±25% jitter to prevent thundering herd
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
            
        return max(0, delay)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds to wait before half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: Optional[float] = None     # Operation timeout
    monitor_window: float = 300.0       # Time window for failure rate calculation


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    """
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.logger = logging.getLogger(__name__)
        
        # Sliding window for failure rate monitoring
        self.call_history: List[Tuple[float, bool]] = []  # (timestamp, success)
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable):
        """Decorator for circuit breaker."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker '{self.name}' is OPEN"
            )
        
        try:
            start_time = time.time()
            
            # Apply timeout if configured
            if self.config.timeout:
                # For sync calls, we can't easily implement timeout
                # In production, consider using signal.alarm or threading.Timer
                pass
                
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            self._record_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker '{self.name}' is OPEN"
            )
        
        try:
            start_time = time.time()
            
            if self.config.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs)
                
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            return result
            
        except asyncio.TimeoutError as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
    
    def _can_execute(self) -> bool:
        """Check if operation can be executed."""
        with self._lock:
            current_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                if (self.last_failure_time and 
                    current_time - self.last_failure_time >= self.config.recovery_timeout):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    return True
                return False
            
            else:  # HALF_OPEN
                return True
    
    def _record_success(self, execution_time: float):
        """Record successful operation."""
        with self._lock:
            current_time = time.time()
            self.call_history.append((current_time, True))
            self._cleanup_history(current_time)
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery
    
    def _record_failure(self, exception: Exception, execution_time: float):
        """Record failed operation."""
        with self._lock:
            current_time = time.time()
            self.call_history.append((current_time, False))
            self._cleanup_history(current_time)
            
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.logger.warning(
                        f"Circuit breaker '{self.name}' OPEN after {self.failure_count} failures"
                    )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker '{self.name}' back to OPEN")
    
    def _cleanup_history(self, current_time: float):
        """Remove old entries from call history."""
        cutoff_time = current_time - self.config.monitor_window
        self.call_history = [
            (timestamp, success) for timestamp, success in self.call_history
            if timestamp >= cutoff_time
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            current_time = time.time()
            self._cleanup_history(current_time)
            
            total_calls = len(self.call_history)
            if total_calls == 0:
                failure_rate = 0.0
                success_rate = 0.0
            else:
                successes = sum(1 for _, success in self.call_history if success)
                success_rate = successes / total_calls
                failure_rate = 1.0 - success_rate
            
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_calls_window": total_calls,
                "failure_rate": failure_rate,
                "success_rate": success_rate,
                "last_failure_time": self.last_failure_time
            }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


def retry_with_backoff(policy: RetryPolicy):
    """
    Decorator for automatic retry with exponential backoff.
    """
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(1, policy.max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        # Check if exception is retryable
                        if not any(isinstance(e, exc_type) for exc_type in policy.retryable_exceptions):
                            raise
                        
                        # Don't delay on last attempt
                        if attempt < policy.max_attempts:
                            delay = policy.get_delay(attempt)
                            await asyncio.sleep(delay)
                
                raise last_exception
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(1, policy.max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        # Check if exception is retryable
                        if not any(isinstance(e, exc_type) for exc_type in policy.retryable_exceptions):
                            raise
                        
                        # Don't delay on last attempt
                        if attempt < policy.max_attempts:
                            delay = policy.get_delay(attempt)
                            time.sleep(delay)
                
                raise last_exception
            return sync_wrapper
    return decorator


class TimeoutManager:
    """
    Manages operation timeouts with graceful degradation.
    """
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.logger = logging.getLogger(__name__)
    
    @asynccontextmanager
    async def timeout(self, seconds: Optional[float] = None):
        """Context manager for timeout operations."""
        timeout_duration = seconds or self.default_timeout
        
        try:
            async with asyncio.timeout(timeout_duration):
                yield
        except asyncio.TimeoutError:
            self.logger.warning(f"Operation timed out after {timeout_duration} seconds")
            raise


class BulkheadIsolation:
    """
    Implements bulkhead pattern for resource isolation.
    """
    
    def __init__(self, name: str, pool_size: int = 10):
        self.name = name
        self.semaphore = asyncio.Semaphore(pool_size)
        self.pool_size = pool_size
        self.active_tasks = 0
        self.logger = logging.getLogger(__name__)
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire resource from bulkhead pool."""
        await self.semaphore.acquire()
        self.active_tasks += 1
        
        try:
            yield
        finally:
            self.active_tasks -= 1
            self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "pool_size": self.pool_size,
            "active_tasks": self.active_tasks,
            "available_slots": self.semaphore._value
        }


class ResilienceManager:
    """
    Central manager for all resilience patterns.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self.timeout_manager = TimeoutManager()
        self.logger = logging.getLogger(__name__)
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            if not config:
                config = CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(config, name)
        return self.circuit_breakers[name]
    
    def get_bulkhead(self, name: str, pool_size: int = 10) -> BulkheadIsolation:
        """Get or create bulkhead."""
        if name not in self.bulkheads:
            self.bulkheads[name] = BulkheadIsolation(name, pool_size)
        return self.bulkheads[name]
    
    async def execute_with_resilience(self, 
                                    func: Callable, 
                                    circuit_breaker_name: Optional[str] = None,
                                    bulkhead_name: Optional[str] = None,
                                    timeout: Optional[float] = None,
                                    retry_policy: Optional[RetryPolicy] = None,
                                    *args, **kwargs) -> Any:
        """Execute function with full resilience patterns."""
        
        # Apply retry if specified
        if retry_policy:
            func = retry_with_backoff(retry_policy)(func)
        
        # Apply circuit breaker if specified
        if circuit_breaker_name:
            circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
            func = circuit_breaker(func)
        
        # Execute with bulkhead and timeout
        async def execute():
            if bulkhead_name:
                bulkhead = self.get_bulkhead(bulkhead_name)
                async with bulkhead.acquire():
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        if timeout:
            async with self.timeout_manager.timeout(timeout):
                return await execute()
        else:
            return await execute()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall resilience health status."""
        circuit_breaker_stats = {
            name: cb.get_stats() 
            for name, cb in self.circuit_breakers.items()
        }
        
        bulkhead_stats = {
            name: bh.get_stats()
            for name, bh in self.bulkheads.items()
        }
        
        # Calculate overall health
        open_breakers = sum(
            1 for stats in circuit_breaker_stats.values() 
            if stats["state"] == "open"
        )
        
        saturated_bulkheads = sum(
            1 for stats in bulkhead_stats.values()
            if stats["available_slots"] == 0
        )
        
        health_score = 1.0
        if circuit_breaker_stats:
            health_score *= (1.0 - open_breakers / len(circuit_breaker_stats))
        if bulkhead_stats:
            health_score *= (1.0 - saturated_bulkheads / len(bulkhead_stats))
        
        return {
            "overall_health_score": health_score,
            "circuit_breakers": circuit_breaker_stats,
            "bulkheads": bulkhead_stats,
            "open_circuit_breakers": open_breakers,
            "saturated_bulkheads": saturated_bulkheads
        }


# Global instance
resilience_manager = ResilienceManager()


# Convenience decorators
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker pattern."""
    cb = resilience_manager.get_circuit_breaker(name, config)
    return cb


def bulkhead(name: str, pool_size: int = 10):
    """Decorator for bulkhead pattern."""
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                bh = resilience_manager.get_bulkhead(name, pool_size)
                async with bh.acquire():
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Sync version would need different implementation
                return func(*args, **kwargs)
            return sync_wrapper
    return decorator


# Example usage decorators
@dataclass
class ResilientOperation:
    """Configuration for resilient operation."""
    circuit_breaker_name: Optional[str] = None
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    bulkhead_name: Optional[str] = None
    bulkhead_size: int = 10
    timeout: Optional[float] = None
    retry_policy: Optional[RetryPolicy] = None


def resilient(config: ResilientOperation):
    """
    Comprehensive resilience decorator.
    
    Example:
        @resilient(ResilientOperation(
            circuit_breaker_name="sensor_api",
            bulkhead_name="sensor_pool",
            timeout=30.0,
            retry_policy=RetryPolicy(max_attempts=3)
        ))
        async def read_sensor():
            # Implementation here
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await resilience_manager.execute_with_resilience(
                func,
                circuit_breaker_name=config.circuit_breaker_name,
                bulkhead_name=config.bulkhead_name,
                timeout=config.timeout,
                retry_policy=config.retry_policy,
                *args, **kwargs
            )
        return wrapper
    return decorator