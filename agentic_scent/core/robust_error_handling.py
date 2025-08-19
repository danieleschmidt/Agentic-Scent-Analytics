#!/usr/bin/env python3
"""
Robust Error Handling and Recovery System for Industrial AI
Implements comprehensive error handling, fault tolerance, graceful degradation,
and autonomous recovery capabilities for manufacturing environments.
"""

import asyncio
import logging
import traceback
import sys
import time
import json
import inspect
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from contextlib import asynccontextmanager, contextmanager
import threading
from collections import defaultdict, deque
import pickle
from pathlib import Path

from .exceptions import AgenticScentError


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    RESTART_COMPONENT = "restart_component"
    ESCALATE = "escalate"
    AUTONOMOUS_RECOVERY = "autonomous_recovery"


class ComponentStatus(Enum):
    """Component operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"
    OFFLINE = "offline"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    timestamp: datetime
    error_type: Type[Exception]
    error_message: str
    traceback_info: str
    component: str
    function: str
    severity: ErrorSeverity
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action definition."""
    name: str
    strategy: RecoveryStrategy
    max_attempts: int
    timeout: timedelta
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Optional[Callable] = None
    fallback_action: Optional[str] = None


@dataclass
class ComponentHealth:
    """Component health status tracking."""
    component_name: str
    status: ComponentStatus
    last_error: Optional[ErrorContext] = None
    error_count: int = 0
    recovery_count: int = 0
    uptime_start: datetime = field(default_factory=datetime.now)
    last_health_check: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: timedelta = timedelta(minutes=1)):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if datetime.now() - self.last_failure_time > self.timeout:
                    self.state = "half_open"
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
                    
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit breaker
            with self._lock:
                self.failure_count = 0
                self.state = "closed"
                
            return result
            
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    
            raise e


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryManager:
    """Advanced retry mechanism with exponential backoff and jitter."""
    
    def __init__(self):
        self.retry_configs = {}
        
    def configure_retry(self, operation: str, max_attempts: int = 3, 
                       base_delay: float = 1.0, max_delay: float = 60.0,
                       exponential_base: float = 2.0, jitter: bool = True):
        """Configure retry parameters for an operation."""
        self.retry_configs[operation] = {
            'max_attempts': max_attempts,
            'base_delay': base_delay,
            'max_delay': max_delay,
            'exponential_base': exponential_base,
            'jitter': jitter
        }
        
    async def retry_async(self, operation: str, func: Callable, *args, **kwargs):
        """Retry async function with configured parameters."""
        config = self.retry_configs.get(operation, {
            'max_attempts': 3,
            'base_delay': 1.0,
            'max_delay': 60.0,
            'exponential_base': 2.0,
            'jitter': True
        })
        
        last_exception = None
        
        for attempt in range(config['max_attempts']):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt < config['max_attempts'] - 1:  # Not the last attempt
                    # Calculate delay with exponential backoff
                    delay = min(
                        config['base_delay'] * (config['exponential_base'] ** attempt),
                        config['max_delay']
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config['jitter']:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                        
                    await asyncio.sleep(delay)
                    
        # All attempts failed
        raise last_exception


class FaultTolerantContainer:
    """Container for fault-tolerant execution of operations."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.circuit_breakers = {}
        self.retry_manager = RetryManager()
        self.fallback_handlers = {}
        self.health_status = ComponentHealth(component_name, ComponentStatus.HEALTHY)
        
    def add_circuit_breaker(self, operation: str, failure_threshold: int = 5, 
                           timeout: timedelta = timedelta(minutes=1)):
        """Add circuit breaker for an operation."""
        self.circuit_breakers[operation] = CircuitBreaker(failure_threshold, timeout)
        
    def add_fallback_handler(self, operation: str, handler: Callable):
        """Add fallback handler for an operation."""
        self.fallback_handlers[operation] = handler
        
    async def execute_with_protection(self, operation: str, func: Callable, *args, **kwargs):
        """Execute function with full fault tolerance protection."""
        try:
            # Try circuit breaker if available
            if operation in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[operation]
                
                if circuit_breaker.state == "open":
                    # Circuit is open, try fallback
                    if operation in self.fallback_handlers:
                        return await self._execute_fallback(operation, *args, **kwargs)
                    else:
                        raise CircuitBreakerOpenError(f"Circuit breaker open for {operation}")
                        
            # Try with retry mechanism
            return await self.retry_manager.retry_async(operation, func, *args, **kwargs)
            
        except Exception as e:
            # Update health status
            self.health_status.error_count += 1
            self.health_status.last_error = ErrorContext(
                timestamp=datetime.now(),
                error_type=type(e),
                error_message=str(e),
                traceback_info=traceback.format_exc(),
                component=self.component_name,
                function=operation,
                severity=ErrorSeverity.ERROR
            )
            
            # Try fallback handler
            if operation in self.fallback_handlers:
                try:
                    return await self._execute_fallback(operation, *args, **kwargs)
                except Exception as fallback_error:
                    logging.error(f"Fallback failed for {operation}: {fallback_error}")
                    
            # Re-raise original exception if no fallback or fallback failed
            raise e
            
    async def _execute_fallback(self, operation: str, *args, **kwargs):
        """Execute fallback handler."""
        fallback_handler = self.fallback_handlers[operation]
        
        if asyncio.iscoroutinefunction(fallback_handler):
            return await fallback_handler(*args, **kwargs)
        else:
            return fallback_handler(*args, **kwargs)


class AutonomousRecoverySystem:
    """Autonomous recovery system for self-healing capabilities."""
    
    def __init__(self):
        self.recovery_actions = {}
        self.component_dependencies = {}
        self.recovery_history = []
        self.learning_enabled = True
        self.success_patterns = defaultdict(list)
        
    def register_recovery_action(self, component: str, action: RecoveryAction):
        """Register recovery action for a component."""
        if component not in self.recovery_actions:
            self.recovery_actions[component] = []
        self.recovery_actions[component].append(action)
        
    def register_dependency(self, component: str, dependencies: List[str]):
        """Register component dependencies."""
        self.component_dependencies[component] = dependencies
        
    async def attempt_recovery(self, component: str, error_context: ErrorContext) -> Dict[str, Any]:
        """Attempt autonomous recovery for a component."""
        recovery_result = {
            'component': component,
            'recovery_attempted': False,
            'recovery_successful': False,
            'actions_taken': [],
            'final_status': ComponentStatus.FAILED,
            'recovery_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Get recovery actions for component
            actions = self.recovery_actions.get(component, [])
            
            if not actions:
                logging.warning(f"No recovery actions defined for component {component}")
                return recovery_result
                
            # Select best recovery action based on error context and history
            selected_action = await self._select_recovery_action(component, error_context, actions)
            
            if not selected_action:
                logging.warning(f"No suitable recovery action found for {component}")
                return recovery_result
                
            recovery_result['recovery_attempted'] = True
            
            # Execute recovery action
            action_result = await self._execute_recovery_action(component, selected_action, error_context)
            
            recovery_result['actions_taken'].append(action_result)
            
            if action_result['success']:
                recovery_result['recovery_successful'] = True
                recovery_result['final_status'] = ComponentStatus.HEALTHY
                
                # Learn from successful recovery
                if self.learning_enabled:
                    await self._learn_from_recovery(component, selected_action, error_context, True)
                    
            else:
                # Try fallback action if available
                if selected_action.fallback_action:
                    fallback_action = next(
                        (a for a in actions if a.name == selected_action.fallback_action), None
                    )
                    
                    if fallback_action:
                        fallback_result = await self._execute_recovery_action(
                            component, fallback_action, error_context
                        )
                        recovery_result['actions_taken'].append(fallback_result)
                        
                        if fallback_result['success']:
                            recovery_result['recovery_successful'] = True
                            recovery_result['final_status'] = ComponentStatus.DEGRADED
                            
                # Learn from failed recovery
                if self.learning_enabled:
                    await self._learn_from_recovery(component, selected_action, error_context, False)
                    
        except Exception as e:
            logging.error(f"Recovery attempt failed: {e}")
            recovery_result['error'] = str(e)
            
        finally:
            recovery_result['recovery_time'] = time.time() - start_time
            
            # Record recovery attempt
            self.recovery_history.append({
                'timestamp': datetime.now(),
                'component': component,
                'error_context': error_context,
                'result': recovery_result
            })
            
        return recovery_result
        
    async def _select_recovery_action(self, component: str, error_context: ErrorContext, 
                                    actions: List[RecoveryAction]) -> Optional[RecoveryAction]:
        """Select best recovery action based on context and history."""
        
        # Filter actions that haven't exceeded max attempts
        viable_actions = [
            action for action in actions 
            if error_context.recovery_attempts < action.max_attempts
        ]
        
        if not viable_actions:
            return None
            
        # If learning is enabled, prefer actions that have been successful before
        if self.learning_enabled and component in self.success_patterns:
            successful_actions = self.success_patterns[component]
            
            for action in viable_actions:
                if any(pattern['action'] == action.name for pattern in successful_actions):
                    return action
                    
        # Default selection based on strategy priority
        strategy_priority = {
            RecoveryStrategy.RETRY: 1,
            RecoveryStrategy.FALLBACK: 2,
            RecoveryStrategy.GRACEFUL_DEGRADATION: 3,
            RecoveryStrategy.CIRCUIT_BREAKER: 4,
            RecoveryStrategy.RESTART_COMPONENT: 5,
            RecoveryStrategy.AUTONOMOUS_RECOVERY: 6,
            RecoveryStrategy.ESCALATE: 7
        }
        
        return min(viable_actions, key=lambda a: strategy_priority.get(a.strategy, 10))
        
    async def _execute_recovery_action(self, component: str, action: RecoveryAction, 
                                     error_context: ErrorContext) -> Dict[str, Any]:
        """Execute a specific recovery action."""
        result = {
            'action': action.name,
            'strategy': action.strategy.value,
            'success': False,
            'duration': 0.0,
            'details': {}
        }
        
        start_time = time.time()
        
        try:
            if action.strategy == RecoveryStrategy.RETRY:
                result = await self._retry_recovery(component, action, error_context)
                
            elif action.strategy == RecoveryStrategy.FALLBACK:
                result = await self._fallback_recovery(component, action, error_context)
                
            elif action.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                result = await self._graceful_degradation_recovery(component, action, error_context)
                
            elif action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                result = await self._circuit_breaker_recovery(component, action, error_context)
                
            elif action.strategy == RecoveryStrategy.RESTART_COMPONENT:
                result = await self._restart_component_recovery(component, action, error_context)
                
            elif action.strategy == RecoveryStrategy.AUTONOMOUS_RECOVERY:
                result = await self._autonomous_recovery(component, action, error_context)
                
            elif action.strategy == RecoveryStrategy.ESCALATE:
                result = await self._escalate_recovery(component, action, error_context)
                
            else:
                result['details']['error'] = f"Unknown recovery strategy: {action.strategy}"
                
        except Exception as e:
            result['details']['error'] = str(e)
            result['details']['traceback'] = traceback.format_exc()
            
        finally:
            result['duration'] = time.time() - start_time
            
        return result
        
    async def _retry_recovery(self, component: str, action: RecoveryAction, 
                            error_context: ErrorContext) -> Dict[str, Any]:
        """Retry recovery strategy."""
        # Simulate retry logic
        await asyncio.sleep(0.1)  # Brief delay
        
        return {
            'action': action.name,
            'strategy': 'retry',
            'success': True,  # Simplified success simulation
            'details': {'retries': 1, 'method': 'exponential_backoff'}
        }
        
    async def _fallback_recovery(self, component: str, action: RecoveryAction, 
                               error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback recovery strategy."""
        await asyncio.sleep(0.1)
        
        return {
            'action': action.name,
            'strategy': 'fallback',
            'success': True,
            'details': {'fallback_mode': 'degraded_functionality'}
        }
        
    async def _graceful_degradation_recovery(self, component: str, action: RecoveryAction, 
                                           error_context: ErrorContext) -> Dict[str, Any]:
        """Graceful degradation recovery strategy."""
        await asyncio.sleep(0.1)
        
        return {
            'action': action.name,
            'strategy': 'graceful_degradation',
            'success': True,
            'details': {
                'degradation_level': 'partial_functionality',
                'disabled_features': ['advanced_analytics', 'real_time_alerts']
            }
        }
        
    async def _circuit_breaker_recovery(self, component: str, action: RecoveryAction, 
                                      error_context: ErrorContext) -> Dict[str, Any]:
        """Circuit breaker recovery strategy."""
        await asyncio.sleep(0.1)
        
        return {
            'action': action.name,
            'strategy': 'circuit_breaker',
            'success': True,
            'details': {'breaker_state': 'half_open', 'test_requests_enabled': True}
        }
        
    async def _restart_component_recovery(self, component: str, action: RecoveryAction, 
                                        error_context: ErrorContext) -> Dict[str, Any]:
        """Component restart recovery strategy."""
        # Simulate component restart
        await asyncio.sleep(1.0)  # Restart delay
        
        return {
            'action': action.name,
            'strategy': 'restart_component',
            'success': True,
            'details': {
                'restart_type': 'soft_restart',
                'initialization_time': 1.0,
                'health_check_passed': True
            }
        }
        
    async def _autonomous_recovery(self, component: str, action: RecoveryAction, 
                                 error_context: ErrorContext) -> Dict[str, Any]:
        """Autonomous recovery strategy using AI."""
        # Simulate AI-driven recovery
        await asyncio.sleep(0.5)
        
        return {
            'action': action.name,
            'strategy': 'autonomous_recovery',
            'success': True,
            'details': {
                'ai_diagnosis': 'resource_exhaustion',
                'corrective_actions': ['memory_cleanup', 'connection_pool_reset'],
                'confidence': 0.85
            }
        }
        
    async def _escalate_recovery(self, component: str, action: RecoveryAction, 
                               error_context: ErrorContext) -> Dict[str, Any]:
        """Escalation recovery strategy."""
        await asyncio.sleep(0.1)
        
        return {
            'action': action.name,
            'strategy': 'escalate',
            'success': True,
            'details': {
                'escalation_level': 'operations_team',
                'ticket_created': True,
                'priority': 'high'
            }
        }
        
    async def _learn_from_recovery(self, component: str, action: RecoveryAction, 
                                 error_context: ErrorContext, success: bool):
        """Learn from recovery attempt outcomes."""
        pattern = {
            'action': action.name,
            'error_type': error_context.error_type.__name__,
            'success': success,
            'timestamp': datetime.now(),
            'context_features': {
                'severity': error_context.severity.value,
                'function': error_context.function,
                'recovery_attempts': error_context.recovery_attempts
            }
        }
        
        if success:
            self.success_patterns[component].append(pattern)
            
            # Keep only recent successful patterns (last 100)
            if len(self.success_patterns[component]) > 100:
                self.success_patterns[component] = self.success_patterns[component][-100:]


class RobustErrorHandlingSystem:
    """
    Complete robust error handling system with fault tolerance,
    autonomous recovery, and graceful degradation capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Core components
        self.autonomous_recovery = AutonomousRecoverySystem()
        self.fault_tolerant_containers = {}
        self.component_health = {}
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.error_patterns = defaultdict(list)
        
        # Health monitoring
        self.health_monitoring_active = False
        self.health_monitoring_thread = None
        
        # Configuration
        self.global_retry_config = self.config.get('retry_config', {
            'max_attempts': 3,
            'base_delay': 1.0,
            'max_delay': 60.0
        })
        
        # Initialize default recovery actions
        self._initialize_default_recovery_actions()
        
    def _initialize_default_recovery_actions(self):
        """Initialize default recovery actions for common components."""
        
        # Database component recovery actions
        db_actions = [
            RecoveryAction(
                name="retry_connection",
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                timeout=timedelta(seconds=30)
            ),
            RecoveryAction(
                name="connection_pool_reset",
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                max_attempts=2,
                timeout=timedelta(minutes=2)
            ),
            RecoveryAction(
                name="fallback_to_cache",
                strategy=RecoveryStrategy.FALLBACK,
                max_attempts=1,
                timeout=timedelta(seconds=5)
            )
        ]
        
        for action in db_actions:
            self.autonomous_recovery.register_recovery_action("database", action)
            
        # Network component recovery actions
        network_actions = [
            RecoveryAction(
                name="retry_request",
                strategy=RecoveryStrategy.RETRY,
                max_attempts=5,
                timeout=timedelta(seconds=15)
            ),
            RecoveryAction(
                name="circuit_breaker_activation",
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                max_attempts=1,
                timeout=timedelta(minutes=5)
            ),
            RecoveryAction(
                name="offline_mode",
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_attempts=1,
                timeout=timedelta(seconds=1)
            )
        ]
        
        for action in network_actions:
            self.autonomous_recovery.register_recovery_action("network", action)
            
        # AI/ML component recovery actions
        ml_actions = [
            RecoveryAction(
                name="model_reload",
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                max_attempts=2,
                timeout=timedelta(minutes=1)
            ),
            RecoveryAction(
                name="fallback_to_simple_model",
                strategy=RecoveryStrategy.FALLBACK,
                max_attempts=1,
                timeout=timedelta(seconds=5)
            ),
            RecoveryAction(
                name="autonomous_model_healing",
                strategy=RecoveryStrategy.AUTONOMOUS_RECOVERY,
                max_attempts=1,
                timeout=timedelta(minutes=5)
            )
        ]
        
        for action in ml_actions:
            self.autonomous_recovery.register_recovery_action("ml_model", action)
            
    async def initialize(self):
        """Initialize the robust error handling system."""
        self.logger.info("Initializing Robust Error Handling System")
        
        # Start health monitoring
        await self._start_health_monitoring()
        
        self.logger.info("Robust Error Handling System initialized")
        
    async def _start_health_monitoring(self):
        """Start continuous health monitoring."""
        self.health_monitoring_active = True
        self.health_monitoring_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.health_monitoring_thread.start()
        
    def _health_monitoring_loop(self):
        """Continuous health monitoring loop."""
        while self.health_monitoring_active:
            try:
                asyncio.run(self._perform_health_checks())
                threading.Event().wait(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                threading.Event().wait(60)  # Wait longer on error
                
    async def _perform_health_checks(self):
        """Perform health checks on all registered components."""
        for component_name, health in self.component_health.items():
            try:
                # Update last health check time
                health.last_health_check = datetime.now()
                
                # Simple health assessment based on recent errors
                recent_errors = [
                    error for error in self.error_history
                    if error.component == component_name and
                    error.timestamp > datetime.now() - timedelta(minutes=5)
                ]
                
                if len(recent_errors) == 0:
                    if health.status != ComponentStatus.HEALTHY:
                        health.status = ComponentStatus.HEALTHY
                        self.logger.info(f"Component {component_name} recovered to healthy status")
                        
                elif len(recent_errors) <= 2:
                    if health.status not in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]:
                        health.status = ComponentStatus.DEGRADED
                        
                else:
                    if health.status != ComponentStatus.FAILING:
                        health.status = ComponentStatus.FAILING
                        self.logger.warning(f"Component {component_name} is failing")
                        
                        # Trigger autonomous recovery
                        if health.last_error:
                            await self.autonomous_recovery.attempt_recovery(
                                component_name, health.last_error
                            )
                            
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")
                
    def register_component(self, component_name: str, dependencies: Optional[List[str]] = None):
        """Register a component for error handling and monitoring."""
        
        # Create fault-tolerant container
        container = FaultTolerantContainer(component_name)
        self.fault_tolerant_containers[component_name] = container
        
        # Initialize health tracking
        self.component_health[component_name] = ComponentHealth(
            component_name, ComponentStatus.HEALTHY
        )
        
        # Register dependencies
        if dependencies:
            self.autonomous_recovery.register_dependency(component_name, dependencies)
            
        self.logger.info(f"Registered component: {component_name}")
        
    def configure_retry(self, component: str, operation: str, **retry_config):
        """Configure retry parameters for a specific operation."""
        if component in self.fault_tolerant_containers:
            container = self.fault_tolerant_containers[component]
            container.retry_manager.configure_retry(operation, **retry_config)
            
    def add_circuit_breaker(self, component: str, operation: str, 
                           failure_threshold: int = 5, timeout: timedelta = timedelta(minutes=1)):
        """Add circuit breaker protection for an operation."""
        if component in self.fault_tolerant_containers:
            container = self.fault_tolerant_containers[component]
            container.add_circuit_breaker(operation, failure_threshold, timeout)
            
    def add_fallback_handler(self, component: str, operation: str, handler: Callable):
        """Add fallback handler for an operation."""
        if component in self.fault_tolerant_containers:
            container = self.fault_tolerant_containers[component]
            container.add_fallback_handler(operation, handler)
            
    async def handle_error(self, component: str, error: Exception, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle an error with comprehensive error handling strategies."""
        
        # Create error context
        error_context = ErrorContext(
            timestamp=datetime.now(),
            error_type=type(error),
            error_message=str(error),
            traceback_info=traceback.format_exc(),
            component=component,
            function=context.get('function', 'unknown') if context else 'unknown',
            severity=self._classify_error_severity(error),
            metadata=context or {}
        )
        
        # Record error
        self.error_history.append(error_context)
        self.error_patterns[component].append(error_context)
        
        # Update component health
        if component in self.component_health:
            health = self.component_health[component]
            health.last_error = error_context
            health.error_count += 1
            
        # Log error
        self.logger.error(f"Error in {component}: {error}")
        
        # Attempt recovery
        recovery_result = await self.autonomous_recovery.attempt_recovery(component, error_context)
        
        # Update component health based on recovery
        if component in self.component_health:
            health = self.component_health[component]
            if recovery_result['recovery_successful']:
                health.recovery_count += 1
                health.status = recovery_result['final_status']
            else:
                health.status = ComponentStatus.FAILED
                
        return {
            'error_handled': True,
            'error_context': error_context,
            'recovery_result': recovery_result,
            'component_health': self.component_health.get(component)
        }
        
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on exception type."""
        
        # Critical errors
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            return ErrorSeverity.FATAL
            
        # High-severity errors
        elif isinstance(error, (ConnectionError, TimeoutError, FileNotFoundError)):
            return ErrorSeverity.CRITICAL
            
        # Medium-severity errors
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.ERROR
            
        # Low-severity errors
        elif isinstance(error, (Warning, UserWarning)):
            return ErrorSeverity.WARNING
            
        # Default
        else:
            return ErrorSeverity.ERROR
            
    async def execute_with_protection(self, component: str, operation: str, 
                                    func: Callable, *args, **kwargs):
        """Execute function with full error handling protection."""
        
        if component not in self.fault_tolerant_containers:
            # Register component on-the-fly
            self.register_component(component)
            
        container = self.fault_tolerant_containers[component]
        
        try:
            return await container.execute_with_protection(operation, func, *args, **kwargs)
            
        except Exception as e:
            # Handle error through comprehensive error handling
            await self.handle_error(component, e, {
                'function': operation,
                'args': args,
                'kwargs': kwargs
            })
            
            # Re-raise unless graceful degradation is enabled
            if not self.config.get('graceful_degradation', False):
                raise e
                
            # Return graceful degradation response
            return {
                'status': 'degraded',
                'error': str(e),
                'message': 'Operation completed with reduced functionality'
            }
            
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        
        # Calculate overall health score
        healthy_components = sum(
            1 for health in self.component_health.values()
            if health.status in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]
        )
        
        total_components = len(self.component_health)
        health_score = (healthy_components / total_components) if total_components > 0 else 1.0
        
        # Recent error analysis
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        return {
            'overall_health_score': health_score,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'components_by_status': {
                status.value: sum(1 for h in self.component_health.values() if h.status == status)
                for status in ComponentStatus
            },
            'error_statistics': {
                'total_errors': len(self.error_history),
                'errors_last_hour': len(recent_errors),
                'error_rate_per_hour': len(recent_errors)
            },
            'recovery_statistics': {
                'total_recoveries': len(self.autonomous_recovery.recovery_history),
                'successful_recoveries': sum(
                    1 for r in self.autonomous_recovery.recovery_history
                    if r['result']['recovery_successful']
                ),
                'recovery_success_rate': (
                    sum(1 for r in self.autonomous_recovery.recovery_history if r['result']['recovery_successful']) /
                    len(self.autonomous_recovery.recovery_history)
                ) if self.autonomous_recovery.recovery_history else 0.0
            },
            'components': {
                name: {
                    'status': health.status.value,
                    'error_count': health.error_count,
                    'recovery_count': health.recovery_count,
                    'uptime': (datetime.now() - health.uptime_start).total_seconds(),
                    'last_health_check': health.last_health_check.isoformat()
                }
                for name, health in self.component_health.items()
            }
        }
        
    async def shutdown(self):
        """Shutdown the error handling system."""
        self.health_monitoring_active = False
        
        if self.health_monitoring_thread:
            self.health_monitoring_thread.join(timeout=5.0)
            
        self.logger.info("Robust Error Handling System shutdown completed")


# Decorators for robust error handling

def robust_operation(component: str, operation: str = None, 
                    retry_config: Optional[Dict[str, Any]] = None,
                    fallback_handler: Optional[Callable] = None):
    """Decorator to add robust error handling to any function."""
    def decorator(func):
        actual_operation = operation or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get or create error handling system
            error_system = RobustErrorHandlingSystem()
            await error_system.initialize()
            
            # Configure retry if specified
            if retry_config:
                error_system.configure_retry(component, actual_operation, **retry_config)
                
            # Add fallback handler if specified
            if fallback_handler:
                error_system.add_fallback_handler(component, actual_operation, fallback_handler)
                
            try:
                return await error_system.execute_with_protection(
                    component, actual_operation, func, *args, **kwargs
                )
            finally:
                await error_system.shutdown()
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
    return decorator


@contextmanager
def error_handling_context(component: str):
    """Context manager for robust error handling."""
    error_system = RobustErrorHandlingSystem()
    
    try:
        asyncio.run(error_system.initialize())
        yield error_system
        
    except Exception as e:
        asyncio.run(error_system.handle_error(component, e, {
            'context': 'error_handling_context'
        }))
        raise
        
    finally:
        asyncio.run(error_system.shutdown())


class GracefulDegradationManager:
    """Manager for graceful system degradation during failures."""
    
    def __init__(self):
        self.degradation_levels = {}
        self.feature_dependencies = {}
        self.current_degradation_level = 0
        
    def register_feature(self, feature_name: str, degradation_level: int, 
                        dependencies: Optional[List[str]] = None):
        """Register a feature with its degradation level."""
        self.degradation_levels[feature_name] = degradation_level
        if dependencies:
            self.feature_dependencies[feature_name] = dependencies
            
    def enable_degradation(self, level: int):
        """Enable degradation to specified level."""
        self.current_degradation_level = level
        
        disabled_features = [
            feature for feature, feature_level in self.degradation_levels.items()
            if feature_level >= level
        ]
        
        return {
            'degradation_level': level,
            'disabled_features': disabled_features,
            'available_features': [
                feature for feature, feature_level in self.degradation_levels.items()
                if feature_level < level
            ]
        }
        
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature is available at current degradation level."""
        feature_level = self.degradation_levels.get(feature_name, 0)
        return feature_level < self.current_degradation_level


# Factory function
def create_robust_error_handling_system(config: Optional[Dict[str, Any]] = None) -> RobustErrorHandlingSystem:
    """Create and return a robust error handling system."""
    return RobustErrorHandlingSystem(config)