"""Individual task execution agents with quantum-inspired capabilities."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import uuid
from enum import Enum
import numpy as np

from ..core.task import Task, TaskStatus
from ..core.config import PlannerConfig


logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskAgent:
    """Individual agent for executing tasks with quantum-inspired load balancing."""
    
    def __init__(self, agent_id: Optional[str] = None, config: Optional[PlannerConfig] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config or PlannerConfig.create_default()
        
        # Agent state
        self.state = AgentState.IDLE
        self.current_task: Optional[Task] = None
        self.capabilities: Dict[str, float] = {}
        
        # Quantum-inspired properties
        self.quantum_efficiency = 1.0  # Quantum coherence for task execution
        self.entanglement_partners: List[str] = []  # Entangled agents for coordination
        
        # Performance tracking
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = timedelta(0)
        self.average_execution_time = timedelta(0)
        self.success_rate = 1.0
        
        # Resource management
        self.resource_usage: Dict[str, float] = {}
        self.max_resources: Dict[str, float] = {"cpu": 1.0, "memory": 1.0, "io": 1.0}
        
        # Task execution
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.execution_lock = asyncio.Lock()
        
        logger.info(f"TaskAgent {self.agent_id} initialized")
    
    async def start(self):
        """Start the agent's task execution loop."""
        logger.info(f"Agent {self.agent_id} starting")
        self.state = AgentState.IDLE
        
        # Start task execution loop
        asyncio.create_task(self._execution_loop())
    
    async def stop(self):
        """Stop the agent gracefully."""
        logger.info(f"Agent {self.agent_id} stopping")
        self.state = AgentState.MAINTENANCE
        
        # Cancel current task if running
        if self.current_task:
            await self._cancel_current_task()
    
    async def assign_task(self, task: Task) -> bool:
        """Assign a task to this agent."""
        if self.state != AgentState.IDLE:
            return False
        
        if not self._can_execute_task(task):
            return False
        
        await self.task_queue.put(task)
        logger.info(f"Agent {self.agent_id} assigned task {task.id}")
        return True
    
    async def _execution_loop(self):
        """Main task execution loop."""
        while self.state != AgentState.MAINTENANCE:
            try:
                # Wait for task with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                # Execute task
                await self._execute_task(task)
                
            except asyncio.TimeoutError:
                # No task available, apply quantum decoherence
                await self._apply_quantum_decoherence()
                continue
                
            except Exception as e:
                logger.error(f"Agent {self.agent_id} execution error: {e}")
                self.state = AgentState.ERROR
                await asyncio.sleep(5)  # Recovery delay
                self.state = AgentState.IDLE
    
    async def _execute_task(self, task: Task):
        """Execute a single task."""
        async with self.execution_lock:
            self.current_task = task
            self.state = AgentState.WORKING
            task.status = TaskStatus.RUNNING
            task.actual_start = datetime.now()
            
            logger.info(f"Agent {self.agent_id} executing task {task.id}")
            
            try:
                # Quantum-enhanced execution
                await self._quantum_enhanced_execution(task)
                
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.actual_finish = datetime.now()
                
                self._update_performance_metrics(task, success=True)
                logger.info(f"Agent {self.agent_id} completed task {task.id}")
                
            except Exception as e:
                # Task failed
                task.status = TaskStatus.FAILED
                task.actual_finish = datetime.now()
                
                self._update_performance_metrics(task, success=False)
                logger.error(f"Agent {self.agent_id} failed task {task.id}: {e}")
                
            finally:
                self.current_task = None
                self.state = AgentState.IDLE
                self._release_resources()
    
    async def _quantum_enhanced_execution(self, task: Task):
        """Execute task with quantum-inspired optimizations."""
        # Allocate resources
        self._allocate_resources(task)
        
        # Apply quantum superposition for parallel execution paths
        execution_paths = self._generate_execution_paths(task)
        
        # Execute with quantum interference optimization
        if len(execution_paths) > 1:
            # Parallel execution with interference
            results = await asyncio.gather(
                *[self._execute_path(path, task) for path in execution_paths],
                return_exceptions=True
            )
            
            # Select best result using quantum measurement
            await self._quantum_measurement_selection(results, task)
        else:
            # Single path execution
            await self._execute_path(execution_paths[0], task)
    
    def _generate_execution_paths(self, task: Task) -> List[Dict[str, Any]]:
        """Generate multiple execution paths using quantum superposition."""
        paths = []
        
        # Base execution path
        base_path = {
            "method": "standard",
            "parallelism": 1,
            "optimization_level": "normal",
            "amplitude": 1.0
        }
        paths.append(base_path)
        
        # Quantum-inspired alternative paths
        if self.quantum_efficiency > 0.8:
            # High-performance path
            high_perf_path = {
                "method": "optimized",
                "parallelism": min(4, len(task.resources_required) + 1),
                "optimization_level": "high",
                "amplitude": 0.7
            }
            paths.append(high_perf_path)
        
        if task.priority.value >= 3:  # High priority tasks
            # Priority execution path
            priority_path = {
                "method": "priority",
                "parallelism": 2,
                "optimization_level": "priority",
                "amplitude": 0.5
            }
            paths.append(priority_path)
        
        return paths
    
    async def _execute_path(self, path: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Execute a specific execution path."""
        method = path["method"]
        parallelism = path["parallelism"]
        
        execution_time = task.estimated_duration.total_seconds()
        
        # Apply quantum efficiency
        efficiency_factor = self.quantum_efficiency * path["amplitude"]
        actual_time = execution_time / efficiency_factor
        
        # Simulate task execution
        await asyncio.sleep(min(actual_time, 0.1))  # Cap simulation time
        
        # Calculate success probability
        base_success = task.success_probability
        method_bonus = {"standard": 0, "optimized": 0.1, "priority": 0.05}[method]
        success_prob = min(1.0, base_success + method_bonus + efficiency_factor * 0.1)
        
        # Determine success
        success = np.random.random() < success_prob
        
        return {
            "success": success,
            "execution_time": actual_time,
            "method": method,
            "efficiency": efficiency_factor,
            "path": path
        }
    
    async def _quantum_measurement_selection(self, results: List[Any], task: Task):
        """Select best execution result using quantum measurement."""
        valid_results = [r for r in results if isinstance(r, dict) and not isinstance(r, Exception)]
        
        if not valid_results:
            raise Exception("All execution paths failed")
        
        # Calculate quantum probability amplitudes
        amplitudes = []
        for result in valid_results:
            success_weight = 2.0 if result["success"] else 0.1
            efficiency_weight = result["efficiency"]
            speed_weight = 1.0 / max(0.1, result["execution_time"])
            
            amplitude = success_weight * efficiency_weight * speed_weight
            amplitudes.append(amplitude)
        
        # Normalize amplitudes
        total_amplitude = sum(amplitudes)
        probabilities = [a / total_amplitude for a in amplitudes] if total_amplitude > 0 else [1.0 / len(amplitudes)] * len(amplitudes)
        
        # Quantum measurement (probabilistic selection)
        selected_idx = np.random.choice(len(valid_results), p=probabilities)
        selected_result = valid_results[selected_idx]
        
        # Apply selected result
        if not selected_result["success"]:
            raise Exception(f"Task execution failed with method {selected_result['method']}")
    
    def _allocate_resources(self, task: Task):
        """Allocate resources for task execution."""
        for resource, amount in task.resources_required.items():
            if resource not in self.max_resources:
                self.max_resources[resource] = 1.0
                
            self.resource_usage[resource] = min(
                self.max_resources[resource],
                self.resource_usage.get(resource, 0) + amount
            )
    
    def _release_resources(self):
        """Release all allocated resources."""
        self.resource_usage.clear()
    
    def _can_execute_task(self, task: Task) -> bool:
        """Check if agent can execute the given task."""
        # Check resource availability
        for resource, required in task.resources_required.items():
            available = self.max_resources.get(resource, 1.0) - self.resource_usage.get(resource, 0)
            if required > available:
                return False
        
        # Check capabilities
        for capability, required_level in task.metadata.get("required_capabilities", {}).items():
            agent_level = self.capabilities.get(capability, 0)
            if agent_level < required_level:
                return False
        
        return True
    
    def _update_performance_metrics(self, task: Task, success: bool):
        """Update agent performance metrics."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        if task.actual_start and task.actual_finish:
            execution_time = task.actual_finish - task.actual_start
            self.total_execution_time += execution_time
            
            total_tasks = self.tasks_completed + self.tasks_failed
            self.average_execution_time = self.total_execution_time / total_tasks
        
        total_tasks = self.tasks_completed + self.tasks_failed
        self.success_rate = self.tasks_completed / total_tasks if total_tasks > 0 else 1.0
        
        # Update quantum efficiency based on performance
        self._update_quantum_efficiency()
    
    def _update_quantum_efficiency(self):
        """Update quantum efficiency based on recent performance."""
        # Increase efficiency with successful executions
        if self.success_rate > 0.9:
            self.quantum_efficiency = min(1.0, self.quantum_efficiency + 0.01)
        elif self.success_rate < 0.7:
            self.quantum_efficiency = max(0.1, self.quantum_efficiency - 0.02)
        
        # Apply quantum decoherence over time
        self.quantum_efficiency *= 0.999
    
    async def _apply_quantum_decoherence(self):
        """Apply quantum decoherence when idle."""
        # Gradual efficiency decay when not working
        self.quantum_efficiency *= 0.995
        
        # Minimum efficiency threshold
        self.quantum_efficiency = max(0.5, self.quantum_efficiency)
    
    async def _cancel_current_task(self):
        """Cancel currently executing task."""
        if self.current_task:
            self.current_task.status = TaskStatus.CANCELLED
            self.current_task.actual_finish = datetime.now()
            logger.info(f"Agent {self.agent_id} cancelled task {self.current_task.id}")
    
    def add_entanglement(self, other_agent_id: str):
        """Create quantum entanglement with another agent."""
        if other_agent_id not in self.entanglement_partners:
            self.entanglement_partners.append(other_agent_id)
            logger.info(f"Agent {self.agent_id} entangled with {other_agent_id}")
    
    def remove_entanglement(self, other_agent_id: str):
        """Remove quantum entanglement with another agent."""
        if other_agent_id in self.entanglement_partners:
            self.entanglement_partners.remove(other_agent_id)
            logger.info(f"Agent {self.agent_id} disentangled from {other_agent_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "current_task": self.current_task.id if self.current_task else None,
            "quantum_efficiency": self.quantum_efficiency,
            "entanglement_partners": self.entanglement_partners,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time.total_seconds(),
            "resource_usage": self.resource_usage.copy(),
            "capabilities": self.capabilities.copy(),
        }
    
    def get_load_factor(self) -> float:
        """Calculate current load factor for load balancing."""
        base_load = 0.0 if self.state == AgentState.IDLE else 1.0
        
        # Resource utilization load
        resource_load = sum(
            usage / self.max_resources.get(resource, 1.0)
            for resource, usage in self.resource_usage.items()
        ) / max(1, len(self.resource_usage))
        
        # Queue load
        queue_load = self.task_queue.qsize() * 0.1
        
        # Efficiency factor (higher efficiency = lower effective load)
        efficiency_factor = 1.0 / max(0.1, self.quantum_efficiency)
        
        total_load = (base_load + resource_load + queue_load) * efficiency_factor
        return min(10.0, total_load)  # Cap maximum load
    
    def set_capabilities(self, capabilities: Dict[str, float]):
        """Set agent capabilities."""
        self.capabilities = capabilities.copy()
        logger.info(f"Agent {self.agent_id} capabilities updated: {capabilities}")
    
    def add_capability(self, capability: str, level: float):
        """Add or update a specific capability."""
        self.capabilities[capability] = level
        logger.info(f"Agent {self.agent_id} added capability {capability}: {level}")