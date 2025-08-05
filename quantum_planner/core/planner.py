"""Main quantum task planner implementation."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from .task import Task, TaskStatus, TaskPriority
from .config import PlannerConfig
from ..algorithms.quantum_optimizer import QuantumOptimizer
from ..algorithms.scheduler import QuantumScheduler
from ..agents.coordinator import TaskCoordinator


logger = logging.getLogger(__name__)


class QuantumTaskPlanner:
    """Main quantum-inspired task planning system."""
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig.create_default()
        self.tasks: Dict[str, Task] = {}
        self.task_graph: Dict[str, Set[str]] = {}
        self.completed_tasks: Set[str] = set()
        self.running_tasks: Set[str] = set()
        
        # Initialize components
        self.optimizer = QuantumOptimizer(self.config.get_quantum_params())
        self.scheduler = QuantumScheduler(self.config.get_scheduling_params())
        self.coordinator = TaskCoordinator(self.config)
        
        # State tracking
        self.is_running = False
        self.current_schedule: Optional[Dict[str, datetime]] = None
        self.resource_usage: Dict[str, float] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        logger.info(f"QuantumTaskPlanner initialized with {len(self.tasks)} tasks")
    
    async def add_task(self, task: Task) -> str:
        """Add a task to the planner."""
        if task.id in self.tasks:
            raise ValueError(f"Task {task.id} already exists")
        
        self.tasks[task.id] = task
        self.task_graph[task.id] = set(dep.task_id for dep in task.dependencies)
        
        # Check for circular dependencies
        if self._has_circular_dependency(task.id):
            del self.tasks[task.id]
            del self.task_graph[task.id]
            raise ValueError(f"Adding task {task.id} would create circular dependency")
        
        # Update quantum entanglements
        for entangled_id in task.entangled_tasks:
            if entangled_id in self.tasks:
                self.tasks[entangled_id].add_entanglement(task.id)
        
        logger.info(f"Added task {task.id}: {task.name}")
        return task.id
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove a task from the planner."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Can't remove running or completed tasks
        if task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]:
            raise ValueError(f"Cannot remove task {task_id} with status {task.status}")
        
        # Remove dependencies on this task
        for other_task in self.tasks.values():
            other_task.dependencies = [
                dep for dep in other_task.dependencies 
                if dep.task_id != task_id
            ]
            other_task.entangled_tasks.discard(task_id)
        
        del self.tasks[task_id]
        del self.task_graph[task_id]
        
        logger.info(f"Removed task {task_id}")
        return True
    
    async def update_task(self, task_id: str, **updates) -> bool:
        """Update task properties."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        task.updated_at = datetime.now()
        logger.info(f"Updated task {task_id}")
        return True
    
    async def optimize_schedule(self) -> Dict[str, datetime]:
        """Generate optimal schedule using quantum optimization."""
        if not self.tasks:
            return {}
        
        logger.info("Starting quantum optimization...")
        
        # Prepare task data for optimization
        task_data = self._prepare_optimization_data()
        
        # Run quantum optimization
        optimal_assignment = await self.optimizer.optimize(task_data)
        
        # Convert to schedule
        schedule = self.scheduler.create_schedule(
            self.tasks, 
            optimal_assignment, 
            self.completed_tasks
        )
        
        self.current_schedule = schedule
        logger.info(f"Generated schedule for {len(schedule)} tasks")
        
        return schedule
    
    async def execute_schedule(self, schedule: Optional[Dict[str, datetime]] = None) -> Dict[str, Any]:
        """Execute the task schedule using multi-agent coordination."""
        if schedule is None:
            schedule = await self.optimize_schedule()
        
        if not schedule:
            return {"status": "no_tasks", "completed": 0, "failed": 0}
        
        logger.info(f"Executing schedule with {len(schedule)} tasks")
        self.is_running = True
        
        try:
            results = await self.coordinator.execute_tasks(
                schedule, 
                self.tasks, 
                self.completed_tasks,
                self.running_tasks
            )
            
            # Update task statuses
            for task_id, result in results.items():
                task = self.tasks.get(task_id)
                if task:
                    if result["success"]:
                        task.status = TaskStatus.COMPLETED
                        task.actual_finish = result["finish_time"]
                        self.completed_tasks.add(task_id)
                    else:
                        task.status = TaskStatus.FAILED
                
                self.running_tasks.discard(task_id)
            
            # Calculate performance metrics
            self.performance_metrics = self._calculate_metrics(results)
            
            return {
                "status": "completed",
                "completed": len([r for r in results.values() if r["success"]]),
                "failed": len([r for r in results.values() if not r["success"]]),
                "metrics": self.performance_metrics
            }
            
        finally:
            self.is_running = False
    
    async def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to be executed."""
        ready_tasks = []
        for task in self.tasks.values():
            if task.is_ready(self.completed_tasks) and task.status == TaskStatus.PENDING:
                ready_tasks.append(task)
        
        # Sort by quantum priority
        ready_tasks.sort(key=lambda t: t.calculate_quantum_priority(), reverse=True)
        return ready_tasks
    
    async def get_critical_path(self) -> List[str]:
        """Calculate the critical path through the task network."""
        return self.scheduler.find_critical_path(self.tasks, self.completed_tasks)
    
    async def predict_completion_time(self) -> Optional[datetime]:
        """Predict when all tasks will be completed."""
        if not self.current_schedule:
            await self.optimize_schedule()
        
        return self.scheduler.predict_completion_time(
            self.tasks, 
            self.current_schedule, 
            self.completed_tasks
        )
    
    async def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        utilization = {}
        
        for task in self.tasks.values():
            if task.status == TaskStatus.RUNNING:
                for resource, amount in task.resources_required.items():
                    utilization[resource] = utilization.get(resource, 0) + amount
        
        return utilization
    
    async def get_status(self) -> Dict[str, Any]:
        """Get overall planner status."""
        total_tasks = len(self.tasks)
        completed = len(self.completed_tasks)
        running = len(self.running_tasks)
        pending = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
        failed = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        
        return {
            "is_running": self.is_running,
            "total_tasks": total_tasks,
            "completed": completed,
            "running": running,
            "pending": pending,
            "failed": failed,
            "completion_rate": completed / total_tasks if total_tasks > 0 else 0,
            "resource_utilization": await self.get_resource_utilization(),
            "estimated_completion": await self.predict_completion_time(),
            "performance_metrics": self.performance_metrics
        }
    
    def _prepare_optimization_data(self) -> Dict[str, Any]:
        """Prepare task data for quantum optimization."""
        task_list = list(self.tasks.values())
        
        # Create adjacency matrix for dependencies
        n_tasks = len(task_list)
        task_indices = {task.id: i for i, task in enumerate(task_list)}
        adjacency_matrix = np.zeros((n_tasks, n_tasks))
        
        for i, task in enumerate(task_list):
            for dep in task.dependencies:
                if dep.task_id in task_indices:
                    j = task_indices[dep.task_id]
                    adjacency_matrix[i][j] = 1
        
        # Create priority vector
        priorities = np.array([task.calculate_quantum_priority() for task in task_list])
        
        # Create duration vector
        durations = np.array([task.estimated_duration.total_seconds() for task in task_list])
        
        return {
            "tasks": task_list,
            "task_indices": task_indices,
            "adjacency_matrix": adjacency_matrix,
            "priorities": priorities,
            "durations": durations,
            "n_tasks": n_tasks
        }
    
    def _has_circular_dependency(self, start_task_id: str) -> bool:
        """Check for circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep_id in self.task_graph.get(task_id, set()):
                if dfs(dep_id):
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        return dfs(start_task_id)
    
    def _calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not results:
            return {}
        
        total_tasks = len(results)
        successful = len([r for r in results.values() if r["success"]])
        failed = total_tasks - successful
        
        execution_times = [
            (r["finish_time"] - r["start_time"]).total_seconds()
            for r in results.values() if r["success"]
        ]
        
        return {
            "success_rate": successful / total_tasks,
            "failure_rate": failed / total_tasks,
            "average_execution_time": np.mean(execution_times) if execution_times else 0,
            "total_execution_time": sum(execution_times),
            "throughput": successful / (max(execution_times) if execution_times else 1),
        }