"""Quantum-inspired scheduling algorithms."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
import numpy as np
from collections import defaultdict, deque

from ..core.task import Task, TaskStatus


logger = logging.getLogger(__name__)


class QuantumScheduler:
    """Quantum-inspired task scheduler with temporal optimization."""
    
    def __init__(self, params: Dict[str, Any]):
        self.time_horizon_days = params.get("time_horizon_days", 30)
        self.resource_buffer_percent = params.get("resource_buffer_percent", 0.1)
        self.priority_weight = params.get("priority_weight", 1.0)
        self.deadline_weight = params.get("deadline_weight", 2.0)
        self.max_concurrent_tasks = params.get("max_concurrent_tasks", 10)
        
        logger.info(f"QuantumScheduler initialized with {self.time_horizon_days} day horizon")
    
    def create_schedule(
        self, 
        tasks: Dict[str, Task], 
        optimization_result: Dict[str, Any],
        completed_tasks: Set[str]
    ) -> Dict[str, datetime]:
        """Create temporal schedule from optimization results."""
        logger.info("Creating temporal schedule from quantum optimization")
        
        if not tasks or not optimization_result:
            return {}
        
        priority_assignment = optimization_result.get("priority_assignment", {})
        
        # Sort tasks by quantum priority
        sorted_tasks = sorted(
            [t for t in tasks.values() if t.id not in completed_tasks],
            key=lambda t: priority_assignment.get(t.id, {}).get("quantum_priority", 0),
            reverse=True
        )
        
        # Create schedule using critical path method with quantum priorities
        schedule = {}
        current_time = datetime.now()
        resource_timeline = defaultdict(list)
        
        # First pass: calculate earliest start times respecting dependencies
        earliest_times = self._calculate_earliest_times(tasks, completed_tasks, current_time)
        
        # Second pass: schedule tasks considering resources and quantum priorities
        for task in sorted_tasks:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
                continue
            
            # Find optimal start time
            optimal_start = self._find_optimal_start_time(
                task, 
                earliest_times,
                resource_timeline,
                priority_assignment.get(task.id, {})
            )
            
            if optimal_start:
                schedule[task.id] = optimal_start
                self._update_resource_timeline(task, optimal_start, resource_timeline)
        
        logger.info(f"Created schedule for {len(schedule)} tasks")
        return schedule
    
    def _calculate_earliest_times(
        self, 
        tasks: Dict[str, Task], 
        completed_tasks: Set[str],
        start_time: datetime
    ) -> Dict[str, datetime]:
        """Calculate earliest possible start times using topological sort."""
        earliest_times = {}
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        # Build dependency graph
        for task in tasks.values():
            if task.id in completed_tasks:
                earliest_times[task.id] = start_time
                continue
                
            for dep in task.dependencies:
                if dep.task_id not in completed_tasks:
                    graph[dep.task_id].append((task.id, dep.lag))
                    in_degree[task.id] += 1
        
        # Topological sort with time calculation
        queue = deque()
        
        # Initialize tasks with no dependencies
        for task in tasks.values():
            if task.id not in completed_tasks and in_degree[task.id] == 0:
                earliest_time = max(start_time, task.earliest_start or start_time)
                earliest_times[task.id] = earliest_time
                queue.append(task.id)
        
        # Process tasks in dependency order
        while queue:
            current_id = queue.popleft()
            current_task = tasks[current_id]
            current_earliest = earliest_times[current_id]
            current_finish = current_earliest + current_task.estimated_duration
            
            for next_id, lag in graph[current_id]:
                if next_id in completed_tasks:
                    continue
                    
                next_task = tasks[next_id]
                dependency_finish_time = current_finish + lag
                next_earliest = max(
                    dependency_finish_time,
                    next_task.earliest_start or start_time
                )
                
                if next_id not in earliest_times:
                    earliest_times[next_id] = next_earliest
                else:
                    earliest_times[next_id] = max(earliest_times[next_id], next_earliest)
                
                in_degree[next_id] -= 1
                if in_degree[next_id] == 0:
                    queue.append(next_id)
        
        return earliest_times
    
    def _find_optimal_start_time(
        self,
        task: Task,
        earliest_times: Dict[str, datetime],
        resource_timeline: Dict[str, List[Tuple[datetime, datetime, float]]],
        quantum_info: Dict[str, Any]
    ) -> Optional[datetime]:
        """Find optimal start time considering resources and quantum properties."""
        earliest_start = earliest_times.get(task.id, datetime.now())
        latest_start = task.latest_finish - task.estimated_duration if task.latest_finish else None
        
        # Quantum-enhanced time slot evaluation
        quantum_priority = quantum_info.get("quantum_priority", 0.5)
        phase = quantum_info.get("phase", 0)
        
        # Search for available time slots
        current_time = earliest_start
        max_search_time = current_time + timedelta(days=self.time_horizon_days)
        
        best_time = None
        best_score = float('-inf')
        
        while current_time <= max_search_time:
            if latest_start and current_time > latest_start:
                break
            
            # Check resource availability
            if self._check_resource_availability(task, current_time, resource_timeline):
                # Calculate quantum-enhanced score
                score = self._calculate_time_slot_score(
                    task, current_time, quantum_priority, phase
                )
                
                if score > best_score:
                    best_score = score
                    best_time = current_time
            
            # Move to next time slot (1 hour granularity)
            current_time += timedelta(hours=1)
        
        return best_time
    
    def _check_resource_availability(
        self,
        task: Task,
        start_time: datetime,
        resource_timeline: Dict[str, List[Tuple[datetime, datetime, float]]]
    ) -> bool:
        """Check if resources are available for the task at given time."""
        end_time = start_time + task.estimated_duration
        
        for resource, required_amount in task.resources_required.items():
            current_usage = 0
            
            for usage_start, usage_end, usage_amount in resource_timeline[resource]:
                # Check for overlap
                if not (end_time <= usage_start or start_time >= usage_end):
                    current_usage += usage_amount
            
            # Apply resource buffer
            max_capacity = 1.0  # Assume normalized resource capacity
            available_capacity = max_capacity * (1 - self.resource_buffer_percent)
            
            if current_usage + required_amount > available_capacity:
                return False
        
        return True
    
    def _calculate_time_slot_score(
        self,
        task: Task, 
        start_time: datetime,
        quantum_priority: float,
        phase: float
    ) -> float:
        """Calculate quantum-enhanced score for time slot."""
        base_score = 0
        
        # Priority component
        base_score += self.priority_weight * quantum_priority
        
        # Deadline pressure component
        if task.latest_finish:
            time_to_deadline = (task.latest_finish - start_time).total_seconds()
            deadline_pressure = 1.0 / (1.0 + time_to_deadline / 3600)  # Normalize by hours
            base_score += self.deadline_weight * deadline_pressure
        
        # Quantum phase interference
        time_offset = (start_time - datetime.now()).total_seconds() / 3600  # Hours from now
        phase_factor = np.cos(phase + time_offset * 0.1)  # Oscillating preference
        base_score += 0.5 * phase_factor
        
        # Early start bonus
        current_time = datetime.now()
        if start_time <= current_time + timedelta(hours=1):
            base_score += 1.0
        
        return base_score
    
    def _update_resource_timeline(
        self,
        task: Task,
        start_time: datetime,
        resource_timeline: Dict[str, List[Tuple[datetime, datetime, float]]]
    ):
        """Update resource timeline with scheduled task."""
        end_time = start_time + task.estimated_duration
        
        for resource, amount in task.resources_required.items():
            resource_timeline[resource].append((start_time, end_time, amount))
    
    def find_critical_path(
        self, 
        tasks: Dict[str, Task], 
        completed_tasks: Set[str]
    ) -> List[str]:
        """Find critical path through task network."""
        logger.info("Calculating critical path")
        
        # Calculate forward pass (earliest times)
        earliest_times = self._calculate_earliest_times(tasks, completed_tasks, datetime.now())
        
        # Calculate backward pass (latest times)
        latest_times = self._calculate_latest_times(tasks, completed_tasks, earliest_times)
        
        # Find critical tasks (where earliest == latest)
        critical_tasks = []
        for task_id, task in tasks.items():
            if task_id in completed_tasks:
                continue
                
            earliest = earliest_times.get(task_id, datetime.now())
            latest = latest_times.get(task_id, datetime.now())
            
            # Tasks with zero slack are critical
            slack = (latest - earliest).total_seconds()
            if slack <= 3600:  # 1 hour tolerance
                critical_tasks.append(task_id)
        
        # Sort critical tasks by dependency order
        critical_path = self._sort_by_dependencies(critical_tasks, tasks)
        
        logger.info(f"Critical path contains {len(critical_path)} tasks")
        return critical_path
    
    def _calculate_latest_times(
        self,
        tasks: Dict[str, Task],
        completed_tasks: Set[str], 
        earliest_times: Dict[str, datetime]
    ) -> Dict[str, datetime]:
        """Calculate latest start times using backward pass."""
        latest_times = {}
        
        # Find project end time
        project_end = max(
            earliest_times[task_id] + task.estimated_duration
            for task_id, task in tasks.items()
            if task_id not in completed_tasks and task_id in earliest_times
        ) if earliest_times else datetime.now()
        
        # Initialize latest times for tasks with no successors
        for task_id, task in tasks.items():
            if task_id in completed_tasks:
                continue
                
            has_successors = any(
                task_id in [dep.task_id for dep in other_task.dependencies]
                for other_task in tasks.values()
                if other_task.id not in completed_tasks
            )
            
            if not has_successors:
                latest_finish = task.latest_finish or project_end
                latest_times[task_id] = latest_finish - task.estimated_duration
        
        # Backward pass
        changed = True
        while changed:
            changed = False
            
            for task_id, task in tasks.items():
                if task_id in completed_tasks or task_id in latest_times:
                    continue
                
                # Find successors and their latest start times
                successor_constraints = []
                for other_task in tasks.values():
                    if other_task.id in completed_tasks:
                        continue
                        
                    for dep in other_task.dependencies:
                        if dep.task_id == task_id and other_task.id in latest_times:
                            constraint_time = latest_times[other_task.id] - dep.lag
                            successor_constraints.append(constraint_time)
                
                if successor_constraints:
                    latest_finish = min(successor_constraints)
                    latest_times[task_id] = latest_finish - task.estimated_duration
                    changed = True
        
        return latest_times
    
    def _sort_by_dependencies(self, task_ids: List[str], tasks: Dict[str, Task]) -> List[str]:
        """Sort tasks by dependency order using topological sort."""
        # Build subgraph with only critical tasks
        subgraph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for task_id in task_ids:
            task = tasks[task_id]
            for dep in task.dependencies:
                if dep.task_id in task_ids:
                    subgraph[dep.task_id].append(task_id)
                    in_degree[task_id] += 1
        
        # Topological sort
        queue = deque([task_id for task_id in task_ids if in_degree[task_id] == 0])
        sorted_tasks = []
        
        while queue:
            current = queue.popleft()
            sorted_tasks.append(current)
            
            for successor in subgraph[current]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        return sorted_tasks
    
    def predict_completion_time(
        self,
        tasks: Dict[str, Task],
        schedule: Dict[str, datetime],
        completed_tasks: Set[str]
    ) -> Optional[datetime]:
        """Predict when all tasks will be completed."""
        if not schedule:
            return None
        
        latest_completion = None
        
        for task_id, start_time in schedule.items():
            if task_id in completed_tasks:
                continue
                
            task = tasks.get(task_id)
            if not task:
                continue
            
            completion_time = start_time + task.estimated_duration
            
            if latest_completion is None or completion_time > latest_completion:
                latest_completion = completion_time
        
        return latest_completion
    
    async def optimize_schedule_realtime(
        self,
        tasks: Dict[str, Task],
        current_schedule: Dict[str, datetime],
        completed_tasks: Set[str],
        running_tasks: Set[str]
    ) -> Dict[str, datetime]:
        """Optimize schedule in real-time based on current progress."""
        logger.info("Performing real-time schedule optimization")
        
        # Identify tasks that need rescheduling
        tasks_to_reschedule = set()
        current_time = datetime.now()
        
        for task_id, scheduled_time in current_schedule.items():
            if task_id in completed_tasks or task_id in running_tasks:
                continue
                
            task = tasks.get(task_id)
            if not task:
                continue
            
            # Check if task is overdue or dependencies have changed
            if scheduled_time < current_time or not task.is_ready(completed_tasks):
                tasks_to_reschedule.add(task_id)
        
        if not tasks_to_reschedule:
            return current_schedule
        
        # Reschedule affected tasks
        updated_schedule = current_schedule.copy()
        
        # Remove tasks that need rescheduling
        for task_id in tasks_to_reschedule:
            updated_schedule.pop(task_id, None)
        
        # Create new schedule for affected tasks
        affected_tasks = {
            task_id: task for task_id, task in tasks.items()
            if task_id in tasks_to_reschedule
        }
        
        # Simple rescheduling (in real implementation, would use full optimization)
        earliest_times = self._calculate_earliest_times(affected_tasks, completed_tasks, current_time)
        
        for task_id in sorted(tasks_to_reschedule):
            task = tasks[task_id]
            if task.is_ready(completed_tasks):
                earliest_start = max(
                    current_time,
                    earliest_times.get(task_id, current_time),
                    task.earliest_start or current_time
                )
                updated_schedule[task_id] = earliest_start
        
        logger.info(f"Rescheduled {len(tasks_to_reschedule)} tasks")
        return updated_schedule