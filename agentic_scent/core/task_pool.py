"""
Auto-scaling task pool for concurrent processing and load balancing.
"""

import asyncio
import time
import logging
from typing import Callable, Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import concurrent.futures
from collections import deque

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    queue_time: float = 0.0
    worker_id: Optional[str] = None


@dataclass
class PoolMetrics:
    """Task pool metrics."""
    active_workers: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_execution_time: float
    average_queue_time: float
    cpu_utilization: float
    memory_utilization: float


class TaskWrapper:
    """Wrapper for queued tasks."""
    
    def __init__(self, task_id: str, func: Callable, args: tuple, kwargs: dict,
                 priority: TaskPriority = TaskPriority.MEDIUM, timeout: Optional[float] = None):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.priority = priority
        self.timeout = timeout
        self.queued_time = time.time()
        self.started_time: Optional[float] = None
        self.completed_time: Optional[float] = None


class AutoScalingTaskPool:
    """
    Auto-scaling task pool with load balancing and priority queues.
    """
    
    def __init__(self, min_workers: int = 2, max_workers: int = 10, 
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3,
                 scale_check_interval: float = 30.0):
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval
        
        # Task queues by priority
        self.task_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        
        # Worker management
        self.workers: Dict[str, asyncio.Task] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        
        # Results and metrics
        self.results: Dict[str, TaskResult] = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Control flags
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._metrics_history: List[PoolMetrics] = []
        
        logger.info(f"TaskPool initialized: {min_workers}-{max_workers} workers")
    
    async def start(self):
        """Start the task pool."""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start initial workers
        for i in range(self.min_workers):
            await self._add_worker()
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"TaskPool started with {len(self.workers)} workers")
    
    async def stop(self):
        """Stop the task pool."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop all workers
        for worker_task in list(self.workers.values()):
            worker_task.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers.values(), return_exceptions=True)
        
        self.workers.clear()
        logger.info("TaskPool stopped")
    
    async def submit_task(self, task_id: str, func: Callable, *args,
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         timeout: Optional[float] = None, **kwargs) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_id: Unique task identifier
            func: Function to execute
            *args: Function arguments
            priority: Task priority
            timeout: Execution timeout
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for result retrieval
        """
        
        if not self.is_running:
            raise RuntimeError("TaskPool is not running")
        
        task = TaskWrapper(task_id, func, args, kwargs, priority, timeout)
        self.task_queues[priority].append(task)
        
        logger.debug(f"Task submitted: {task_id} (priority: {priority.name})")
        return task_id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get task result, waiting if necessary."""
        
        start_time = time.time()
        
        while task_id not in self.results:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
        
        return self.results[task_id]
    
    async def get_result_nowait(self, task_id: str) -> Optional[TaskResult]:
        """Get task result if available."""
        return self.results.get(task_id)
    
    async def _add_worker(self) -> str:
        """Add a new worker."""
        worker_id = f"worker_{len(self.workers):03d}"
        worker_task = asyncio.create_task(self._worker_loop(worker_id))
        
        self.workers[worker_id] = worker_task
        self.worker_stats[worker_id] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "started_time": time.time()
        }
        
        logger.debug(f"Added worker: {worker_id}")
        return worker_id
    
    async def _remove_worker(self, worker_id: str):
        """Remove a worker."""
        if worker_id in self.workers:
            self.workers[worker_id].cancel()
            try:
                await self.workers[worker_id]
            except asyncio.CancelledError:
                pass
            
            del self.workers[worker_id]
            del self.worker_stats[worker_id]
            
            logger.debug(f"Removed worker: {worker_id}")
    
    async def _worker_loop(self, worker_id: str):
        """Worker execution loop."""
        
        logger.debug(f"Worker {worker_id} started")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                # Get next task from priority queues
                task = await self._get_next_task()
                
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                await self._execute_task(worker_id, task)
        
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} cancelled")
            raise
        
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
        
        finally:
            logger.debug(f"Worker {worker_id} stopped")
    
    async def _get_next_task(self) -> Optional[TaskWrapper]:
        """Get next task from priority queues."""
        
        # Check queues in priority order (CRITICAL -> LOW)
        for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
            queue = self.task_queues[priority]
            if queue:
                return queue.popleft()
        
        return None
    
    async def _execute_task(self, worker_id: str, task: TaskWrapper):
        """Execute a single task."""
        
        task.started_time = time.time()
        queue_time = task.started_time - task.queued_time
        
        try:
            # Execute with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._run_task_function(task.func, *task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                result = await self._run_task_function(task.func, *task.args, **task.kwargs)
            
            task.completed_time = time.time()
            execution_time = task.completed_time - task.started_time
            
            # Store result
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                queue_time=queue_time,
                worker_id=worker_id
            )
            
            self.results[task.task_id] = task_result
            self.completed_tasks += 1
            
            # Update worker stats
            stats = self.worker_stats[worker_id]
            stats["tasks_completed"] += 1
            stats["total_execution_time"] += execution_time
            
            logger.debug(f"Task completed: {task.task_id} in {execution_time:.3f}s")
        
        except Exception as e:
            task.completed_time = time.time()
            execution_time = task.completed_time - task.started_time
            
            # Store error result
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time=execution_time,
                queue_time=queue_time,
                worker_id=worker_id
            )
            
            self.results[task.task_id] = task_result
            self.failed_tasks += 1
            
            # Update worker stats
            stats = self.worker_stats[worker_id]
            stats["tasks_failed"] += 1
            
            logger.error(f"Task failed: {task.task_id} - {e}")
    
    async def _run_task_function(self, func: Callable, *args, **kwargs) -> Any:
        """Run task function (async or sync)."""
        
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
    
    async def _monitoring_loop(self):
        """Monitor pool performance and auto-scale."""
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                self._metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self._metrics_history) > 100:
                    self._metrics_history = self._metrics_history[-50:]
                
                # Auto-scaling decisions
                await self._auto_scale(metrics)
                
                logger.debug(f"Pool metrics: workers={metrics.active_workers}, "
                           f"queued={metrics.queued_tasks}, utilization={metrics.cpu_utilization:.2f}")
            
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            await asyncio.sleep(self.scale_check_interval)
    
    async def _collect_metrics(self) -> PoolMetrics:
        """Collect current pool metrics."""
        
        # Task queue sizes
        queued_tasks = sum(len(queue) for queue in self.task_queues.values())
        
        # Calculate averages
        all_results = list(self.results.values())
        if all_results:
            avg_execution = sum(r.execution_time for r in all_results) / len(all_results)
            avg_queue = sum(r.queue_time for r in all_results) / len(all_results)
        else:
            avg_execution = avg_queue = 0.0
        
        # System utilization (mock values - in production use psutil)
        try:
            import psutil
            cpu_utilization = psutil.cpu_percent()
            memory_utilization = psutil.virtual_memory().percent
        except ImportError:
            cpu_utilization = min(95.0, len(self.workers) * 10.0)  # Mock
            memory_utilization = 65.0  # Mock
        
        return PoolMetrics(
            active_workers=len(self.workers),
            queued_tasks=queued_tasks,
            completed_tasks=self.completed_tasks,
            failed_tasks=self.failed_tasks,
            average_execution_time=avg_execution,
            average_queue_time=avg_queue,
            cpu_utilization=cpu_utilization,
            memory_utilization=memory_utilization
        )
    
    async def _auto_scale(self, metrics: PoolMetrics):
        """Auto-scale workers based on metrics."""
        
        current_workers = metrics.active_workers
        queue_load = metrics.queued_tasks / max(current_workers, 1)
        
        # Scale up conditions
        should_scale_up = (
            queue_load > self.scale_up_threshold and
            current_workers < self.max_workers and
            metrics.cpu_utilization < 90.0  # Don't scale up if CPU is too high
        )
        
        # Scale down conditions  
        should_scale_down = (
            queue_load < self.scale_down_threshold and
            current_workers > self.min_workers and
            metrics.queued_tasks == 0
        )
        
        if should_scale_up:
            await self._add_worker()
            logger.info(f"Scaled up: {current_workers} -> {len(self.workers)} workers")
        
        elif should_scale_down:
            # Remove least utilized worker
            worker_to_remove = min(
                self.worker_stats.keys(),
                key=lambda w: self.worker_stats[w]["tasks_completed"]
            )
            await self._remove_worker(worker_to_remove)
            logger.info(f"Scaled down: {current_workers} -> {len(self.workers)} workers")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status."""
        
        metrics = asyncio.create_task(self._collect_metrics()) if self.is_running else None
        
        return {
            "is_running": self.is_running,
            "active_workers": len(self.workers),
            "worker_ids": list(self.workers.keys()),
            "queued_tasks_by_priority": {
                priority.name: len(queue) 
                for priority, queue in self.task_queues.items()
            },
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "pending_results": len(self.results),
            "configuration": {
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold
            }
        }


async def create_optimized_task_pool(min_workers: int = 2, max_workers: int = 10) -> AutoScalingTaskPool:
    """Create and start optimized task pool."""
    
    pool = AutoScalingTaskPool(
        min_workers=min_workers,
        max_workers=max_workers,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        scale_check_interval=15.0
    )
    
    await pool.start()
    return pool