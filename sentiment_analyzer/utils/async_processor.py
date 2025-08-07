"""
Async task processing and queue management for sentiment analysis
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class Task:
    id: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __lt__(self, other):
        # For priority queue ordering (higher priority value = higher priority)
        return self.priority.value > other.priority.value


class TaskQueue:
    """Async task queue with priority support"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue = asyncio.PriorityQueue(maxsize=max_size)
        self._results: Dict[str, TaskResult] = {}
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0
        }
    
    async def put_task(self, task: Task) -> bool:
        """Add task to queue"""
        try:
            await self._queue.put(task)
            self._stats["total_tasks"] += 1
            
            # Initialize result
            self._results[task.id] = TaskResult(
                task_id=task.id,
                status=TaskStatus.PENDING
            )
            
            logger.debug(f"Task {task.id} added to queue")
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Queue full, cannot add task {task.id}")
            return False
    
    async def get_task(self) -> Optional[Task]:
        """Get next task from queue"""
        try:
            task = await self._queue.get()
            
            # Update result status
            if task.id in self._results:
                self._results[task.id].status = TaskStatus.RUNNING
                self._results[task.id].started_at = datetime.now(timezone.utc)
            
            logger.debug(f"Task {task.id} retrieved from queue")
            return task
            
        except Exception as e:
            logger.error(f"Error getting task from queue: {e}")
            return None
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result by ID"""
        return self._results.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self._stats,
            "pending_tasks": self._queue.qsize(),
            "active_results": len(self._results)
        }
    
    def cleanup_old_results(self, max_age_hours: int = 24):
        """Clean up old task results"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        
        to_remove = [
            task_id for task_id, result in self._results.items()
            if result.created_at.timestamp() < cutoff_time
        ]
        
        for task_id in to_remove:
            del self._results[task_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old task results")


class TaskProcessor:
    """Async task processor with worker pool"""
    
    def __init__(self, 
                 num_workers: int = 4,
                 max_concurrent: int = 100,
                 enable_metrics: bool = True):
        self.num_workers = num_workers
        self.max_concurrent = max_concurrent
        self.enable_metrics = enable_metrics
        
        self.queue = TaskQueue()
        self.workers: List[asyncio.Task] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.running = False
        
        # Metrics
        self.metrics = {
            "total_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0.0,
            "worker_utilization": 0.0
        }
        
        logger.info(f"TaskProcessor initialized with {num_workers} workers")
    
    async def start(self):
        """Start the task processor"""
        if self.running:
            logger.warning("Task processor already running")
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Task processor started with {len(self.workers)} workers")
    
    async def stop(self):
        """Stop the task processor"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("Task processor stopped")
    
    async def _worker(self, worker_name: str):
        """Worker coroutine"""
        logger.debug(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get next task
                task = await self.queue.get_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process task with semaphore
                async with self.semaphore:
                    await self._process_task(task, worker_name)
                
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.debug(f"Worker {worker_name} stopped")
    
    async def _process_task(self, task: Task, worker_name: str):
        """Process a single task"""
        start_time = time.time()
        result = self.queue.get_result(task.id)
        
        if not result:
            logger.error(f"No result object found for task {task.id}")
            return
        
        try:
            logger.debug(f"Worker {worker_name} processing task {task.id}")
            
            # Execute task with timeout
            if asyncio.iscoroutinefunction(task.func):
                if task.timeout:
                    task_result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    task_result = await task.func(*task.args, **task.kwargs)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    if task.timeout:
                        task_result = await asyncio.wait_for(
                            loop.run_in_executor(
                                executor, 
                                lambda: task.func(*task.args, **task.kwargs)
                            ),
                            timeout=task.timeout
                        )
                    else:
                        task_result = await loop.run_in_executor(
                            executor,
                            lambda: task.func(*task.args, **task.kwargs)
                        )
            
            # Update result
            processing_time = (time.time() - start_time) * 1000
            result.status = TaskStatus.COMPLETED
            result.result = task_result
            result.completed_at = datetime.now(timezone.utc)
            result.processing_time_ms = processing_time
            
            # Update metrics
            self.metrics["total_processed"] += 1
            self.metrics["successful_tasks"] += 1
            self._update_avg_processing_time(processing_time)
            
            logger.debug(f"Task {task.id} completed in {processing_time:.2f}ms")
            
        except asyncio.TimeoutError:
            result.status = TaskStatus.FAILED
            result.error = "Task timeout"
            result.completed_at = datetime.now(timezone.utc)
            self.metrics["failed_tasks"] += 1
            logger.warning(f"Task {task.id} timed out")
            
        except Exception as e:
            # Check if we should retry
            if task.retries < task.max_retries:
                task.retries += 1
                logger.info(f"Retrying task {task.id} (attempt {task.retries}/{task.max_retries})")
                
                # Add delay before retry
                await asyncio.sleep(task.retry_delay * task.retries)
                
                # Re-queue the task
                await self.queue.put_task(task)
                return
            
            # Final failure
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            self.metrics["failed_tasks"] += 1
            
            logger.error(f"Task {task.id} failed: {e}")
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time metric"""
        current_avg = self.metrics["avg_processing_time"]
        total_processed = self.metrics["total_processed"]
        
        # Calculate running average
        new_avg = ((current_avg * (total_processed - 1)) + processing_time) / total_processed
        self.metrics["avg_processing_time"] = new_avg
    
    async def submit_task(self, 
                         func: Callable,
                         *args,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[float] = None,
                         max_retries: int = 3,
                         **kwargs) -> str:
        """Submit a task for processing"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        success = await self.queue.put_task(task)
        if not success:
            raise Exception("Failed to queue task - queue may be full")
        
        return task_id
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a specific task"""
        return self.queue.get_result(task_id)
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for a task to complete"""
        start_time = time.time()
        
        while True:
            result = self.queue.get_result(task_id)
            
            if not result:
                raise ValueError(f"Task {task_id} not found")
            
            if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics"""
        queue_stats = self.queue.get_stats()
        
        return {
            **self.metrics,
            "queue_stats": queue_stats,
            "num_workers": self.num_workers,
            "running": self.running,
            "active_workers": len([w for w in self.workers if not w.done()]),
            "max_concurrent": self.max_concurrent
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "status": "healthy" if self.running else "stopped",
            "workers_active": len([w for w in self.workers if not w.done()]),
            "workers_total": len(self.workers),
            "queue_size": self.queue._queue.qsize(),
            "metrics": self.get_metrics()
        }


# Global task processor instance
_global_processor: Optional[TaskProcessor] = None


def get_global_processor() -> TaskProcessor:
    """Get or create global task processor"""
    global _global_processor
    
    if _global_processor is None:
        import os
        
        num_workers = int(os.getenv("TASK_PROCESSOR_WORKERS", "4"))
        max_concurrent = int(os.getenv("TASK_PROCESSOR_MAX_CONCURRENT", "100"))
        
        _global_processor = TaskProcessor(
            num_workers=num_workers,
            max_concurrent=max_concurrent
        )
        
        logger.info("Global task processor initialized")
    
    return _global_processor


async def start_global_processor():
    """Start the global task processor"""
    processor = get_global_processor()
    if not processor.running:
        await processor.start()


async def stop_global_processor():
    """Stop the global task processor"""
    global _global_processor
    if _global_processor and _global_processor.running:
        await _global_processor.stop()
        _global_processor = None