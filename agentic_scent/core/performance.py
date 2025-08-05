"""
Performance optimization and scaling systems.
"""

import asyncio
import logging
import time
import functools
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
import threading
import multiprocessing
import psutil

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    REDIS = "redis" 
    DISK = "disk"


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    cache_hit: bool = False
    concurrent_tasks: int = 0
    queue_size: int = 0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + self.ttl
    
    def touch(self):
        """Update last access time."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class AsyncCache:
    """
    High-performance async cache with multiple storage levels.
    """
    
    def __init__(self, 
                 max_memory_size_mb: int = 256,
                 default_ttl: timedelta = timedelta(hours=1),
                 redis_url: Optional[str] = None):
        self.max_memory_size_mb = max_memory_size_mb
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)
        
        # Memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_size_mb = 0.0
        self._cache_lock = asyncio.Lock()
        
        # Redis cache (if available)
        self._redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                self._redis_client.ping()
                self.logger.info("Connected to Redis cache")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
                self._redis_client = None
        
        # Cache statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_usage_mb": 0.0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try memory cache first
        async with self._cache_lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                if entry.is_expired():
                    await self._remove_entry(key)
                else:
                    entry.touch()
                    self._stats["hits"] += 1
                    return entry.value
        
        # Try Redis cache
        if self._redis_client:
            try:
                value = await asyncio.get_event_loop().run_in_executor(
                    None, self._redis_client.get, key
                )
                if value is not None:
                    # Deserialize and store in memory for faster access
                    import pickle
                    deserialized_value = pickle.loads(value.encode('latin-1'))
                    await self.set(key, deserialized_value, store_in_memory=True)
                    self._stats["hits"] += 1
                    return deserialized_value
            except Exception as e:
                self.logger.error(f"Redis get error: {e}")
        
        self._stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None, 
                 store_in_memory: bool = True) -> bool:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        
        # Estimate size
        import sys
        size_bytes = sys.getsizeof(value)
        size_mb = size_bytes / (1024 * 1024)
        
        # Store in memory cache
        if store_in_memory and size_mb < self.max_memory_size_mb / 10:  # Don't store huge objects
            async with self._cache_lock:
                # Evict if necessary
                await self._ensure_memory_capacity(size_mb)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    ttl=ttl,
                    size_bytes=size_bytes
                )
                
                self._memory_cache[key] = entry
                self._memory_size_mb += size_mb
                self._stats["memory_usage_mb"] = self._memory_size_mb
        
        # Store in Redis cache
        if self._redis_client:
            try:
                import pickle
                serialized_value = pickle.dumps(value).decode('latin-1')
                ttl_seconds = int(ttl.total_seconds()) if ttl else None
                
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self._redis_client.setex(key, ttl_seconds, serialized_value) if ttl_seconds else self._redis_client.set(key, serialized_value)
                )
            except Exception as e:
                self.logger.error(f"Redis set error: {e}")
                return False
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        removed = False
        
        # Remove from memory
        async with self._cache_lock:
            if key in self._memory_cache:
                entry = self._memory_cache.pop(key)
                self._memory_size_mb -= entry.size_bytes / (1024 * 1024)
                removed = True
        
        # Remove from Redis
        if self._redis_client:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._redis_client.delete, key
                )
                removed = True
            except Exception as e:
                self.logger.error(f"Redis delete error: {e}")
        
        return removed
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._cache_lock:
            self._memory_cache.clear()
            self._memory_size_mb = 0.0
        
        if self._redis_client:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._redis_client.flushdb
                )
            except Exception as e:
                self.logger.error(f"Redis clear error: {e}")
    
    async def _ensure_memory_capacity(self, required_mb: float):
        """Ensure sufficient memory capacity by evicting entries."""
        while (self._memory_size_mb + required_mb) > self.max_memory_size_mb:
            if not self._memory_cache:
                break
            
            # Find least recently used entry
            lru_key = min(self._memory_cache.keys(), 
                         key=lambda k: self._memory_cache[k].last_accessed)
            
            await self._remove_entry(lru_key)
            self._stats["evictions"] += 1
    
    async def _remove_entry(self, key: str):
        """Remove entry from memory cache."""
        if key in self._memory_cache:
            entry = self._memory_cache.pop(key)
            self._memory_size_mb -= entry.size_bytes / (1024 * 1024)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "total_entries": len(self._memory_cache),
            "redis_available": self._redis_client is not None
        }


def cached(ttl: timedelta = timedelta(minutes=15), key_generator: Optional[Callable] = None):
    """
    Decorator for caching async function results.
    """
    def decorator(func: Callable) -> Callable:
        cache = AsyncCache(default_ttl=ttl)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try cache first
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result)
            
            return result
        
        wrapper._cache = cache  # Expose cache for management
        return wrapper
    
    return decorator


class TaskPool:
    """
    High-performance task pool with load balancing and auto-scaling.
    """
    
    def __init__(self, 
                 max_workers: int = None,
                 max_concurrent_tasks: int = 100,
                 queue_size_limit: int = 1000,
                 use_processes: bool = False):
        
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.queue_size_limit = queue_size_limit
        self.use_processes = use_processes
        self.logger = logging.getLogger(__name__)
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Task management
        self._task_queue = asyncio.Queue(maxsize=queue_size_limit)
        self._active_tasks: Set[asyncio.Task] = set()
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = time.time()
        
        # Performance monitoring
        self._performance_history: deque = deque(maxlen=1000)
        
        # Auto-scaling
        self._scaling_enabled = True
        self._scaling_check_interval = 10  # seconds
        self._scaling_task: Optional[asyncio.Task] = None
        
        # Load balancing
        self._worker_loads: Dict[int, float] = defaultdict(float)
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task for execution."""
        if self._task_queue.qsize() >= self.queue_size_limit:
            raise asyncio.QueueFull("Task queue is full")
        
        # Create task info
        task_info = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "submitted_at": time.time(),
            "task_id": id((func, args, kwargs))
        }
        
        await self._task_queue.put(task_info)
        
        # Process task
        return await self._execute_task(task_info)
    
    async def _execute_task(self, task_info: Dict[str, Any]) -> Any:
        """Execute a single task."""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Wait for available slot
            while len(self._active_tasks) >= self.max_concurrent_tasks:
                await asyncio.sleep(0.01)
            
            # Execute task
            func = task_info["func"]
            args = task_info["args"]
            kwargs = task_info["kwargs"]
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
            
            self._completed_tasks += 1
            
            # Record performance metrics
            execution_time = time.time() - start_time
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = memory_after - memory_before
            
            metrics = PerformanceMetrics(
                operation_name=func.__name__,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_percent=psutil.cpu_percent(),
                concurrent_tasks=len(self._active_tasks),
                queue_size=self._task_queue.qsize()
            )
            
            self._performance_history.append(metrics)
            
            return result
            
        except Exception as e:
            self._failed_tasks += 1
            self.logger.error(f"Task execution failed: {e}")
            raise
    
    async def start_auto_scaling(self):
        """Start auto-scaling monitoring."""
        if not self._scaling_task:
            self._scaling_task = asyncio.create_task(self._auto_scaling_loop())
    
    async def stop_auto_scaling(self):
        """Stop auto-scaling monitoring."""
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
            self._scaling_task = None
    
    async def _auto_scaling_loop(self):
        """Auto-scaling monitoring loop."""
        while True:
            try:
                await self._check_scaling_needs()
                await asyncio.sleep(self._scaling_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(self._scaling_check_interval)
    
    async def _check_scaling_needs(self):
        """Check if scaling adjustments are needed."""
        if not self._performance_history:
            return
        
        # Analyze recent performance
        recent_metrics = list(self._performance_history)[-10:]  # Last 10 tasks
        
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        avg_queue_size = sum(m.queue_size for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        
        # Scale up if needed
        if (avg_queue_size > self.queue_size_limit * 0.8 and 
            avg_cpu < 80 and 
            self.max_workers < multiprocessing.cpu_count() * 2):
            
            self._scale_up()
        
        # Scale down if underutilized
        elif (avg_queue_size < self.queue_size_limit * 0.2 and 
              avg_cpu < 30 and 
              self.max_workers > 1):
            
            self._scale_down()
    
    def _scale_up(self):
        """Scale up worker count."""
        new_max_workers = min(self.max_workers + 1, multiprocessing.cpu_count() * 2)
        if new_max_workers > self.max_workers:
            self.logger.info(f"Scaling up workers: {self.max_workers} -> {new_max_workers}")
            
            # Create new executor with more workers
            old_executor = self.executor
            if self.use_processes:
                self.executor = ProcessPoolExecutor(max_workers=new_max_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=new_max_workers)
            
            self.max_workers = new_max_workers
            
            # Shutdown old executor
            old_executor.shutdown(wait=False)
    
    def _scale_down(self):
        """Scale down worker count."""
        new_max_workers = max(self.max_workers - 1, 1)
        if new_max_workers < self.max_workers:
            self.logger.info(f"Scaling down workers: {self.max_workers} -> {new_max_workers}")
            
            # Create new executor with fewer workers
            old_executor = self.executor
            if self.use_processes:
                self.executor = ProcessPoolExecutor(max_workers=new_max_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=new_max_workers)
            
            self.max_workers = new_max_workers
            
            # Shutdown old executor
            old_executor.shutdown(wait=False)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._performance_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self._performance_history[-100:])  # Last 100 tasks
        
        return {
            "total_completed": self._completed_tasks,
            "total_failed": self._failed_tasks,
            "current_workers": self.max_workers,
            "active_tasks": len(self._active_tasks),
            "queue_size": self._task_queue.qsize(),
            "avg_execution_time": sum(m.execution_time for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            "avg_cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "throughput_per_second": self._completed_tasks / (time.time() - self._start_time),
            "success_rate": self._completed_tasks / (self._completed_tasks + self._failed_tasks) if (self._completed_tasks + self._failed_tasks) > 0 else 0
        }
    
    async def shutdown(self):
        """Shutdown task pool."""
        await self.stop_auto_scaling()
        self.executor.shutdown(wait=True)


class LoadBalancer:
    """
    Load balancer for distributing work across multiple instances.
    """
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances: List[Any] = []
        self.instance_loads: Dict[int, float] = {}
        self.current_index = 0
        self.logger = logging.getLogger(__name__)
    
    def add_instance(self, instance: Any):
        """Add instance to load balancer."""
        self.instances.append(instance)
        self.instance_loads[id(instance)] = 0.0
        self.logger.info(f"Added instance to load balancer. Total instances: {len(self.instances)}")
    
    def remove_instance(self, instance: Any):
        """Remove instance from load balancer."""
        if instance in self.instances:
            self.instances.remove(instance)
            self.instance_loads.pop(id(instance), None)
            self.logger.info(f"Removed instance from load balancer. Total instances: {len(self.instances)}")
    
    def get_next_instance(self) -> Optional[Any]:
        """Get next instance based on load balancing strategy."""
        if not self.instances:
            return None
        
        if self.strategy == "round_robin":
            instance = self.instances[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.instances)
            return instance
        
        elif self.strategy == "least_loaded":
            # Find instance with lowest load
            min_load_instance = min(self.instances, 
                                  key=lambda inst: self.instance_loads.get(id(inst), 0.0))
            return min_load_instance
        
        elif self.strategy == "random":
            import random
            return random.choice(self.instances)
        
        else:
            # Default to round robin
            return self.get_next_instance()
    
    def update_load(self, instance: Any, load: float):
        """Update load for an instance."""
        self.instance_loads[id(instance)] = load
    
    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution."""
        return {
            f"instance_{i}": self.instance_loads.get(id(inst), 0.0)
            for i, inst in enumerate(self.instances)
        }


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.cache = AsyncCache(
            max_memory_size_mb=self.config.get("cache_size_mb", 256),
            redis_url=self.config.get("redis_url")
        )
        
        self.task_pool = TaskPool(
            max_workers=self.config.get("max_workers"),
            max_concurrent_tasks=self.config.get("max_concurrent_analyses", 100),
            use_processes=self.config.get("use_processes", False)
        )
        
        self.load_balancer = LoadBalancer(
            strategy=self.config.get("load_balancing_strategy", "least_loaded")
        )
        
        # Performance monitoring
        self._monitoring_enabled = True
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start performance optimization systems."""
        await self.task_pool.start_auto_scaling()
        
        if self._monitoring_enabled:
            self._monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        self.logger.info("Performance optimization systems started")
    
    async def stop(self):
        """Stop performance optimization systems."""
        await self.task_pool.stop_auto_scaling()
        await self.task_pool.shutdown()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance optimization systems stopped")
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance continuously."""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Log performance warnings
                if cpu_percent > 90:
                    self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                if memory.percent > 90:
                    self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
                # Check cache performance
                cache_stats = self.cache.get_stats()
                if cache_stats["hit_rate"] < 0.5:
                    self.logger.info(f"Low cache hit rate: {cache_stats['hit_rate']:.2%}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system performance status."""
        return {
            "cache": self.cache.get_stats(),
            "task_pool": self.task_pool.get_performance_stats(),
            "load_balancer": {
                "instances": len(self.load_balancer.instances),
                "load_distribution": self.load_balancer.get_load_distribution()
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        }