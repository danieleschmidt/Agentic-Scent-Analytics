"""
High-Performance Engine for Industrial Scent Analytics

Implements advanced performance optimization techniques including:
- Adaptive load balancing
- Intelligent caching strategies
- Real-time auto-scaling
- Memory-efficient processing
- GPU acceleration support
- Distributed computing capabilities
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, PriorityQueue
import multiprocessing as mp
from pathlib import Path
import json
import weakref
import gc

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing execution modes."""
    CPU_SINGLE = "cpu_single"
    CPU_MULTI = "cpu_multi"
    GPU_SINGLE = "gpu_single"
    GPU_MULTI = "gpu_multi"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RESOURCE_USAGE = "resource_usage"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    throughput: float = 0.0  # requests per second
    latency: float = 0.0     # average response time
    queue_size: int = 0
    active_workers: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0


@dataclass
class WorkerNode:
    """Worker node information."""
    node_id: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    current_load: float
    last_response_time: float
    status: str = "active"  # active, busy, offline
    capabilities: List[str] = field(default_factory=list)


@dataclass
class ProcessingTask:
    """High-performance processing task."""
    task_id: str
    task_type: str
    data: Any
    priority: int
    created_at: datetime
    deadline: Optional[datetime] = None
    estimated_duration: float = 1.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None


class AdaptiveCache:
    """
    Intelligent caching system with adaptive strategies and predictive prefetching.
    """
    
    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        self.access_patterns: Dict[str, List[str]] = {}  # For predictive caching
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with adaptive strategy."""
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
                # Trigger predictive prefetch
                self._update_access_patterns(key)
                asyncio.create_task(self._predictive_prefetch(key))
                
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
                
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache with intelligent eviction."""
        with self.lock:
            # Evict items if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_items()
                
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            
            if ttl:
                # Schedule TTL expiration
                asyncio.create_task(self._expire_after_ttl(key, ttl))
                
    def _evict_items(self) -> None:
        """Intelligently evict items based on strategy."""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._remove_item(oldest_key)
            
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            least_used_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            self._remove_item(least_used_key)
            
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on access patterns
            scores = {}
            current_time = time.time()
            
            for key in self.cache.keys():
                recency_score = 1.0 / (current_time - self.access_times.get(key, 0) + 1)
                frequency_score = self.access_counts.get(key, 0)
                prediction_score = self._calculate_prediction_score(key)
                
                # Weighted combination
                scores[key] = (0.4 * recency_score + 
                             0.4 * frequency_score + 
                             0.2 * prediction_score)
                             
            # Remove item with lowest score
            worst_key = min(scores.keys(), key=lambda k: scores[k])
            self._remove_item(worst_key)
            
    def _remove_item(self, key: str) -> None:
        """Remove item and its metadata."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_counts:
            del self.access_counts[key]
            
    def _update_access_patterns(self, key: str) -> None:
        """Update access patterns for predictive caching."""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
            
        # Track recent access sequence
        self.access_patterns[key].append(key)
        
        # Limit pattern history
        if len(self.access_patterns[key]) > 10:
            self.access_patterns[key] = self.access_patterns[key][-10:]
            
    async def _predictive_prefetch(self, accessed_key: str) -> None:
        """Predictively prefetch likely next items."""
        # Simple pattern prediction - in production, use ML models
        predicted_keys = self._predict_next_keys(accessed_key)
        
        for predicted_key in predicted_keys:
            if predicted_key not in self.cache:
                # Trigger prefetch (would call data loading function)
                pass
                
    def _predict_next_keys(self, current_key: str) -> List[str]:
        """Predict next likely accessed keys."""
        # Simple heuristic - in practice, use ML models
        predicted = []
        
        # Pattern-based prediction
        if current_key in self.access_patterns:
            pattern = self.access_patterns[current_key]
            if len(pattern) > 1:
                # Look for sequence patterns
                last_sequence = pattern[-3:] if len(pattern) >= 3 else pattern
                predicted.extend(last_sequence)
                
        return predicted[:3]  # Limit predictions
        
    def _calculate_prediction_score(self, key: str) -> float:
        """Calculate prediction score for cache retention."""
        if key not in self.access_patterns:
            return 0.0
            
        pattern_length = len(self.access_patterns[key])
        recent_accesses = self.access_patterns[key][-5:] if pattern_length >= 5 else self.access_patterns[key]
        
        # Score based on pattern regularity
        if len(set(recent_accesses)) == 1:  # Repeated access
            return 1.0
        else:
            return 0.5
            
    async def _expire_after_ttl(self, key: str, ttl: float) -> None:
        """Expire cache item after TTL."""
        await asyncio.sleep(ttl)
        with self.lock:
            if key in self.cache:
                self._remove_item(key)
                
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
        
    def clear(self) -> None:
        """Clear all cache data."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.access_patterns.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": self.get_hit_rate(),
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "strategy": self.strategy.value
            }


class LoadBalancer:
    """
    Intelligent load balancer with adaptive strategies and real-time optimization.
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.workers: List[WorkerNode] = []
        self.current_index = 0
        self.performance_history: Dict[str, List[float]] = {}
        self.lock = threading.RLock()
        
    def add_worker(self, worker: WorkerNode) -> None:
        """Add worker node to the pool."""
        with self.lock:
            self.workers.append(worker)
            self.performance_history[worker.node_id] = []
            
    def remove_worker(self, node_id: str) -> None:
        """Remove worker node from the pool."""
        with self.lock:
            self.workers = [w for w in self.workers if w.node_id != node_id]
            if node_id in self.performance_history:
                del self.performance_history[node_id]
                
    def select_worker(self, task: ProcessingTask) -> Optional[WorkerNode]:
        """Select optimal worker for task."""
        with self.lock:
            available_workers = [w for w in self.workers if w.status == "active"]
            
            if not available_workers:
                return None
                
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                worker = available_workers[self.current_index % len(available_workers)]
                self.current_index += 1
                return worker
                
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(available_workers, key=lambda w: w.current_load)
                
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return min(available_workers, key=lambda w: w.last_response_time)
                
            elif self.strategy == LoadBalancingStrategy.RESOURCE_USAGE:
                return self._select_by_resource_usage(available_workers, task)
                
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                return self._adaptive_selection(available_workers, task)
                
            else:
                return available_workers[0]  # Fallback
                
    def _select_by_resource_usage(self, workers: List[WorkerNode], task: ProcessingTask) -> WorkerNode:
        """Select worker based on resource requirements."""
        requirements = task.resource_requirements
        
        # Score workers based on resource fit
        scores = []
        for worker in workers:
            score = 0.0
            
            # CPU score
            if "cpu_cores" in requirements:
                if worker.cpu_cores >= requirements["cpu_cores"]:
                    score += 1.0 - (worker.current_load / 100.0)
                else:
                    score -= 1.0  # Penalty for insufficient CPU
                    
            # Memory score
            if "memory_gb" in requirements:
                available_memory = worker.memory_gb * (1.0 - worker.current_load / 100.0)
                if available_memory >= requirements["memory_gb"]:
                    score += 0.5
                else:
                    score -= 0.5
                    
            # GPU score
            if "gpu_required" in requirements and requirements["gpu_required"]:
                if worker.gpu_count > 0:
                    score += 1.0
                else:
                    score -= 2.0  # Heavy penalty for missing GPU
                    
            scores.append(score)
            
        # Select worker with highest score
        best_index = np.argmax(scores)
        return workers[best_index]
        
    def _adaptive_selection(self, workers: List[WorkerNode], task: ProcessingTask) -> WorkerNode:
        """Adaptive worker selection using ML-like scoring."""
        scores = []
        
        for worker in workers:
            # Base score from current load
            load_score = 1.0 - (worker.current_load / 100.0)
            
            # Historical performance score
            history = self.performance_history.get(worker.node_id, [])
            if history:
                avg_performance = np.mean(history[-10:])  # Last 10 tasks
                performance_score = min(1.0, 1.0 / (avg_performance + 0.1))
            else:
                performance_score = 0.5  # Neutral for new workers
                
            # Task type compatibility score
            compatibility_score = 1.0
            if task.task_type in worker.capabilities:
                compatibility_score = 1.2  # Bonus for specialized workers
                
            # Resource availability score
            resource_score = self._calculate_resource_score(worker, task)
            
            # Combined score with weights
            total_score = (0.3 * load_score + 
                         0.3 * performance_score + 
                         0.2 * compatibility_score + 
                         0.2 * resource_score)
                         
            scores.append(total_score)
            
        # Select worker with highest score
        best_index = np.argmax(scores)
        return workers[best_index]
        
    def _calculate_resource_score(self, worker: WorkerNode, task: ProcessingTask) -> float:
        """Calculate resource availability score."""
        requirements = task.resource_requirements
        score = 1.0
        
        # Check CPU availability
        if "cpu_cores" in requirements:
            available_cpu = worker.cpu_cores * (1.0 - worker.current_load / 100.0)
            if available_cpu < requirements["cpu_cores"]:
                score *= 0.5
                
        # Check memory availability
        if "memory_gb" in requirements:
            available_memory = worker.memory_gb * (1.0 - worker.current_load / 100.0)
            if available_memory < requirements["memory_gb"]:
                score *= 0.5
                
        # Check GPU availability
        if requirements.get("gpu_required", False):
            if worker.gpu_count == 0:
                score *= 0.1  # Heavy penalty
                
        return score
        
    def update_worker_performance(self, node_id: str, response_time: float) -> None:
        """Update worker performance metrics."""
        with self.lock:
            if node_id in self.performance_history:
                self.performance_history[node_id].append(response_time)
                
                # Limit history size
                if len(self.performance_history[node_id]) > 100:
                    self.performance_history[node_id] = self.performance_history[node_id][-100:]
                    
            # Update worker's last response time
            for worker in self.workers:
                if worker.node_id == node_id:
                    worker.last_response_time = response_time
                    break
                    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            return {
                "total_workers": len(self.workers),
                "active_workers": len([w for w in self.workers if w.status == "active"]),
                "average_load": np.mean([w.current_load for w in self.workers]) if self.workers else 0.0,
                "strategy": self.strategy.value,
                "worker_details": [
                    {
                        "node_id": w.node_id,
                        "current_load": w.current_load,
                        "last_response_time": w.last_response_time,
                        "status": w.status
                    } for w in self.workers
                ]
            }


class HighPerformanceEngine:
    """
    Main high-performance processing engine with auto-scaling and optimization.
    """
    
    def __init__(self, 
                 max_workers: int = None,
                 processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE,
                 cache_size: int = 10000):
        
        self.max_workers = max_workers or mp.cpu_count()
        self.processing_mode = processing_mode
        
        # Core components
        self.cache = AdaptiveCache(max_size=cache_size)
        self.load_balancer = LoadBalancer()
        self.task_queue = PriorityQueue()
        self.result_queue = Queue()
        
        # Workers and executors
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        self.workers: List[WorkerNode] = []
        
        # Performance monitoring
        self.performance_metrics: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.processed_tasks = 0
        self.failed_tasks = 0
        
        # Auto-scaling
        self.auto_scale_enabled = True
        self.scale_up_threshold = 0.8  # 80% load
        self.scale_down_threshold = 0.3  # 30% load
        self.last_scale_time = time.time()
        self.scale_cooldown = 60  # seconds
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize system
        self._initialize_workers()
        
    def _initialize_workers(self) -> None:
        """Initialize worker nodes."""
        # Create local worker nodes
        cpu_cores = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        gpu_count = self._detect_gpu_count()
        
        for i in range(min(self.max_workers, cpu_cores)):
            worker = WorkerNode(
                node_id=f"local_worker_{i}",
                cpu_cores=1,  # Single core per worker for fine-grained control
                memory_gb=memory_gb / self.max_workers,
                gpu_count=gpu_count if i == 0 else 0,  # GPU only on first worker
                current_load=0.0,
                last_response_time=0.0,
                capabilities=["cpu_processing", "sensor_analysis", "ml_inference"]
            )
            
            if gpu_count > 0 and i == 0:
                worker.capabilities.append("gpu_processing")
                
            self.workers.append(worker)
            self.load_balancer.add_worker(worker)
            
        # Initialize executors
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        if self.processing_mode in [ProcessingMode.CPU_MULTI, ProcessingMode.DISTRIBUTED]:
            self.process_executor = ProcessPoolExecutor(max_workers=min(self.max_workers, cpu_cores))
            
        self.logger.info(f"Initialized {len(self.workers)} workers with {cpu_cores} CPU cores, {memory_gb:.1f}GB RAM, {gpu_count} GPUs")
        
    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        if GPU_AVAILABLE:
            try:
                return cp.cuda.runtime.getDeviceCount()
            except Exception:
                return 0
        return 0
        
    async def start(self) -> None:
        """Start the high-performance engine."""
        self.running = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start task processing
        asyncio.create_task(self._task_processing_loop())
        
        self.logger.info("High-performance engine started")
        
    async def stop(self) -> None:
        """Stop the high-performance engine."""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            
        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
            
        self.logger.info("High-performance engine stopped")
        
    async def submit_task(self, task: ProcessingTask) -> str:
        """Submit task for high-performance processing."""
        # Check cache first
        cache_key = self._generate_cache_key(task)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            if task.callback:
                task.callback(cached_result)
            return task.task_id
            
        # Add to queue with priority
        priority = -task.priority  # Negative for max-heap behavior
        self.task_queue.put((priority, time.time(), task))
        
        return task.task_id
        
    async def _task_processing_loop(self) -> None:
        """Main task processing loop."""
        while self.running:
            try:
                if not self.task_queue.empty():
                    _, submit_time, task = self.task_queue.get(timeout=0.1)
                    
                    # Select optimal worker
                    worker = self.load_balancer.select_worker(task)
                    
                    if worker:
                        # Process task
                        asyncio.create_task(self._process_task(task, worker, submit_time))
                    else:
                        # No workers available, requeue
                        self.task_queue.put((-task.priority, submit_time, task))
                        await asyncio.sleep(0.1)
                        
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")
                await asyncio.sleep(1.0)
                
    async def _process_task(self, task: ProcessingTask, worker: WorkerNode, submit_time: float) -> None:
        """Process individual task."""
        start_time = time.time()
        
        try:
            # Update worker load
            worker.current_load = min(100.0, worker.current_load + 10.0)
            
            # Execute task based on processing mode
            if self.processing_mode == ProcessingMode.GPU_SINGLE and GPU_AVAILABLE:
                result = await self._execute_gpu_task(task)
            elif self.processing_mode == ProcessingMode.CPU_MULTI:
                result = await self._execute_process_task(task)
            else:
                result = await self._execute_thread_task(task)
                
            # Cache result
            cache_key = self._generate_cache_key(task)
            self.cache.set(cache_key, result)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.load_balancer.update_worker_performance(worker.node_id, processing_time)
            self.processed_tasks += 1
            
            # Execute callback
            if task.callback:
                task.callback(result)
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            self.failed_tasks += 1
            
            if task.callback:
                task.callback({"error": str(e)})
                
        finally:
            # Update worker load
            worker.current_load = max(0.0, worker.current_load - 10.0)
            
    async def _execute_thread_task(self, task: ProcessingTask) -> Any:
        """Execute task in thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_executor,
            self._process_task_sync,
            task
        )
        
    async def _execute_process_task(self, task: ProcessingTask) -> Any:
        """Execute task in process executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_executor,
            self._process_task_sync,
            task
        )
        
    async def _execute_gpu_task(self, task: ProcessingTask) -> Any:
        """Execute task on GPU."""
        if not GPU_AVAILABLE:
            return await self._execute_thread_task(task)
            
        # Move data to GPU and process
        try:
            if isinstance(task.data, np.ndarray):
                gpu_data = cp.asarray(task.data)
                # Perform GPU computation (example)
                gpu_result = cp.mean(gpu_data, axis=0)
                return cp.asnumpy(gpu_result)
            else:
                return await self._execute_thread_task(task)
        except Exception:
            # Fallback to CPU
            return await self._execute_thread_task(task)
            
    def _process_task_sync(self, task: ProcessingTask) -> Any:
        """Synchronous task processing function."""
        # Mock processing based on task type
        if task.task_type == "sensor_analysis":
            return self._analyze_sensor_data(task.data)
        elif task.task_type == "quality_prediction":
            return self._predict_quality(task.data)
        elif task.task_type == "anomaly_detection":
            return self._detect_anomalies(task.data)
        else:
            # Generic processing
            time.sleep(task.estimated_duration)
            return {"processed": True, "task_id": task.task_id}
            
    def _analyze_sensor_data(self, data: Any) -> Dict[str, Any]:
        """High-performance sensor data analysis."""
        if isinstance(data, dict) and "values" in data:
            values = np.array(list(data["values"].values()))
            
            # Optimized statistical analysis
            result = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "quality_score": float(np.clip(1.0 - np.std(values) / (np.mean(values) + 1e-10), 0.0, 1.0))
            }
            
            return result
            
        return {"error": "Invalid sensor data format"}
        
    def _predict_quality(self, data: Any) -> Dict[str, Any]:
        """High-performance quality prediction."""
        # Mock ML inference
        if isinstance(data, dict):
            # Simulate complex prediction
            time.sleep(0.1)  # Simulate processing time
            
            quality_score = np.random.beta(8, 2)  # Skewed towards high quality
            
            return {
                "predicted_quality": float(quality_score),
                "confidence": float(np.random.uniform(0.7, 0.95)),
                "risk_factors": ["temperature_variation", "humidity_drift"] if quality_score < 0.8 else []
            }
            
        return {"error": "Invalid data for quality prediction"}
        
    def _detect_anomalies(self, data: Any) -> Dict[str, Any]:
        """High-performance anomaly detection."""
        if isinstance(data, dict) and "values" in data:
            values = np.array(list(data["values"].values()))
            
            # Statistical anomaly detection
            mean_val = np.mean(values)
            std_val = np.std(values)
            z_scores = np.abs((values - mean_val) / (std_val + 1e-10))
            
            anomalies = []
            for i, z_score in enumerate(z_scores):
                if z_score > 2.5:  # Threshold for anomaly
                    anomalies.append({
                        "channel": i,
                        "value": float(values[i]),
                        "z_score": float(z_score),
                        "severity": "high" if z_score > 3.0 else "medium"
                    })
                    
            return {
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies,
                "overall_anomaly_score": float(np.max(z_scores) if len(z_scores) > 0 else 0.0)
            }
            
        return {"error": "Invalid data for anomaly detection"}
        
    def _generate_cache_key(self, task: ProcessingTask) -> str:
        """Generate cache key for task."""
        # Create deterministic key based on task content
        content = f"{task.task_type}_{str(task.data)}"
        return hashlib.md5(content.encode()).hexdigest()
        
    async def _monitoring_loop(self) -> None:
        """Continuous performance monitoring and auto-scaling."""
        while self.running:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                self.performance_metrics.append(metrics)
                
                # Limit metrics history
                if len(self.performance_metrics) > 1000:
                    self.performance_metrics = self.performance_metrics[-1000:]
                    
                # Auto-scaling decisions
                if self.auto_scale_enabled:
                    await self._auto_scale_workers(metrics)
                    
                # Memory management
                await self._manage_memory()
                
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5.0)
                
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # GPU metrics
        gpu_usage = 0.0
        if GPU_AVAILABLE:
            try:
                gpu_usage = cp.cuda.runtime.memGetInfo()[1] / cp.cuda.runtime.memGetInfo()[0] * 100
            except Exception:
                pass
                
        # Application metrics
        current_time = time.time()
        uptime = current_time - self.start_time
        throughput = self.processed_tasks / uptime if uptime > 0 else 0.0
        
        # Calculate average latency from recent tasks
        recent_metrics = self.performance_metrics[-10:] if self.performance_metrics else []
        avg_latency = np.mean([m.latency for m in recent_metrics]) if recent_metrics else 0.0
        
        # Error rate
        total_tasks = self.processed_tasks + self.failed_tasks
        error_rate = self.failed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            throughput=throughput,
            latency=avg_latency,
            queue_size=self.task_queue.qsize(),
            active_workers=len([w for w in self.workers if w.status == "active"]),
            cache_hit_rate=self.cache.get_hit_rate(),
            error_rate=error_rate
        )
        
    async def _auto_scale_workers(self, metrics: PerformanceMetrics) -> None:
        """Auto-scale workers based on performance metrics."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
            
        # Scale up conditions
        if (metrics.cpu_usage > self.scale_up_threshold * 100 or 
            metrics.queue_size > len(self.workers) * 2):
            
            if len(self.workers) < self.max_workers:
                await self._scale_up()
                self.last_scale_time = current_time
                
        # Scale down conditions
        elif (metrics.cpu_usage < self.scale_down_threshold * 100 and 
              metrics.queue_size < len(self.workers) * 0.5 and
              len(self.workers) > 1):
            
            await self._scale_down()
            self.last_scale_time = current_time
            
    async def _scale_up(self) -> None:
        """Add more workers."""
        new_worker_id = len(self.workers)
        
        worker = WorkerNode(
            node_id=f"auto_worker_{new_worker_id}",
            cpu_cores=1,
            memory_gb=psutil.virtual_memory().total / (1024**3) / self.max_workers,
            gpu_count=0,
            current_load=0.0,
            last_response_time=0.0,
            capabilities=["cpu_processing", "sensor_analysis"]
        )
        
        self.workers.append(worker)
        self.load_balancer.add_worker(worker)
        
        self.logger.info(f"Scaled up: Added worker {worker.node_id}")
        
    async def _scale_down(self) -> None:
        """Remove excess workers."""
        # Find worker with lowest load
        idle_workers = [w for w in self.workers if w.current_load < 10.0]
        
        if idle_workers:
            worker_to_remove = min(idle_workers, key=lambda w: w.current_load)
            
            self.workers.remove(worker_to_remove)
            self.load_balancer.remove_worker(worker_to_remove.node_id)
            
            self.logger.info(f"Scaled down: Removed worker {worker_to_remove.node_id}")
            
    async def _manage_memory(self) -> None:
        """Intelligent memory management."""
        memory_info = psutil.virtual_memory()
        
        # If memory usage is high, trigger cleanup
        if memory_info.percent > 85.0:
            # Clear old cache entries
            cache_size = len(self.cache.cache)
            if cache_size > self.cache.max_size * 0.7:
                # Clear 30% of cache
                items_to_remove = int(cache_size * 0.3)
                oldest_keys = sorted(
                    self.cache.access_times.keys(),
                    key=lambda k: self.cache.access_times[k]
                )[:items_to_remove]
                
                for key in oldest_keys:
                    self.cache._remove_item(key)
                    
            # Trigger garbage collection
            gc.collect()
            
            self.logger.info(f"Memory cleanup: Freed cache entries, memory usage: {memory_info.percent:.1f}%")
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_metrics:
            return {"status": "no_metrics"}
            
        recent_metrics = self.performance_metrics[-20:]  # Last 20 measurements
        
        summary = {
            "uptime_seconds": time.time() - self.start_time,
            "total_tasks_processed": self.processed_tasks,
            "total_tasks_failed": self.failed_tasks,
            "current_metrics": {
                "cpu_usage": recent_metrics[-1].cpu_usage,
                "memory_usage": recent_metrics[-1].memory_usage,
                "gpu_usage": recent_metrics[-1].gpu_usage,
                "throughput": recent_metrics[-1].throughput,
                "queue_size": recent_metrics[-1].queue_size,
                "active_workers": recent_metrics[-1].active_workers,
                "cache_hit_rate": recent_metrics[-1].cache_hit_rate,
                "error_rate": recent_metrics[-1].error_rate
            },
            "performance_trends": {
                "avg_cpu_usage": float(np.mean([m.cpu_usage for m in recent_metrics])),
                "avg_memory_usage": float(np.mean([m.memory_usage for m in recent_metrics])),
                "avg_throughput": float(np.mean([m.throughput for m in recent_metrics])),
                "avg_latency": float(np.mean([m.latency for m in recent_metrics]))
            },
            "cache_stats": self.cache.get_stats(),
            "load_balancer_stats": self.load_balancer.get_load_stats(),
            "processing_mode": self.processing_mode.value,
            "auto_scaling_enabled": self.auto_scale_enabled
        }
        
        return summary


async def create_high_performance_engine(
    max_workers: Optional[int] = None,
    processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE
) -> HighPerformanceEngine:
    """Create and initialize high-performance engine."""
    engine = HighPerformanceEngine(
        max_workers=max_workers,
        processing_mode=processing_mode
    )
    
    await engine.start()
    return engine


async def demonstrate_high_performance():
    """Demonstration of high-performance capabilities."""
    print("⚡ High-Performance Engine Demo")
    print("=" * 50)
    
    # Create engine
    engine = await create_high_performance_engine(
        max_workers=4,
        processing_mode=ProcessingMode.ADAPTIVE
    )
    
    # Submit test tasks
    tasks = []
    for i in range(10):
        task = ProcessingTask(
            task_id=f"test_task_{i}",
            task_type="sensor_analysis",
            data={
                "sensor_id": f"sensor_{i}",
                "values": {f"ch_{j}": np.random.normal(0, 1) for j in range(16)}
            },
            priority=i % 3,  # Different priorities
            created_at=datetime.now(),
            estimated_duration=0.1
        )
        tasks.append(task)
        await engine.submit_task(task)
        
    print(f"\n🚀 Submitted {len(tasks)} tasks")
    
    # Wait for processing
    await asyncio.sleep(2.0)
    
    # Get performance summary
    summary = engine.get_performance_summary()
    
    print("\n📊 Performance Summary:")
    print(f"  Uptime: {summary['uptime_seconds']:.1f}s")
    print(f"  Tasks Processed: {summary['total_tasks_processed']}")
    print(f"  Tasks Failed: {summary['total_tasks_failed']}")
    print(f"  Current CPU Usage: {summary['current_metrics']['cpu_usage']:.1f}%")
    print(f"  Current Memory Usage: {summary['current_metrics']['memory_usage']:.1f}%")
    print(f"  Throughput: {summary['current_metrics']['throughput']:.2f} tasks/sec")
    print(f"  Cache Hit Rate: {summary['current_metrics']['cache_hit_rate']:.1%}")
    print(f"  Active Workers: {summary['current_metrics']['active_workers']}")
    print(f"  Processing Mode: {summary['processing_mode']}")
    
    # Cache performance
    cache_stats = summary['cache_stats']
    print(f"\n💾 Cache Performance:")
    print(f"  Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Strategy: {cache_stats['strategy']}")
    
    # Load balancer stats
    lb_stats = summary['load_balancer_stats']
    print(f"\n⚖️ Load Balancer:")
    print(f"  Total Workers: {lb_stats['total_workers']}")
    print(f"  Active Workers: {lb_stats['active_workers']}")
    print(f"  Average Load: {lb_stats['average_load']:.1f}%")
    
    # Stop engine
    await engine.stop()
    
    print("\n✅ High-performance demonstration completed!")
    
    return engine


if __name__ == "__main__":
    asyncio.run(demonstrate_high_performance())
