"""
Load balancing and auto-scaling utilities for sentiment analysis
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import statistics
import threading

from .monitoring import system_monitor, performance_monitor

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RESOURCE_BASED = "resource_based"


@dataclass
class WorkerNode:
    id: str
    endpoint: str
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def utilization(self) -> float:
        return self.current_connections / self.max_connections if self.max_connections > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "endpoint": self.endpoint,
            "weight": self.weight,
            "max_connections": self.max_connections,
            "current_connections": self.current_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "avg_response_time_ms": self.avg_response_time_ms,
            "success_rate": self.success_rate,
            "utilization": self.utilization,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "is_healthy": self.is_healthy,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "created_at": self.created_at.isoformat()
        }


class LoadBalancer:
    """Intelligent load balancer for sentiment analysis workers"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_BASED):
        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self._round_robin_index = 0
        self._lock = threading.RLock()
        
        # Health check configuration
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5   # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "balanced_requests": 0,
            "failed_balancing": 0,
            "avg_balancing_time_ms": 0.0
        }
        
        logger.info(f"LoadBalancer initialized with strategy: {strategy.value}")
    
    def add_worker(self, worker_id: str, endpoint: str, weight: float = 1.0, 
                   max_connections: int = 100) -> bool:
        """Add a worker node"""
        with self._lock:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already exists")
                return False
            
            worker = WorkerNode(
                id=worker_id,
                endpoint=endpoint,
                weight=weight,
                max_connections=max_connections
            )
            
            self.workers[worker_id] = worker
            logger.info(f"Added worker {worker_id} at {endpoint}")
            return True
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker node"""
        with self._lock:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found")
                return False
            
            del self.workers[worker_id]
            logger.info(f"Removed worker {worker_id}")
            return True
    
    def update_worker_metrics(self, worker_id: str, 
                             response_time_ms: float = None,
                             success: bool = True,
                             cpu_usage: float = None,
                             memory_usage: float = None):
        """Update worker performance metrics"""
        with self._lock:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            worker.total_requests += 1
            
            if not success:
                worker.failed_requests += 1
            
            if response_time_ms is not None:
                # Update average response time (exponential moving average)
                alpha = 0.3
                if worker.avg_response_time_ms == 0.0:
                    worker.avg_response_time_ms = response_time_ms
                else:
                    worker.avg_response_time_ms = (
                        alpha * response_time_ms + 
                        (1 - alpha) * worker.avg_response_time_ms
                    )
            
            if cpu_usage is not None:
                worker.cpu_usage = cpu_usage
            
            if memory_usage is not None:
                worker.memory_usage = memory_usage
    
    def get_next_worker(self) -> Optional[WorkerNode]:
        """Get next worker based on load balancing strategy"""
        start_time = time.time()
        
        with self._lock:
            healthy_workers = [w for w in self.workers.values() if w.is_healthy]
            
            if not healthy_workers:
                self.metrics["failed_balancing"] += 1
                return None
            
            self.metrics["total_requests"] += 1
            
            # Apply load balancing strategy
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                worker = self._round_robin_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                worker = self._least_connections_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                worker = self._weighted_round_robin_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                worker = self._response_time_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                worker = self._resource_based_select(healthy_workers)
            else:
                worker = healthy_workers[0]  # Fallback
            
            if worker:
                worker.current_connections += 1
                self.metrics["balanced_requests"] += 1
            
            # Update metrics
            balancing_time = (time.time() - start_time) * 1000
            self._update_avg_balancing_time(balancing_time)
            
            return worker
    
    def _round_robin_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin selection"""
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker
    
    def _least_connections_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least connections"""
        return min(workers, key=lambda w: w.current_connections)
    
    def _weighted_round_robin_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin selection"""
        # Simple weighted selection based on weight
        total_weight = sum(w.weight for w in workers)
        if total_weight == 0:
            return workers[0]
        
        # Calculate selection probability
        selection_point = (self._round_robin_index % int(total_weight * 10)) / 10.0
        current_weight = 0
        
        for worker in workers:
            current_weight += worker.weight
            if selection_point <= current_weight:
                self._round_robin_index += 1
                return worker
        
        return workers[-1]  # Fallback
    
    def _response_time_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with best response time"""
        # Prefer workers with lower average response time
        available_workers = [w for w in workers if w.current_connections < w.max_connections]
        if not available_workers:
            available_workers = workers
        
        return min(available_workers, key=lambda w: w.avg_response_time_ms or float('inf'))
    
    def _resource_based_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on resource utilization and performance"""
        available_workers = [w for w in workers if w.current_connections < w.max_connections]
        if not available_workers:
            available_workers = workers
        
        def score_worker(worker: WorkerNode) -> float:
            # Composite score based on multiple factors (lower is better)
            utilization_score = worker.utilization * 40
            response_time_score = (worker.avg_response_time_ms or 0) / 100.0
            cpu_score = worker.cpu_usage * 20
            memory_score = worker.memory_usage * 20
            failure_rate_score = (1 - worker.success_rate) * 100
            
            return utilization_score + response_time_score + cpu_score + memory_score + failure_rate_score
        
        return min(available_workers, key=score_worker)
    
    def _update_avg_balancing_time(self, balancing_time: float):
        """Update average balancing time"""
        current_avg = self.metrics["avg_balancing_time_ms"]
        total_requests = self.metrics["total_requests"]
        
        if total_requests == 1:
            self.metrics["avg_balancing_time_ms"] = balancing_time
        else:
            self.metrics["avg_balancing_time_ms"] = (
                (current_avg * (total_requests - 1) + balancing_time) / total_requests
            )
    
    def release_worker(self, worker_id: str):
        """Release worker connection"""
        with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.current_connections = max(0, worker.current_connections - 1)
    
    async def start_health_checks(self, health_check_func: Optional[Callable] = None):
        """Start periodic health checks"""
        if self._health_check_task and not self._health_check_task.done():
            logger.warning("Health checks already running")
            return
        
        self._health_check_task = asyncio.create_task(
            self._health_check_loop(health_check_func)
        )
        logger.info("Started health check loop")
    
    async def stop_health_checks(self):
        """Stop health checks"""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health check loop")
    
    async def _health_check_loop(self, health_check_func: Optional[Callable]):
        """Health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks(health_check_func)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _perform_health_checks(self, health_check_func: Optional[Callable]):
        """Perform health checks on all workers"""
        with self._lock:
            workers_to_check = list(self.workers.values())
        
        for worker in workers_to_check:
            try:
                is_healthy = True
                
                if health_check_func:
                    # Custom health check
                    if asyncio.iscoroutinefunction(health_check_func):
                        is_healthy = await asyncio.wait_for(
                            health_check_func(worker),
                            timeout=self.health_check_timeout
                        )
                    else:
                        is_healthy = health_check_func(worker)
                else:
                    # Basic health check
                    is_healthy = worker.success_rate > 0.5 and worker.utilization < 0.95
                
                with self._lock:
                    worker.is_healthy = is_healthy
                    worker.last_health_check = datetime.now(timezone.utc)
                
                if not is_healthy:
                    logger.warning(f"Worker {worker.id} marked as unhealthy")
                
            except Exception as e:
                logger.error(f"Health check failed for worker {worker.id}: {e}")
                with self._lock:
                    worker.is_healthy = False
                    worker.last_health_check = datetime.now(timezone.utc)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self._lock:
            worker_stats = [worker.to_dict() for worker in self.workers.values()]
            healthy_workers = sum(1 for w in self.workers.values() if w.is_healthy)
            
            return {
                "strategy": self.strategy.value,
                "total_workers": len(self.workers),
                "healthy_workers": healthy_workers,
                "unhealthy_workers": len(self.workers) - healthy_workers,
                "metrics": self.metrics,
                "workers": worker_stats
            }


class AutoScaler:
    """Auto-scaling controller for sentiment analysis workers"""
    
    def __init__(self, 
                 load_balancer: LoadBalancer,
                 min_workers: int = 1,
                 max_workers: int = 10,
                 target_utilization: float = 0.7,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 cooldown_period: int = 300):  # 5 minutes
        
        self.load_balancer = load_balancer
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.last_scale_action = 0
        self.scaling_enabled = True
        self._scaling_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.scaling_history = []
        self.metrics = {
            "scale_up_events": 0,
            "scale_down_events": 0,
            "last_scale_time": None,
            "current_desired_workers": min_workers
        }
        
        logger.info(f"AutoScaler initialized - min: {min_workers}, max: {max_workers}")
    
    def start(self):
        """Start auto-scaling"""
        if self._scaling_task and not self._scaling_task.done():
            logger.warning("Auto-scaler already running")
            return
        
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaler started")
    
    def stop(self):
        """Stop auto-scaling"""
        if self._scaling_task and not self._scaling_task.done():
            self._scaling_task.cancel()
        logger.info("Auto-scaler stopped")
    
    async def _scaling_loop(self):
        """Main scaling loop"""
        while self.scaling_enabled:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._evaluate_scaling()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(30)
    
    async def _evaluate_scaling(self):
        """Evaluate if scaling action is needed"""
        now = time.time()
        
        # Check cooldown period
        if now - self.last_scale_action < self.cooldown_period:
            logger.debug("Scaling in cooldown period")
            return
        
        # Get current metrics
        lb_stats = self.load_balancer.get_stats()
        healthy_workers = lb_stats["healthy_workers"]
        
        if healthy_workers == 0:
            logger.error("No healthy workers available")
            return
        
        # Calculate utilization metrics
        workers = lb_stats["workers"]
        utilizations = [w["utilization"] for w in workers if w["is_healthy"]]
        
        if not utilizations:
            return
        
        avg_utilization = statistics.mean(utilizations)
        max_utilization = max(utilizations)
        
        # Get system metrics
        system_metrics = system_monitor.get_system_metrics()
        cpu_usage = system_metrics.get("cpu_percent", type('', (), {'value': 0})).value
        memory_usage = system_metrics.get("memory_percent", type('', (), {'value': 0})).value
        
        # Scaling decision logic
        should_scale_up = (
            (avg_utilization > self.scale_up_threshold or 
             max_utilization > 0.95 or
             cpu_usage > 80 or
             memory_usage > 85) and
            healthy_workers < self.max_workers
        )
        
        should_scale_down = (
            avg_utilization < self.scale_down_threshold and
            cpu_usage < 50 and
            memory_usage < 60 and
            healthy_workers > self.min_workers
        )
        
        # Execute scaling action
        if should_scale_up:
            await self._scale_up()
        elif should_scale_down:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up workers"""
        lb_stats = self.load_balancer.get_stats()
        current_workers = lb_stats["healthy_workers"]
        
        if current_workers >= self.max_workers:
            logger.info("Cannot scale up - at maximum worker limit")
            return
        
        # Calculate how many workers to add
        workers_to_add = min(2, self.max_workers - current_workers)  # Add up to 2 at a time
        
        logger.info(f"Scaling up: adding {workers_to_add} workers")
        
        # This would integrate with your container orchestrator
        # For now, just log the action
        success = await self._provision_workers(workers_to_add)
        
        if success:
            self.metrics["scale_up_events"] += 1
            self.metrics["last_scale_time"] = datetime.now(timezone.utc)
            self.metrics["current_desired_workers"] += workers_to_add
            self.last_scale_action = time.time()
            
            self.scaling_history.append({
                "timestamp": datetime.now(timezone.utc),
                "action": "scale_up",
                "workers_added": workers_to_add,
                "total_workers": current_workers + workers_to_add
            })
    
    async def _scale_down(self):
        """Scale down workers"""
        lb_stats = self.load_balancer.get_stats()
        current_workers = lb_stats["healthy_workers"]
        
        if current_workers <= self.min_workers:
            logger.info("Cannot scale down - at minimum worker limit")
            return
        
        # Calculate how many workers to remove
        workers_to_remove = min(1, current_workers - self.min_workers)  # Remove 1 at a time
        
        logger.info(f"Scaling down: removing {workers_to_remove} workers")
        
        # This would integrate with your container orchestrator
        success = await self._deprovision_workers(workers_to_remove)
        
        if success:
            self.metrics["scale_down_events"] += 1
            self.metrics["last_scale_time"] = datetime.now(timezone.utc)
            self.metrics["current_desired_workers"] -= workers_to_remove
            self.last_scale_action = time.time()
            
            self.scaling_history.append({
                "timestamp": datetime.now(timezone.utc),
                "action": "scale_down",
                "workers_removed": workers_to_remove,
                "total_workers": current_workers - workers_to_remove
            })
    
    async def _provision_workers(self, count: int) -> bool:
        """Provision new worker instances"""
        # This would integrate with Kubernetes, Docker, or cloud APIs
        # For demonstration, we'll just simulate the action
        logger.info(f"Provisioning {count} new worker instances...")
        
        # Simulate deployment time
        await asyncio.sleep(2)
        
        # Add mock workers to load balancer
        for i in range(count):
            worker_id = f"auto-worker-{int(time.time())}-{i}"
            endpoint = f"http://worker-{worker_id}:8000"
            self.load_balancer.add_worker(worker_id, endpoint)
        
        return True
    
    async def _deprovision_workers(self, count: int) -> bool:
        """Deprovision worker instances"""
        logger.info(f"Deprovisioning {count} worker instances...")
        
        # Find workers to remove (prefer least utilized)
        with self.load_balancer._lock:
            workers = list(self.load_balancer.workers.values())
            workers_to_remove = sorted(workers, key=lambda w: w.utilization)[:count]
        
        for worker in workers_to_remove:
            logger.info(f"Removing worker {worker.id}")
            self.load_balancer.remove_worker(worker.id)
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get auto-scaler metrics"""
        return {
            "enabled": self.scaling_enabled,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "target_utilization": self.target_utilization,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "cooldown_period": self.cooldown_period,
            "metrics": self.metrics,
            "scaling_history": self.scaling_history[-10:]  # Last 10 events
        }