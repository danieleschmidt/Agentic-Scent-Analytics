"""
Performance and scalability tests.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from agentic_scent.core.performance import (
    AsyncCache, TaskPool, LoadBalancer, PerformanceOptimizer, cached
)
from agentic_scent.sensors.base import SensorReading, SensorType


class TestAsyncCache:
    """Test async caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_basic_operations(self):
        """Test basic cache set/get operations."""
        cache = AsyncCache(max_memory_size_mb=10, default_ttl=timedelta(minutes=5))
        
        # Test set and get
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Test non-existent key
        missing_value = await cache.get("missing_key")
        assert missing_value is None
        
        # Test delete
        deleted = await cache.delete("test_key")
        assert deleted is True
        
        value_after_delete = await cache.get("test_key")
        assert value_after_delete is None
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = AsyncCache(max_memory_size_mb=10)
        
        # Set value with short TTL
        await cache.set("expiring_key", "expiring_value", ttl=timedelta(milliseconds=100))
        
        # Should be available immediately
        value = await cache.get("expiring_key")
        assert value == "expiring_value"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        expired_value = await cache.get("expiring_key")
        assert expired_value is None
    
    @pytest.mark.asyncio
    async def test_cache_memory_eviction(self):
        """Test cache memory-based eviction."""
        cache = AsyncCache(max_memory_size_mb=1)  # Very small cache
        
        # Fill cache beyond capacity
        large_data = "x" * 1024 * 100  # 100KB strings
        
        keys = []
        for i in range(20):  # Try to store 2MB of data in 1MB cache
            key = f"large_key_{i}"
            await cache.set(key, large_data)
            keys.append(key)
        
        # Some keys should have been evicted
        remaining_keys = []
        for key in keys:
            value = await cache.get(key)
            if value is not None:
                remaining_keys.append(key)
        
        assert len(remaining_keys) < len(keys)  # Some eviction occurred
        
        # Get cache stats
        stats = cache.get_stats()
        assert stats["evictions"] > 0
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = AsyncCache(max_memory_size_mb=10)
        
        # Generate hits and misses
        await cache.set("hit_key", "hit_value")
        
        # Generate hits
        for _ in range(5):
            await cache.get("hit_key")
        
        # Generate misses
        for i in range(3):
            await cache.get(f"miss_key_{i}")
        
        stats = cache.get_stats()
        assert stats["hits"] >= 5
        assert stats["misses"] >= 3
        assert stats["hit_rate"] > 0.5
    
    def test_cached_decorator(self):
        """Test the cached decorator functionality."""
        call_count = 0
        
        @cached(ttl=timedelta(seconds=1))
        async def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate expensive operation
            return x + y
        
        async def test_caching():
            # First call should execute function
            result1 = await expensive_function(1, 2)
            assert result1 == 3
            assert call_count == 1
            
            # Second call should use cache
            result2 = await expensive_function(1, 2)
            assert result2 == 3
            assert call_count == 1  # Function not called again
            
            # Different parameters should execute function
            result3 = await expensive_function(2, 3)
            assert result3 == 5
            assert call_count == 2
        
        asyncio.run(test_caching())


class TestTaskPool:
    """Test task pool functionality."""
    
    @pytest.mark.asyncio
    async def test_task_pool_basic_execution(self):
        """Test basic task execution in task pool."""
        pool = TaskPool(max_workers=2, max_concurrent_tasks=5)
        
        def simple_task(x):
            return x * 2
        
        # Submit and execute task
        result = await pool.submit(simple_task, 5)
        assert result == 10
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_task_pool_async_execution(self):
        """Test async task execution in task pool."""
        pool = TaskPool(max_workers=2, max_concurrent_tasks=5)
        
        async def async_task(x, delay=0.1):
            await asyncio.sleep(delay)
            return x ** 2
        
        # Submit async task
        result = await pool.submit(async_task, 4, delay=0.05)
        assert result == 16
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_task_pool_concurrent_execution(self):
        """Test concurrent task execution."""
        pool = TaskPool(max_workers=4, max_concurrent_tasks=10)
        
        def cpu_task(n):
            # Simple CPU-bound calculation
            total = 0
            for i in range(n):
                total += i
            return total
        
        # Submit multiple tasks
        start_time = time.time()
        tasks = [pool.submit(cpu_task, 1000) for _ in range(8)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify results
        expected_result = sum(range(1000))
        assert all(result == expected_result for result in results)
        
        # Should complete reasonably quickly with parallel execution
        assert end_time - start_time < 5.0
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_task_pool_performance_metrics(self):
        """Test task pool performance metrics."""
        pool = TaskPool(max_workers=2, max_concurrent_tasks=5)
        
        def tracked_task(x):
            time.sleep(0.1)  # Simulate work
            return x
        
        # Execute several tasks
        for i in range(5):
            await pool.submit(tracked_task, i)
        
        # Get performance stats
        stats = pool.get_performance_stats()
        
        assert stats["total_completed"] == 5
        assert stats["total_failed"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["avg_execution_time"] > 0
        assert stats["throughput_per_second"] > 0
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_task_pool_error_handling(self):
        """Test task pool error handling."""
        pool = TaskPool(max_workers=2, max_concurrent_tasks=5)
        
        def failing_task():
            raise ValueError("Task failed")
        
        # Submit failing task
        with pytest.raises(ValueError, match="Task failed"):
            await pool.submit(failing_task)
        
        # Check that failure was recorded
        stats = pool.get_performance_stats()
        assert stats["total_failed"] >= 1
        
        await pool.shutdown()


class TestLoadBalancer:
    """Test load balancer functionality."""
    
    def test_load_balancer_round_robin(self):
        """Test round-robin load balancing."""
        balancer = LoadBalancer(strategy="round_robin")
        
        # Add instances
        instances = ["instance1", "instance2", "instance3"]
        for instance in instances:
            balancer.add_instance(instance)
        
        # Test round-robin distribution
        selected = []
        for _ in range(6):  # Two full rounds
            instance = balancer.get_next_instance()
            selected.append(instance)
        
        # Should cycle through instances
        assert selected == instances * 2
    
    def test_load_balancer_least_loaded(self):
        """Test least-loaded load balancing."""
        balancer = LoadBalancer(strategy="least_loaded")
        
        # Add instances with different loads
        instances = ["low_load", "medium_load", "high_load"]
        for instance in instances:
            balancer.add_instance(instance)
        
        # Set loads
        balancer.update_load("low_load", 0.1)
        balancer.update_load("medium_load", 0.5)
        balancer.update_load("high_load", 0.9)
        
        # Should consistently select least loaded
        for _ in range(5):
            instance = balancer.get_next_instance()
            assert instance == "low_load"
    
    def test_load_balancer_instance_management(self):
        """Test adding and removing instances."""
        balancer = LoadBalancer()
        
        # Initially empty
        assert balancer.get_next_instance() is None
        
        # Add instances
        balancer.add_instance("instance1")
        balancer.add_instance("instance2")
        assert len(balancer.instances) == 2
        
        # Remove instance
        balancer.remove_instance("instance1")
        assert len(balancer.instances) == 1
        assert balancer.get_next_instance() == "instance2"
    
    def test_load_distribution_tracking(self):
        """Test load distribution tracking."""
        balancer = LoadBalancer()
        
        instances = ["inst1", "inst2", "inst3"]
        loads = [0.2, 0.5, 0.8]
        
        for instance, load in zip(instances, loads):
            balancer.add_instance(instance)
            balancer.update_load(instance, load)
        
        distribution = balancer.get_load_distribution()
        
        assert len(distribution) == 3
        assert distribution["instance_0"] == 0.2
        assert distribution["instance_1"] == 0.5
        assert distribution["instance_2"] == 0.8


class TestPerformanceOptimizer:
    """Test performance optimizer functionality."""
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        config = {
            "cache_size_mb": 128,
            "max_concurrent_analyses": 50,
            "load_balancing_strategy": "least_loaded"
        }
        
        optimizer = PerformanceOptimizer(config)
        
        assert optimizer.cache.max_memory_size_mb == 128
        assert optimizer.task_pool.max_concurrent_tasks == 50
        assert optimizer.load_balancer.strategy == "least_loaded"
        
        await optimizer.stop()  # Cleanup
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_system_status(self):
        """Test system status reporting."""
        optimizer = PerformanceOptimizer()
        await optimizer.start()
        
        # Generate some activity
        await optimizer.cache.set("test_key", "test_value")
        await optimizer.task_pool.submit(lambda x: x + 1, 5)
        
        # Get system status
        status = optimizer.get_system_status()
        
        assert "cache" in status
        assert "task_pool" in status
        assert "load_balancer" in status
        assert "system" in status
        
        # Cache status
        assert "total_entries" in status["cache"]
        assert "hit_rate" in status["cache"]
        
        # Task pool status
        assert "total_completed" in status["task_pool"]
        assert "current_workers" in status["task_pool"]
        
        # System status
        assert "cpu_percent" in status["system"]
        assert "memory_percent" in status["system"]
        
        await optimizer.stop()


class TestPerformanceBenchmarks:
    """Performance benchmarks and stress tests."""
    
    @pytest.mark.asyncio
    async def test_sensor_analysis_throughput(self):
        """Test sensor analysis throughput."""
        # This would normally use real agents, using simplified version for testing
        async def mock_analysis(reading):
            # Simulate analysis work
            await asyncio.sleep(0.01)
            return {"anomaly_detected": sum(reading.values) > 1000}
        
        # Generate test sensor readings
        readings = []
        for i in range(100):
            reading = SensorReading(
                sensor_id=f"perf_sensor_{i%10}",
                sensor_type=SensorType.E_NOSE,
                values=[100 + i] * 8,
                timestamp=datetime.now()
            )
            readings.append(reading)
        
        # Measure throughput
        start_time = time.time()
        
        # Process readings concurrently
        semaphore = asyncio.Semaphore(20)  # Limit concurrency
        
        async def process_reading(reading):
            async with semaphore:
                return await mock_analysis(reading)
        
        tasks = [process_reading(reading) for reading in readings]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = len(readings) / total_time
        
        # Should achieve reasonable throughput
        assert throughput > 50  # At least 50 analyses per second
        assert len(results) == 100
        assert all(isinstance(result, dict) for result in results)
    
    @pytest.mark.asyncio 
    async def test_cache_performance_under_load(self):
        """Test cache performance under high load."""
        cache = AsyncCache(max_memory_size_mb=50)
        
        # Test data
        test_data = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
        
        # Measure set performance
        start_time = time.time()
        
        set_tasks = [cache.set(key, value) for key, value in test_data.items()]
        await asyncio.gather(*set_tasks)
        
        set_time = time.time() - start_time
        
        # Measure get performance
        start_time = time.time()
        
        get_tasks = [cache.get(key) for key in test_data.keys()]
        retrieved_values = await asyncio.gather(*get_tasks)
        
        get_time = time.time() - start_time
        
        # Performance assertions
        assert set_time < 5.0  # Should set 1000 items in under 5 seconds
        assert get_time < 2.0  # Should get 1000 items in under 2 seconds
        
        # Verify data integrity
        non_none_values = [v for v in retrieved_values if v is not None]
        assert len(non_none_values) >= 800  # At least 80% should be retrieved (some may be evicted)
    
    def test_memory_usage_optimization(self):
        """Test memory usage stays within bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create objects that should manage memory efficiently
        cache = AsyncCache(max_memory_size_mb=100)  # Limited cache
        task_pool = TaskPool(max_workers=4)  # Limited workers
        
        # Generate some load
        large_data = "x" * 1024 * 10  # 10KB strings
        
        async def memory_test():
            # Fill cache to capacity
            for i in range(1000):
                await cache.set(f"memory_key_{i}", large_data)
            
            # Execute some tasks
            def memory_task(data):
                return len(data)
            
            tasks = [task_pool.submit(memory_task, large_data) for _ in range(50)]
            await asyncio.gather(*tasks)
            
            await task_pool.shutdown()
        
        asyncio.run(memory_test())
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 150MB)
        assert memory_increase < 150
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_coordination(self):
        """Test performance of concurrent agent coordination."""
        from agentic_scent.agents.orchestrator import AgentOrchestrator
        from agentic_scent.agents.base import MockLLMAgent, AgentConfig, AgentCapability
        
        orchestrator = AgentOrchestrator()
        
        # Create multiple mock agents
        agents = []
        for i in range(5):
            config = AgentConfig(
                agent_id=f"perf_agent_{i}",
                capabilities=[AgentCapability.ANOMALY_DETECTION],
                confidence_threshold=0.7
            )
            agent = MockLLMAgent(config)
            await agent.start()
            agents.append(agent)
            orchestrator.register_agent(f"agent_{i}", agent)
        
        # Create test sensor readings
        readings = []
        for i in range(20):
            reading = SensorReading(
                sensor_id=f"coord_sensor_{i}",
                sensor_type=SensorType.E_NOSE,
                values=[100 + i * 10] * 8,
                timestamp=datetime.now()
            )
            readings.append(reading)
        
        # Measure coordination performance
        start_time = time.time()
        
        coordination_tasks = [
            orchestrator.coordinate_analysis(reading) 
            for reading in readings
        ]
        results = await asyncio.gather(*coordination_tasks)
        
        end_time = time.time()
        coordination_time = end_time - start_time
        
        # Performance checks
        assert coordination_time < 10.0  # Should complete within 10 seconds
        assert len(results) == 20
        
        # Verify results structure
        for result in results:
            assert isinstance(result, dict)
            assert len(result) == 5  # One result per agent
        
        # Cleanup
        for agent in agents:
            await agent.stop()
        await orchestrator.stop_monitoring()