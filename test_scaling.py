#!/usr/bin/env python3
"""
Test script for performance scaling and optimization features.
"""

import asyncio
import sys
import time
import traceback
from datetime import datetime

# Test imports
try:
    from agentic_scent.core.caching import create_optimized_cache, cache_key
    from agentic_scent.core.task_pool import create_optimized_task_pool, TaskPriority
    from agentic_scent.core.metrics import create_metrics_system
    from agentic_scent.core.factory import ScentAnalyticsFactory
    from agentic_scent.agents.quality_control import QualityControlAgent
    print("✅ All scaling modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


async def test_caching_system():
    """Test multi-level caching system."""
    print("\n📋 Testing Caching System")
    print("-" * 40)
    
    try:
        # Create optimized cache
        cache = await create_optimized_cache(
            memory_size=100,
            memory_ttl=30,
            redis_url=None,  # Uses mock Redis
            redis_ttl=60
        )
        
        # Test cache operations
        test_data = {
            "sensor_reading": [100, 150, 120, 180, 95],
            "timestamp": datetime.now().isoformat(),
            "analysis_result": "normal_operation"
        }
        
        key = cache_key("test_analysis", sensor_id="e_nose_01", batch_id="BATCH_001")
        
        # Set value
        await cache.set(key, test_data, ttl=60)
        print(f"  ✅ Cached data with key: {key}")
        
        # Get value
        cached_result = await cache.get(key)
        if cached_result and cached_result["analysis_result"] == "normal_operation":
            print("  ✅ Cache retrieval successful")
        else:
            print("  ❌ Cache retrieval failed")
            return False
        
        # Test cache statistics
        stats = cache.get_stats()
        print(f"  ✅ Cache stats - Memory: {stats['memory_cache']['hit_rate']:.2f} hit rate")
        print(f"                   Redis: {stats['redis_cache']['hit_rate']:.2f} hit rate")
        
        # Test cache miss
        missing_result = await cache.get("non_existent_key")
        if missing_result is None:
            print("  ✅ Cache miss handled correctly")
        
        print("✅ Caching system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Caching system test failed: {e}")
        traceback.print_exc()
        return False


async def test_task_pool():
    """Test auto-scaling task pool."""
    print("\n📋 Testing Task Pool System")
    print("-" * 40)
    
    try:
        # Create task pool
        task_pool = await create_optimized_task_pool(min_workers=2, max_workers=6)
        
        # Test basic task submission
        async def sample_task(task_id: str, duration: float = 0.1):
            await asyncio.sleep(duration)
            return f"Task {task_id} completed in {duration}s"
        
        # Submit high-priority tasks
        task_ids = []
        for i in range(5):
            task_id = f"test_task_{i}"
            await task_pool.submit_task(
                task_id,
                sample_task,
                task_id,
                0.2,  # duration
                priority=TaskPriority.HIGH
            )
            task_ids.append(task_id)
        
        print(f"  ✅ Submitted {len(task_ids)} tasks")
        
        # Wait for results
        results = []
        for task_id in task_ids:
            try:
                result = await task_pool.get_result(task_id, timeout=5.0)
                if result.success:
                    results.append(result)
                    print(f"  ✅ {task_id}: {result.execution_time:.3f}s execution time")
                else:
                    print(f"  ❌ {task_id} failed: {result.error}")
            except Exception as e:
                print(f"  ❌ {task_id} timeout or error: {e}")
        
        # Check pool status
        status = task_pool.get_status()
        print(f"  ✅ Pool status: {status['active_workers']} workers, "
              f"{status['completed_tasks']} completed")
        
        # Stop task pool
        await task_pool.stop()
        print("  ✅ Task pool stopped")
        
        success_rate = len(results) / len(task_ids)
        if success_rate >= 0.8:
            print(f"✅ Task pool test passed ({success_rate:.1%} success rate)")
            return True
        else:
            print(f"❌ Task pool test failed ({success_rate:.1%} success rate)")
            return False
        
    except Exception as e:
        print(f"❌ Task pool test failed: {e}")
        traceback.print_exc()
        return False


async def test_metrics_system():
    """Test metrics collection and export."""
    print("\n📋 Testing Metrics System")
    print("-" * 40)
    
    try:
        # Create metrics system
        metrics, profiler = create_metrics_system(enable_prometheus=True)
        
        # Record some metrics
        metrics.record_sensor_reading("e_nose_01", "MOS")
        metrics.record_analysis_duration("qc_agent_01", "anomaly_detection", 0.25)
        metrics.record_anomaly_detection("qc_agent_01", "medium")
        metrics.record_llm_request("openai", "gpt-4", "success", tokens_used=150)
        metrics.set_system_metrics(cpu_percent=45.2, memory_percent=62.8)
        
        print("  ✅ Recorded various metrics")
        
        # Test profiling
        with profiler.profile_context("test_computation"):
            await asyncio.sleep(0.1)  # Simulate work
        
        # Get profile summary
        profile_summary = profiler.get_profile_summary("test_computation")
        if profile_summary["count"] > 0:
            print(f"  ✅ Profiling: {profile_summary['avg_time']:.4f}s average")
        
        # Get metrics summary
        summary = metrics.get_metrics_summary()
        counters = summary.get("counters", {})
        histograms = summary.get("histograms", {})
        
        print(f"  ✅ Metrics collected: {len(counters)} counters, {len(histograms)} histograms")
        
        # Test Prometheus export
        prometheus_metrics = metrics.export_prometheus_metrics()
        if prometheus_metrics:
            print("  ✅ Prometheus metrics exported")
            # Show sample of exported metrics
            lines = prometheus_metrics.split('\n')[:5]
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    print(f"    {line}")
                    break
        else:
            print("  ⚠️  Prometheus export not available (mock mode)")
        
        print("✅ Metrics system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Metrics system test failed: {e}")
        traceback.print_exc()
        return False


async def test_optimized_factory():
    """Test factory with scaling optimizations."""
    print("\n📋 Testing Optimized Factory System")
    print("-" * 40)
    
    try:
        # Create optimized factory
        factory = ScentAnalyticsFactory(
            production_line="test_optimization_line",
            e_nose_config={"sensors": ["MOS", "PID"], "channels": 16},
            enable_scaling=True
        )
        
        # Register an agent
        agent = QualityControlAgent(
            llm_model="gpt-4",
            agent_id="optimization_test_agent"
        )
        factory.register_agent(agent)
        
        # Test performance metrics before starting
        initial_metrics = factory.get_performance_metrics()
        print(f"  ✅ Initial metrics: {initial_metrics['factory_status']['active_agents']} agents")
        
        # Start monitoring (this initializes async systems)
        await factory.start_monitoring("OPT_BATCH_001")
        
        # Let it run briefly
        await asyncio.sleep(0.5)
        
        # Test performance metrics during operation
        runtime_metrics = factory.get_performance_metrics()
        
        # Check if performance systems are active
        has_task_pool = "task_pool" in runtime_metrics
        has_cache = "cache" in runtime_metrics
        has_metrics = "system_metrics" in runtime_metrics
        
        print(f"  ✅ Performance systems active: Task Pool={has_task_pool}, "
              f"Cache={has_cache}, Metrics={has_metrics}")
        
        # Test metrics export
        json_export = await factory.export_metrics("json")
        if json_export and len(json_export) > 100:
            print("  ✅ JSON metrics export successful")
        
        prometheus_export = await factory.export_metrics("prometheus")
        if prometheus_export:
            print("  ✅ Prometheus metrics export successful")
        else:
            print("  ⚠️  Prometheus export not available")
        
        # Test cached analysis
        sensor_data = {
            "values": [100, 150, 120, 180, 95],
            "sensor_id": "e_nose_01",
            "timestamp": datetime.now().isoformat()
        }
        
        # First call (should compute)
        start_time = time.time()
        result1 = await factory.cached_analysis(sensor_data, "test_analysis_key")
        first_duration = time.time() - start_time
        
        # Second call (should use cache if implemented)
        start_time = time.time()
        result2 = await factory.cached_analysis(sensor_data, "test_analysis_key")
        second_duration = time.time() - start_time
        
        print(f"  ✅ Cached analysis: first={first_duration:.4f}s, second={second_duration:.4f}s")
        
        # Stop monitoring
        await factory.stop_monitoring()
        print("  ✅ Factory stopped gracefully")
        
        print("✅ Optimized factory test passed")
        return True
        
    except Exception as e:
        print(f"❌ Optimized factory test failed: {e}")
        traceback.print_exc()
        return False


async def test_load_simulation():
    """Test system under load."""
    print("\n📋 Testing Load Simulation")
    print("-" * 40)
    
    try:
        # Create task pool for load testing
        task_pool = await create_optimized_task_pool(min_workers=2, max_workers=8)
        
        # Submit many concurrent tasks
        async def cpu_intensive_task(task_id: str, iterations: int = 1000):
            total = 0
            for i in range(iterations):
                total += i * i
                if i % 100 == 0:
                    await asyncio.sleep(0.001)  # Yield occasionally
            return f"Task {task_id}: computed {total}"
        
        # Submit load
        num_tasks = 20
        task_ids = []
        
        start_time = time.time()
        for i in range(num_tasks):
            task_id = f"load_task_{i}"
            priority = TaskPriority.HIGH if i < 5 else TaskPriority.MEDIUM
            await task_pool.submit_task(
                task_id,
                cpu_intensive_task,
                task_id,
                500,  # iterations
                priority=priority
            )
            task_ids.append(task_id)
        
        submission_time = time.time() - start_time
        print(f"  ✅ Submitted {num_tasks} tasks in {submission_time:.3f}s")
        
        # Wait for completion
        completed = 0
        failed = 0
        
        for task_id in task_ids:
            try:
                result = await task_pool.get_result(task_id, timeout=10.0)
                if result.success:
                    completed += 1
                else:
                    failed += 1
            except:
                failed += 1
        
        total_time = time.time() - start_time
        
        # Get final pool status
        final_status = task_pool.get_status()
        
        print(f"  ✅ Load test results:")
        print(f"    Completed: {completed}/{num_tasks} ({completed/num_tasks:.1%})")
        print(f"    Total time: {total_time:.3f}s")
        print(f"    Throughput: {completed/total_time:.1f} tasks/sec")
        print(f"    Final workers: {final_status['active_workers']}")
        
        await task_pool.stop()
        
        success_rate = completed / num_tasks
        if success_rate >= 0.8:
            print(f"✅ Load simulation test passed ({success_rate:.1%} success)")
            return True
        else:
            print(f"❌ Load simulation test failed ({success_rate:.1%} success)")
            return False
        
    except Exception as e:
        print(f"❌ Load simulation test failed: {e}")
        traceback.print_exc()
        return False


async def run_scaling_tests():
    """Run all scaling and optimization tests."""
    print("🚀 Running Agentic Scent Analytics Scaling Tests")
    print("=" * 60)
    
    test_functions = [
        ("Caching System", test_caching_system),
        ("Task Pool System", test_task_pool),
        ("Metrics System", test_metrics_system),
        ("Optimized Factory", test_optimized_factory),
        ("Load Simulation", test_load_simulation)
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_name, test_func in test_functions:
        try:
            if await test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test suite failed")
        except Exception as e:
            print(f"❌ {test_name} test suite crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Scaling Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All scaling tests passed! System is optimized for production.")
        return True
    else:
        print(f"⚠️  {total - passed} test suite(s) failed. Review scaling implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_scaling_tests())
    sys.exit(0 if success else 1)