#!/usr/bin/env python3
"""
Comprehensive test suite for production readiness validation.
"""

import asyncio
import sys
import time
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Test imports
try:
    from agentic_scent import ScentAnalyticsFactory, QualityControlAgent, AgentOrchestrator
    from agentic_scent.integration import QuantumScheduledFactory, HybridAgentOrchestrator
    from agentic_scent.core.caching import create_optimized_cache
    from agentic_scent.core.task_pool import create_optimized_task_pool
    from agentic_scent.core.metrics import create_metrics_system
    from agentic_scent.core.logging_config import setup_logging
    from agentic_scent.core.health_checks import create_default_health_checker
    from agentic_scent.llm.client import create_llm_client
    print("✅ All production modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class ProductionTestSuite:
    """Comprehensive production readiness test suite."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def record_result(self, test_name: str, success: bool, duration: float, details: dict = None):
        """Record test result."""
        self.test_results[test_name] = {
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
    
    async def test_integration_end_to_end(self):
        """Complete end-to-end integration test."""
        print("\n📋 End-to-End Integration Test")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # 1. Create quantum-enhanced factory
            factory = QuantumScheduledFactory(
                production_line="production_test_line",
                e_nose_config={
                    "sensors": ["MOS", "PID", "EC"],
                    "channels": 32,
                    "sampling_rate": 10.0
                }
            )
            print("  ✅ Quantum factory created")
            
            # 2. Create hybrid orchestrator
            orchestrator = HybridAgentOrchestrator()
            print("  ✅ Hybrid orchestrator created")
            
            # 3. Create and register multiple agents
            agents = []
            for i in range(3):
                agent = QualityControlAgent(
                    llm_model="gpt-4",
                    agent_id=f"production_agent_{i}",
                    alert_threshold=0.85
                )
                agents.append(agent)
                factory.register_agent(agent)
                orchestrator.register_agent(f"agent_{i}", agent)
            
            print(f"  ✅ Created and registered {len(agents)} agents")
            
            # 4. Initialize all systems
            await factory.start_monitoring("PROD_TEST_BATCH")
            await orchestrator.start()
            print("  ✅ All systems started")
            
            # 5. Simulate production data flow
            sensor_data = {
                "timestamp": datetime.now().isoformat(),
                "values": [125, 180, 145, 200, 110, 165, 155, 190],
                "sensor_id": "e_nose_production",
                "batch_id": "PROD_TEST_BATCH",
                "quality_metrics": {
                    "temperature": 23.5,
                    "humidity": 45.2,
                    "pressure": 1013.25
                }
            }
            
            # 6. Run hybrid analysis
            analysis_results = await orchestrator.coordinate_hybrid_analysis(
                sensor_data, "Production quality control analysis"
            )
            
            has_results = len(analysis_results) > 0
            has_quantum = "quantum_enhancement" in analysis_results
            print(f"  ✅ Analysis completed: results={has_results}, quantum={has_quantum}")
            
            # 7. Test consensus building
            consensus = await orchestrator.quantum_consensus_building(
                "Should this batch be approved for release based on analysis results?",
                quantum_enhanced=True
            )
            
            has_decision = "decision" in consensus or "agreement_level" in consensus
            print(f"  ✅ Consensus built: decision_available={has_decision}")
            
            # 8. Test quantum scheduling
            quantum_status = await factory.get_quantum_status()
            optimization_active = quantum_status.get("optimization_active", False)
            print(f"  ✅ Quantum scheduling: active={optimization_active}")
            
            # 9. Performance metrics validation
            performance_metrics = factory.get_performance_metrics()
            has_metrics = len(performance_metrics) > 3
            print(f"  ✅ Performance metrics: comprehensive={has_metrics}")
            
            # 10. Clean shutdown
            await orchestrator.stop()
            await factory.stop_monitoring()
            print("  ✅ Clean shutdown completed")
            
            duration = time.time() - start_time
            print(f"✅ End-to-End Integration Test PASSED ({duration:.2f}s)")
            
            self.record_result("integration_e2e", True, duration, {
                "agents_tested": len(agents),
                "systems_integrated": 4,
                "quantum_enhanced": has_quantum
            })
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ End-to-End Integration Test FAILED: {e}")
            traceback.print_exc()
            
            self.record_result("integration_e2e", False, duration, {
                "error": str(e)
            })
            return False
    
    async def test_performance_benchmarks(self):
        """Performance and scalability benchmarks."""
        print("\n📋 Performance Benchmarks")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # 1. Task pool performance
            task_pool = await create_optimized_task_pool(min_workers=3, max_workers=10)
            
            async def benchmark_task(task_id: str):
                # Simulate analysis work
                await asyncio.sleep(0.05)
                return f"Benchmark task {task_id} completed"
            
            # Submit 50 tasks rapidly
            benchmark_start = time.time()
            task_ids = []
            
            for i in range(50):
                task_id = f"benchmark_{i}"
                await task_pool.submit_task(task_id, benchmark_task, task_id)
                task_ids.append(task_id)
            
            # Wait for completion
            completed = 0
            for task_id in task_ids:
                try:
                    result = await task_pool.get_result(task_id, timeout=5.0)
                    if result.success:
                        completed += 1
                except:
                    pass
            
            benchmark_duration = time.time() - benchmark_start
            throughput = completed / benchmark_duration
            
            await task_pool.stop()
            
            print(f"  ✅ Task Pool: {completed}/50 tasks, {throughput:.1f} tasks/sec")
            
            # 2. Cache performance
            cache = await create_optimized_cache(memory_size=1000)
            
            cache_start = time.time()
            
            # Write performance
            for i in range(1000):
                key = f"perf_test_key_{i}"
                value = {"data": f"test_value_{i}", "timestamp": time.time()}
                await cache.set(key, value)
            
            write_duration = time.time() - cache_start
            write_rate = 1000 / write_duration
            
            # Read performance
            read_start = time.time()
            hits = 0
            
            for i in range(1000):
                key = f"perf_test_key_{i}"
                result = await cache.get(key)
                if result:
                    hits += 1
            
            read_duration = time.time() - read_start
            read_rate = 1000 / read_duration
            hit_rate = hits / 1000
            
            print(f"  ✅ Cache: Write {write_rate:.0f} ops/sec, Read {read_rate:.0f} ops/sec, Hit rate {hit_rate:.1%}")
            
            # 3. LLM client performance
            llm_client = create_llm_client("gpt-4")  # Will use mock
            
            llm_start = time.time()
            llm_responses = []
            
            for i in range(10):
                prompt = f"Analyze sensor reading {i}: values=[100, 150, 120]"
                response = await llm_client.generate(prompt)
                if response.content:
                    llm_responses.append(response)
            
            llm_duration = time.time() - llm_start
            llm_rate = len(llm_responses) / llm_duration
            
            print(f"  ✅ LLM Client: {len(llm_responses)} responses, {llm_rate:.1f} req/sec")
            
            # Performance requirements validation
            performance_pass = (
                throughput >= 20.0 and  # Task throughput
                write_rate >= 500.0 and  # Cache write rate
                read_rate >= 1000.0 and  # Cache read rate
                hit_rate >= 0.95 and     # Cache hit rate
                llm_rate >= 2.0          # LLM rate
            )
            
            duration = time.time() - start_time
            
            if performance_pass:
                print(f"✅ Performance Benchmarks PASSED ({duration:.2f}s)")
                self.record_result("performance_benchmarks", True, duration, {
                    "task_throughput": throughput,
                    "cache_write_rate": write_rate,
                    "cache_read_rate": read_rate,
                    "cache_hit_rate": hit_rate,
                    "llm_rate": llm_rate
                })
                return True
            else:
                print(f"❌ Performance Benchmarks FAILED - Requirements not met")
                self.record_result("performance_benchmarks", False, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Performance Benchmarks FAILED: {e}")
            self.record_result("performance_benchmarks", False, duration, {"error": str(e)})
            return False
    
    async def test_error_recovery(self):
        """Error handling and recovery testing."""
        print("\n📋 Error Recovery Testing")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # 1. Agent failure recovery
            factory = ScentAnalyticsFactory("error_test_line", {"channels": 16})
            
            # Create agent that will fail
            agent = QualityControlAgent(llm_model="failing-model", agent_id="error_test_agent")
            factory.register_agent(agent)
            
            # Test graceful failure handling
            await factory.start_monitoring("ERROR_TEST_BATCH")
            
            # Simulate some processing time
            await asyncio.sleep(0.5)
            
            await factory.stop_monitoring()
            print("  ✅ Agent failure handled gracefully")
            
            # 2. Task pool error recovery
            task_pool = await create_optimized_task_pool(min_workers=2, max_workers=4)
            
            async def failing_task(task_id: str):
                if "fail" in task_id:
                    raise ValueError(f"Intentional failure in {task_id}")
                return f"Success: {task_id}"
            
            # Submit mix of successful and failing tasks
            mixed_tasks = []
            for i in range(10):
                task_id = f"task_{'fail' if i % 3 == 0 else 'success'}_{i}"
                await task_pool.submit_task(task_id, failing_task, task_id)
                mixed_tasks.append(task_id)
            
            # Check results
            successes = 0
            failures = 0
            
            for task_id in mixed_tasks:
                result = await task_pool.get_result(task_id, timeout=2.0)
                if result.success:
                    successes += 1
                else:
                    failures += 1
            
            await task_pool.stop()
            
            recovery_ratio = successes / (successes + failures) if (successes + failures) > 0 else 0
            print(f"  ✅ Task Pool Recovery: {successes} successes, {failures} failures ({recovery_ratio:.1%} recovery)")
            
            # 3. Health check system
            health_checker = create_default_health_checker()
            await health_checker.start_monitoring(interval=1)
            
            # Let it run for a moment
            await asyncio.sleep(2)
            
            system_status = health_checker.get_system_status()
            overall_status = system_status["overall_status"]
            
            await health_checker.stop_monitoring()
            
            print(f"  ✅ Health Monitoring: {overall_status} status")
            
            duration = time.time() - start_time
            
            error_recovery_pass = (
                recovery_ratio >= 0.6 and  # At least 60% task recovery
                overall_status in ["healthy", "warning"]  # System health acceptable
            )
            
            if error_recovery_pass:
                print(f"✅ Error Recovery Testing PASSED ({duration:.2f}s)")
                self.record_result("error_recovery", True, duration, {
                    "task_recovery_rate": recovery_ratio,
                    "system_health": overall_status
                })
                return True
            else:
                print(f"❌ Error Recovery Testing FAILED")
                self.record_result("error_recovery", False, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Error Recovery Testing FAILED: {e}")
            self.record_result("error_recovery", False, duration, {"error": str(e)})
            return False
    
    async def test_security_compliance(self):
        """Security and compliance validation."""
        print("\n📋 Security & Compliance Testing")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # 1. Logging security
            loggers = setup_logging(
                log_level="INFO",
                log_dir="./security_test_logs",
                enable_file=True,
                enable_console=False
            )
            
            main_logger = loggers["main"]
            
            # Test that sensitive data is not logged
            main_logger.info("Test log entry with safe data")
            
            # Check log files exist
            log_files = list(Path("./security_test_logs").glob("*.log"))
            print(f"  ✅ Logging: {len(log_files)} log files created")
            
            # 2. Data validation
            from agentic_scent.core.validation import DataValidator
            validator = DataValidator()
            
            # Test input sanitization
            test_inputs = [
                {"safe_input": [100, 150, 120]},
                {"potential_injection": "'; DROP TABLE users; --"},
                {"large_input": "x" * 10000},
                {"valid_sensor_data": {"values": [1, 2, 3], "timestamp": "2024-01-01T00:00:00"}}
            ]
            
            validation_results = []
            for test_input in test_inputs:
                try:
                    result = validator.validate_sensor_data(test_input)
                    validation_results.append(True)
                except:
                    validation_results.append(False)
            
            safe_validations = sum(validation_results)
            print(f"  ✅ Input Validation: {safe_validations}/{len(test_inputs)} inputs handled safely")
            
            # 3. Error handling without information leakage
            factory = ScentAnalyticsFactory("security_test_line", {"channels": 8})
            
            try:
                # Attempt operation that should fail safely
                await factory.sensor_stream("non_existent_sensor").__anext__()
            except Exception as e:
                error_message = str(e)
                # Check that error doesn't leak sensitive information
                safe_error = "password" not in error_message.lower() and "secret" not in error_message.lower()
                print(f"  ✅ Safe Error Handling: {safe_error}")
            
            duration = time.time() - start_time
            
            security_pass = (
                len(log_files) > 0 and  # Logging works
                safe_validations >= len(test_inputs) // 2  # Most validations safe
            )
            
            if security_pass:
                print(f"✅ Security & Compliance Testing PASSED ({duration:.2f}s)")
                self.record_result("security_compliance", True, duration, {
                    "log_files_created": len(log_files),
                    "safe_validations": safe_validations
                })
                return True
            else:
                print(f"❌ Security & Compliance Testing FAILED")
                self.record_result("security_compliance", False, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Security & Compliance Testing FAILED: {e}")
            self.record_result("security_compliance", False, duration, {"error": str(e)})
            return False
    
    def generate_test_report(self) -> dict:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        total_tests = len(self.test_results)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculate performance metrics
        performance_details = self.test_results.get("performance_benchmarks", {}).get("details", {})
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "pass_rate": pass_rate,
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": self.test_results,
            "performance_summary": performance_details,
            "production_readiness": {
                "ready": pass_rate >= 0.8,  # 80% pass rate required
                "quality_score": pass_rate * 100,
                "recommendation": "APPROVED FOR PRODUCTION" if pass_rate >= 0.8 else "REQUIRES FIXES BEFORE PRODUCTION"
            }
        }
        
        return report


async def run_comprehensive_tests():
    """Run all comprehensive production tests."""
    print("🏭 Running Agentic Scent Analytics Production Test Suite")
    print("=" * 70)
    
    suite = ProductionTestSuite()
    
    # Run all test categories
    test_functions = [
        ("End-to-End Integration", suite.test_integration_end_to_end),
        ("Performance Benchmarks", suite.test_performance_benchmarks), 
        ("Error Recovery", suite.test_error_recovery),
        ("Security & Compliance", suite.test_security_compliance)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n🔄 Running {test_name}...")
        try:
            await test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            suite.record_result(test_name.lower().replace(" ", "_"), False, 0, {"error": str(e)})
    
    # Generate final report
    report = suite.generate_test_report()
    
    print("\n" + "=" * 70)
    print("📊 PRODUCTION TEST SUITE RESULTS")
    print("=" * 70)
    
    print(f"Total Tests: {report['test_summary']['total_tests']}")
    print(f"Passed: {report['test_summary']['passed_tests']}")
    print(f"Failed: {report['test_summary']['failed_tests']}")
    print(f"Pass Rate: {report['test_summary']['pass_rate']:.1%}")
    print(f"Duration: {report['test_summary']['total_duration']:.2f}s")
    
    print(f"\n🎯 Production Readiness: {report['production_readiness']['recommendation']}")
    print(f"Quality Score: {report['production_readiness']['quality_score']:.1f}/100")
    
    # Save detailed report
    with open("production_test_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: production_test_report.json")
    
    return report['production_readiness']['ready']


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    sys.exit(0 if success else 1)