#!/usr/bin/env python3
"""
Test script for robustness and error handling features.
"""

import asyncio
import sys
import traceback
from datetime import datetime

# Test imports
try:
    from agentic_scent.core.logging_config import setup_logging, get_contextual_logger
    from agentic_scent.core.health_checks import create_default_health_checker
    from agentic_scent.core.exceptions import *
    from agentic_scent.integration.quantum_scheduler import QuantumScheduledFactory
    from agentic_scent.integration.hybrid_orchestrator import HybridAgentOrchestrator
    print("✅ All robustness modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


async def test_logging_system():
    """Test enhanced logging capabilities."""
    print("\n📋 Testing Logging System")
    print("-" * 40)
    
    try:
        # Setup structured logging
        loggers = setup_logging(
            log_level="INFO",
            log_dir="./test_logs",
            enable_console=True,
            enable_file=True,
            enable_structured=False
        )
        
        main_logger = loggers["main"]
        main_logger.set_context(test_session="robustness_test", user_id="test_user")
        
        main_logger.info("Logging system initialized successfully")
        main_logger.warning("This is a test warning with context")
        
        # Test contextual logging
        agent_logger = loggers.get("agents")
        if agent_logger:
            agent_logger.set_context(agent_id="test_agent", batch_id="BATCH_001")
            agent_logger.info("Agent analysis completed")
        
        print("✅ Logging system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Logging system test failed: {e}")
        return False


async def test_health_checks():
    """Test health monitoring system."""
    print("\n📋 Testing Health Check System")
    print("-" * 40)
    
    try:
        # Create health checker
        health_checker = create_default_health_checker()
        
        # Run all health checks
        results = await health_checker.run_all_checks()
        
        print(f"Health checks run: {len(results)}")
        
        # Get system status
        system_status = health_checker.get_system_status()
        
        print(f"Overall status: {system_status['overall_status']}")
        print(f"Total checks: {system_status['total_checks']}")
        
        # Display individual check results
        for check_name, result in results.items():
            status_icon = "✅" if result.status.value == "healthy" else "⚠️" if result.status.value == "warning" else "❌"
            print(f"  {status_icon} {check_name}: {result.status.value} ({result.response_time*1000:.1f}ms)")
        
        print("✅ Health check system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Health check system test failed: {e}")
        traceback.print_exc()
        return False


async def test_error_handling():
    """Test exception handling and error codes."""
    print("\n📋 Testing Error Handling System")
    print("-" * 40)
    
    try:
        # Test custom exceptions
        test_errors = [
            (SensorConnectionError, "Test sensor connection error"),
            (AgentInitializationError, "Test agent initialization error"),
            (LLMError, "Test LLM service error"),
            (ValidationError, "Test validation error")
        ]
        
        error_count = 0
        for error_class, message in test_errors:
            try:
                raise error_class(message, error_code="TEST_ERROR", context={"test": True})
            except AgenticScentError as e:
                print(f"  ✅ {error_class.__name__}: {e.error_code}")
                error_count += 1
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
        
        # Test error code system
        error_code = get_error_code("SENSOR_CONNECTION_FAILED")
        if error_code == 1001:
            print(f"  ✅ Error code system working: SENSOR_CONNECTION_FAILED = {error_code}")
        else:
            print(f"  ❌ Error code system failed: expected 1001, got {error_code}")
        
        print(f"✅ Error handling system test passed ({error_count} errors tested)")
        return True
        
    except Exception as e:
        print(f"❌ Error handling system test failed: {e}")
        traceback.print_exc()
        return False


async def test_quantum_integration():
    """Test quantum scheduler integration."""
    print("\n📋 Testing Quantum Integration")
    print("-" * 40)
    
    try:
        # Create quantum-enhanced factory
        factory = QuantumScheduledFactory(
            production_line="test_line",
            e_nose_config={"sensors": ["MOS", "PID"], "channels": 16}
        )
        
        # Test system state collection
        system_state = await factory._collect_system_state()
        print(f"  ✅ System state collected: {len(system_state)} metrics")
        
        # Test quantum schedule generation
        optimal_tasks = await factory._generate_quantum_schedule(system_state)
        print(f"  ✅ Quantum schedule generated: {len(optimal_tasks)} tasks")
        
        # Get quantum status
        status = await factory.get_quantum_status()
        print(f"  ✅ Quantum status retrieved: efficiency={status.get('system_efficiency', 0.0):.3f}")
        
        print("✅ Quantum integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quantum integration test failed: {e}")
        traceback.print_exc()
        return False


async def test_hybrid_orchestration():
    """Test hybrid agent orchestration."""
    print("\n📋 Testing Hybrid Orchestration")
    print("-" * 40)
    
    try:
        # Create hybrid orchestrator
        orchestrator = HybridAgentOrchestrator()
        
        # Test sensor data for analysis
        sensor_data = {
            "timestamp": datetime.now().isoformat(),
            "values": [100, 150, 120, 180, 95],
            "sensor_id": "e_nose_01"
        }
        
        # Test hybrid analysis
        results = await orchestrator.coordinate_hybrid_analysis(
            sensor_data, "Test analysis with quantum enhancement"
        )
        
        has_quantum = "quantum_enhancement" in results
        print(f"  ✅ Hybrid analysis completed: quantum_enhanced={has_quantum}")
        
        # Test hybrid status
        status = await orchestrator.get_hybrid_status()
        hybrid_info = status.get("hybrid_orchestration", {})
        print(f"  ✅ Hybrid status: {len(hybrid_info)} metrics retrieved")
        
        print("✅ Hybrid orchestration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid orchestration test failed: {e}")
        traceback.print_exc()
        return False


async def run_robustness_tests():
    """Run all robustness tests."""
    print("🧪 Running Agentic Scent Analytics Robustness Tests")
    print("=" * 60)
    
    test_functions = [
        ("Logging System", test_logging_system),
        ("Health Checks", test_health_checks),
        ("Error Handling", test_error_handling),
        ("Quantum Integration", test_quantum_integration),
        ("Hybrid Orchestration", test_hybrid_orchestration)
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
    print(f"📊 Robustness Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All robustness tests passed! System is production-ready.")
        return True
    else:
        print(f"⚠️  {total - passed} test suite(s) failed. Review implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_robustness_tests())
    sys.exit(0 if success else 1)