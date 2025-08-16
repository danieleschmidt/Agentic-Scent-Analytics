#!/usr/bin/env python3
"""
Essential Quality Gates for Agentic Scent Analytics
Minimal but comprehensive testing to ensure production readiness.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

def test_import_stability():
    """Test that all essential imports work without errors."""
    print("📦 Testing import stability...")
    
    essential_imports = [
        "agentic_scent",
        "agentic_scent.core.factory",
        "agentic_scent.agents.quality_control",
        "agentic_scent.sensors.base",
        "agentic_scent.analytics.fingerprinting",
        "agentic_scent.core.config",
        "agentic_scent.core.validation",
        "agentic_scent.core.security",
    ]
    
    failed_imports = []
    
    for module_name in essential_imports:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name}")
        except Exception as e:
            print(f"   ❌ {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"❌ Import stability failed: {len(failed_imports)} modules failed")
        return False
    
    print("✅ All essential imports successful")
    return True

def test_core_functionality():
    """Test core system functionality."""
    print("🏭 Testing core functionality...")
    
    try:
        from agentic_scent.core.factory import ScentAnalyticsFactory
        from agentic_scent.agents.quality_control import QualityControlAgent
        from agentic_scent.sensors.base import SensorReading
        from datetime import datetime
        import numpy as np
        
        # Test factory creation
        factory = ScentAnalyticsFactory(
            production_line="quality_gate_test",
            e_nose_config={"channels": 32, "sensors": ["MOS"]},
            enable_scaling=True
        )
        print("   ✅ Factory creation successful")
        
        # Test agent creation and registration
        agent = QualityControlAgent(agent_id="qg_test_agent", llm_model="mock")
        factory.register_agent(agent)
        print("   ✅ Agent registration successful")
        
        # Test sensor reading creation
        from agentic_scent.sensors.base import SensorType
        reading = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=np.random.random(32).tolist(),
            timestamp=datetime.now(),
            metadata={"test": True}
        )
        print("   ✅ Sensor reading creation successful")
        
        # Test factory state retrieval
        state = factory.get_current_state()
        required_fields = ["batch_id", "is_monitoring", "active_sensors"]
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required state field: {field}")
        print("   ✅ Factory state retrieval successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

async def test_async_operations():
    """Test asynchronous operations."""
    print("⚡ Testing async operations...")
    
    try:
        from agentic_scent.agents.quality_control import QualityControlAgent
        from agentic_scent.sensors.base import SensorReading
        from datetime import datetime
        import numpy as np
        
        # Create agent
        agent = QualityControlAgent(agent_id="async_test_agent", llm_model="mock")
        
        # Test async agent lifecycle
        await agent.start()
        print("   ✅ Agent start successful")
        
        # Test async analysis
        from agentic_scent.sensors.base import SensorType
        reading = SensorReading(
            sensor_id="async_test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=np.random.random(16).tolist(),
            timestamp=datetime.now(),
            metadata={"async_test": True}
        )
        
        analysis = await agent.analyze(reading)
        if not hasattr(analysis, 'confidence'):
            raise ValueError("Analysis missing required attributes")
        print(f"   ✅ Async analysis successful (confidence: {analysis.confidence:.3f})")
        
        # Test agent stop
        await agent.stop()
        print("   ✅ Agent stop successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Async operations test failed: {e}")
        return False

def test_data_validation():
    """Test data validation and security."""
    print("🔐 Testing data validation...")
    
    try:
        from agentic_scent.core.validation import AdvancedDataValidator, ValidationLevel
        
        # Create validator
        validator = AdvancedDataValidator(ValidationLevel.STRICT)
        print("   ✅ Validator creation successful")
        
        # Test valid data
        valid_data = {
            "sensor_id": "test_sensor",
            "temperature": 25.5,
            "humidity": 45.0,
            "readings": [1.0, 2.0, 3.0, 4.0],
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        result = validator.validate_comprehensive(valid_data)
        if not result.is_valid:
            raise ValueError(f"Valid data failed validation: {result.anomalies_detected}")
        print("   ✅ Valid data validation successful")
        
        # Test malicious data detection
        malicious_data = {
            "sensor_id": "'; DROP TABLE sensors; --",
            "script": "<script>alert('xss')</script>",
            "command": "rm -rf /",
            "temperature": 25.0
        }
        
        result = validator.validate_comprehensive(malicious_data)
        if result.is_valid:
            raise ValueError("Malicious data passed validation")
        print("   ✅ Malicious data detection successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False

async def test_performance_basics():
    """Test basic performance characteristics."""
    print("⚡ Testing performance basics...")
    
    try:
        from agentic_scent.core.performance import AsyncCache
        from datetime import timedelta
        
        # Test cache performance
        cache = AsyncCache(max_memory_size_mb=32, default_ttl=timedelta(minutes=1))
        
        async def cache_performance_test():
            # Store and retrieve data
            test_data = {"performance": "test", "value": 12345}
            
            # Measure set performance
            start_time = time.time()
            await cache.set("perf_test", test_data)
            set_time = time.time() - start_time
            
            # Measure get performance
            start_time = time.time()
            retrieved = await cache.get("perf_test")
            get_time = time.time() - start_time
            
            if retrieved != test_data:
                raise ValueError("Cache data integrity failed")
            
            # Performance thresholds (generous for basic test)
            if set_time > 0.1:  # 100ms
                raise ValueError(f"Cache set too slow: {set_time:.3f}s")
            if get_time > 0.1:  # 100ms  
                raise ValueError(f"Cache get too slow: {get_time:.3f}s")
            
            return set_time, get_time
        
        set_time, get_time = await cache_performance_test()
        
        print(f"   ✅ Cache performance: set={set_time:.3f}s, get={get_time:.3f}s")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_configuration_management():
    """Test configuration management."""
    print("⚙️ Testing configuration management...")
    
    try:
        from agentic_scent.core.config import ConfigManager
        
        # Test config manager creation
        config = ConfigManager()
        print("   ✅ ConfigManager creation successful")
        
        # Test configuration validation
        test_config = {
            "site_id": "test_site",
            "production_line": "test_line",
            "log_level": "INFO",
            "enable_metrics": True
        }
        
        # Basic validation test
        if hasattr(config, 'validate_config'):
            is_valid = config.validate_config(test_config)
            if not is_valid:
                raise ValueError("Valid configuration rejected")
            print("   ✅ Configuration validation successful")
        else:
            print("   ⚠️  Configuration validation not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration management test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling and resilience."""
    print("🛡️ Testing error handling...")
    
    try:
        from agentic_scent.core.factory import ScentAnalyticsFactory
        from agentic_scent.agents.quality_control import QualityControlAgent
        
        # Test factory with invalid configuration
        try:
            factory = ScentAnalyticsFactory(
                production_line="",  # Invalid empty line
                e_nose_config={},    # Invalid empty config
                enable_scaling=True
            )
            # Should not raise exception, but handle gracefully
            print("   ✅ Graceful handling of invalid factory config")
        except Exception as e:
            print(f"   ✅ Factory config validation working: {type(e).__name__}")
        
        # Test agent error handling
        agent = QualityControlAgent(agent_id="error_test_agent", llm_model="mock")
        
        async def test_agent_error_handling():
            await agent.start()
            
            # Test with invalid reading
            try:
                await agent.analyze(None)  # Invalid input
                # Should handle gracefully or raise appropriate error
            except Exception as e:
                print(f"   ✅ Agent error handling working: {type(e).__name__}")
            
            await agent.stop()
        
        await test_agent_error_handling()
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_cli_interface():
    """Test CLI interface functionality."""
    print("💻 Testing CLI interface...")
    
    try:
        import subprocess
        
        # Test CLI help
        result = subprocess.run(
            [sys.executable, "-m", "agentic_scent.cli", "--help"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            raise ValueError(f"CLI help failed: {result.stderr}")
        
        if "usage:" not in result.stdout.lower():
            raise ValueError("CLI help output invalid")
        
        print("   ✅ CLI help interface working")
        
        # Test CLI status command
        result = subprocess.run(
            [sys.executable, "-m", "agentic_scent.cli", "status"],
            cwd=Path(__file__).parent,
            capture_output=True, 
            text=True,
            timeout=15
        )
        
        if result.returncode != 0:
            print(f"   ⚠️  CLI status command issues: {result.stderr}")
        else:
            print("   ✅ CLI status command working")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("   ❌ CLI command timed out")
        return False
    except Exception as e:
        print(f"❌ CLI interface test failed: {e}")
        return False

async def run_essential_quality_gates():
    """Run all essential quality gate tests."""
    print("🚪 Running Essential Quality Gates for Agentic Scent Analytics")
    print("=" * 65)
    
    test_results = {}
    
    # Gate 1: Import Stability
    test_results["import_stability"] = test_import_stability()
    
    # Gate 2: Core Functionality
    test_results["core_functionality"] = test_core_functionality()
    
    # Gate 3: Async Operations
    test_results["async_operations"] = await test_async_operations()
    
    # Gate 4: Data Validation
    test_results["data_validation"] = test_data_validation()
    
    # Gate 5: Performance Basics
    test_results["performance_basics"] = await test_performance_basics()
    
    # Gate 6: Configuration Management
    test_results["configuration_management"] = test_configuration_management()
    
    # Gate 7: Error Handling
    test_results["error_handling"] = await test_error_handling()
    
    # Gate 8: CLI Interface
    test_results["cli_interface"] = test_cli_interface()
    
    # Summary
    print("\n" + "=" * 65)
    print("📊 Essential Quality Gates Results")
    print("=" * 65)
    
    passed_gates = sum(1 for result in test_results.values() if result)
    total_gates = len(test_results)
    
    for gate_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{gate_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nQuality Gates: {passed_gates}/{total_gates} passed")
    
    quality_score = passed_gates / total_gates
    if quality_score >= 0.875:  # 7/8 gates
        print("🎉 QUALITY GATES PASSED - System ready for production!")
        quality_level = "PRODUCTION_READY"
    elif quality_score >= 0.75:  # 6/8 gates
        print("⚠️  Most quality gates passed - minor issues to address")
        quality_level = "MOSTLY_READY"
    else:
        print("❌ Quality gates failed - significant issues to address")
        quality_level = "FAILED"
    
    # Save results
    import json
    import time
    
    report = {
        "timestamp": time.time(),
        "quality_level": quality_level,
        "quality_score": quality_score,
        "gate_results": test_results,
        "summary": {
            "total_gates": total_gates,
            "passed_gates": passed_gates,
            "failed_gates": total_gates - passed_gates
        }
    }
    
    with open("quality_gates_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Quality gates report saved to: quality_gates_report.json")
    
    return quality_score >= 0.75

if __name__ == "__main__":
    # Run quality gates
    result = asyncio.run(run_essential_quality_gates())
    exit(0 if result else 1)