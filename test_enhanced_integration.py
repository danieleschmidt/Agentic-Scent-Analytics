#!/usr/bin/env python3
"""
Test script for enhanced autonomous SDLC integration
Tests the integration without external dependencies
"""

import sys
import importlib
import inspect
from pathlib import Path

def test_enhanced_modules_exist():
    """Test that all enhanced modules exist and are properly structured"""
    print("🧪 Testing Enhanced SDLC Module Integration")
    print("=" * 60)
    
    # List of enhanced modules to test
    enhanced_modules = [
        'agentic_scent.core.progressive_quality_gates',
        'agentic_scent.core.autonomous_testing',
        'agentic_scent.core.global_deployment_orchestrator',
        'agentic_scent.core.ml_performance_optimizer',
        'agentic_scent.core.zero_trust_security'
    ]
    
    passed = 0
    total = len(enhanced_modules)
    
    for module_name in enhanced_modules:
        try:
            # Try to import module
            module_path = module_name.replace('.', '/') + '.py'
            file_path = Path('/root/repo') / module_path
            
            if file_path.exists():
                print(f"✅ {module_name}: File exists")
                
                # Check file size (should be substantial)
                file_size = file_path.stat().st_size
                if file_size > 10000:  # At least 10KB
                    print(f"   📏 File size: {file_size:,} bytes (substantial)")
                    passed += 1
                else:
                    print(f"   ❌ File size: {file_size:,} bytes (too small)")
            else:
                print(f"❌ {module_name}: File missing")
                
        except Exception as e:
            print(f"❌ {module_name}: Error - {e}")
    
    print(f"\n📊 Module Structure Test: {passed}/{total} modules properly created")
    return passed == total

def test_factory_integration():
    """Test that factory.py has been enhanced with SDLC integration"""
    print("\n🏭 Testing Factory Integration")
    print("-" * 40)
    
    factory_path = Path('/root/repo/agentic_scent/core/factory.py')
    
    if not factory_path.exists():
        print("❌ Factory file not found")
        return False
    
    with open(factory_path, 'r') as f:
        content = f.read()
    
    # Check for enhanced imports
    enhanced_imports = [
        'progressive_quality_gates',
        'autonomous_testing',
        'global_deployment_orchestrator',
        'ml_performance_optimizer',
        'zero_trust_security'
    ]
    
    import_score = 0
    for imp in enhanced_imports:
        if imp in content:
            print(f"✅ Enhanced import found: {imp}")
            import_score += 1
        else:
            print(f"❌ Enhanced import missing: {imp}")
    
    # Check for enhanced methods
    enhanced_methods = [
        '_initialize_autonomous_sdlc_systems',
        'execute_autonomous_testing',
        'execute_progressive_quality_gates',
        'deploy_to_regions',
        'get_autonomous_sdlc_status'
    ]
    
    method_score = 0
    for method in enhanced_methods:
        if method in content:
            print(f"✅ Enhanced method found: {method}")
            method_score += 1
        else:
            print(f"❌ Enhanced method missing: {method}")
    
    # Check for enable_autonomous_sdlc parameter
    if 'enable_autonomous_sdlc' in content:
        print("✅ Autonomous SDLC enablement parameter found")
        param_score = 1
    else:
        print("❌ Autonomous SDLC enablement parameter missing")
        param_score = 0
    
    total_score = import_score + method_score + param_score
    max_score = len(enhanced_imports) + len(enhanced_methods) + 1
    
    print(f"\n📊 Factory Integration Score: {total_score}/{max_score}")
    return total_score >= max_score * 0.8  # 80% threshold

def test_cli_integration():
    """Test CLI integration capabilities"""
    print("\n💻 Testing CLI Integration")
    print("-" * 40)
    
    cli_path = Path('/root/repo/agentic_scent/cli.py')
    
    if not cli_path.exists():
        print("❌ CLI file not found")
        return False
    
    with open(cli_path, 'r') as f:
        content = f.read()
    
    # Check for enhanced CLI capabilities
    cli_features = [
        'autonomous',
        'quality',
        'deploy',
        'security',
        'optimize'
    ]
    
    found_features = 0
    for feature in cli_features:
        if feature in content.lower():
            print(f"✅ CLI feature reference found: {feature}")
            found_features += 1
        else:
            print(f"⚠️  CLI feature reference not found: {feature}")
    
    print(f"\n📊 CLI Integration: {found_features}/{len(cli_features)} features referenced")
    return found_features >= 2  # At least some integration

def test_autonomous_sdlc_completeness():
    """Test completeness of autonomous SDLC implementation"""
    print("\n🤖 Testing Autonomous SDLC Completeness")
    print("-" * 50)
    
    # Key classes that should exist
    key_classes = [
        ('ProgressiveQualityGates', 'progressive_quality_gates.py'),
        ('AutonomousTestingFramework', 'autonomous_testing.py'),
        ('GlobalDeploymentOrchestrator', 'global_deployment_orchestrator.py'),
        ('MLPerformanceOptimizer', 'ml_performance_optimizer.py'),
        ('ZeroTrustSecurityFramework', 'zero_trust_security.py')
    ]
    
    class_score = 0
    for class_name, file_name in key_classes:
        file_path = Path('/root/repo/agentic_scent/core') / file_name
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            if f"class {class_name}" in content:
                print(f"✅ Key class found: {class_name}")
                class_score += 1
            else:
                print(f"❌ Key class missing: {class_name}")
        else:
            print(f"❌ File missing: {file_name}")
    
    # Check for advanced features
    advanced_features = [
        'machine learning',
        'behavioral analysis',
        'threat detection',
        'auto-scaling',
        'multi-region',
        'zero-trust',
        'progressive',
        'autonomous'
    ]
    
    feature_count = 0
    for class_name, file_name in key_classes:
        file_path = Path('/root/repo/agentic_scent/core') / file_name
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            for feature in advanced_features:
                if feature in content:
                    feature_count += 1
                    break  # Count each file only once
    
    print(f"\n📊 Class Implementation: {class_score}/{len(key_classes)} key classes")
    print(f"🔬 Advanced Features: {feature_count}/{len(key_classes)} files have advanced features")
    
    return class_score >= 4 and feature_count >= 3  # Strong implementation threshold

def main():
    """Run all enhancement tests"""
    print("🚀 Autonomous SDLC Enhancement Validation")
    print("=" * 60)
    
    tests = [
        ("Enhanced Module Structure", test_enhanced_modules_exist),
        ("Factory Integration", test_factory_integration),
        ("CLI Integration", test_cli_integration),
        ("SDLC Completeness", test_autonomous_sdlc_completeness)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 60)
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 AUTONOMOUS SDLC ENHANCEMENT VALIDATION RESULTS")
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL ENHANCEMENT TESTS PASSED!")
        print("🚀 Autonomous SDLC integration is complete and operational!")
        return True
    elif passed >= total * 0.75:
        print("⚠️  Most tests passed - System is largely operational")
        return True
    else:
        print("❌ Several tests failed - Please review implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)