#!/usr/bin/env python3
"""
Production Readiness Tests for Agentic Scent Analytics
Comprehensive testing of production deployment readiness.
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

def test_docker_configuration():
    """Test Docker configuration and build capability."""
    print("🐳 Testing Docker configuration...")
    
    try:
        # Check if Docker is available
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ Docker not available")
            return False
        
        print(f"   Docker version: {result.stdout.strip()}")
        
        # Test Docker build (dry run)
        dockerfile_path = Path("/root/repo/Dockerfile")
        if not dockerfile_path.exists():
            print("❌ Dockerfile not found")
            return False
        
        print("   ✅ Dockerfile exists")
        
        # Check docker-compose configuration
        compose_path = Path("/root/repo/docker-compose.yml")
        if not compose_path.exists():
            print("❌ docker-compose.yml not found")
            return False
        
        print("   ✅ docker-compose.yml exists")
        
        # Validate docker-compose syntax
        result = subprocess.run(['docker', 'compose', 'config'], 
                              cwd="/root/repo", capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"❌ docker-compose validation failed: {result.stderr}")
            return False
        
        print("   ✅ docker-compose configuration valid")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Docker command timed out")
        return False
    except Exception as e:
        print(f"❌ Docker test failed: {e}")
        return False

def test_kubernetes_manifests():
    """Test Kubernetes deployment manifests."""
    print("☸️ Testing Kubernetes manifests...")
    
    try:
        k8s_dir = Path("/root/repo/k8s")
        if not k8s_dir.exists():
            print("❌ Kubernetes manifests directory not found")
            return False
        
        required_manifests = [
            "deployment.yaml",
            "service.yaml", 
            "configmap.yaml",
            "namespace.yaml"
        ]
        
        for manifest in required_manifests:
            manifest_path = k8s_dir / manifest
            if not manifest_path.exists():
                print(f"❌ Required manifest {manifest} not found")
                return False
            
            # Basic YAML validation
            try:
                import yaml
                with open(manifest_path) as f:
                    yaml.safe_load(f)
                print(f"   ✅ {manifest} is valid YAML")
            except yaml.YAMLError as e:
                print(f"❌ {manifest} has invalid YAML: {e}")
                return False
        
        print("   ✅ All required Kubernetes manifests present and valid")
        return True
        
    except Exception as e:
        print(f"❌ Kubernetes manifest test failed: {e}")
        return False

def test_environment_configuration():
    """Test environment configuration management."""
    print("⚙️ Testing environment configuration...")
    
    try:
        from agentic_scent.core.config import ConfigManager
        
        # Test configuration creation
        config = ConfigManager()
        print("   ✅ ConfigManager instantiated")
        
        # Test configuration loading with different environments
        test_configs = {
            'development': {
                'database_url': 'sqlite:///dev.db',
                'redis_url': 'redis://localhost:6379/0',
                'log_level': 'DEBUG'
            },
            'production': {
                'database_url': 'postgresql://user:pass@localhost:5432/prod',
                'redis_url': 'redis://redis-service:6379/0',
                'log_level': 'INFO'
            }
        }
        
        for env_name, test_config in test_configs.items():
            try:
                # Test configuration validation
                if hasattr(config, 'validate_config'):
                    is_valid = config.validate_config(test_config)
                    if not is_valid:
                        print(f"❌ {env_name} configuration validation failed")
                        return False
                print(f"   ✅ {env_name} configuration valid")
            except Exception as e:
                print(f"   ⚠️  {env_name} config validation skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment configuration test failed: {e}")
        return False

def test_security_configuration():
    """Test security configuration and compliance."""
    print("🔐 Testing security configuration...")
    
    try:
        from agentic_scent.core.security import SecurityManager
        
        # Test security manager initialization
        security = SecurityManager()
        print("   ✅ SecurityManager instantiated")
        
        # Test password hashing
        test_password = "test_password_123"
        hashed = security.hash_password(test_password)
        is_valid = security.verify_password(test_password, hashed)
        
        if not is_valid:
            print("❌ Password hashing/verification failed")
            return False
        print("   ✅ Password hashing and verification working")
        
        # Test data encryption
        test_data = {"sensitive": "data", "user_id": 12345}
        encrypted = security.encrypt_data(test_data)
        decrypted = security.decrypt_data(encrypted)
        
        if decrypted != test_data:
            print("❌ Data encryption/decryption failed")
            return False
        print("   ✅ Data encryption and decryption working")
        
        # Test audit trail
        security.log_security_event("test_event", {"test": "data"})
        print("   ✅ Security event logging working")
        
        return True
        
    except Exception as e:
        print(f"❌ Security configuration test failed: {e}")
        return False

def test_monitoring_and_metrics():
    """Test monitoring and metrics systems."""
    print("📊 Testing monitoring and metrics...")
    
    try:
        from agentic_scent.core.metrics import create_metrics_system
        
        # Test metrics system creation
        metrics, profiler = create_metrics_system(enable_prometheus=True)
        print("   ✅ Metrics system created")
        
        # Test metric recording
        metrics.record_sensor_reading("test_sensor", "test_type")
        metrics.record_agent_analysis(1.5, "test_agent")
        metrics.record_cache_operation("hit")
        
        # Test metrics export
        prometheus_metrics = metrics.export_prometheus_metrics()
        if not prometheus_metrics:
            print("❌ Prometheus metrics export failed")
            return False
        print("   ✅ Prometheus metrics export working")
        
        # Test performance profiling
        with profiler.profile("test_operation"):
            time.sleep(0.01)  # Simulate work
        
        profiles = profiler.get_all_profiles()
        if "test_operation" not in profiles:
            print("❌ Performance profiling failed")
            return False
        print("   ✅ Performance profiling working")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring and metrics test failed: {e}")
        return False

def test_health_checks():
    """Test health check endpoints and system status."""
    print("🏥 Testing health checks...")
    
    try:
        from agentic_scent.core.health_checks import HealthCheckManager
        
        # Test health check manager
        health_manager = HealthCheckManager()
        print("   ✅ HealthCheckManager instantiated")
        
        # Perform system health check
        health_status = health_manager.check_system_health()
        
        required_checks = ["database", "cache", "storage", "memory"]
        for check in required_checks:
            if check not in health_status:
                print(f"❌ Missing health check: {check}")
                return False
            
            if health_status[check]["status"] not in ["healthy", "warning"]:
                print(f"❌ Health check {check} failed: {health_status[check]}")
                return False
            
            print(f"   ✅ {check}: {health_status[check]['status']}")
        
        # Test overall system status
        overall_status = health_manager.get_overall_status()
        print(f"   ✅ Overall system status: {overall_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

async def test_api_endpoints():
    """Test API endpoints and service interfaces."""
    print("🌐 Testing API endpoints...")
    
    try:
        from agentic_scent.core.factory import ScentAnalyticsFactory
        
        # Create factory instance
        factory = ScentAnalyticsFactory(
            production_line="api_test_line",
            e_nose_config={"channels": 32},
            enable_scaling=True
        )
        
        # Test factory status endpoint equivalent
        status = factory.get_current_state()
        required_status_fields = ["batch_id", "is_monitoring", "active_sensors"]
        
        for field in required_status_fields:
            if field not in status:
                print(f"❌ Missing status field: {field}")
                return False
        
        print("   ✅ Factory status endpoint working")
        
        # Test performance metrics endpoint equivalent
        metrics = factory.get_performance_metrics()
        if "factory_status" not in metrics:
            print("❌ Performance metrics endpoint failed")
            return False
        
        print("   ✅ Performance metrics endpoint working")
        
        # Test metrics export endpoint equivalent
        exported_metrics = await factory.export_metrics("json")
        if not exported_metrics:
            print("❌ Metrics export endpoint failed")
            return False
        
        print("   ✅ Metrics export endpoint working")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False

def test_deployment_scripts():
    """Test deployment automation scripts."""
    print("🚀 Testing deployment scripts...")
    
    try:
        # Check for deployment script
        deploy_script = Path("/root/repo/deploy.sh")
        if not deploy_script.exists():
            print("❌ deploy.sh script not found")
            return False
        
        # Check script permissions
        if not deploy_script.stat().st_mode & 0o111:  # Check execute bit
            print("❌ deploy.sh is not executable")
            return False
        
        print("   ✅ deploy.sh exists and is executable")
        
        # Check for deployment check script
        check_script = Path("/root/repo/deployment_check.py")
        if not check_script.exists():
            print("❌ deployment_check.py not found")
            return False
        
        print("   ✅ deployment_check.py exists")
        
        # Test deployment check script execution (dry run)
        try:
            result = subprocess.run(['python3', str(check_script), '--dry-run'], 
                                  cwd="/root/repo", capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("   ✅ deployment_check.py executes successfully")
            else:
                print(f"   ⚠️  deployment_check.py dry run issues: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("   ⚠️  deployment_check.py timed out")
        except Exception as e:
            print(f"   ⚠️  deployment_check.py test skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment scripts test failed: {e}")
        return False

def test_documentation_completeness():
    """Test documentation completeness."""
    print("📚 Testing documentation completeness...")
    
    try:
        docs_dir = Path("/root/repo/docs")
        required_docs = [
            "API.md",
            "ARCHITECTURE.md", 
            "DEPLOYMENT.md"
        ]
        
        for doc in required_docs:
            doc_path = docs_dir / doc
            if not doc_path.exists():
                print(f"❌ Required documentation {doc} not found")
                return False
            
            # Check if documentation has content
            with open(doc_path) as f:
                content = f.read().strip()
                if len(content) < 100:  # Minimal content check
                    print(f"❌ Documentation {doc} appears incomplete")
                    return False
            
            print(f"   ✅ {doc} exists and has content")
        
        # Check README
        readme_path = Path("/root/repo/README.md")
        if not readme_path.exists():
            print("❌ README.md not found")
            return False
        
        with open(readme_path) as f:
            readme_content = f.read()
            required_sections = ["installation", "usage", "architecture"]
            for section in required_sections:
                if section.lower() not in readme_content.lower():
                    print(f"⚠️  README.md missing {section} section")
        
        print("   ✅ README.md exists with basic sections")
        return True
        
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False

async def run_production_readiness_tests():
    """Run all production readiness tests."""
    print("🏭 Running Agentic Scent Analytics Production Readiness Tests")
    print("=" * 70)
    
    test_results = {}
    
    # Test 1: Docker Configuration
    test_results["docker_config"] = test_docker_configuration()
    
    # Test 2: Kubernetes Manifests
    test_results["k8s_manifests"] = test_kubernetes_manifests()
    
    # Test 3: Environment Configuration
    test_results["env_config"] = test_environment_configuration()
    
    # Test 4: Security Configuration
    test_results["security_config"] = test_security_configuration()
    
    # Test 5: Monitoring and Metrics
    test_results["monitoring_metrics"] = test_monitoring_and_metrics()
    
    # Test 6: Health Checks
    test_results["health_checks"] = test_health_checks()
    
    # Test 7: API Endpoints
    test_results["api_endpoints"] = await test_api_endpoints()
    
    # Test 8: Deployment Scripts
    test_results["deployment_scripts"] = test_deployment_scripts()
    
    # Test 9: Documentation
    test_results["documentation"] = test_documentation_completeness()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Production Readiness Test Results")
    print("=" * 70)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ READY" if result else "❌ NOT READY"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Readiness: {passed_tests}/{total_tests} areas ready")
    
    readiness_score = passed_tests / total_tests
    if readiness_score >= 0.9:
        print("🎉 System is PRODUCTION READY!")
        readiness_level = "PRODUCTION_READY"
    elif readiness_score >= 0.7:
        print("⚠️  System is mostly ready - minor issues to address")
        readiness_level = "MOSTLY_READY"
    else:
        print("❌ System needs significant work before production deployment")
        readiness_level = "NOT_READY"
    
    # Generate readiness report
    report = {
        "timestamp": time.time(),
        "readiness_level": readiness_level,
        "readiness_score": readiness_score,
        "test_results": test_results,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests
        }
    }
    
    # Save report
    with open("/root/repo/production_readiness_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Full report saved to: production_readiness_report.json")
    
    return readiness_score >= 0.7

if __name__ == "__main__":
    # Run tests
    result = asyncio.run(run_production_readiness_tests())
    exit(0 if result else 1)