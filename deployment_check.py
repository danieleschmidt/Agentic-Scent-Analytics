#!/usr/bin/env python3
"""
Production Deployment Readiness Check for Agentic Scent Analytics
"""

import os
import sys
import json
import subprocess
import time
from typing import Dict, List, Any
from datetime import datetime
import asyncio

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("📦 Checking Dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'click', 'asyncio',
        'typing-extensions', 'psutil', 'prometheus-client'
    ]
    
    missing_packages = []
    installed_packages = {}
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            installed_packages[package] = version
            print(f"  ✅ {package}: {version}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}: NOT INSTALLED")
    
    return len(missing_packages) == 0, installed_packages, missing_packages


def check_file_structure():
    """Check if all critical files are present."""
    print("\n📁 Checking File Structure...")
    
    critical_files = [
        'setup.py',
        'requirements.txt',
        'README.md',
        'CLAUDE.md',
        'agentic_scent/__init__.py',
        'agentic_scent/core/factory.py',
        'agentic_scent/agents/quality_control.py',
        'agentic_scent/sensors/base.py',
        'agentic_scent/analytics/fingerprinting.py',
        'deploy.sh',
        'docker-compose.yml',
        'k8s/deployment.yaml'
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in critical_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(full_path):
            present_files.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}: MISSING")
    
    return len(missing_files) == 0, present_files, missing_files


def check_imports():
    """Check if all modules can be imported successfully."""
    print("\n🐍 Checking Module Imports...")
    
    critical_imports = [
        'agentic_scent',
        'agentic_scent.core.factory',
        'agentic_scent.agents.quality_control',
        'agentic_scent.sensors.base',
        'agentic_scent.analytics.fingerprinting',
        'agentic_scent.core.security',
        'agentic_scent.core.validation',
        'agentic_scent.core.caching',
        'agentic_scent.core.metrics'
    ]
    
    failed_imports = []
    successful_imports = []
    
    for module_name in critical_imports:
        try:
            __import__(module_name)
            successful_imports.append(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            failed_imports.append((module_name, str(e)))
            print(f"  ❌ {module_name}: {e}")
    
    return len(failed_imports) == 0, successful_imports, failed_imports


def run_test_suite():
    """Run the basic test suite."""
    print("\n🧪 Running Test Suite...")
    
    test_results = {}
    
    # Run basic tests
    try:
        result = subprocess.run([sys.executable, 'run_basic_tests.py'], 
                              capture_output=True, text=True, timeout=60)
        basic_tests_passed = result.returncode == 0
        test_results['basic_tests'] = {
            'passed': basic_tests_passed,
            'output': result.stdout if basic_tests_passed else result.stderr
        }
        status = "✅ PASSED" if basic_tests_passed else "❌ FAILED"
        print(f"  Basic Tests: {status}")
    except Exception as e:
        test_results['basic_tests'] = {'passed': False, 'error': str(e)}
        print(f"  Basic Tests: ❌ ERROR - {e}")
    
    # Run scaling tests if available
    try:
        result = subprocess.run([sys.executable, 'test_scaling.py'], 
                              capture_output=True, text=True, timeout=120)
        scaling_tests_passed = result.returncode == 0
        test_results['scaling_tests'] = {
            'passed': scaling_tests_passed,
            'output': result.stdout if scaling_tests_passed else result.stderr
        }
        status = "✅ PASSED" if scaling_tests_passed else "❌ FAILED"
        print(f"  Scaling Tests: {status}")
    except Exception as e:
        test_results['scaling_tests'] = {'passed': False, 'error': str(e)}
        print(f"  Scaling Tests: ❌ ERROR - {e}")
    
    # Run security tests
    try:
        result = subprocess.run([sys.executable, 'test_security.py'], 
                              capture_output=True, text=True, timeout=120)
        security_tests_passed = result.returncode == 0
        test_results['security_tests'] = {
            'passed': security_tests_passed,
            'output': result.stdout if security_tests_passed else result.stderr
        }
        status = "✅ PASSED" if security_tests_passed else "⚠️ PARTIAL"
        print(f"  Security Tests: {status}")
    except Exception as e:
        test_results['security_tests'] = {'passed': False, 'error': str(e)}
        print(f"  Security Tests: ❌ ERROR - {e}")
    
    return test_results


def check_configuration():
    """Check configuration files and settings."""
    print("\n⚙️ Checking Configuration...")
    
    config_checks = {}
    
    # Check setup.py
    try:
        with open('setup.py', 'r') as f:
            setup_content = f.read()
            has_version = 'version=' in setup_content
            has_dependencies = 'install_requires=' in setup_content
            has_entry_points = 'entry_points=' in setup_content
            
            config_checks['setup.py'] = {
                'exists': True,
                'has_version': has_version,
                'has_dependencies': has_dependencies,
                'has_entry_points': has_entry_points
            }
            
            print(f"  setup.py: ✅ Present")
            print(f"    Version defined: {'✅' if has_version else '❌'}")
            print(f"    Dependencies defined: {'✅' if has_dependencies else '❌'}")
            print(f"    Entry points defined: {'✅' if has_entry_points else '❌'}")
            
    except FileNotFoundError:
        config_checks['setup.py'] = {'exists': False}
        print(f"  setup.py: ❌ Missing")
    
    # Check Docker configuration
    try:
        with open('Dockerfile', 'r') as f:
            dockerfile_content = f.read()
            has_python_base = 'FROM python' in dockerfile_content
            has_workdir = 'WORKDIR' in dockerfile_content
            has_copy = 'COPY' in dockerfile_content
            has_expose = 'EXPOSE' in dockerfile_content
            
            config_checks['Dockerfile'] = {
                'exists': True,
                'has_python_base': has_python_base,
                'has_workdir': has_workdir,
                'has_copy': has_copy,
                'has_expose': has_expose
            }
            
            print(f"  Dockerfile: ✅ Present")
            
    except FileNotFoundError:
        config_checks['Dockerfile'] = {'exists': False}
        print(f"  Dockerfile: ⚠️ Missing (optional)")
    
    # Check Kubernetes configuration
    k8s_files = ['k8s/deployment.yaml', 'k8s/service.yaml']
    k8s_present = 0
    for k8s_file in k8s_files:
        if os.path.exists(k8s_file):
            k8s_present += 1
            print(f"  {k8s_file}: ✅ Present")
        else:
            print(f"  {k8s_file}: ⚠️ Missing")
    
    config_checks['kubernetes'] = {
        'files_present': k8s_present,
        'total_files': len(k8s_files)
    }
    
    return config_checks


def check_system_requirements():
    """Check system requirements and resources."""
    print("\n💻 Checking System Requirements...")
    
    import psutil
    
    # CPU check
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"  CPU Cores: {cpu_count}")
    print(f"  CPU Usage: {cpu_percent}%")
    
    # Memory check
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    
    print(f"  Total Memory: {memory_gb:.1f} GB")
    print(f"  Available Memory: {memory_available_gb:.1f} GB")
    
    # Disk check
    disk = psutil.disk_usage('/')
    disk_total_gb = disk.total / (1024**3)
    disk_free_gb = disk.free / (1024**3)
    
    print(f"  Disk Total: {disk_total_gb:.1f} GB")
    print(f"  Disk Free: {disk_free_gb:.1f} GB")
    
    # Requirements assessment
    meets_requirements = (
        cpu_count >= 2 and 
        memory_gb >= 4.0 and 
        disk_free_gb >= 10.0 and
        cpu_percent < 80
    )
    
    return {
        'cpu_cores': cpu_count,
        'cpu_usage_percent': cpu_percent,
        'memory_total_gb': memory_gb,
        'memory_available_gb': memory_available_gb,
        'disk_total_gb': disk_total_gb,
        'disk_free_gb': disk_free_gb,
        'meets_requirements': meets_requirements
    }


def generate_deployment_report(checks: Dict[str, Any]):
    """Generate comprehensive deployment readiness report."""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'deployment_readiness': {
            'overall_status': 'UNKNOWN',
            'readiness_score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        },
        'checks': checks
    }
    
    # Calculate readiness score
    score_components = []
    critical_issues = []
    warnings = []
    recommendations = []
    
    # Dependencies check (25% weight)
    if checks['dependencies']['success']:
        score_components.append(25)
    else:
        critical_issues.append(f"Missing dependencies: {checks['dependencies']['missing']}")
        recommendations.append("Install missing dependencies with: pip install -r requirements.txt")
    
    # File structure check (20% weight)
    if checks['file_structure']['success']:
        score_components.append(20)
    else:
        critical_issues.append(f"Missing critical files: {checks['file_structure']['missing']}")
        recommendations.append("Ensure all required files are present in the repository")
    
    # Imports check (20% weight)
    if checks['imports']['success']:
        score_components.append(20)
    else:
        critical_issues.append("Some modules cannot be imported")
        recommendations.append("Fix import errors before deployment")
    
    # Tests check (20% weight)
    test_results = checks['tests']
    passing_tests = sum(1 for test_name, result in test_results.items() 
                       if isinstance(result, dict) and result.get('passed', False))
    total_tests = len(test_results)
    
    if total_tests > 0:
        test_score = (passing_tests / total_tests) * 20
        score_components.append(test_score)
        
        if passing_tests == total_tests:
            pass  # All good
        elif passing_tests >= total_tests * 0.8:
            warnings.append(f"Some tests failed ({passing_tests}/{total_tests} passed)")
        else:
            critical_issues.append(f"Many tests failed ({passing_tests}/{total_tests} passed)")
            recommendations.append("Fix failing tests before production deployment")
    
    # System requirements check (15% weight)
    if checks['system_requirements']['meets_requirements']:
        score_components.append(15)
    else:
        warnings.append("System may not meet minimum requirements")
        recommendations.append("Ensure adequate system resources for production workload")
    
    # Calculate final score
    readiness_score = sum(score_components)
    
    # Determine overall status
    if readiness_score >= 90 and len(critical_issues) == 0:
        overall_status = 'READY'
    elif readiness_score >= 70 and len(critical_issues) == 0:
        overall_status = 'MOSTLY_READY'
    elif readiness_score >= 50:
        overall_status = 'NEEDS_WORK'
    else:
        overall_status = 'NOT_READY'
    
    # Add general recommendations
    if readiness_score >= 90:
        recommendations.append("System is ready for production deployment")
        recommendations.append("Consider setting up monitoring and alerting")
        recommendations.append("Prepare rollback procedures")
    
    report['deployment_readiness'].update({
        'overall_status': overall_status,
        'readiness_score': readiness_score,
        'critical_issues': critical_issues,
        'warnings': warnings,
        'recommendations': recommendations
    })
    
    return report


def print_deployment_report(report: Dict[str, Any]):
    """Print formatted deployment readiness report."""
    
    readiness = report['deployment_readiness']
    
    print("\n" + "=" * 80)
    print("🚀 PRODUCTION DEPLOYMENT READINESS REPORT")
    print("=" * 80)
    
    # Overall status
    status_emoji = {
        'READY': '🟢',
        'MOSTLY_READY': '🟡', 
        'NEEDS_WORK': '🟠',
        'NOT_READY': '🔴'
    }
    
    emoji = status_emoji.get(readiness['overall_status'], '❓')
    print(f"Overall Status: {emoji} {readiness['overall_status']}")
    print(f"Readiness Score: {readiness['readiness_score']:.1f}/100")
    print(f"Generated: {report['timestamp']}")
    
    # Critical issues
    if readiness['critical_issues']:
        print(f"\n❌ CRITICAL ISSUES ({len(readiness['critical_issues'])})")
        for i, issue in enumerate(readiness['critical_issues'], 1):
            print(f"  {i}. {issue}")
    
    # Warnings
    if readiness['warnings']:
        print(f"\n⚠️ WARNINGS ({len(readiness['warnings'])})")
        for i, warning in enumerate(readiness['warnings'], 1):
            print(f"  {i}. {warning}")
    
    # Recommendations
    if readiness['recommendations']:
        print(f"\n💡 RECOMMENDATIONS ({len(readiness['recommendations'])})")
        for i, rec in enumerate(readiness['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Detailed results
    print(f"\n📊 DETAILED RESULTS")
    print("-" * 40)
    
    checks = report['checks']
    
    deps_status = "✅ PASS" if checks['dependencies']['success'] else "❌ FAIL"
    print(f"Dependencies: {deps_status}")
    
    files_status = "✅ PASS" if checks['file_structure']['success'] else "❌ FAIL"
    print(f"File Structure: {files_status}")
    
    imports_status = "✅ PASS" if checks['imports']['success'] else "❌ FAIL"
    print(f"Module Imports: {imports_status}")
    
    # Test results summary
    test_results = checks['tests']
    passing_tests = sum(1 for test_name, result in test_results.items() 
                       if isinstance(result, dict) and result.get('passed', False))
    total_tests = len(test_results)
    print(f"Test Results: {passing_tests}/{total_tests} passed")
    
    sys_req_status = "✅ PASS" if checks['system_requirements']['meets_requirements'] else "⚠️ WARNING"
    print(f"System Requirements: {sys_req_status}")
    
    # Next steps
    print(f"\n🎯 NEXT STEPS")
    print("-" * 40)
    
    if readiness['overall_status'] == 'READY':
        print("1. Review deployment configuration")
        print("2. Set up monitoring and logging")
        print("3. Prepare production environment")
        print("4. Execute deployment plan")
        print("5. Validate production deployment")
        
    elif readiness['overall_status'] in ['MOSTLY_READY', 'NEEDS_WORK']:
        print("1. Address critical issues listed above")
        print("2. Fix failing tests")
        print("3. Re-run deployment readiness check")
        print("4. Proceed with deployment when score > 90")
        
    else:
        print("1. Address all critical issues")
        print("2. Ensure all dependencies are installed")
        print("3. Fix broken imports and missing files")
        print("4. Run full test suite and fix failures")
        print("5. Re-run this check before proceeding")


def save_report(report: Dict[str, Any], filename: str = "deployment_readiness_report.json"):
    """Save deployment report to file."""
    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Report saved to: {filename}")
    except Exception as e:
        print(f"\n❌ Failed to save report: {e}")


def main():
    """Run complete deployment readiness check."""
    
    print("🔍 Agentic Scent Analytics - Production Deployment Readiness Check")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all checks
    checks = {}
    
    # 1. Dependencies check
    deps_success, installed, missing = check_dependencies()
    checks['dependencies'] = {
        'success': deps_success,
        'installed': installed,
        'missing': missing
    }
    
    # 2. File structure check
    files_success, present, missing_files = check_file_structure()
    checks['file_structure'] = {
        'success': files_success,
        'present': present,
        'missing': missing_files
    }
    
    # 3. Import check
    imports_success, successful, failed = check_imports()
    checks['imports'] = {
        'success': imports_success,
        'successful': successful,
        'failed': failed
    }
    
    # 4. Test suite
    test_results = run_test_suite()
    checks['tests'] = test_results
    
    # 5. Configuration check
    config_results = check_configuration()
    checks['configuration'] = config_results
    
    # 6. System requirements
    system_results = check_system_requirements()
    checks['system_requirements'] = system_results
    
    # Generate comprehensive report
    report = generate_deployment_report(checks)
    
    # Print results
    duration = time.time() - start_time
    report['check_duration_seconds'] = duration
    
    print_deployment_report(report)
    
    # Save report
    save_report(report)
    
    print(f"\n⏱️ Total check duration: {duration:.2f} seconds")
    
    # Exit with appropriate code
    readiness_score = report['deployment_readiness']['readiness_score']
    critical_issues = len(report['deployment_readiness']['critical_issues'])
    
    if readiness_score >= 90 and critical_issues == 0:
        print("\n🎉 System is READY for production deployment!")
        return 0
    elif readiness_score >= 70:
        print("\n⚠️ System is MOSTLY READY - address warnings before deployment")
        return 1
    else:
        print("\n❌ System is NOT READY for production deployment")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)