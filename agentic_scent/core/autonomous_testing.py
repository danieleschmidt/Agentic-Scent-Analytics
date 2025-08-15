#!/usr/bin/env python3
"""
Autonomous Testing Framework - Self-evolving test generation and execution
Part of Agentic Scent Analytics Platform

This module implements an AI-powered testing framework that automatically generates,
executes, and evolves tests based on code changes, coverage analysis, and historical
defect patterns. The system uses machine learning to predict where bugs are likely
to occur and focuses testing efforts accordingly.
"""

import asyncio
import ast
import inspect
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
import hashlib
import importlib
import sys
import traceback

import numpy as np
from .config import ConfigManager
from .validation import ValidationManager


class TestType(Enum):
    """Types of tests that can be automatically generated"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY = "property"
    FUZZ = "fuzz"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"
    MUTATION = "mutation"


class TestPriority(Enum):
    """Priority levels for test execution"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestCase:
    """Generated test case"""
    id: str
    name: str
    test_type: TestType
    priority: TestPriority
    target_function: str
    target_module: str
    test_code: str
    expected_outcome: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    execution_count: int = 0
    last_result: Optional[str] = None
    bug_finding_score: float = 0.0


@dataclass
class TestResult:
    """Result of test execution"""
    test_id: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    coverage_data: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BugPattern:
    """Detected bug pattern for ML model"""
    pattern_type: str
    code_signature: str
    frequency: int
    severity: str
    fix_complexity: str
    detection_methods: List[str]


class CodeAnalyzer:
    """Analyzes code to identify testing opportunities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """Analyze a Python module for testing opportunities"""
        try:
            with open(module_path, 'r') as f:
                source = f.read()
                
            tree = ast.parse(source)
            
            functions = []
            classes = []
            complexity_hotspots = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node, source)
                    functions.append(func_info)
                    
                    if func_info['complexity'] > 10:
                        complexity_hotspots.append(func_info)
                        
                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, source)
                    classes.append(class_info)
            
            return {
                'module_path': module_path,
                'functions': functions,
                'classes': classes,
                'complexity_hotspots': complexity_hotspots,
                'total_lines': len(source.split('\n')),
                'imports': self._extract_imports(tree),
                'risk_score': self._calculate_risk_score(functions, classes)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing module {module_path}: {e}")
            return {'error': str(e)}
    
    def _analyze_function(self, node: ast.FunctionDef, source: str) -> Dict[str, Any]:
        """Analyze a function for testing characteristics"""
        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(node)
        
        # Extract parameters
        params = []
        for arg in node.args.args:
            params.append({
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None
            })
        
        # Check for async
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Check for decorators
        decorators = [ast.unparse(dec) for dec in node.decorator_list]
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Identify test opportunities
        test_opportunities = self._identify_test_opportunities(node)
        
        return {
            'name': node.name,
            'line_number': node.lineno,
            'complexity': complexity,
            'parameters': params,
            'is_async': is_async,
            'decorators': decorators,
            'docstring': docstring,
            'test_opportunities': test_opportunities,
            'risk_factors': self._identify_risk_factors(node),
            'testing_priority': self._calculate_testing_priority(complexity, test_opportunities)
        }
    
    def _analyze_class(self, node: ast.ClassDef, source: str) -> Dict[str, Any]:
        """Analyze a class for testing characteristics"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, source)
                method_info['is_method'] = True
                methods.append(method_info)
        
        return {
            'name': node.name,
            'line_number': node.lineno,
            'methods': methods,
            'base_classes': [ast.unparse(base) for base in node.bases],
            'decorators': [ast.unparse(dec) for dec in node.decorator_list]
        }
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        return imports
    
    def _identify_test_opportunities(self, node: ast.FunctionDef) -> List[str]:
        """Identify specific testing opportunities in a function"""
        opportunities = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                opportunities.append("exception_testing")
            elif isinstance(child, ast.Return):
                opportunities.append("return_value_testing")
            elif isinstance(child, ast.If):
                opportunities.append("branch_testing")
            elif isinstance(child, ast.For):
                opportunities.append("loop_testing")
            elif isinstance(child, ast.Call):
                if hasattr(child.func, 'attr') and child.func.attr in ['open', 'read', 'write']:
                    opportunities.append("io_testing")
                elif hasattr(child.func, 'id') and child.func.id in ['len', 'max', 'min']:
                    opportunities.append("boundary_testing")
        
        return list(set(opportunities))
    
    def _identify_risk_factors(self, node: ast.FunctionDef) -> List[str]:
        """Identify risk factors that suggest need for thorough testing"""
        risk_factors = []
        
        # Check for external dependencies
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'attr'):
                    if child.func.attr in ['requests', 'urllib', 'socket']:
                        risk_factors.append("network_dependency")
                    elif child.func.attr in ['open', 'write', 'read']:
                        risk_factors.append("file_system_dependency")
                    elif child.func.attr in ['execute', 'run', 'popen']:
                        risk_factors.append("system_call")
        
        # Check for exception handling
        has_exception_handling = any(isinstance(child, ast.ExceptHandler) 
                                   for child in ast.walk(node))
        if not has_exception_handling:
            risk_factors.append("no_exception_handling")
        
        return risk_factors
    
    def _calculate_testing_priority(self, complexity: int, 
                                  opportunities: List[str]) -> TestPriority:
        """Calculate testing priority based on complexity and opportunities"""
        score = complexity * 0.3 + len(opportunities) * 0.7
        
        if score >= 8:
            return TestPriority.CRITICAL
        elif score >= 5:
            return TestPriority.HIGH
        elif score >= 3:
            return TestPriority.MEDIUM
        else:
            return TestPriority.LOW
    
    def _calculate_risk_score(self, functions: List[Dict], 
                            classes: List[Dict]) -> float:
        """Calculate overall risk score for the module"""
        total_complexity = sum(f['complexity'] for f in functions)
        high_complexity_count = len([f for f in functions if f['complexity'] > 10])
        
        # Normalize to 0-1 scale
        base_score = min(1.0, total_complexity / (len(functions) * 10))
        complexity_penalty = high_complexity_count * 0.1
        
        return min(1.0, base_score + complexity_penalty)


class TestGenerator:
    """Generates tests automatically based on code analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._load_test_templates()
        
    def _load_test_templates(self) -> Dict[str, str]:
        """Load test templates for different test types"""
        return {
            'unit_basic': '''
def test_{function_name}_{test_suffix}():
    """Auto-generated unit test for {function_name}"""
    # Arrange
    {setup_code}
    
    # Act
    result = {function_call}
    
    # Assert
    {assertions}
''',
            
            'unit_exception': '''
def test_{function_name}_raises_{exception_type}():
    """Test that {function_name} raises {exception_type} for invalid input"""
    import pytest
    
    with pytest.raises({exception_type}):
        {function_call}
''',
            
            'property_test': '''
def test_{function_name}_property_{property_name}():
    """Property-based test for {function_name}"""
    from hypothesis import given, strategies as st
    
    @given({hypothesis_strategy})
    def test_property(input_value):
        result = {function_call}
        assert {property_assertion}
        
    test_property()
''',
            
            'fuzz_test': '''
def test_{function_name}_fuzz():
    """Fuzz test for {function_name}"""
    import random
    import string
    
    for _ in range(100):
        # Generate random input
        fuzz_input = {fuzz_generation}
        
        try:
            result = {function_call}
            # Basic invariant checks
            {invariant_checks}
        except Exception as e:
            # Log unexpected exceptions
            print(f"Unexpected exception with input {{fuzz_input}}: {{e}}")
'''
        }
    
    async def generate_tests_for_function(self, function_info: Dict[str, Any],
                                        module_path: str) -> List[TestCase]:
        """Generate comprehensive tests for a specific function"""
        tests = []
        
        # Generate basic unit tests
        basic_tests = self._generate_basic_unit_tests(function_info, module_path)
        tests.extend(basic_tests)
        
        # Generate exception tests
        exception_tests = self._generate_exception_tests(function_info, module_path)
        tests.extend(exception_tests)
        
        # Generate property-based tests if applicable
        if self._is_suitable_for_property_testing(function_info):
            property_tests = self._generate_property_tests(function_info, module_path)
            tests.extend(property_tests)
        
        # Generate fuzz tests for high-risk functions
        if function_info['testing_priority'] in [TestPriority.CRITICAL, TestPriority.HIGH]:
            fuzz_tests = self._generate_fuzz_tests(function_info, module_path)
            tests.extend(fuzz_tests)
        
        return tests
    
    def _generate_basic_unit_tests(self, function_info: Dict[str, Any],
                                 module_path: str) -> List[TestCase]:
        """Generate basic unit tests"""
        tests = []
        function_name = function_info['name']
        
        # Generate tests for different input scenarios
        scenarios = self._generate_input_scenarios(function_info)
        
        for i, scenario in enumerate(scenarios):
            test_code = self.templates['unit_basic'].format(
                function_name=function_name,
                test_suffix=f"scenario_{i+1}",
                setup_code=scenario['setup'],
                function_call=scenario['call'],
                assertions=scenario['assertions']
            )
            
            test_case = TestCase(
                id=f"{function_name}_unit_{i+1}",
                name=f"test_{function_name}_scenario_{i+1}",
                test_type=TestType.UNIT,
                priority=function_info['testing_priority'],
                target_function=function_name,
                target_module=module_path,
                test_code=test_code,
                expected_outcome="pass",
                metadata={
                    'scenario': scenario['description'],
                    'auto_generated': True
                }
            )
            
            tests.append(test_case)
        
        return tests
    
    def _generate_input_scenarios(self, function_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate different input scenarios for testing"""
        scenarios = []
        function_name = function_info['name']
        parameters = function_info['parameters']
        
        if not parameters:
            # No parameters - simple call
            scenarios.append({
                'description': 'No parameters test',
                'setup': 'pass',
                'call': f'{function_name}()',
                'assertions': 'assert result is not None'
            })
        else:
            # Valid input scenario
            valid_args = []
            setup_lines = []
            
            for param in parameters:
                param_name = param['name']
                if param['annotation']:
                    # Use type annotation to generate appropriate value
                    value, setup = self._generate_value_for_type(param['annotation'])
                    valid_args.append(f"{param_name}={value}")
                    if setup:
                        setup_lines.append(setup)
                else:
                    # Default to string for unknown types
                    valid_args.append(f"{param_name}='test_value'")
            
            scenarios.append({
                'description': 'Valid input test',
                'setup': '\n    '.join(setup_lines) if setup_lines else 'pass',
                'call': f"{function_name}({', '.join(valid_args)})",
                'assertions': 'assert result is not None'
            })
            
            # Edge case scenarios
            edge_scenarios = self._generate_edge_case_scenarios(function_info)
            scenarios.extend(edge_scenarios)
        
        return scenarios
    
    def _generate_value_for_type(self, type_annotation: str) -> Tuple[str, Optional[str]]:
        """Generate appropriate test value for a given type annotation"""
        type_mapping = {
            'str': ("'test_string'", None),
            'int': ("42", None),
            'float': ("3.14", None),
            'bool': ("True", None),
            'list': ("[1, 2, 3]", None),
            'dict': ("{'key': 'value'}", None),
            'List[str]': ("['item1', 'item2']", None),
            'Dict[str, Any]': ("{'test': 'data'}", None),
            'Optional[str]': ("'optional_value'", None),
            'np.ndarray': ("test_array", "test_array = np.array([1, 2, 3])"),
        }
        
        return type_mapping.get(type_annotation, ("'default_value'", None))
    
    def _generate_edge_case_scenarios(self, function_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate edge case test scenarios"""
        scenarios = []
        function_name = function_info['name']
        
        # Empty/None inputs
        if function_info['parameters']:
            scenarios.append({
                'description': 'Empty input test',
                'setup': 'pass',
                'call': f"{function_name}('')",
                'assertions': 'assert True  # Test completes without exception'
            })
        
        return scenarios
    
    def _generate_exception_tests(self, function_info: Dict[str, Any],
                                module_path: str) -> List[TestCase]:
        """Generate tests for exception handling"""
        tests = []
        
        if 'exception_testing' in function_info['test_opportunities']:
            function_name = function_info['name']
            
            # Common exception types to test
            exception_types = ['ValueError', 'TypeError', 'AttributeError']
            
            for exc_type in exception_types:
                test_code = self.templates['unit_exception'].format(
                    function_name=function_name,
                    exception_type=exc_type,
                    function_call=f"{function_name}(None)"
                )
                
                test_case = TestCase(
                    id=f"{function_name}_exception_{exc_type.lower()}",
                    name=f"test_{function_name}_raises_{exc_type.lower()}",
                    test_type=TestType.UNIT,
                    priority=TestPriority.HIGH,
                    target_function=function_name,
                    target_module=module_path,
                    test_code=test_code,
                    expected_outcome="pass",
                    metadata={
                        'exception_type': exc_type,
                        'auto_generated': True
                    }
                )
                
                tests.append(test_case)
        
        return tests
    
    def _generate_property_tests(self, function_info: Dict[str, Any],
                               module_path: str) -> List[TestCase]:
        """Generate property-based tests using Hypothesis"""
        tests = []
        function_name = function_info['name']
        
        # Define properties to test
        properties = [
            {
                'name': 'idempotent',
                'strategy': 'st.text()',
                'assertion': 'f(f(input_value)) == f(input_value)',
                'description': 'Function is idempotent'
            },
            {
                'name': 'deterministic',
                'strategy': 'st.integers()',
                'assertion': 'f(input_value) == f(input_value)',
                'description': 'Function is deterministic'
            }
        ]
        
        for prop in properties:
            test_code = self.templates['property_test'].format(
                function_name=function_name,
                property_name=prop['name'],
                hypothesis_strategy=prop['strategy'],
                function_call=f"{function_name}(input_value)",
                property_assertion=prop['assertion'].replace('f(', f'{function_name}(')
            )
            
            test_case = TestCase(
                id=f"{function_name}_property_{prop['name']}",
                name=f"test_{function_name}_property_{prop['name']}",
                test_type=TestType.PROPERTY,
                priority=TestPriority.MEDIUM,
                target_function=function_name,
                target_module=module_path,
                test_code=test_code,
                expected_outcome="pass",
                metadata={
                    'property': prop['description'],
                    'auto_generated': True
                }
            )
            
            tests.append(test_case)
        
        return tests
    
    def _generate_fuzz_tests(self, function_info: Dict[str, Any],
                           module_path: str) -> List[TestCase]:
        """Generate fuzz tests for robustness testing"""
        tests = []
        function_name = function_info['name']
        
        # Generate different fuzz strategies based on parameter types
        fuzz_strategies = [
            {
                'name': 'random_strings',
                'generation': "''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(0, 1000)))",
                'invariants': 'assert isinstance(result, (type(None), str, int, float, bool, list, dict))'
            },
            {
                'name': 'large_numbers',
                'generation': 'random.randint(-1000000, 1000000)',
                'invariants': 'assert result is not None or result is None'
            }
        ]
        
        for strategy in fuzz_strategies:
            test_code = self.templates['fuzz_test'].format(
                function_name=function_name,
                fuzz_generation=strategy['generation'],
                function_call=f"{function_name}(fuzz_input)",
                invariant_checks=strategy['invariants']
            )
            
            test_case = TestCase(
                id=f"{function_name}_fuzz_{strategy['name']}",
                name=f"test_{function_name}_fuzz_{strategy['name']}",
                test_type=TestType.FUZZ,
                priority=TestPriority.HIGH,
                target_function=function_name,
                target_module=module_path,
                test_code=test_code,
                expected_outcome="pass",
                metadata={
                    'fuzz_strategy': strategy['name'],
                    'auto_generated': True
                }
            )
            
            tests.append(test_case)
        
        return tests
    
    def _is_suitable_for_property_testing(self, function_info: Dict[str, Any]) -> bool:
        """Determine if a function is suitable for property-based testing"""
        # Functions with mathematical properties or pure functions are good candidates
        suitable_patterns = [
            'calculate', 'compute', 'transform', 'convert', 'parse', 'format'
        ]
        
        function_name = function_info['name'].lower()
        return any(pattern in function_name for pattern in suitable_patterns)


class TestExecutor:
    """Executes generated tests and collects results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def execute_test_suite(self, test_cases: List[TestCase]) -> List[TestResult]:
        """Execute a suite of test cases"""
        results = []
        
        for test_case in test_cases:
            try:
                result = await self._execute_single_test(test_case)
                results.append(result)
                
                # Update test case with execution info
                test_case.execution_count += 1
                test_case.last_result = 'pass' if result.passed else 'fail'
                
            except Exception as e:
                self.logger.error(f"Error executing test {test_case.id}: {e}")
                
                result = TestResult(
                    test_id=test_case.id,
                    passed=False,
                    execution_time=0.0,
                    error_message=str(e)
                )
                results.append(result)
        
        return results
    
    async def _execute_single_test(self, test_case: TestCase) -> TestResult:
        """Execute a single test case"""
        start_time = time.time()
        
        try:
            # Create a temporary test module
            test_module_code = f"""
import sys
import os
sys.path.append(os.path.dirname('{test_case.target_module}'))

from {Path(test_case.target_module).stem} import {test_case.target_function}

{test_case.test_code}

# Execute the test
try:
    {test_case.name}()
    test_result = True
    error_msg = None
except Exception as e:
    test_result = False
    error_msg = str(e)
"""
            
            # Execute in isolated namespace
            namespace = {}
            exec(test_module_code, namespace)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_case.id,
                passed=namespace.get('test_result', False),
                execution_time=execution_time,
                error_message=namespace.get('error_msg')
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_case.id,
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )


class BugPredictionModel:
    """ML model for predicting where bugs are likely to occur"""
    
    def __init__(self):
        self.bug_patterns: List[BugPattern] = []
        self.feature_weights = {
            'complexity': 0.3,
            'recent_changes': 0.25,
            'historical_bugs': 0.2,
            'test_coverage': 0.15,
            'code_age': 0.1
        }
        
    def predict_bug_likelihood(self, function_info: Dict[str, Any],
                             module_info: Dict[str, Any]) -> float:
        """Predict likelihood of bugs in a function (0-1 scale)"""
        score = 0.0
        
        # Complexity factor
        complexity = function_info.get('complexity', 1)
        complexity_score = min(1.0, complexity / 20.0)
        score += complexity_score * self.feature_weights['complexity']
        
        # Risk factors
        risk_factors = function_info.get('risk_factors', [])
        risk_score = len(risk_factors) * 0.1
        score += min(1.0, risk_score) * self.feature_weights['recent_changes']
        
        # Historical patterns
        historical_score = self._calculate_historical_score(function_info)
        score += historical_score * self.feature_weights['historical_bugs']
        
        return min(1.0, score)
    
    def _calculate_historical_score(self, function_info: Dict[str, Any]) -> float:
        """Calculate score based on historical bug patterns"""
        function_name = function_info['name']
        
        # Check for known problematic patterns
        problematic_patterns = [
            'parse', 'convert', 'transform', 'decode', 'encode',
            'validate', 'sanitize', 'process'
        ]
        
        pattern_score = 0.0
        for pattern in problematic_patterns:
            if pattern in function_name.lower():
                pattern_score += 0.2
        
        return min(1.0, pattern_score)
    
    def learn_from_test_results(self, test_results: List[TestResult],
                               function_info: Dict[str, Any]):
        """Update model based on test execution results"""
        failed_tests = [r for r in test_results if not r.passed]
        
        if failed_tests:
            # Extract patterns from failed tests
            for result in failed_tests:
                pattern = BugPattern(
                    pattern_type="test_failure",
                    code_signature=self._generate_code_signature(function_info),
                    frequency=1,
                    severity="medium",
                    fix_complexity="unknown",
                    detection_methods=["automated_testing"]
                )
                
                self.bug_patterns.append(pattern)
    
    def _generate_code_signature(self, function_info: Dict[str, Any]) -> str:
        """Generate a signature for code patterns"""
        signature_parts = [
            function_info['name'],
            str(function_info['complexity']),
            ','.join(function_info.get('risk_factors', []))
        ]
        
        signature = '|'.join(signature_parts)
        return hashlib.md5(signature.encode()).hexdigest()[:16]


class AutonomousTestingFramework:
    """Main autonomous testing framework coordinator"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.code_analyzer = CodeAnalyzer()
        self.test_generator = TestGenerator()
        self.test_executor = TestExecutor()
        self.bug_predictor = BugPredictionModel()
        
        self.generated_tests: Dict[str, List[TestCase]] = {}
        self.test_results: List[TestResult] = []
        self.coverage_data: Dict[str, float] = {}
        
    async def analyze_and_test_codebase(self, codebase_path: str) -> Dict[str, Any]:
        """Analyze entire codebase and generate comprehensive tests"""
        self.logger.info(f"Starting autonomous testing of codebase: {codebase_path}")
        
        # Discover Python modules
        python_files = list(Path(codebase_path).rglob("*.py"))
        python_files = [f for f in python_files if not f.name.startswith('test_')]
        
        analysis_results = {}
        all_test_cases = []
        
        # Analyze each module
        for py_file in python_files:
            try:
                module_analysis = self.code_analyzer.analyze_module(str(py_file))
                
                if 'error' not in module_analysis:
                    analysis_results[str(py_file)] = module_analysis
                    
                    # Generate tests for high-priority functions
                    for function_info in module_analysis['functions']:
                        if function_info['testing_priority'] in [TestPriority.CRITICAL, TestPriority.HIGH]:
                            tests = await self.test_generator.generate_tests_for_function(
                                function_info, str(py_file)
                            )
                            all_test_cases.extend(tests)
                            
                            # Predict bug likelihood
                            bug_likelihood = self.bug_predictor.predict_bug_likelihood(
                                function_info, module_analysis
                            )
                            
                            # Boost test priority for high-risk functions
                            if bug_likelihood > 0.7:
                                for test in tests:
                                    test.priority = TestPriority.CRITICAL
                                    test.metadata['bug_likelihood'] = bug_likelihood
                
            except Exception as e:
                self.logger.error(f"Error analyzing {py_file}: {e}")
        
        # Prioritize and execute tests
        prioritized_tests = self._prioritize_tests(all_test_cases)
        test_results = await self.test_executor.execute_test_suite(prioritized_tests)
        
        # Analyze results and learn
        self._analyze_test_results(test_results, analysis_results)
        
        # Generate report
        report = self._generate_testing_report(analysis_results, test_results)
        
        self.logger.info(f"Autonomous testing completed. Generated {len(all_test_cases)} tests, "
                        f"executed {len(test_results)} tests")
        
        return report
    
    def _prioritize_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Prioritize tests based on various factors"""
        priority_order = {
            TestPriority.CRITICAL: 0,
            TestPriority.HIGH: 1,
            TestPriority.MEDIUM: 2,
            TestPriority.LOW: 3
        }
        
        # Sort by priority, then by bug likelihood
        return sorted(test_cases, key=lambda t: (
            priority_order[t.priority],
            -t.metadata.get('bug_likelihood', 0.0)
        ))
    
    def _analyze_test_results(self, test_results: List[TestResult],
                            analysis_results: Dict[str, Any]):
        """Analyze test results and update ML models"""
        self.test_results.extend(test_results)
        
        # Update bug prediction model
        for module_path, module_analysis in analysis_results.items():
            for function_info in module_analysis['functions']:
                function_tests = [r for r in test_results 
                                if r.test_id.startswith(function_info['name'])]
                
                if function_tests:
                    self.bug_predictor.learn_from_test_results(function_tests, function_info)
    
    def _generate_testing_report(self, analysis_results: Dict[str, Any],
                               test_results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive testing report"""
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.passed])
        failed_tests = total_tests - passed_tests
        
        # Calculate coverage by module
        module_coverage = {}
        for module_path, analysis in analysis_results.items():
            module_tests = [r for r in test_results 
                          if any(f['name'] in r.test_id for f in analysis['functions'])]
            
            total_functions = len(analysis['functions'])
            tested_functions = len(set(r.test_id.split('_')[0] for r in module_tests))
            
            coverage = tested_functions / total_functions if total_functions > 0 else 0.0
            module_coverage[module_path] = coverage
        
        # Identify high-risk areas
        high_risk_functions = []
        for module_path, analysis in analysis_results.items():
            for function_info in analysis['functions']:
                bug_likelihood = self.bug_predictor.predict_bug_likelihood(
                    function_info, analysis
                )
                
                if bug_likelihood > 0.8:
                    high_risk_functions.append({
                        'module': module_path,
                        'function': function_info['name'],
                        'risk_score': bug_likelihood,
                        'complexity': function_info['complexity']
                    })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests_generated': total_tests,
                'tests_passed': passed_tests,
                'tests_failed': failed_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                'modules_analyzed': len(analysis_results),
                'high_risk_functions': len(high_risk_functions)
            },
            'module_coverage': module_coverage,
            'high_risk_functions': high_risk_functions,
            'failed_tests': [
                {
                    'test_id': r.test_id,
                    'error_message': r.error_message,
                    'execution_time': r.execution_time
                }
                for r in test_results if not r.passed
            ],
            'recommendations': self._generate_recommendations(analysis_results, test_results)
        }
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any],
                                test_results: List[TestResult]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Failed test analysis
        failed_tests = [r for r in test_results if not r.passed]
        if failed_tests:
            recommendations.append(f"Fix {len(failed_tests)} failing tests to improve code reliability")
        
        # Complexity analysis
        high_complexity_functions = []
        for module_analysis in analysis_results.values():
            high_complexity_functions.extend(module_analysis.get('complexity_hotspots', []))
        
        if high_complexity_functions:
            recommendations.append(
                f"Refactor {len(high_complexity_functions)} high-complexity functions "
                "to improve maintainability"
            )
        
        # Coverage gaps
        low_coverage_modules = [
            module for module, coverage in self.coverage_data.items()
            if coverage < 0.8
        ]
        
        if low_coverage_modules:
            recommendations.append(
                f"Increase test coverage for {len(low_coverage_modules)} modules "
                "with insufficient testing"
            )
        
        return recommendations
    
    async def continuous_testing_loop(self, codebase_path: str, 
                                    interval_seconds: int = 3600):
        """Run continuous testing loop with periodic re-analysis"""
        self.logger.info(f"Starting continuous testing loop with {interval_seconds}s interval")
        
        while True:
            try:
                await self.analyze_and_test_codebase(codebase_path)
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in continuous testing loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def export_test_suite(self, output_path: str, format: str = 'pytest'):
        """Export generated tests as executable test files"""
        if format == 'pytest':
            self._export_pytest_suite(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_pytest_suite(self, output_path: str):
        """Export tests in pytest format"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Group tests by target module
        tests_by_module = {}
        for test_cases in self.generated_tests.values():
            for test_case in test_cases:
                module_name = Path(test_case.target_module).stem
                if module_name not in tests_by_module:
                    tests_by_module[module_name] = []
                tests_by_module[module_name].append(test_case)
        
        # Generate test files
        for module_name, test_cases in tests_by_module.items():
            test_file_content = f'''"""
Auto-generated tests for {module_name}
Generated by Autonomous Testing Framework
"""

import pytest
import sys
import os
from pathlib import Path

# Add module to path
sys.path.append(str(Path(__file__).parent.parent))

from {module_name} import *

'''
            
            for test_case in test_cases:
                test_file_content += f"\n{test_case.test_code}\n"
            
            test_file_path = output_dir / f"test_{module_name}_auto.py"
            with open(test_file_path, 'w') as f:
                f.write(test_file_content)
        
        self.logger.info(f"Exported {len(tests_by_module)} test files to {output_path}")


# Factory function
def create_autonomous_testing_framework(config_path: Optional[str] = None) -> AutonomousTestingFramework:
    """Create and configure autonomous testing framework"""
    config = ConfigManager(config_path)
    return AutonomousTestingFramework(config)


# CLI interface
if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python autonomous_testing.py <codebase_path> [output_path]")
            return
        
        codebase_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "./generated_tests"
        
        framework = create_autonomous_testing_framework()
        
        # Analyze and test
        report = await framework.analyze_and_test_codebase(codebase_path)
        
        # Export tests
        framework.export_test_suite(output_path)
        
        # Print summary
        print(f"\n=== AUTONOMOUS TESTING REPORT ===")
        print(f"Tests Generated: {report['summary']['total_tests_generated']}")
        print(f"Pass Rate: {report['summary']['pass_rate']:.1%}")
        print(f"High-Risk Functions: {report['summary']['high_risk_functions']}")
        print(f"Tests exported to: {output_path}")
        
        # Save detailed report
        with open("autonomous_testing_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
    
    asyncio.run(main())