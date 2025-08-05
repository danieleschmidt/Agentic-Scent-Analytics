"""Data validation and sanitization framework."""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import hashlib

from ..core.task import Task, TaskStatus, TaskPriority


logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    field: str
    value: Any
    error_type: str
    message: str
    severity: str = "error"


class DataValidator:
    """Comprehensive data validation and sanitization."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.sanitization_rules: Dict[str, List[Callable]] = {}
        
        # Initialize default rules
        self._setup_default_rules()
        
        logger.info("DataValidator initialized")
    
    def _setup_default_rules(self):
        """Setup default validation and sanitization rules."""
        # String validation rules
        self.add_validation_rule("string", self._validate_string_length)
        self.add_validation_rule("string", self._validate_no_script_injection)
        
        # Numeric validation rules  
        self.add_validation_rule("number", self._validate_numeric_range)
        self.add_validation_rule("number", self._validate_finite_number)
        
        # Date validation rules
        self.add_validation_rule("datetime", self._validate_datetime_range)
        
        # JSON validation rules
        self.add_validation_rule("json", self._validate_json_structure)
        
        # String sanitization rules
        self.add_sanitization_rule("string", self._sanitize_html_entities)
        self.add_sanitization_rule("string", self._sanitize_control_characters)
        
        # Numeric sanitization rules
        self.add_sanitization_rule("number", self._sanitize_numeric_bounds)
    
    def add_validation_rule(self, data_type: str, rule_func: Callable):
        """Add custom validation rule."""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []
        self.validation_rules[data_type].append(rule_func)
    
    def add_sanitization_rule(self, data_type: str, rule_func: Callable):
        """Add custom sanitization rule."""
        if data_type not in self.sanitization_rules:
            self.sanitization_rules[data_type] = []
        self.sanitization_rules[data_type].append(rule_func)
    
    def validate(self, data: Any, data_type: str, context: Optional[Dict] = None) -> List[ValidationError]:
        """Validate data against rules."""
        errors = []
        context = context or {}
        
        rules = self.validation_rules.get(data_type, [])
        
        for rule in rules:
            try:
                rule_errors = rule(data, context)
                if rule_errors:
                    errors.extend(rule_errors)
            except Exception as e:
                errors.append(ValidationError(
                    field="validation_rule",
                    value=data,
                    error_type="rule_exception",
                    message=f"Validation rule failed: {str(e)}",
                    severity="error"
                ))
        
        return errors
    
    def sanitize(self, data: Any, data_type: str, context: Optional[Dict] = None) -> Any:
        """Sanitize data using rules."""
        context = context or {}
        sanitized_data = data
        
        rules = self.sanitization_rules.get(data_type, [])
        
        for rule in rules:
            try:
                sanitized_data = rule(sanitized_data, context)
            except Exception as e:
                logger.error(f"Sanitization rule failed: {e}")
                # Continue with current data if sanitization fails
        
        return sanitized_data
    
    def validate_and_sanitize(self, data: Any, data_type: str, context: Optional[Dict] = None) -> tuple[Any, List[ValidationError]]:
        """Validate and sanitize data in one operation."""
        errors = self.validate(data, data_type, context)
        sanitized_data = self.sanitize(data, data_type, context)
        return sanitized_data, errors
    
    # Default validation rules
    def _validate_string_length(self, data: str, context: Dict) -> List[ValidationError]:
        """Validate string length constraints."""
        errors = []
        
        if not isinstance(data, str):
            return [ValidationError(
                field="string_type",
                value=data,
                error_type="type_error",
                message="Expected string type"
            )]
        
        min_length = context.get("min_length", 0)
        max_length = context.get("max_length", 10000)
        
        if len(data) < min_length:
            errors.append(ValidationError(
                field="string_length",
                value=data,
                error_type="length_error",
                message=f"String too short: {len(data)} < {min_length}"
            ))
        
        if len(data) > max_length:
            errors.append(ValidationError(
                field="string_length",
                value=data,
                error_type="length_error", 
                message=f"String too long: {len(data)} > {max_length}"
            ))
        
        return errors
    
    def _validate_no_script_injection(self, data: str, context: Dict) -> List[ValidationError]:
        """Check for potential script injection."""
        errors = []
        
        if not isinstance(data, str):
            return []
        
        # Check for common script injection patterns
        dangerous_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe\b',
            r'<object\b',
            r'<embed\b'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                errors.append(ValidationError(
                    field="script_injection",
                    value=data,
                    error_type="security_error",
                    message=f"Potential script injection detected: {pattern}",
                    severity="critical"
                ))
        
        return errors
    
    def _validate_numeric_range(self, data: Union[int, float], context: Dict) -> List[ValidationError]:
        """Validate numeric range constraints."""
        errors = []
        
        if not isinstance(data, (int, float)):
            return [ValidationError(
                field="numeric_type",
                value=data,
                error_type="type_error",
                message="Expected numeric type"
            )]
        
        min_value = context.get("min_value", float("-inf"))
        max_value = context.get("max_value", float("inf"))
        
        if data < min_value:
            errors.append(ValidationError(
                field="numeric_range",
                value=data,
                error_type="range_error",
                message=f"Value too small: {data} < {min_value}"
            ))
        
        if data > max_value:
            errors.append(ValidationError(
                field="numeric_range", 
                value=data,
                error_type="range_error",
                message=f"Value too large: {data} > {max_value}"
            ))
        
        return errors
    
    def _validate_finite_number(self, data: Union[int, float], context: Dict) -> List[ValidationError]:
        """Validate number is finite."""
        errors = []
        
        if isinstance(data, float):
            import math
            if not math.isfinite(data):
                errors.append(ValidationError(
                    field="numeric_finite",
                    value=data,
                    error_type="value_error",
                    message=f"Number must be finite: {data}"
                ))
        
        return errors
    
    def _validate_datetime_range(self, data: datetime, context: Dict) -> List[ValidationError]:
        """Validate datetime range constraints."""
        errors = []
        
        if not isinstance(data, datetime):
            return [ValidationError(
                field="datetime_type",
                value=data,
                error_type="type_error",
                message="Expected datetime type"
            )]
        
        min_date = context.get("min_date")
        max_date = context.get("max_date")
        
        if min_date and data < min_date:
            errors.append(ValidationError(
                field="datetime_range",
                value=data,
                error_type="range_error",
                message=f"Date too early: {data} < {min_date}"
            ))
        
        if max_date and data > max_date:
            errors.append(ValidationError(
                field="datetime_range",
                value=data,
                error_type="range_error",
                message=f"Date too late: {data} > {max_date}"
            ))
        
        return errors
    
    def _validate_json_structure(self, data: str, context: Dict) -> List[ValidationError]:
        """Validate JSON structure."""
        errors = []
        
        if not isinstance(data, str):
            return []
        
        try:
            parsed = json.loads(data)
            
            # Validate JSON depth
            max_depth = context.get("max_depth", 10)
            if self._json_depth(parsed) > max_depth:
                errors.append(ValidationError(
                    field="json_depth",
                    value=data,
                    error_type="structure_error",
                    message=f"JSON too deeply nested: > {max_depth}"
                ))
                
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                field="json_syntax",
                value=data,
                error_type="syntax_error",
                message=f"Invalid JSON: {str(e)}"
            ))
        
        return errors
    
    def _json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate JSON nesting depth."""
        if isinstance(obj, dict):
            return max([self._json_depth(v, current_depth + 1) for v in obj.values()], default=current_depth)
        elif isinstance(obj, list):
            return max([self._json_depth(item, current_depth + 1) for item in obj], default=current_depth)
        else:
            return current_depth
    
    # Default sanitization rules
    def _sanitize_html_entities(self, data: str, context: Dict) -> str:
        """Sanitize HTML entities."""
        if not isinstance(data, str):
            return data
        
        # Basic HTML entity encoding
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
        }
        
        for char, entity in html_escape_table.items():
            data = data.replace(char, entity)
        
        return data
    
    def _sanitize_control_characters(self, data: str, context: Dict) -> str:
        """Remove control characters."""
        if not isinstance(data, str):
            return data
        
        # Remove control characters except whitespace
        sanitized = ''.join(char for char in data if ord(char) >= 32 or char in '\t\n\r')
        return sanitized
    
    def _sanitize_numeric_bounds(self, data: Union[int, float], context: Dict) -> Union[int, float]:
        """Clamp numeric values to bounds."""
        if not isinstance(data, (int, float)):
            return data
        
        min_value = context.get("min_value", float("-inf"))
        max_value = context.get("max_value", float("inf"))
        
        return max(min_value, min(max_value, data))


class TaskValidator:
    """Specialized validator for task objects."""
    
    def __init__(self):
        self.data_validator = DataValidator()
        
        # Task-specific validation contexts
        self.task_contexts = {
            "name": {"min_length": 1, "max_length": 200},
            "description": {"max_length": 2000},
            "priority": {"min_value": 1, "max_value": 4},
            "amplitude": {"min_value": 0.0, "max_value": 1.0},
            "phase": {"min_value": 0.0, "max_value": 6.28},  # 2π
            "success_probability": {"min_value": 0.0, "max_value": 1.0},
            "estimated_duration": {"min_value": 0},
            "earliest_start": {"min_date": datetime.now() - timedelta(days=1)},
            "latest_finish": {"max_date": datetime.now() + timedelta(days=365)}
        }
        
        logger.info("TaskValidator initialized")
    
    def validate_task(self, task: Task) -> List[ValidationError]:
        """Validate a complete task object."""
        errors = []
        
        # Validate basic fields
        errors.extend(self._validate_task_field("name", task.name, "string"))
        errors.extend(self._validate_task_field("description", task.description, "string"))
        errors.extend(self._validate_task_field("amplitude", task.amplitude, "number"))
        errors.extend(self._validate_task_field("phase", task.phase, "number"))
        errors.extend(self._validate_task_field("success_probability", task.success_probability, "number"))
        
        # Validate datetime fields
        if task.earliest_start:
            errors.extend(self._validate_task_field("earliest_start", task.earliest_start, "datetime"))
        
        if task.latest_finish:
            errors.extend(self._validate_task_field("latest_finish", task.latest_finish, "datetime"))
        
        # Validate logical constraints
        errors.extend(self._validate_task_logic(task))
        
        # Validate dependencies
        errors.extend(self._validate_task_dependencies(task))
        
        return errors
    
    def sanitize_task(self, task: Task) -> Task:
        """Sanitize task data."""
        # Sanitize string fields
        task.name = self.data_validator.sanitize(task.name, "string", self.task_contexts["name"])
        task.description = self.data_validator.sanitize(task.description, "string", self.task_contexts["description"])
        
        # Sanitize numeric fields
        task.amplitude = self.data_validator.sanitize(task.amplitude, "number", self.task_contexts["amplitude"])
        task.phase = self.data_validator.sanitize(task.phase, "number", self.task_contexts["phase"])
        task.success_probability = self.data_validator.sanitize(task.success_probability, "number", self.task_contexts["success_probability"])
        
        # Ensure task has a name
        if not task.name.strip():
            task.name = f"Task-{task.id[:8]}"
        
        return task
    
    def validate_and_sanitize_task(self, task: Task) -> tuple[Task, List[ValidationError]]:
        """Validate and sanitize task in one operation."""
        sanitized_task = self.sanitize_task(task)
        errors = self.validate_task(sanitized_task)
        return sanitized_task, errors
    
    def _validate_task_field(self, field_name: str, value: Any, data_type: str) -> List[ValidationError]:
        """Validate a specific task field."""
        context = self.task_contexts.get(field_name, {})
        errors = self.data_validator.validate(value, data_type, context)
        
        # Update field name in errors
        for error in errors:
            error.field = f"task.{field_name}"
        
        return errors
    
    def _validate_task_logic(self, task: Task) -> List[ValidationError]:
        """Validate logical constraints within task."""
        errors = []
        
        # Check date logic
        if task.earliest_start and task.latest_finish:
            if task.earliest_start >= task.latest_finish:
                errors.append(ValidationError(
                    field="task.date_logic",
                    value=f"{task.earliest_start} >= {task.latest_finish}",
                    error_type="logic_error",
                    message="Earliest start must be before latest finish"
                ))
        
        # Check duration vs. time constraints
        if task.earliest_start and task.latest_finish and task.estimated_duration:
            available_time = task.latest_finish - task.earliest_start
            if task.estimated_duration > available_time:
                errors.append(ValidationError(
                    field="task.duration_logic",
                    value=f"{task.estimated_duration} > {available_time}",
                    error_type="logic_error",
                    message="Estimated duration exceeds available time window"
                ))
        
        # Check resource requirements
        for resource, amount in task.resources_required.items():
            if amount < 0:
                errors.append(ValidationError(
                    field=f"task.resources.{resource}",
                    value=amount,
                    error_type="value_error",
                    message=f"Resource requirement cannot be negative: {resource}={amount}"
                ))
        
        return errors
    
    def _validate_task_dependencies(self, task: Task) -> List[ValidationError]:
        """Validate task dependencies."""
        errors = []
        
        # Check for self-dependency
        for dep in task.dependencies:
            if dep.task_id == task.id:
                errors.append(ValidationError(
                    field="task.dependencies",
                    value=dep.task_id,
                    error_type="logic_error",
                    message="Task cannot depend on itself"
                ))
        
        # Check for duplicate dependencies
        dep_ids = [dep.task_id for dep in task.dependencies]
        if len(dep_ids) != len(set(dep_ids)):
            duplicates = [dep_id for dep_id in dep_ids if dep_ids.count(dep_id) > 1]
            errors.append(ValidationError(
                field="task.dependencies",
                value=duplicates,
                error_type="logic_error",
                message=f"Duplicate dependencies found: {duplicates}"
            ))
        
        return errors
    
    def calculate_data_integrity_hash(self, task: Task) -> str:
        """Calculate integrity hash for task data."""
        # Create deterministic representation of task
        task_data = {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "priority": task.priority.value,
            "estimated_duration": task.estimated_duration.total_seconds(),
            "dependencies": sorted([dep.task_id for dep in task.dependencies]),
            "resources_required": sorted(task.resources_required.items()),
            "success_probability": task.success_probability
        }
        
        # Calculate SHA-256 hash
        data_str = json.dumps(task_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_data_integrity(self, task: Task, expected_hash: str) -> bool:
        """Verify task data hasn't been tampered with."""
        current_hash = self.calculate_data_integrity_hash(task)
        return current_hash == expected_hash