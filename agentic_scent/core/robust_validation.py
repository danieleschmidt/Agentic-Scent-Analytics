"""
Robust Validation Framework for Industrial Scent Analytics

Provides comprehensive data validation, input sanitization, schema enforcement,
and integrity checking for production-grade industrial applications.
"""

import asyncio
import json
import hashlib
import hmac
import time
import re
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np
from pathlib import Path
import uuid

try:
    import jsonschema
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(Enum):
    """Supported data types for validation."""
    SENSOR_READING = "sensor_reading"
    BATCH_DATA = "batch_data"
    CONFIGURATION = "configuration"
    CALIBRATION_DATA = "calibration_data"
    USER_INPUT = "user_input"
    API_REQUEST = "api_request"
    ANALYSIS_RESULT = "analysis_result"


@dataclass
class ValidationRule:
    """Single validation rule."""
    field_path: str
    rule_type: str
    parameters: Dict[str, Any]
    severity: ValidationSeverity = ValidationSeverity.ERROR
    error_message: str = ""
    custom_validator: Optional[Callable] = None


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class SecurityContext:
    """Security context for validation."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    encryption_key: Optional[bytes] = None


class RobustValidator:
    """
    Comprehensive validation framework with:
    - Schema validation
    - Data type checking
    - Range validation
    - Pattern matching
    - Integrity verification
    - Security validation
    - Custom rule support
    """
    
    def __init__(self, enable_encryption: bool = True):
        self.validation_rules: Dict[DataType, List[ValidationRule]] = {}
        self.validation_schemas: Dict[DataType, Dict[str, Any]] = {}
        self.enable_encryption = enable_encryption and CRYPTOGRAPHY_AVAILABLE
        self.security_policies: Dict[str, Any] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize default validation rules
        self._initialize_default_rules()
        self._initialize_default_schemas()
        self._initialize_security_policies()
        
    def _initialize_default_rules(self):
        """Initialize default validation rules for each data type."""
        
        # Sensor reading validation rules
        self.validation_rules[DataType.SENSOR_READING] = [
            ValidationRule(
                field_path="timestamp",
                rule_type="datetime_format",
                parameters={"format": "iso8601"},
                severity=ValidationSeverity.ERROR,
                error_message="Invalid timestamp format"
            ),
            ValidationRule(
                field_path="sensor_id",
                rule_type="pattern",
                parameters={"pattern": r"^[a-zA-Z0-9_-]+$"},
                severity=ValidationSeverity.ERROR,
                error_message="Invalid sensor ID format"
            ),
            ValidationRule(
                field_path="values",
                rule_type="sensor_values",
                parameters={"min_channels": 1, "max_channels": 1000},
                severity=ValidationSeverity.ERROR,
                error_message="Invalid sensor values"
            ),
            ValidationRule(
                field_path="quality_score",
                rule_type="range",
                parameters={"min": 0.0, "max": 1.0},
                severity=ValidationSeverity.WARNING,
                error_message="Quality score outside valid range"
            )
        ]
        
        # Batch data validation rules
        self.validation_rules[DataType.BATCH_DATA] = [
            ValidationRule(
                field_path="batch_id",
                rule_type="pattern",
                parameters={"pattern": r"^[A-Z0-9_-]+$"},
                severity=ValidationSeverity.ERROR,
                error_message="Invalid batch ID format"
            ),
            ValidationRule(
                field_path="start_time",
                rule_type="datetime_format",
                parameters={"format": "iso8601"},
                severity=ValidationSeverity.ERROR,
                error_message="Invalid start time format"
            ),
            ValidationRule(
                field_path="process_parameters",
                rule_type="required_fields",
                parameters={"required": ["temperature", "pressure", "humidity"]},
                severity=ValidationSeverity.WARNING,
                error_message="Missing required process parameters"
            )
        ]
        
        # Configuration validation rules
        self.validation_rules[DataType.CONFIGURATION] = [
            ValidationRule(
                field_path="sampling_rate",
                rule_type="range",
                parameters={"min": 0.1, "max": 1000.0},
                severity=ValidationSeverity.ERROR,
                error_message="Sampling rate outside valid range"
            ),
            ValidationRule(
                field_path="timeout",
                rule_type="range",
                parameters={"min": 1.0, "max": 300.0},
                severity=ValidationSeverity.WARNING,
                error_message="Timeout value may cause issues"
            )
        ]
        
        # User input validation rules
        self.validation_rules[DataType.USER_INPUT] = [
            ValidationRule(
                field_path="*",
                rule_type="sql_injection",
                parameters={},
                severity=ValidationSeverity.CRITICAL,
                error_message="Potential SQL injection detected"
            ),
            ValidationRule(
                field_path="*",
                rule_type="xss_protection",
                parameters={},
                severity=ValidationSeverity.CRITICAL,
                error_message="Potential XSS attack detected"
            ),
            ValidationRule(
                field_path="*",
                rule_type="max_length",
                parameters={"max_length": 10000},
                severity=ValidationSeverity.ERROR,
                error_message="Input exceeds maximum length"
            )
        ]
        
    def _initialize_default_schemas(self):
        """Initialize JSON schemas for validation."""
        
        if not JSONSCHEMA_AVAILABLE:
            self.logger.warning("JSONSchema not available, schema validation disabled")
            return
            
        # Sensor reading schema
        self.validation_schemas[DataType.SENSOR_READING] = {
            "type": "object",
            "properties": {
                "sensor_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "timestamp": {"type": "string", "format": "date-time"},
                "values": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                },
                "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                "metadata": {"type": "object"}
            },
            "required": ["sensor_id", "timestamp", "values"],
            "additionalProperties": False
        }
        
        # Batch data schema
        self.validation_schemas[DataType.BATCH_DATA] = {
            "type": "object",
            "properties": {
                "batch_id": {"type": "string", "pattern": "^[A-Z0-9_-]+$"},
                "product_type": {"type": "string", "minLength": 1},
                "start_time": {"type": "string", "format": "date-time"},
                "end_time": {"type": "string", "format": "date-time"},
                "process_parameters": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"},
                        "pressure": {"type": "number"},
                        "humidity": {"type": "number"}
                    }
                },
                "quality_data": {"type": "array"},
                "status": {
                    "type": "string",
                    "enum": ["in_progress", "completed", "failed", "cancelled"]
                }
            },
            "required": ["batch_id", "product_type", "start_time"],
            "additionalProperties": True
        }
        
        # Configuration schema
        self.validation_schemas[DataType.CONFIGURATION] = {
            "type": "object",
            "properties": {
                "site_id": {"type": "string", "minLength": 1},
                "production_line": {"type": "string", "minLength": 1},
                "sampling_rate": {"type": "number", "minimum": 0.1, "maximum": 1000},
                "timeout": {"type": "number", "minimum": 1, "maximum": 300},
                "e_nose_config": {
                    "type": "object",
                    "properties": {
                        "sensors": {"type": "array", "items": {"type": "string"}},
                        "channels": {"type": "integer", "minimum": 1, "maximum": 1000}
                    }
                }
            },
            "required": ["site_id", "production_line"],
            "additionalProperties": True
        }
        
    def _initialize_security_policies(self):
        """Initialize security validation policies."""
        
        self.security_policies = {
            "max_input_size": 1024 * 1024,  # 1MB
            "max_nested_depth": 10,
            "allowed_file_types": [".json", ".csv", ".txt"],
            "blocked_patterns": [
                r"<script[^>]*>.*?</script>",  # XSS
                r"(union|select|insert|update|delete|drop)\s+",  # SQL injection
                r"javascript:",  # JavaScript URLs
                r"data:.*base64",  # Data URLs
                r"\.\.[\\/]",  # Path traversal
            ],
            "rate_limits": {
                "requests_per_minute": 100,
                "requests_per_hour": 1000
            },
            "encryption": {
                "algorithm": "AES-256",
                "key_rotation_days": 30
            }
        }
        
    async def validate(
        self, 
        data: Dict[str, Any], 
        data_type: DataType,
        security_context: Optional[SecurityContext] = None,
        strict_mode: bool = True
    ) -> ValidationResult:
        """Comprehensive validation of input data."""
        
        start_time = time.time()
        result = ValidationResult(is_valid=True)
        
        try:
            # Security validation first
            if security_context:
                security_result = await self._validate_security(
                    data, security_context, strict_mode
                )
                result.errors.extend(security_result.errors)
                result.warnings.extend(security_result.warnings)
                
            # Schema validation
            if JSONSCHEMA_AVAILABLE and data_type in self.validation_schemas:
                schema_result = await self._validate_schema(data, data_type)
                result.errors.extend(schema_result.errors)
                result.warnings.extend(schema_result.warnings)
                
            # Rule-based validation
            if data_type in self.validation_rules:
                rules_result = await self._validate_rules(data, data_type)
                result.errors.extend(rules_result.errors)
                result.warnings.extend(rules_result.warnings)
                
            # Data sanitization
            result.sanitized_data = await self._sanitize_data(
                data, data_type, security_context
            )
            
            # Integrity verification
            integrity_result = await self._verify_integrity(data, data_type)
            result.errors.extend(integrity_result.errors)
            result.warnings.extend(integrity_result.warnings)
            
            # Determine overall validity
            result.is_valid = len(result.errors) == 0 and (
                not strict_mode or len(result.warnings) == 0
            )
            
            # Add metadata
            result.processing_time = time.time() - start_time
            result.validation_metadata = {
                "data_type": data_type.value,
                "strict_mode": strict_mode,
                "schema_validated": JSONSCHEMA_AVAILABLE,
                "security_validated": security_context is not None,
                "rules_applied": len(self.validation_rules.get(data_type, [])),
                "timestamp": datetime.now().isoformat()
            }
            
            # Audit logging
            await self._log_validation(
                data_type, result, security_context
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            result.is_valid = False
            result.errors.append({
                "type": "validation_exception",
                "message": str(e),
                "severity": ValidationSeverity.CRITICAL.value
            })
            result.processing_time = time.time() - start_time
            return result
            
    async def _validate_security(
        self, 
        data: Dict[str, Any],
        security_context: SecurityContext,
        strict_mode: bool
    ) -> ValidationResult:
        """Validate security aspects of the data."""
        
        result = ValidationResult(is_valid=True)
        
        # Input size validation
        data_size = len(json.dumps(data))
        if data_size > self.security_policies["max_input_size"]:
            result.errors.append({
                "type": "input_size_exceeded",
                "message": f"Input size {data_size} exceeds limit {self.security_policies['max_input_size']}",
                "severity": ValidationSeverity.ERROR.value
            })
            
        # Nested depth validation
        max_depth = self._calculate_nested_depth(data)
        if max_depth > self.security_policies["max_nested_depth"]:
            result.errors.append({
                "type": "nesting_too_deep",
                "message": f"Data nesting depth {max_depth} exceeds limit",
                "severity": ValidationSeverity.ERROR.value
            })
            
        # Pattern-based security checks
        for pattern in self.security_policies["blocked_patterns"]:
            matches = self._find_pattern_matches(data, pattern)
            if matches:
                result.errors.append({
                    "type": "security_pattern_detected",
                    "message": f"Blocked pattern detected: {pattern}",
                    "severity": ValidationSeverity.CRITICAL.value,
                    "matches": matches
                })
                
        # Permission validation
        if security_context.permissions:
            required_permissions = self._get_required_permissions(data)
            missing_permissions = set(required_permissions) - set(security_context.permissions)
            if missing_permissions:
                result.errors.append({
                    "type": "insufficient_permissions",
                    "message": f"Missing permissions: {list(missing_permissions)}",
                    "severity": ValidationSeverity.ERROR.value
                })
                
        return result
        
    async def _validate_schema(
        self, 
        data: Dict[str, Any], 
        data_type: DataType
    ) -> ValidationResult:
        """Validate data against JSON schema."""
        
        result = ValidationResult(is_valid=True)
        
        if data_type not in self.validation_schemas:
            return result
            
        try:
            schema = self.validation_schemas[data_type]
            validate(instance=data, schema=schema)
            
        except ValidationError as e:
            result.errors.append({
                "type": "schema_validation",
                "message": f"Schema validation failed: {e.message}",
                "severity": ValidationSeverity.ERROR.value,
                "field_path": e.absolute_path,
                "failed_value": e.instance
            })
            
        except Exception as e:
            result.errors.append({
                "type": "schema_validation_exception",
                "message": f"Schema validation error: {e}",
                "severity": ValidationSeverity.ERROR.value
            })
            
        return result
        
    async def _validate_rules(
        self, 
        data: Dict[str, Any], 
        data_type: DataType
    ) -> ValidationResult:
        """Validate data against custom rules."""
        
        result = ValidationResult(is_valid=True)
        
        rules = self.validation_rules.get(data_type, [])
        
        for rule in rules:
            try:
                rule_result = await self._apply_validation_rule(data, rule)
                if not rule_result:
                    error_entry = {
                        "type": "rule_validation",
                        "message": rule.error_message or f"Rule {rule.rule_type} failed",
                        "severity": rule.severity.value,
                        "field_path": rule.field_path,
                        "rule_type": rule.rule_type
                    }
                    
                    if rule.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        result.errors.append(error_entry)
                    else:
                        result.warnings.append(error_entry)
                        
            except Exception as e:
                result.errors.append({
                    "type": "rule_validation_exception",
                    "message": f"Rule validation error: {e}",
                    "severity": ValidationSeverity.ERROR.value,
                    "rule_type": rule.rule_type
                })
                
        return result
        
    async def _apply_validation_rule(
        self, 
        data: Dict[str, Any], 
        rule: ValidationRule
    ) -> bool:
        """Apply a single validation rule."""
        
        # Get field value
        if rule.field_path == "*":
            # Apply to all fields
            values = self._get_all_values(data)
        else:
            values = [self._get_field_value(data, rule.field_path)]
            
        # Apply rule based on type
        for value in values:
            if value is None:
                continue
                
            if rule.custom_validator:
                if not rule.custom_validator(value, rule.parameters):
                    return False
                continue
                
            # Built-in rule types
            if rule.rule_type == "range":
                if not self._validate_range(value, rule.parameters):
                    return False
                    
            elif rule.rule_type == "pattern":
                if not self._validate_pattern(value, rule.parameters):
                    return False
                    
            elif rule.rule_type == "datetime_format":
                if not self._validate_datetime(value, rule.parameters):
                    return False
                    
            elif rule.rule_type == "sensor_values":
                if not self._validate_sensor_values(value, rule.parameters):
                    return False
                    
            elif rule.rule_type == "required_fields":
                if not self._validate_required_fields(value, rule.parameters):
                    return False
                    
            elif rule.rule_type == "sql_injection":
                if not self._validate_sql_injection(value):
                    return False
                    
            elif rule.rule_type == "xss_protection":
                if not self._validate_xss(value):
                    return False
                    
            elif rule.rule_type == "max_length":
                if not self._validate_max_length(value, rule.parameters):
                    return False
                    
        return True
        
    def _validate_range(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate numeric range."""
        if not isinstance(value, (int, float)):
            return False
        return params.get("min", float("-inf")) <= value <= params.get("max", float("inf"))
        
    def _validate_pattern(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate regex pattern."""
        if not isinstance(value, str):
            return False
        pattern = params.get("pattern", ".*")
        return bool(re.match(pattern, value))
        
    def _validate_datetime(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate datetime format."""
        if not isinstance(value, str):
            return False
        try:
            if params.get("format") == "iso8601":
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False
            
    def _validate_sensor_values(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate sensor values structure."""
        if not isinstance(value, dict):
            return False
        
        channel_count = len(value)
        min_channels = params.get("min_channels", 1)
        max_channels = params.get("max_channels", 1000)
        
        if not (min_channels <= channel_count <= max_channels):
            return False
            
        # Validate all values are numeric
        for v in value.values():
            if not isinstance(v, (int, float, np.number)):
                return False
                
        return True
        
    def _validate_required_fields(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate required fields presence."""
        if not isinstance(value, dict):
            return False
        
        required_fields = params.get("required", [])
        return all(field in value for field in required_fields)
        
    def _validate_sql_injection(self, value: Any) -> bool:
        """Check for SQL injection patterns."""
        if not isinstance(value, str):
            return True
        
        sql_patterns = [
            r"(union|select|insert|update|delete|drop)\s+",
            r"(or|and)\s+\d+\s*=\s*\d+",
            r"['\"]\s*(or|and)\s+['\"]\s*[^'\"]*['\"]\s*=\s*['\"].*",
            r"--",
            r"/\*.*\*/"
        ]
        
        value_lower = value.lower()
        return not any(re.search(pattern, value_lower) for pattern in sql_patterns)
        
    def _validate_xss(self, value: Any) -> bool:
        """Check for XSS patterns."""
        if not isinstance(value, str):
            return True
        
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>"
        ]
        
        value_lower = value.lower()
        return not any(re.search(pattern, value_lower) for pattern in xss_patterns)
        
    def _validate_max_length(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate maximum length."""
        max_length = params.get("max_length", 1000)
        if isinstance(value, str):
            return len(value) <= max_length
        elif isinstance(value, (list, dict)):
            return len(str(value)) <= max_length
        return True
        
    def _get_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get field value using dot notation."""
        try:
            keys = field_path.split(".")
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except Exception:
            return None
            
    def _get_all_values(self, data: Dict[str, Any]) -> List[Any]:
        """Get all values from nested dictionary."""
        values = []
        
        def extract_values(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    extract_values(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_values(item)
            else:
                values.append(obj)
                
        extract_values(data)
        return values
        
    def _calculate_nested_depth(self, data: Dict[str, Any]) -> int:
        """Calculate maximum nesting depth."""
        def get_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                return max([get_depth(v, current_depth + 1) for v in obj.values()], default=current_depth)
            elif isinstance(obj, list):
                return max([get_depth(item, current_depth + 1) for item in obj], default=current_depth)
            else:
                return current_depth
                
        return get_depth(data)
        
    def _find_pattern_matches(self, data: Dict[str, Any], pattern: str) -> List[str]:
        """Find all matches of a pattern in the data."""
        matches = []
        
        def search_in_data(obj):
            if isinstance(obj, str):
                if re.search(pattern, obj, re.IGNORECASE):
                    matches.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    search_in_data(v)
            elif isinstance(obj, list):
                for item in obj:
                    search_in_data(item)
                    
        search_in_data(data)
        return matches
        
    def _get_required_permissions(self, data: Dict[str, Any]) -> List[str]:
        """Determine required permissions based on data content."""
        permissions = ["read"]
        
        # Check for operations that require write permissions
        write_indicators = ["batch_id", "calibration", "configuration"]
        if any(indicator in str(data).lower() for indicator in write_indicators):
            permissions.append("write")
            
        # Check for admin operations
        admin_indicators = ["security", "user", "permission", "system"]
        if any(indicator in str(data).lower() for indicator in admin_indicators):
            permissions.append("admin")
            
        return permissions
        
    async def _sanitize_data(
        self, 
        data: Dict[str, Any], 
        data_type: DataType,
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Sanitize input data."""
        
        sanitized = json.loads(json.dumps(data))  # Deep copy
        
        def sanitize_value(value):
            if isinstance(value, str):
                # Remove potential XSS
                value = re.sub(r"<script[^>]*>.*?</script>", "", value, flags=re.IGNORECASE)
                value = re.sub(r"javascript:", "", value, flags=re.IGNORECASE)
                value = re.sub(r"on\w+\s*=", "", value, flags=re.IGNORECASE)
                
                # Trim whitespace
                value = value.strip()
                
                # Limit length
                max_len = 10000
                if len(value) > max_len:
                    value = value[:max_len]
                    
            elif isinstance(value, dict):
                for k, v in value.items():
                    value[k] = sanitize_value(v)
            elif isinstance(value, list):
                value = [sanitize_value(item) for item in value]
                
            return value
            
        return sanitize_value(sanitized)
        
    async def _verify_integrity(
        self, 
        data: Dict[str, Any], 
        data_type: DataType
    ) -> ValidationResult:
        """Verify data integrity."""
        
        result = ValidationResult(is_valid=True)
        
        # Check for data consistency
        if data_type == DataType.SENSOR_READING:
            if "timestamp" in data and "values" in data:
                timestamp_str = data["timestamp"]
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    
                    # Check if timestamp is reasonable (not too far in past/future)
                    now = datetime.now(timestamp.tzinfo)
                    if abs((now - timestamp).total_seconds()) > 86400:  # 24 hours
                        result.warnings.append({
                            "type": "timestamp_out_of_range",
                            "message": "Timestamp is more than 24 hours from current time",
                            "severity": ValidationSeverity.WARNING.value
                        })
                        
                except Exception:
                    pass  # Already caught by datetime validation
                    
        # Check for data completeness
        if isinstance(data, dict):
            if len(data) == 0:
                result.warnings.append({
                    "type": "empty_data",
                    "message": "Data object is empty",
                    "severity": ValidationSeverity.WARNING.value
                })
                
        return result
        
    async def _log_validation(
        self, 
        data_type: DataType,
        result: ValidationResult,
        security_context: Optional[SecurityContext] = None
    ):
        """Log validation results for audit trail."""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "data_type": data_type.value,
            "is_valid": result.is_valid,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "processing_time": result.processing_time,
            "user_id": security_context.user_id if security_context else None,
            "session_id": security_context.session_id if security_context else None,
            "source_ip": security_context.source_ip if security_context else None
        }
        
        self.audit_log.append(log_entry)
        
        # Limit audit log size
        max_entries = 10000
        if len(self.audit_log) > max_entries:
            self.audit_log = self.audit_log[-max_entries:]
            
        # Log critical issues
        critical_errors = [
            error for error in result.errors 
            if error.get("severity") == ValidationSeverity.CRITICAL.value
        ]
        
        if critical_errors:
            self.logger.critical(
                f"Critical validation errors detected: {len(critical_errors)} errors"
            )
            
    def add_custom_rule(self, data_type: DataType, rule: ValidationRule):
        """Add custom validation rule."""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []
        self.validation_rules[data_type].append(rule)
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation system summary."""
        total_validations = len(self.audit_log)
        if total_validations == 0:
            return {"total_validations": 0}
            
        successful_validations = sum(1 for entry in self.audit_log if entry["is_valid"])
        avg_processing_time = np.mean([entry["processing_time"] for entry in self.audit_log])
        
        recent_entries = self.audit_log[-100:]  # Last 100 entries
        error_rate = sum(1 for entry in recent_entries if not entry["is_valid"]) / len(recent_entries)
        
        return {
            "total_validations": total_validations,
            "success_rate": successful_validations / total_validations,
            "recent_error_rate": error_rate,
            "average_processing_time": float(avg_processing_time),
            "supported_data_types": [dt.value for dt in self.validation_rules.keys()],
            "security_features_enabled": {
                "encryption": self.enable_encryption,
                "pattern_blocking": len(self.security_policies["blocked_patterns"]) > 0,
                "permission_checking": True,
                "audit_logging": True
            }
        }


async def create_robust_validator(enable_encryption: bool = True) -> RobustValidator:
    """Create configured robust validator."""
    return RobustValidator(enable_encryption=enable_encryption)


async def demonstrate_robust_validation():
    """Demonstration of robust validation capabilities."""
    print("🔒 Robust Validation Framework Demo")
    print("=" * 50)
    
    # Create validator
    validator = await create_robust_validator()
    
    # Test data samples
    test_cases = [
        {
            "name": "Valid Sensor Reading",
            "data": {
                "sensor_id": "e_nose_01",
                "timestamp": datetime.now().isoformat(),
                "values": {"ch1": 0.5, "ch2": 1.2, "ch3": 0.8},
                "quality_score": 0.95
            },
            "data_type": DataType.SENSOR_READING
        },
        {
            "name": "Invalid Sensor Reading (Bad Quality Score)",
            "data": {
                "sensor_id": "e_nose_01",
                "timestamp": datetime.now().isoformat(),
                "values": {"ch1": 0.5},
                "quality_score": 1.5  # Invalid: > 1.0
            },
            "data_type": DataType.SENSOR_READING
        },
        {
            "name": "Security Threat (SQL Injection)",
            "data": {
                "user_input": "'; DROP TABLE users; --",
                "comment": "This looks suspicious"
            },
            "data_type": DataType.USER_INPUT
        },
        {
            "name": "Valid Batch Data",
            "data": {
                "batch_id": "BATCH_2024_001",
                "product_type": "tablet_coating",
                "start_time": datetime.now().isoformat(),
                "process_parameters": {
                    "temperature": 25.0,
                    "pressure": 1013.25,
                    "humidity": 45.0
                },
                "status": "in_progress"
            },
            "data_type": DataType.BATCH_DATA
        }
    ]
    
    # Security context
    security_context = SecurityContext(
        user_id="test_user",
        session_id=str(uuid.uuid4()),
        source_ip="192.168.1.100",
        permissions=["read", "write"]
    )
    
    print("\n📊 Validation Results:")
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        
        result = await validator.validate(
            test_case["data"],
            test_case["data_type"],
            security_context
        )
        
        print(f"    Valid: {result.is_valid}")
        print(f"    Errors: {len(result.errors)}")
        print(f"    Warnings: {len(result.warnings)}")
        print(f"    Processing Time: {result.processing_time:.3f}s")
        
        if result.errors:
            print("    Error Details:")
            for error in result.errors[:2]:  # Show first 2 errors
                print(f"      - {error['type']}: {error['message']}")
                
    # Validation summary
    print("\n📊 Validation System Summary:")
    summary = validator.get_validation_summary()
    print(f"  Total Validations: {summary['total_validations']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Average Processing Time: {summary['average_processing_time']:.3f}s")
    print(f"  Supported Data Types: {len(summary['supported_data_types'])}")
    
    print("\n✅ Robust validation demonstration completed!")
    
    return validator


if __name__ == "__main__":
    asyncio.run(demonstrate_robust_validation())
