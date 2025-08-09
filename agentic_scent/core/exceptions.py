"""
Custom exceptions for agentic scent analytics system.
"""

from typing import Optional, Dict, Any


class AgenticScentError(Exception):
    """Base exception for all agentic scent analytics errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}


class SensorError(AgenticScentError):
    """Sensor-related errors."""
    pass


class SensorConnectionError(SensorError):
    """Sensor connection failures."""
    pass


class SensorCalibrationError(SensorError):
    """Sensor calibration issues."""
    pass


class SensorDataError(SensorError):
    """Invalid or corrupted sensor data."""
    pass


class AgentError(AgenticScentError):
    """Agent-related errors."""
    pass


class AgentInitializationError(AgentError):
    """Agent initialization failures."""
    pass


class AgentCommunicationError(AgentError):
    """Agent communication issues."""
    pass


class LLMError(AgentError):
    """LLM service errors."""
    pass


class AnalysisError(AgenticScentError):
    """Analysis processing errors."""
    pass


class ModelError(AnalysisError):
    """ML model errors."""
    pass


class ValidationError(AgenticScentError):
    """Data validation errors."""
    pass


class ConfigurationError(AgenticScentError):
    """Configuration-related errors."""
    pass


class SecurityError(AgenticScentError):
    """Security and authentication errors."""
    pass


class FactoryError(AgenticScentError):
    """Factory system errors."""
    pass


class IntegrationError(AgenticScentError):
    """System integration errors."""
    pass


class DatabaseError(AgenticScentError):
    """Database operation errors."""
    pass


class CacheError(AgenticScentError):
    """Cache operation errors."""
    pass


class QuantumError(AgenticScentError):
    """Quantum planner integration errors."""
    pass


# Error code mappings
ERROR_CODES = {
    # Sensor errors (1000-1999)
    "SENSOR_CONNECTION_FAILED": 1001,
    "SENSOR_TIMEOUT": 1002,
    "SENSOR_CALIBRATION_FAILED": 1003,
    "SENSOR_DATA_INVALID": 1004,
    "SENSOR_DRIFT_DETECTED": 1005,
    
    # Agent errors (2000-2999)
    "AGENT_INIT_FAILED": 2001,
    "AGENT_COMMUNICATION_FAILED": 2002,
    "LLM_SERVICE_UNAVAILABLE": 2003,
    "LLM_QUOTA_EXCEEDED": 2004,
    "AGENT_TIMEOUT": 2005,
    
    # Analysis errors (3000-3999)
    "ANALYSIS_FAILED": 3001,
    "MODEL_PREDICTION_FAILED": 3002,
    "INSUFFICIENT_DATA": 3003,
    "ANOMALY_DETECTION_FAILED": 3004,
    
    # Validation errors (4000-4999)
    "DATA_VALIDATION_FAILED": 4001,
    "SCHEMA_VALIDATION_FAILED": 4002,
    "PARAMETER_VALIDATION_FAILED": 4003,
    
    # Configuration errors (5000-5999)
    "CONFIG_LOAD_FAILED": 5001,
    "CONFIG_VALIDATION_FAILED": 5002,
    "MISSING_REQUIRED_CONFIG": 5003,
    
    # Security errors (6000-6999)
    "AUTHENTICATION_FAILED": 6001,
    "AUTHORIZATION_FAILED": 6002,
    "ENCRYPTION_FAILED": 6003,
    "AUDIT_LOG_FAILED": 6004,
    
    # System errors (7000-7999)
    "FACTORY_STARTUP_FAILED": 7001,
    "DATABASE_CONNECTION_FAILED": 7002,
    "CACHE_CONNECTION_FAILED": 7003,
    "INTEGRATION_FAILED": 7004,
    
    # Quantum errors (8000-8999)
    "QUANTUM_OPTIMIZATION_FAILED": 8001,
    "QUANTUM_TASK_FAILED": 8002,
    "QUANTUM_COORDINATOR_FAILED": 8003
}


def get_error_code(error_key: str) -> Optional[int]:
    """Get numeric error code for error key."""
    return ERROR_CODES.get(error_key)


def create_error_with_code(exception_class: type, message: str, error_key: str, 
                          context: Optional[Dict[str, Any]] = None) -> AgenticScentError:
    """Create exception with error code."""
    error_code = get_error_code(error_key)
    return exception_class(
        message=message,
        error_code=f"{error_key}:{error_code}" if error_code else error_key,
        context=context
    )