"""
Comprehensive logging configuration for agentic scent analytics.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add context fields
        for attr in ['sensor_id', 'agent_id', 'batch_id', 'session_id', 'user_id']:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        
        return json.dumps(log_entry, default=str)


class ContextualLogger:
    """Logger with contextual information."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set contextual information."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear contextual information."""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Log with contextual information."""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)


def setup_logging(log_level: str = "INFO", 
                  log_dir: Optional[str] = None,
                  enable_console: bool = True,
                  enable_file: bool = True,
                  enable_structured: bool = False,
                  max_file_size: int = 100 * 1024 * 1024,  # 100MB
                  backup_count: int = 5) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging for the system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        enable_console: Enable console output
        enable_file: Enable file output
        enable_structured: Use structured JSON logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Create log directory
    if log_dir is None:
        log_dir = "./logs"
    
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatters
    if enable_structured:
        formatter = StructuredFormatter()
        console_formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    loggers = {}
    
    if enable_file:
        # Main application log
        main_handler = logging.handlers.RotatingFileHandler(
            log_path / "agentic_scent.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        main_handler.setLevel(numeric_level)
        main_handler.setFormatter(formatter)
        root_logger.addHandler(main_handler)
        
        # Specialized loggers
        specialized_logs = {
            "sensors": "sensor_data.log",
            "agents": "agent_analysis.log", 
            "factory": "factory_operations.log",
            "security": "security_audit.log",
            "performance": "performance.log",
            "errors": "errors.log",
            "quantum": "quantum_operations.log"
        }
        
        for logger_name, filename in specialized_logs.items():
            logger = logging.getLogger(f"agentic_scent.{logger_name}")
            logger.setLevel(numeric_level)
            
            handler = logging.handlers.RotatingFileHandler(
                log_path / filename,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            handler.setLevel(numeric_level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            loggers[logger_name] = ContextualLogger(logger)
    
    # Error-only handler for critical issues
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / "critical_errors.log",
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Create main contextual loggers
    loggers.update({
        "main": ContextualLogger(logging.getLogger("agentic_scent")),
        "root": ContextualLogger(root_logger)
    })
    
    # Log configuration
    main_logger = loggers["main"]
    main_logger.info(f"Logging configured: level={log_level}, dir={log_dir}")
    main_logger.info(f"Log rotation: max_size={max_file_size//1024//1024}MB, backups={backup_count}")
    
    return loggers


def get_contextual_logger(name: str) -> ContextualLogger:
    """Get a contextual logger for a specific component."""
    return ContextualLogger(logging.getLogger(name))


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_contextual_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        
        # Set initial context
        if hasattr(self, 'agent_id'):
            self.logger.set_context(agent_id=self.agent_id)
        elif hasattr(self, 'sensor_id'):
            self.logger.set_context(sensor_id=self.sensor_id)
        elif hasattr(self, 'production_line'):
            self.logger.set_context(production_line=self.production_line)


def log_performance(func):
    """Decorator to log function performance."""
    import time
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_contextual_logger(f"{func.__module__}.{func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Function completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function failed after {duration:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_contextual_logger(f"{func.__module__}.{func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Function completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function failed after {duration:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def log_error_context(logger: ContextualLogger, error: Exception, 
                     context: Dict[str, Any] = None):
    """Log error with full context information."""
    
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    
    if hasattr(error, 'error_code'):
        error_info["error_code"] = error.error_code
    
    if hasattr(error, 'context'):
        error_info["error_context"] = error.context
    
    logger.error("Exception occurred", extra={"extra_fields": error_info})