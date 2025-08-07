"""
Logging configuration and utilities
"""

import logging
import logging.config
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            } and not key.startswith('_'):
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class SentimentAnalyzerFilter(logging.Filter):
    """Custom filter for sentiment analyzer logs"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add custom fields
        record.service = "sentiment-analyzer-pro"
        record.version = "1.0.0"
        
        # Filter out noisy logs in production
        if os.getenv("ENVIRONMENT") == "production":
            # Skip debug logs from certain modules
            if record.levelno == logging.DEBUG and record.module in {"urllib3", "requests"}:
                return False
        
        return True


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """Setup logging configuration"""
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    if format_type.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add custom filter
    sentiment_filter = SentimentAnalyzerFilter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sentiment_filter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sentiment_filter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # Set our logger levels
    logging.getLogger("sentiment_analyzer").setLevel(numeric_level)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    logging.info(f"Logging configured - Level: {level}, Format: {format_type}")


def get_logger(name: str) -> logging.Logger:
    """Get configured logger for module"""
    return logging.getLogger(f"sentiment_analyzer.{name}")


class ContextualLogger:
    """Logger with contextual information"""
    
    def __init__(self, name: str, context: Dict[str, Any] = None):
        self.logger = get_logger(name)
        self.context = context or {}
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context"""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def add_context(self, **kwargs):
        """Add context fields"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context"""
        self.context.clear()


# Pre-configured logging setup for different environments
LOGGING_CONFIGS = {
    "development": {
        "level": "DEBUG",
        "format_type": "standard",
        "enable_console": True,
        "log_file": None
    },
    "testing": {
        "level": "WARNING",
        "format_type": "standard", 
        "enable_console": False,
        "log_file": "tests/test.log"
    },
    "production": {
        "level": "INFO",
        "format_type": "json",
        "enable_console": True,
        "log_file": "/var/log/sentiment-analyzer/app.log"
    }
}


def setup_environment_logging(environment: str = None):
    """Setup logging for specific environment"""
    if not environment:
        environment = os.getenv("ENVIRONMENT", "development")
    
    config = LOGGING_CONFIGS.get(environment, LOGGING_CONFIGS["development"])
    setup_logging(**config)
    
    get_logger("config").info(f"Logging setup for environment: {environment}")


# Default setup if imported
if not logging.getLogger().handlers:
    setup_environment_logging()