"""
Input validation and security checks for sentiment analysis
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
import html
import unicodedata

from ..core.models import TextInput, AnalysisConfig, ModelType

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class TextValidator:
    """Validates and sanitizes text input for security and quality"""
    
    # Security patterns to detect
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'<iframe[^>]*>.*?</iframe>',  # Iframe tags
        r'<object[^>]*>.*?</object>',  # Object tags
        r'<embed[^>]*>.*?</embed>',    # Embed tags
        r'<form[^>]*>.*?</form>',      # Form tags
    ]
    
    # SQL injection patterns
    SQL_PATTERNS = [
        r'union\s+select',
        r'drop\s+table',
        r'delete\s+from',
        r'insert\s+into',
        r'update\s+set',
        r'\bor\s+1\s*=\s*1\b',
        r'\band\s+1\s*=\s*1\b',
    ]
    
    # Command injection patterns
    CMD_PATTERNS = [
        r';\s*(rm|del|format|shutdown)',
        r'\$\([^)]+\)',  # Command substitution
        r'`[^`]+`',      # Backtick command execution
        r'\|\s*nc\s',    # Netcat piping
        r'&&\s*(curl|wget)',
    ]
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._compiled_patterns = {
            'suspicious': [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                          for pattern in self.SUSPICIOUS_PATTERNS],
            'sql': [re.compile(pattern, re.IGNORECASE) 
                   for pattern in self.SQL_PATTERNS],
            'cmd': [re.compile(pattern, re.IGNORECASE) 
                   for pattern in self.CMD_PATTERNS]
        }
    
    def validate_text_input(self, text_input: TextInput) -> Tuple[TextInput, List[str]]:
        """Validate and sanitize text input"""
        errors = []
        
        # Basic validation
        if not isinstance(text_input.text, str):
            errors.append("Text must be a string")
            return text_input, errors
        
        # Length validation
        if len(text_input.text) == 0:
            errors.append("Text cannot be empty")
        elif len(text_input.text) > 100000:
            errors.append("Text exceeds maximum length (100,000 characters)")
        
        # Character validation
        if not self._is_valid_text(text_input.text):
            errors.append("Text contains invalid characters")
        
        # Security checks
        security_issues = self._check_security_patterns(text_input.text)
        if security_issues:
            errors.extend(security_issues)
            if self.strict_mode:
                raise SecurityError(f"Security violations detected: {', '.join(security_issues)}")
        
        # Sanitize text
        sanitized_text = self._sanitize_text(text_input.text)
        
        # Create sanitized input
        sanitized_input = TextInput(
            text=sanitized_text,
            language=text_input.language,
            metadata=text_input.metadata
        )
        
        return sanitized_input, errors
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text contains only valid characters"""
        try:
            # Normalize unicode
            normalized = unicodedata.normalize('NFKC', text)
            
            # Check for control characters (except common whitespace)
            for char in normalized:
                if unicodedata.category(char)[0] == 'C' and char not in '\t\n\r ':
                    return False
            
            return True
        except Exception:
            return False
    
    def _check_security_patterns(self, text: str) -> List[str]:
        """Check text for security patterns"""
        issues = []
        
        # Check suspicious patterns
        for pattern in self._compiled_patterns['suspicious']:
            if pattern.search(text):
                issues.append(f"Suspicious HTML/JS pattern detected")
                logger.warning(f"Suspicious pattern found: {pattern.pattern}")
        
        # Check SQL injection patterns
        for pattern in self._compiled_patterns['sql']:
            if pattern.search(text):
                issues.append("Potential SQL injection pattern detected")
                logger.warning(f"SQL injection pattern found: {pattern.pattern}")
        
        # Check command injection patterns
        for pattern in self._compiled_patterns['cmd']:
            if pattern.search(text):
                issues.append("Potential command injection pattern detected")
                logger.warning(f"Command injection pattern found: {pattern.pattern}")
        
        return list(set(issues))  # Remove duplicates
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing/escaping dangerous content"""
        # HTML escape
        sanitized = html.escape(text, quote=False)
        
        # Normalize unicode
        sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Remove null bytes and other control characters
        sanitized = ''.join(char for char in sanitized 
                          if unicodedata.category(char)[0] != 'C' or char in '\t\n\r ')
        
        # Limit consecutive whitespace
        sanitized = re.sub(r'\s{10,}', ' ' * 10, sanitized)
        
        return sanitized.strip()


class ConfigValidator:
    """Validates analysis configuration for security and correctness"""
    
    MAX_TIMEOUT = 300  # 5 minutes
    MAX_RETRIES = 10
    MAX_BATCH_SIZE = 1000
    
    @staticmethod
    def validate_config(config: AnalysisConfig) -> Tuple[AnalysisConfig, List[str]]:
        """Validate and sanitize analysis configuration"""
        errors = []
        
        # Validate models
        if not config.models:
            errors.append("At least one model must be specified")
        
        # Check for valid model types
        valid_models = set(ModelType)
        for model in config.models:
            if model not in valid_models:
                errors.append(f"Invalid model type: {model}")
        
        # Validate timeout
        if config.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
            config.timeout_seconds = 30
        elif config.timeout_seconds > ConfigValidator.MAX_TIMEOUT:
            errors.append(f"Timeout exceeds maximum ({ConfigValidator.MAX_TIMEOUT}s)")
            config.timeout_seconds = ConfigValidator.MAX_TIMEOUT
        
        # Validate retries
        if config.max_retries < 0:
            errors.append("Max retries cannot be negative")
            config.max_retries = 0
        elif config.max_retries > ConfigValidator.MAX_RETRIES:
            errors.append(f"Max retries exceeds maximum ({ConfigValidator.MAX_RETRIES})")
            config.max_retries = ConfigValidator.MAX_RETRIES
        
        # Validate batch size
        if config.batch_size <= 0:
            errors.append("Batch size must be positive")
            config.batch_size = 32
        elif config.batch_size > ConfigValidator.MAX_BATCH_SIZE:
            errors.append(f"Batch size exceeds maximum ({ConfigValidator.MAX_BATCH_SIZE})")
            config.batch_size = ConfigValidator.MAX_BATCH_SIZE
        
        # Validate model names (basic checks)
        if config.transformers_model and not ConfigValidator._is_valid_model_name(config.transformers_model):
            errors.append("Invalid transformers model name")
        
        if config.openai_model and not ConfigValidator._is_valid_model_name(config.openai_model):
            errors.append("Invalid OpenAI model name")
        
        if config.anthropic_model and not ConfigValidator._is_valid_model_name(config.anthropic_model):
            errors.append("Invalid Anthropic model name")
        
        return config, errors
    
    @staticmethod
    def _is_valid_model_name(model_name: str) -> bool:
        """Check if model name is valid"""
        if not model_name or not isinstance(model_name, str):
            return False
        
        # Basic checks
        if len(model_name) > 200:
            return False
        
        # Should contain only alphanumeric, hyphens, underscores, slashes, dots
        if not re.match(r'^[a-zA-Z0-9\-_/.]+$', model_name):
            return False
        
        return True


class RateLimiter:
    """Simple rate limiting for API calls"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}  # IP -> list of timestamps
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for given IP"""
        now = datetime.now(timezone.utc).timestamp()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if client_ip in self._requests:
            self._requests[client_ip] = [
                timestamp for timestamp in self._requests[client_ip] 
                if timestamp > window_start
            ]
        else:
            self._requests[client_ip] = []
        
        # Check rate limit
        if len(self._requests[client_ip]) >= self.max_requests:
            return False
        
        # Record new request
        self._requests[client_ip].append(now)
        return True
    
    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for IP"""
        if client_ip not in self._requests:
            return self.max_requests
        
        now = datetime.now(timezone.utc).timestamp()
        window_start = now - self.window_seconds
        
        current_requests = sum(
            1 for timestamp in self._requests[client_ip] 
            if timestamp > window_start
        )
        
        return max(0, self.max_requests - current_requests)


class AuditLogger:
    """Security audit logging"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger('sentiment_analyzer.security')
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_validation_error(self, client_ip: str, text_hash: str, errors: List[str]):
        """Log validation errors"""
        self.logger.warning(
            f"Validation error - IP: {client_ip}, "
            f"Text hash: {text_hash}, Errors: {', '.join(errors)}"
        )
    
    def log_security_violation(self, client_ip: str, violation_type: str, details: str):
        """Log security violations"""
        self.logger.error(
            f"Security violation - IP: {client_ip}, "
            f"Type: {violation_type}, Details: {details}"
        )
    
    def log_rate_limit_exceeded(self, client_ip: str):
        """Log rate limit violations"""
        self.logger.warning(f"Rate limit exceeded - IP: {client_ip}")
    
    def log_analysis_success(self, client_ip: str, text_length: int, processing_time: float):
        """Log successful analysis"""
        self.logger.info(
            f"Analysis success - IP: {client_ip}, "
            f"Text length: {text_length}, Time: {processing_time:.2f}ms"
        )
    
    def log_analysis_error(self, client_ip: str, error: str):
        """Log analysis errors"""
        self.logger.error(f"Analysis error - IP: {client_ip}, Error: {error}")
    
    @staticmethod
    def hash_text(text: str) -> str:
        """Create hash of text for logging (privacy-preserving)"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]