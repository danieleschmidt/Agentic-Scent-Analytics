"""
Security tests for sentiment analyzer
"""

import pytest
from sentiment_analyzer.security.validator import (
    TextValidator, ConfigValidator, SecurityError, ValidationError,
    RateLimiter, AuditLogger
)
from sentiment_analyzer.core.models import TextInput, AnalysisConfig, ModelType


class TestTextValidator:
    """Test text validation and security"""
    
    def test_basic_validation(self):
        """Test basic text validation"""
        validator = TextValidator()
        
        # Valid text
        text_input = TextInput(text="This is a valid text message.")
        sanitized, errors = validator.validate_text_input(text_input)
        
        assert len(errors) == 0
        assert sanitized.text == text_input.text
    
    def test_empty_text_validation(self):
        """Test empty text validation"""
        validator = TextValidator()
        
        # Empty text
        with pytest.raises(ValueError):
            TextInput(text="")
        
        # Whitespace only
        with pytest.raises(ValueError):
            TextInput(text="   ")
    
    def test_length_validation(self):
        """Test text length validation"""
        validator = TextValidator()
        
        # Very long text
        long_text = "a" * 100001  # Exceeds 100,000 limit
        text_input = TextInput(text=long_text)
        sanitized, errors = validator.validate_text_input(text_input)
        
        assert any("exceeds maximum length" in error for error in errors)
    
    def test_security_pattern_detection(self, security_test_inputs):
        """Test detection of malicious patterns"""
        validator = TextValidator()
        
        for malicious_text in security_test_inputs:
            text_input = TextInput(text=malicious_text)
            sanitized, errors = validator.validate_text_input(text_input)
            
            # Should detect security issues
            assert len(errors) > 0
            security_errors = [e for e in errors if "injection" in e.lower() or "pattern" in e.lower()]
            assert len(security_errors) > 0
    
    def test_strict_mode(self, security_test_inputs):
        """Test strict mode security validation"""
        validator = TextValidator(strict_mode=True)
        
        # Should raise SecurityError in strict mode
        with pytest.raises(SecurityError):
            text_input = TextInput(text=security_test_inputs[0])
            validator.validate_text_input(text_input)
    
    def test_html_sanitization(self):
        """Test HTML content sanitization"""
        validator = TextValidator()
        
        html_text = "This contains <b>bold</b> and <i>italic</i> text."
        text_input = TextInput(text=html_text)
        sanitized, errors = validator.validate_text_input(text_input)
        
        # HTML should be escaped
        assert "&lt;b&gt;" in sanitized.text or sanitized.text != html_text
    
    def test_unicode_normalization(self):
        """Test Unicode text normalization"""
        validator = TextValidator()
        
        # Text with special Unicode characters
        unicode_text = "This has unicode: café, naïve, résumé"
        text_input = TextInput(text=unicode_text)
        sanitized, errors = validator.validate_text_input(text_input)
        
        assert len(errors) == 0
        assert sanitized.text is not None
    
    def test_control_character_filtering(self):
        """Test filtering of control characters"""
        validator = TextValidator()
        
        # Text with control characters
        control_text = "Normal text\x00null\x01control"
        text_input = TextInput(text=control_text)
        sanitized, errors = validator.validate_text_input(text_input)
        
        # Control characters should be filtered
        assert "\x00" not in sanitized.text
        assert "\x01" not in sanitized.text


class TestConfigValidator:
    """Test configuration validation"""
    
    def test_valid_config(self):
        """Test validation of valid configuration"""
        config = AnalysisConfig(
            models=[ModelType.VADER, ModelType.TEXTBLOB],
            timeout_seconds=30,
            max_retries=3
        )
        
        validated_config, errors = ConfigValidator.validate_config(config)
        assert len(errors) == 0
        assert validated_config.timeout_seconds == 30
    
    def test_empty_models_validation(self):
        """Test validation of empty models list"""
        config = AnalysisConfig(models=[])
        
        validated_config, errors = ConfigValidator.validate_config(config)
        assert any("at least one model" in error for error in errors)
    
    def test_timeout_validation(self):
        """Test timeout validation and correction"""
        # Negative timeout
        config = AnalysisConfig(timeout_seconds=-5)
        validated_config, errors = ConfigValidator.validate_config(config)
        
        assert any("positive" in error for error in errors)
        assert validated_config.timeout_seconds == 30  # Should be corrected
        
        # Excessive timeout
        config = AnalysisConfig(timeout_seconds=500)
        validated_config, errors = ConfigValidator.validate_config(config)
        
        assert validated_config.timeout_seconds == 300  # Should be capped
    
    def test_retry_validation(self):
        """Test retry count validation"""
        # Negative retries
        config = AnalysisConfig(max_retries=-1)
        validated_config, errors = ConfigValidator.validate_config(config)
        
        assert validated_config.max_retries == 0  # Should be corrected
        
        # Excessive retries
        config = AnalysisConfig(max_retries=20)
        validated_config, errors = ConfigValidator.validate_config(config)
        
        assert validated_config.max_retries == 10  # Should be capped
    
    def test_batch_size_validation(self):
        """Test batch size validation"""
        # Zero batch size
        config = AnalysisConfig(batch_size=0)
        validated_config, errors = ConfigValidator.validate_config(config)
        
        assert validated_config.batch_size == 32  # Should be corrected
        
        # Excessive batch size
        config = AnalysisConfig(batch_size=2000)
        validated_config, errors = ConfigValidator.validate_config(config)
        
        assert validated_config.batch_size == 1000  # Should be capped
    
    def test_model_name_validation(self):
        """Test model name validation"""
        # Invalid model name
        config = AnalysisConfig(
            transformers_model="<script>alert('xss')</script>",
            openai_model="malicious/model/../../../etc/passwd"
        )
        
        validated_config, errors = ConfigValidator.validate_config(config)
        
        assert any("Invalid transformers model" in error for error in errors)
        assert any("Invalid OpenAI model" in error for error in errors)


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_basic_rate_limiting(self):
        """Test basic rate limiting"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        client_ip = "192.168.1.100"
        
        # Should allow requests within limit
        for i in range(5):
            assert limiter.is_allowed(client_ip) is True
        
        # Should block request exceeding limit
        assert limiter.is_allowed(client_ip) is False
    
    def test_different_clients(self):
        """Test rate limiting for different clients"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        client1 = "192.168.1.100"
        client2 = "192.168.1.101"
        
        # Each client should have independent limits
        for i in range(3):
            assert limiter.is_allowed(client1) is True
            assert limiter.is_allowed(client2) is True
        
        # Both should be blocked after limit
        assert limiter.is_allowed(client1) is False
        assert limiter.is_allowed(client2) is False
    
    def test_remaining_requests(self):
        """Test getting remaining request count"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        client_ip = "192.168.1.100"
        
        # Initially should have full allowance
        assert limiter.get_remaining(client_ip) == 5
        
        # Should decrease with each request
        limiter.is_allowed(client_ip)
        assert limiter.get_remaining(client_ip) == 4
        
        limiter.is_allowed(client_ip)
        assert limiter.get_remaining(client_ip) == 3
    
    def test_window_reset(self):
        """Test rate limit window reset"""
        limiter = RateLimiter(max_requests=2, window_seconds=1)  # 1 second window
        
        client_ip = "192.168.1.100"
        
        # Use up allowance
        assert limiter.is_allowed(client_ip) is True
        assert limiter.is_allowed(client_ip) is True
        assert limiter.is_allowed(client_ip) is False
        
        # Wait for window to reset
        import time
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.is_allowed(client_ip) is True


class TestAuditLogger:
    """Test audit logging functionality"""
    
    def test_logger_initialization(self):
        """Test audit logger initialization"""
        logger = AuditLogger()
        assert logger.logger is not None
    
    def test_validation_error_logging(self, caplog):
        """Test validation error logging"""
        logger = AuditLogger()
        
        client_ip = "192.168.1.100"
        text_hash = "abc123"
        errors = ["Invalid input", "Security violation"]
        
        logger.log_validation_error(client_ip, text_hash, errors)
        
        # Should log validation error
        assert "Validation error" in caplog.text
        assert client_ip in caplog.text
    
    def test_security_violation_logging(self, caplog):
        """Test security violation logging"""
        logger = AuditLogger()
        
        client_ip = "192.168.1.100"
        violation_type = "SQL Injection"
        details = "Detected SQL injection pattern"
        
        logger.log_security_violation(client_ip, violation_type, details)
        
        # Should log security violation
        assert "Security violation" in caplog.text
        assert violation_type in caplog.text
    
    def test_rate_limit_logging(self, caplog):
        """Test rate limit logging"""
        logger = AuditLogger()
        
        client_ip = "192.168.1.100"
        logger.log_rate_limit_exceeded(client_ip)
        
        # Should log rate limit exceeded
        assert "Rate limit exceeded" in caplog.text
        assert client_ip in caplog.text
    
    def test_text_hashing(self):
        """Test text hashing for privacy"""
        text = "This is sensitive text"
        hash1 = AuditLogger.hash_text(text)
        hash2 = AuditLogger.hash_text(text)
        
        # Same text should produce same hash
        assert hash1 == hash2
        
        # Hash should be different from original text
        assert hash1 != text
        
        # Hash should be consistent length
        assert len(hash1) == 16


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security features"""
    
    @pytest.mark.asyncio
    async def test_malicious_input_rejection(self, basic_analyzer, security_test_inputs):
        """Test that malicious inputs are properly handled"""
        validator = TextValidator(strict_mode=False)  # Non-strict for graceful handling
        
        for malicious_text in security_test_inputs[:3]:  # Test a few
            try:
                # Should either sanitize or reject
                text_input = TextInput(text=malicious_text)
                sanitized, errors = validator.validate_text_input(text_input)
                
                if len(errors) == 0:
                    # If no errors, text should be sanitized
                    assert sanitized.text != malicious_text or len(sanitized.text) < len(malicious_text)
                else:
                    # Errors should be security-related
                    assert any("pattern" in error.lower() or "injection" in error.lower() for error in errors)
                    
            except Exception as e:
                # Should handle gracefully
                assert "SecurityError" in str(type(e)) or "ValidationError" in str(type(e))
    
    @pytest.mark.asyncio
    async def test_input_sanitization_effectiveness(self, basic_analyzer):
        """Test that input sanitization is effective"""
        validator = TextValidator()
        
        # Test various malicious patterns
        test_cases = [
            ("<script>alert('xss')</script>Good text", "Good text"),
            ("Normal text<iframe>bad</iframe>", "Normal text"),
            ("Text with\x00null\x01characters", "Text with characters")
        ]
        
        for malicious_text, expected_clean in test_cases:
            text_input = TextInput(text=malicious_text)
            sanitized, errors = validator.validate_text_input(text_input)
            
            # Should be sanitized (may not match exactly due to HTML escaping)
            assert "<script>" not in sanitized.text
            assert "<iframe>" not in sanitized.text
            assert "\x00" not in sanitized.text