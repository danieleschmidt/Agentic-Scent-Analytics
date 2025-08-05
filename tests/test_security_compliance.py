"""
Security and compliance tests.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agentic_scent.core.security import (
    SecurityManager, AuditTrail, CryptographyManager, SecurityContext,
    AuditEvent, AuditEventType, SecurityLevel
)
from agentic_scent.core.validation import SensorReadingValidator, InputSanitizer, ValidationSeverity
from agentic_scent.core.config import ConfigManager, AgenticScentConfig


class TestCryptographyManager:
    """Test cryptography and security functions."""
    
    def test_key_generation(self, temp_dir):
        """Test encryption key generation and loading."""
        crypto = CryptographyManager(key_path=temp_dir / "test_keys")
        
        # Key should be generated
        assert crypto.encryption_key is not None
        assert len(crypto.encryption_key) == 32  # 256-bit key
        
        # Test key persistence
        crypto2 = CryptographyManager(key_path=temp_dir / "test_keys")
        assert crypto2.encryption_key == crypto.encryption_key  # Should load same key
    
    def test_data_encryption_decryption(self, temp_dir):
        """Test data encryption and decryption."""
        crypto = CryptographyManager(key_path=temp_dir / "test_keys")
        
        # Test string encryption
        original_text = "Sensitive manufacturing data: Batch-001, Quality: 98.5%"
        encrypted = crypto.encrypt_data(original_text)
        decrypted = crypto.decrypt_data(encrypted).decode('utf-8')
        
        assert encrypted != original_text.encode()
        assert decrypted == original_text
        
        # Test bytes encryption
        original_bytes = b"Binary sensor data: \x01\x02\x03\x04"
        encrypted_bytes = crypto.encrypt_data(original_bytes)
        decrypted_bytes = crypto.decrypt_data(encrypted_bytes)
        
        assert encrypted_bytes != original_bytes
        assert decrypted_bytes == original_bytes
    
    def test_password_hashing(self, temp_dir):
        """Test password hashing and verification."""
        crypto = CryptographyManager(key_path=temp_dir / "test_keys")
        
        password = "SecurePassword123!"
        hash_value, salt = crypto.hash_password(password)
        
        # Verify correct password
        assert crypto.verify_password(password, hash_value, salt) is True
        
        # Verify incorrect password
        assert crypto.verify_password("WrongPassword", hash_value, salt) is False
        
        # Test with different salt produces different hash
        hash_value2, salt2 = crypto.hash_password(password)
        assert hash_value != hash_value2
        assert salt != salt2
    
    def test_token_generation(self, temp_dir):
        """Test secure token generation."""
        crypto = CryptographyManager(key_path=temp_dir / "test_keys")
        
        token1 = crypto.generate_token()
        token2 = crypto.generate_token()
        
        assert token1 != token2
        assert len(token1) > 32  # URL-safe base64 encoded
        assert all(c.isalnum() or c in '-_' for c in token1)
    
    def test_data_signing_verification(self, temp_dir):
        """Test data signing and verification."""
        crypto = CryptographyManager(key_path=temp_dir / "test_keys")
        
        data = "Critical quality decision: REJECT batch due to contamination"
        signature = crypto.sign_data(data)
        
        # Verify correct signature
        assert crypto.verify_signature(data, signature) is True
        
        # Verify tampered data
        tampered_data = "Critical quality decision: APPROVE batch due to contamination"
        assert crypto.verify_signature(tampered_data, signature) is False


class TestAuditTrail:
    """Test audit trail functionality."""
    
    def test_audit_trail_initialization(self, temp_dir):
        """Test audit trail database initialization."""
        audit = AuditTrail(db_path=temp_dir / "test_audit.db")
        
        # Check database file creation
        assert (temp_dir / "test_audit.db").exists()
        
        # Check database schema
        conn = sqlite3.connect(str(temp_dir / "test_audit.db"))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert "audit_events" in tables
    
    def test_audit_event_logging(self, temp_dir):
        """Test logging of audit events."""
        audit = AuditTrail(db_path=temp_dir / "test_audit.db", enable_encryption=False)
        
        # Create test event
        event = AuditEvent(
            event_type=AuditEventType.QUALITY_DECISION,
            user_id="test_user",
            action="batch_approved",
            details={
                "batch_id": "BATCH-001",
                "decision": "APPROVED",
                "confidence": 0.95
            },
            success=True
        )
        
        # Log event
        success = audit.log_event(event)
        assert success is True
        
        # Verify event was stored
        events = audit.query_events(limit=1)
        assert len(events) == 1
        
        stored_event = events[0]
        assert stored_event["event_type"] == "quality_decision"
        assert stored_event["user_id"] == "test_user"
        assert stored_event["action"] == "batch_approved"
        assert stored_event["success"] == 1  # SQLite boolean as integer
    
    def test_audit_event_querying(self, temp_dir):
        """Test querying audit events with filters."""
        audit = AuditTrail(db_path=temp_dir / "test_audit.db", enable_encryption=False)
        
        # Create multiple test events
        events = [
            AuditEvent(
                event_type=AuditEventType.USER_LOGIN,
                user_id="user1",
                action="login_successful",
                success=True
            ),
            AuditEvent(
                event_type=AuditEventType.QUALITY_DECISION,
                user_id="user2", 
                action="batch_rejected",
                success=True
            ),
            AuditEvent(
                event_type=AuditEventType.SECURITY_VIOLATION,
                user_id="user1",
                action="invalid_access_attempt",
                success=False
            )
        ]
        
        # Log all events
        for event in events:
            audit.log_event(event)
        
        # Test queries
        all_events = audit.query_events(limit=10)
        assert len(all_events) >= 3
        
        user1_events = audit.query_events(user_id="user1", limit=10)
        assert len(user1_events) == 2
        
        security_events = audit.query_events(event_type=AuditEventType.SECURITY_VIOLATION, limit=10)
        assert len(security_events) == 1
        assert security_events[0]["success"] == 0  # Failed event
    
    def test_audit_integrity_verification(self, temp_dir):
        """Test audit event integrity verification."""
        audit = AuditTrail(db_path=temp_dir / "test_audit.db", enable_encryption=True)
        
        # Create and log event
        event = AuditEvent(
            event_type=AuditEventType.AGENT_ACTION,
            agent_id="test_agent",
            action="anomaly_detected",
            details={"confidence": 0.87}
        )
        
        audit.log_event(event)
        
        # Verify integrity
        integrity_ok = audit.verify_integrity(event.event_id)
        assert integrity_ok is True
    
    def test_compliance_report_generation(self, temp_dir):
        """Test compliance report generation."""
        audit = AuditTrail(db_path=temp_dir / "test_audit.db", enable_encryption=False)
        
        # Create events for report
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        events = [
            AuditEvent(
                event_type=AuditEventType.QUALITY_DECISION,
                user_id="qc_manager",
                action="batch_approved"
            ),
            AuditEvent(
                event_type=AuditEventType.SECURITY_VIOLATION,
                user_id="unknown",
                action="unauthorized_access",
                success=False
            )
        ]
        
        for event in events:
            audit.log_event(event)
        
        # Generate compliance report
        report = audit.generate_compliance_report(start_time, end_time, "full")
        
        assert "report_id" in report
        assert "generated_at" in report
        assert "summary" in report
        assert "events" in report
        
        summary = report["summary"]
        assert summary["total_events"] >= 2
        assert "quality_decision" in summary["event_types"]
        assert "security_violation" in summary["event_types"]
        assert summary["security_violations"] >= 1
        assert summary["failed_operations"] >= 1


class TestSecurityManager:
    """Test security manager functionality."""
    
    def test_user_authentication(self, temp_dir):
        """Test user authentication."""
        config = {"enable_encryption": True, "max_failed_attempts": 3}
        security = SecurityManager(config)
        
        # Test successful authentication (using demo credentials)
        context = security.authenticate_user("admin", "admin123", "127.0.0.1")
        
        assert context is not None
        assert context.user_id == "admin"
        assert context.authenticated is True
        assert "admin" in context.permissions
        assert context.session_id is not None
        
        # Test failed authentication
        failed_context = security.authenticate_user("admin", "wrongpassword", "127.0.0.1")
        assert failed_context is None
    
    def test_session_management(self, temp_dir):
        """Test session validation and management."""
        security = SecurityManager()
        
        # Authenticate user
        context = security.authenticate_user("admin", "admin123")
        assert context is not None
        
        session_id = context.session_id
        
        # Validate session
        validated_context = security.validate_session(session_id)
        assert validated_context is not None
        assert validated_context.user_id == context.user_id
        
        # Logout user
        security.logout_user(session_id)
        
        # Session should be invalid after logout
        invalid_context = security.validate_session(session_id)
        assert invalid_context is None
    
    def test_permission_checking(self):
        """Test permission checking."""
        security = SecurityManager()
        
        # Create security context
        context = SecurityContext(
            user_id="test_user",
            authenticated=True,
            permissions=["read", "write"]
        )
        
        # Test permission checks
        assert security.check_permission(context, "read") is True
        assert security.check_permission(context, "write") is True
        assert security.check_permission(context, "admin") is False
        
        # Test admin permission
        admin_context = SecurityContext(
            user_id="admin_user",
            authenticated=True,
            permissions=["admin"]
        )
        
        assert security.check_permission(admin_context, "read") is True  # Admin has all permissions
        assert security.check_permission(admin_context, "admin") is True
    
    def test_failed_attempt_tracking(self):
        """Test failed authentication attempt tracking."""
        config = {"max_failed_attempts": 2}
        security = SecurityManager(config)
        
        # First failed attempt
        result1 = security.authenticate_user("testuser", "wrong1")
        assert result1 is None
        assert security.failed_attempts["testuser"] == 1
        
        # Second failed attempt
        result2 = security.authenticate_user("testuser", "wrong2")
        assert result2 is None
        assert security.failed_attempts["testuser"] == 2
        
        # Third attempt should be blocked
        result3 = security.authenticate_user("testuser", "wrong3")
        assert result3 is None
        # Should still be 2 attempts as the third was blocked


class TestDataValidation:
    """Test data validation and sanitization."""
    
    def test_sensor_reading_validation(self, sample_sensor_reading):
        """Test sensor reading validation."""
        validator = SensorReadingValidator()
        
        # Test valid reading
        result = validator.validate_reading(sample_sensor_reading)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.validation_score > 0.8
        assert result.severity == ValidationSeverity.INFO
    
    def test_invalid_sensor_reading_validation(self):
        """Test validation of invalid sensor readings."""
        from agentic_scent.sensors.base import SensorReading, SensorType
        
        validator = SensorReadingValidator()
        
        # Create invalid reading
        invalid_reading = SensorReading(
            sensor_id="",  # Empty sensor_id
            sensor_type=SensorType.E_NOSE,
            values=[float('inf'), -100, float('nan'), 50000],  # Invalid values
            timestamp=None,  # Missing timestamp
            quality_score=1.5  # Invalid quality score
        )
        
        result = validator.validate_reading(invalid_reading)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert result.validation_score < 0.8
        assert result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
    
    def test_temporal_validation(self):
        """Test temporal consistency validation."""
        from agentic_scent.sensors.base import SensorReading, SensorType
        
        validator = SensorReadingValidator()
        
        # Create sequence of readings
        readings = []
        for i in range(3):
            reading = SensorReading(
                sensor_id="temporal_test_sensor",
                sensor_type=SensorType.E_NOSE,
                values=[100 + i] * 8,
                timestamp=datetime.now() - timedelta(seconds=10-i*3)  # 3 second intervals
            )
            readings.append(reading)
        
        # Validate each reading (builds history)
        results = []
        for reading in readings:
            result = validator.validate_reading(reading)
            results.append(result)
        
        # First reading should have no temporal issues
        assert results[0].is_valid
        
        # Subsequent readings should pass temporal validation
        assert all(r.is_valid for r in results[1:])
    
    def test_input_sanitization(self):
        """Test input sanitization functions."""
        sanitizer = InputSanitizer()
        
        # Test string sanitization
        dirty_string = "  sensor_id_001\x00  "
        clean_string, warnings = sanitizer.sanitize_string(dirty_string, "sensor_id")
        
        assert clean_string == "sensor_id_001"
        assert "Removed null bytes" in warnings
        assert "Trimmed whitespace" in warnings
        
        # Test numeric sanitization
        dirty_number = "123.45abc"
        clean_number, warnings = sanitizer.sanitize_numeric(dirty_number, min_val=0, max_val=1000)
        
        assert clean_number == 123.45
        assert "Removed non-numeric characters" in warnings
        
        # Test list sanitization
        dirty_list = [1, "2", 3.5, "invalid", 5]
        clean_list, warnings = sanitizer.sanitize_list(dirty_list, item_type=float)
        
        assert len(clean_list) >= 4  # Should convert most items
        assert all(isinstance(item, float) for item in clean_list)
    
    def test_validation_statistics(self, sample_sensor_reading):
        """Test validation statistics tracking."""
        validator = SensorReadingValidator()
        
        # Process several readings
        for i in range(10):
            reading = SensorReading(
                sensor_id=f"stats_sensor_{i%3}",  # 3 different sensors
                sensor_type=SensorType.E_NOSE,
                values=[100 + i] * 8,
                timestamp=datetime.now() - timedelta(seconds=i)
            )
            validator.validate_reading(reading)
        
        stats = validator.get_validation_statistics()
        
        assert stats["total_readings_processed"] >= 10
        assert stats["sensors_monitored"] == 3
        assert "validation_rules" in stats
        assert "value_ranges" in stats["validation_rules"]


class TestConfigurationSecurity:
    """Test configuration security features."""
    
    def test_config_hash_verification(self, temp_dir):
        """Test configuration hash verification."""
        config_path = temp_dir / "test_config.json"
        config_manager = ConfigManager(config_path)
        
        # Create and save config
        config = config_manager.load_config()
        config.site_id = "secure_test_site"
        config_manager.save_config(config)
        
        # Load config and verify hash
        loaded_config = config_manager.load_config()
        assert loaded_config.site_id == "secure_test_site"
        assert loaded_config.config_hash is not None
    
    def test_environment_variable_overrides(self, temp_dir):
        """Test environment variable configuration overrides."""
        import os
        
        config_manager = ConfigManager(temp_dir / "env_test_config.json")
        
        # Set environment variables
        test_env_vars = {
            "AGENTIC_SCENT_SITE_ID": "env_test_site",
            "AGENTIC_SCENT_LOG_LEVEL": "DEBUG",
            "AGENTIC_SCENT_MAX_CONCURRENT": "50"
        }
        
        # Mock environment variables
        with patch.dict(os.environ, test_env_vars):
            overrides = config_manager.get_environment_overrides()
            
            assert "site_id" in overrides
            assert overrides["site_id"] == "env_test_site"
            assert overrides["monitoring.log_level"] == "DEBUG"
            assert overrides["performance.max_concurrent_analyses"] == 50
    
    def test_config_validation(self, temp_dir):
        """Test configuration validation."""
        config_manager = ConfigManager(temp_dir / "validation_test.json")
        
        # Test invalid configuration
        config = config_manager.load_config()
        config.environment = "invalid_environment"
        config.monitoring.prometheus_port = 999999  # Invalid port
        
        with pytest.raises(ValueError, match="Configuration validation failed"):
            config_manager.save_config(config)


class TestComplianceFeatures:
    """Test regulatory compliance features."""
    
    def test_gmp_audit_trail(self, temp_dir):
        """Test GMP-compliant audit trail features."""
        audit = AuditTrail(db_path=temp_dir / "gmp_audit.db")
        
        # Log GMP-relevant events
        events = [
            AuditEvent(
                event_type=AuditEventType.BATCH_RELEASE,
                user_id="qc_manager",
                action="batch_approved",
                details={
                    "batch_id": "BATCH-2024-001",
                    "product": "Aspirin 500mg",
                    "test_results": {"potency": 98.5, "dissolution": 95.2},
                    "release_decision": "APPROVED"
                },
                security_level=SecurityLevel.CONFIDENTIAL
            ),
            AuditEvent(
                event_type=AuditEventType.SENSOR_CALIBRATION,
                user_id="technician",
                agent_id="calibration_agent",
                action="sensor_calibrated",
                details={
                    "sensor_id": "e_nose_001",
                    "calibration_standard": "ISO-12345",
                    "reference_values": [100.0, 150.0, 200.0]
                }
            )
        ]
        
        # Log events
        for event in events:
            success = audit.log_event(event)
            assert success is True
        
        # Generate compliance report
        report = audit.generate_compliance_report(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            report_type="quality"
        )
        
        assert report["summary"]["total_events"] >= 2
        assert "batch_release" in report["summary"]["event_types"]
        assert "sensor_calibration" in report["summary"]["event_types"]
    
    def test_data_integrity_verification(self, temp_dir):
        """Test data integrity verification for compliance."""
        audit = AuditTrail(db_path=temp_dir / "integrity_test.db")
        
        # Create critical quality decision event
        event = AuditEvent(
            event_type=AuditEventType.QUALITY_DECISION,
            user_id="qa_supervisor",
            action="lot_disposition",
            details={
                "lot_number": "LOT-2024-0123",
                "disposition": "REJECT",
                "reason": "Contamination detected by AI agent",
                "agent_confidence": 0.94,
                "review_required": True
            },
            security_level=SecurityLevel.RESTRICTED
        )
        
        # Log event
        audit.log_event(event)
        
        # Verify integrity immediately
        integrity_ok = audit.verify_integrity(event.event_id)
        assert integrity_ok is True
        
        # Query and verify integrity of retrieved event
        events = audit.query_events(event_type=AuditEventType.QUALITY_DECISION)
        assert len(events) >= 1
        
        retrieved_event = events[0]
        assert retrieved_event["action"] == "lot_disposition"
        assert retrieved_event["details"]["disposition"] == "REJECT"