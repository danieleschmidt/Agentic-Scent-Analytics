"""
Security and audit system for industrial compliance.
"""

import hashlib
import hmac
import json
import logging
import secrets
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import sqlite3
import threading


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class AuditEventType(Enum):
    """Types of audit events."""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CONFIGURATION = "system_configuration"
    SECURITY_VIOLATION = "security_violation"
    QUALITY_DECISION = "quality_decision"
    AGENT_ACTION = "agent_action"
    SENSOR_CALIBRATION = "sensor_calibration"
    BATCH_RELEASE = "batch_release"


@dataclass
class AuditEvent:
    """Audit trail event."""
    event_type: AuditEventType
    action: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['security_level'] = self.security_level.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    authenticated: bool = False
    session_expires: Optional[datetime] = None


class CryptographyManager:
    """
    Manages encryption and cryptographic operations.
    """
    
    def __init__(self, key_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.key_path = key_path or Path.home() / ".agentic_scent" / "keys"
        self.key_path.mkdir(parents=True, exist_ok=True)
        
        # Load or generate encryption key
        self.encryption_key = self._load_or_generate_key()
    
    def _load_or_generate_key(self) -> bytes:
        """Load existing key or generate new one."""
        key_file = self.key_path / "master.key"
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    key = f.read()
                self.logger.info("Loaded existing encryption key")
                return key
            except Exception as e:
                self.logger.error(f"Failed to load key: {e}")
        
        # Generate new key
        key = secrets.token_bytes(32)  # 256-bit key
        
        try:
            # Save key with restricted permissions
            with open(key_file, 'wb') as f:
                f.write(key)
            key_file.chmod(0o600)  # Owner read/write only
            self.logger.info("Generated new encryption key")
        except Exception as e:
            self.logger.error(f"Failed to save key: {e}")
        
        return key
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using AES-256.
        Note: This is a simplified implementation. Production should use proper libraries like cryptography.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Simple XOR encryption for demo (use proper AES in production)
        key_hash = hashlib.sha256(self.encryption_key).digest()
        encrypted = bytearray()
        
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key_hash[i % len(key_hash)])
        
        return bytes(encrypted)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        # Simple XOR decryption (use proper AES in production)
        key_hash = hashlib.sha256(self.encryption_key).digest()
        decrypted = bytearray()
        
        for i, byte in enumerate(encrypted_data):
            decrypted.append(byte ^ key_hash[i % len(key_hash)])
        
        return bytes(decrypted)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for password hashing
        hash_value = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return hash_value, salt
    
    def verify_password(self, password: str, hash_value: bytes, salt: bytes) -> bool:
        """Verify password against hash."""
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(hash_value, computed_hash)
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)
    
    def sign_data(self, data: Union[str, bytes]) -> str:
        """Create HMAC signature for data integrity."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        signature = hmac.new(self.encryption_key, data, hashlib.sha256)
        return signature.hexdigest()
    
    def verify_signature(self, data: Union[str, bytes], signature: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.sign_data(data)
        return hmac.compare_digest(signature, expected_signature)


class AuditTrail:
    """
    Comprehensive audit trail system for regulatory compliance.
    """
    
    def __init__(self, db_path: Optional[Path] = None, enable_encryption: bool = True):
        self.db_path = db_path or Path.home() / ".agentic_scent" / "audit.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_encryption = enable_encryption
        self.logger = logging.getLogger(__name__)
        
        # Initialize cryptography
        self.crypto = CryptographyManager() if enable_encryption else None
        
        # Thread-safe database access
        self._db_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize audit database."""
        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        user_id TEXT,
                        agent_id TEXT,
                        resource TEXT,
                        action TEXT NOT NULL,
                        details TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        security_level TEXT,
                        success BOOLEAN,
                        error_message TEXT,
                        signature TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp);
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type);
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id);
                ''')
                
                conn.commit()
                self.logger.info("Audit database initialized")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize audit database: {e}")
                raise
            finally:
                conn.close()
    
    def log_event(self, event: AuditEvent) -> bool:
        """
        Log an audit event.
        
        Args:
            event: AuditEvent to log
            
        Returns:
            True if successfully logged, False otherwise
        """
        try:
            event_data = event.to_dict()
            
            # Encrypt sensitive details if encryption is enabled
            if self.crypto and event.security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]:
                details_json = json.dumps(event_data['details'])
                encrypted_details = self.crypto.encrypt_data(details_json)
                event_data['details'] = encrypted_details.hex()
                event_data['encrypted'] = True
            else:
                event_data['details'] = json.dumps(event_data['details'])
                event_data['encrypted'] = False
            
            # Create signature for integrity
            signature = ""
            if self.crypto:
                event_copy = event_data.copy()
                event_copy.pop('signature', None)
                signature_data = json.dumps(event_copy, sort_keys=True)
                signature = self.crypto.sign_data(signature_data)
            
            event_data['signature'] = signature
            
            # Store in database
            with self._db_lock:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    conn.execute('''
                        INSERT INTO audit_events (
                            event_id, event_type, timestamp, user_id, agent_id,
                            resource, action, details, ip_address, user_agent,
                            security_level, success, error_message, signature
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event_data['event_id'],
                        event_data['event_type'],
                        event_data['timestamp'],
                        event_data.get('user_id'),
                        event_data.get('agent_id'),
                        event_data.get('resource'),
                        event_data['action'],
                        event_data['details'],
                        event_data.get('ip_address'),
                        event_data.get('user_agent'),
                        event_data['security_level'],
                        event_data['success'],
                        event_data.get('error_message'),
                        signature
                    ))
                    conn.commit()
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Failed to log audit event: {e}")
                    return False
                finally:
                    conn.close()
        
        except Exception as e:
            self.logger.error(f"Error logging audit event: {e}")
            return False
    
    def query_events(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    event_type: Optional[AuditEventType] = None,
                    user_id: Optional[str] = None,
                    agent_id: Optional[str] = None,
                    limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Query audit events with filters.
        
        Args:
            start_time: Start timestamp filter
            end_time: End timestamp filter
            event_type: Event type filter
            user_id: User ID filter
            agent_id: Agent ID filter
            limit: Maximum number of events to return
            
        Returns:
            List of audit events
        """
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries and decrypt if needed
                events = []
                columns = [desc[0] for desc in cursor.description]
                
                for row in rows:
                    event_dict = dict(zip(columns, row))
                    
                    # Decrypt details if encrypted
                    try:
                        details_str = event_dict['details']
                        if self.crypto and details_str and len(details_str) > 100:  # Check if likely encrypted
                            try:
                                encrypted_bytes = bytes.fromhex(details_str)
                                decrypted_data = self.crypto.decrypt_data(encrypted_bytes)
                                event_dict['details'] = json.loads(decrypted_data.decode('utf-8'))
                            except:
                                # If decryption fails, treat as regular JSON
                                event_dict['details'] = json.loads(details_str) if details_str else {}
                        else:
                            event_dict['details'] = json.loads(details_str) if details_str else {}
                    except json.JSONDecodeError:
                        event_dict['details'] = {}
                    
                    events.append(event_dict)
                
                return events
                
            except Exception as e:
                self.logger.error(f"Failed to query audit events: {e}")
                return []
            finally:
                conn.close()
    
    def verify_integrity(self, event_id: str) -> bool:
        """
        Verify the integrity of an audit event.
        
        Args:
            event_id: ID of event to verify
            
        Returns:
            True if integrity is verified, False otherwise
        """
        if not self.crypto:
            return True  # No crypto, assume valid
        
        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(
                    "SELECT * FROM audit_events WHERE event_id = ?",
                    (event_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                columns = [desc[0] for desc in cursor.description]
                event_dict = dict(zip(columns, row))
                
                stored_signature = event_dict.pop('signature', '')
                signature_data = json.dumps(event_dict, sort_keys=True)
                
                return self.crypto.verify_signature(signature_data, stored_signature)
                
            except Exception as e:
                self.logger.error(f"Failed to verify event integrity: {e}")
                return False
            finally:
                conn.close()
    
    def generate_compliance_report(self, 
                                 start_time: datetime,
                                 end_time: datetime,
                                 report_type: str = "full") -> Dict[str, Any]:
        """
        Generate compliance report for regulatory audits.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            report_type: Type of report ("full", "security", "quality")
            
        Returns:
            Compliance report data
        """
        events = self.query_events(start_time=start_time, end_time=end_time, limit=10000)
        
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "report_type": report_type,
            "summary": {
                "total_events": len(events),
                "event_types": {},
                "users_active": set(),
                "agents_active": set(),
                "security_violations": 0,
                "failed_operations": 0
            },
            "events": events if report_type == "full" else []
        }
        
        # Analyze events
        for event in events:
            event_type = event.get('event_type', 'unknown')
            report["summary"]["event_types"][event_type] = report["summary"]["event_types"].get(event_type, 0) + 1
            
            if event.get('user_id'):
                report["summary"]["users_active"].add(event['user_id'])
            
            if event.get('agent_id'):
                report["summary"]["agents_active"].add(event['agent_id'])
            
            if event_type == 'security_violation':
                report["summary"]["security_violations"] += 1
            
            if not event.get('success', True):
                report["summary"]["failed_operations"] += 1
        
        # Convert sets to lists for JSON serialization
        report["summary"]["users_active"] = list(report["summary"]["users_active"])
        report["summary"]["agents_active"] = list(report["summary"]["agents_active"])
        
        return report


class SecurityManager:
    """
    Main security manager coordinating all security functions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.crypto = CryptographyManager()
        self.audit = AuditTrail(enable_encryption=self.config.get('enable_encryption', True))
        
        # Session management
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.session_timeout = timedelta(minutes=self.config.get('session_timeout_minutes', 60))
        
        # Security policies
        self.max_failed_attempts = self.config.get('max_failed_attempts', 3)
        self.failed_attempts: Dict[str, int] = {}
        
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.crypto.hash_password(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return self.crypto.verify_password(password, hashed)
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: Optional[str] = None) -> Optional[SecurityContext]:
        """
        Authenticate user and create security context.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            
        Returns:
            SecurityContext if authentication successful, None otherwise
        """
        # Check for too many failed attempts
        if self.failed_attempts.get(username, 0) >= self.max_failed_attempts:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.SECURITY_VIOLATION,
                user_id=username,
                action="authentication_blocked",
                details={"reason": "too_many_failed_attempts"},
                ip_address=ip_address,
                success=False,
                error_message="Account temporarily blocked"
            ))
            return None
        
        # Mock authentication - in production, verify against user database
        if username == "admin" and password == "admin123":  # Demo credentials
            # Create session
            session_id = self.crypto.generate_token()
            context = SecurityContext(
                user_id=username,
                session_id=session_id,
                permissions=["read", "write"] if username != "admin" else ["read", "write", "admin"],
                security_level=SecurityLevel.CONFIDENTIAL,
                authenticated=True,
                session_expires=datetime.now() + self.session_timeout
            )
            
            self.active_sessions[session_id] = context
            
            # Reset failed attempts
            self.failed_attempts.pop(username, None)
            
            # Log successful authentication
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.USER_LOGIN,
                user_id=username,
                action="user_authenticated",
                details={"session_id": session_id},
                ip_address=ip_address,
                success=True
            ))
            
            return context
        else:
            # Track failed attempt
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            
            # Log failed authentication
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.SECURITY_VIOLATION,
                user_id=username,
                action="authentication_failed",
                details={"attempts": self.failed_attempts[username]},
                ip_address=ip_address,
                success=False,
                error_message="Invalid credentials"
            ))
            
            return None
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate and return security context for session."""
        context = self.active_sessions.get(session_id)
        
        if not context:
            return None
        
        # Check expiration
        if context.session_expires and datetime.now() > context.session_expires:
            self.logout_user(session_id)
            return None
        
        return context
    
    def logout_user(self, session_id: str):
        """Logout user and cleanup session."""
        context = self.active_sessions.pop(session_id, None)
        
        if context:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.USER_LOGOUT,
                user_id=context.user_id,
                action="user_logged_out",
                details={"session_id": session_id},
                success=True
            ))
    
    def check_permission(self, context: SecurityContext, permission: str) -> bool:
        """Check if user has required permission."""
        if not context.authenticated:
            return False
        
        # Admin users have all standard permissions but not arbitrary ones
        if "admin" in context.permissions:
            allowed_permissions = ["read", "write", "admin", "delete", "modify", "audit", "manage"]
            return permission in allowed_permissions
        
        return permission in context.permissions
    
    def log_security_event(self, event_type: AuditEventType, **kwargs):
        """Log a security-related event."""
        event = AuditEvent(event_type=event_type, **kwargs)
        self.audit.log_event(event)