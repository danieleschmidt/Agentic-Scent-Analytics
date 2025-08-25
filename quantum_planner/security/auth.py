"""
Authentication and authorization management for quantum planner.
"""

import hashlib
import hmac
import secrets
import time
from typing import Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute" 
    ADMIN = "admin"


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    salt: str
    permissions: Set[Permission]
    created_at: float
    last_access: Optional[float] = None


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    permissions: Set[Permission]


class AuthenticationManager:
    """Manages authentication and authorization for quantum planner."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = 3600  # 1 hour
        
    def create_user(self, username: str, password: str, 
                   permissions: Set[Permission]) -> str:
        """Create a new user with hashed password."""
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)
        user_id = secrets.token_urlsafe(16)
        
        user = User(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            salt=salt,
            permissions=permissions,
            created_at=time.time()
        )
        
        self.users[user_id] = user
        return user_id
        
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session ID if successful."""
        user = self._find_user_by_username(username)
        if not user:
            return None
            
        expected_hash = self._hash_password(password, user.salt)
        if not hmac.compare_digest(user.password_hash, expected_hash):
            return None
            
        # Create session
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=time.time(),
            expires_at=time.time() + self.session_timeout,
            permissions=user.permissions
        )
        
        self.sessions[session_id] = session
        user.last_access = time.time()
        
        return session_id
        
    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate session and return session info if valid."""
        session = self.sessions.get(session_id)
        if not session:
            return None
            
        if time.time() > session.expires_at:
            del self.sessions[session_id]
            return None
            
        return session
        
    def check_permission(self, session_id: str, permission: Permission) -> bool:
        """Check if session has required permission."""
        session = self.validate_session(session_id)
        if not session:
            return False
            
        return permission in session.permissions or Permission.ADMIN in session.permissions
        
    def logout(self, session_id: str) -> bool:
        """Logout user by invalidating session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
        
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time > session.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using PBKDF2."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None