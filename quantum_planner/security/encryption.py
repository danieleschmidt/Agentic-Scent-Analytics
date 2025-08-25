"""
Encryption and cryptographic utilities for quantum planner.
"""

import base64
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import bytes, str, Dict, Any


class EncryptionManager:
    """Handles encryption/decryption and key management."""
    
    def __init__(self, master_key: str = None):
        if master_key:
            self.key = self._derive_key(master_key)
        else:
            self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        
    def encrypt(self, data: str) -> str:
        """Encrypt string data and return base64 encoded result."""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
        
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded data and return original string."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_data.decode()
        
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data."""
        import json
        json_data = json.dumps(data)
        return self.encrypt(json_data)
        
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt data back to dictionary."""
        import json
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)
        
    def generate_salt(self) -> str:
        """Generate a random salt."""
        return base64.b64encode(secrets.token_bytes(32)).decode()
        
    def hash_data(self, data: str, salt: str = None) -> str:
        """Hash data with optional salt."""
        if salt is None:
            salt = self.generate_salt()
            
        digest = hashes.Hash(hashes.SHA256())
        digest.update(data.encode())
        digest.update(salt.encode())
        hash_bytes = digest.finalize()
        return base64.b64encode(hash_bytes).decode()
        
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        salt = b'salt_1234567890'  # In production, use random salt per key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
    def get_key_string(self) -> str:
        """Get the encryption key as string for storage."""
        return base64.b64encode(self.key).decode()
        
    @classmethod
    def from_key_string(cls, key_string: str) -> 'EncryptionManager':
        """Create EncryptionManager from stored key string."""
        key = base64.b64decode(key_string.encode())
        manager = cls.__new__(cls)
        manager.key = key
        manager.cipher_suite = Fernet(key)
        return manager