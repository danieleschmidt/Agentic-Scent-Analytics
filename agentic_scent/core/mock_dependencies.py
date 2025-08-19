#!/usr/bin/env python3
"""
Mock Dependencies for Autonomous Enhancement Testing
Provides lightweight mocks for heavy dependencies like numpy, pandas, etc.
"""

import sys
import math
import random
from typing import Any, List, Union, Optional


class MockNumPy:
    """Mock numpy for basic testing without full numpy dependency."""
    
    @staticmethod
    def array(data):
        """Mock numpy array - returns list with additional methods."""
        if isinstance(data, list):
            result = data.copy()
        else:
            result = [data]
            
        # Add basic numpy-like methods
        def mean():
            return sum(result) / len(result) if result else 0
            
        def std():
            if len(result) <= 1:
                return 0
            mean_val = mean()
            variance = sum((x - mean_val) ** 2 for x in result) / len(result)
            return math.sqrt(variance)
            
        result.mean = mean
        result.std = std
        result.shape = (len(result),)
        
        return result
    
    @staticmethod
    def zeros(shape):
        """Create array of zeros."""
        if isinstance(shape, tuple):
            if len(shape) == 2:
                return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
            else:
                return [0.0 for _ in range(shape[0])]
        else:
            return [0.0 for _ in range(shape)]
    
    @staticmethod
    def eye(n):
        """Create identity matrix."""
        result = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            result[i][i] = 1.0
        return result
    
    @staticmethod
    def random():
        """Mock random module."""
        class MockRandom:
            @staticmethod
            def uniform(low, high, size=None):
                if size is None:
                    return random.uniform(low, high)
                elif isinstance(size, int):
                    return [random.uniform(low, high) for _ in range(size)]
                else:  # tuple
                    result = []
                    for i in range(size[0]):
                        if len(size) > 1:
                            result.append([random.uniform(low, high) for _ in range(size[1])])
                        else:
                            result.append(random.uniform(low, high))
                    return result
            
            @staticmethod
            def normal(mean=0, std=1, size=None):
                if size is None:
                    return random.gauss(mean, std)
                elif isinstance(size, int):
                    return [random.gauss(mean, std) for _ in range(size)]
                else:  # tuple
                    result = []
                    for i in range(size[0]):
                        if len(size) > 1:
                            result.append([random.gauss(mean, std) for _ in range(size[1])])
                        else:
                            result.append(random.gauss(mean, std))
                    return result
            
            @staticmethod
            def random():
                return random.random()
            
            @staticmethod
            def choice(arr, size=None, replace=True):
                if size is None:
                    return random.choice(arr)
                return [random.choice(arr) for _ in range(size)]
        
        return MockRandom()
    
    @staticmethod
    def mean(data):
        """Calculate mean of data."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):  # 2D array
                return [sum(row) / len(row) for row in data]
            else:  # 1D array
                return sum(data) / len(data)
        return 0
    
    @staticmethod
    def std(data):
        """Calculate standard deviation."""
        if isinstance(data, list) and len(data) > 1:
            mean_val = MockNumPy.mean(data)
            if isinstance(mean_val, list):  # 2D
                return 0  # Simplified for 2D
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return math.sqrt(variance)
        return 0
    
    @staticmethod
    def var(data):
        """Calculate variance."""
        if isinstance(data, list) and len(data) > 1:
            mean_val = MockNumPy.mean(data)
            if isinstance(mean_val, list):  # 2D
                return 0  # Simplified
            return sum((x - mean_val) ** 2 for x in data) / len(data)
        return 0
    
    @staticmethod
    def dot(a, b):
        """Matrix/vector dot product."""
        if isinstance(a, list) and isinstance(b, list):
            if isinstance(a[0], list) and isinstance(b[0], list):
                # Matrix multiplication
                result = []
                for i in range(len(a)):
                    row = []
                    for j in range(len(b[0])):
                        sum_val = sum(a[i][k] * b[k][j] for k in range(len(b)))
                        row.append(sum_val)
                    result.append(row)
                return result
            elif not isinstance(a[0], list) and not isinstance(b[0], list):
                # Vector dot product
                return sum(a[i] * b[i] for i in range(min(len(a), len(b))))
        return 0
    
    @staticmethod
    def linalg():
        """Linear algebra functions."""
        class MockLinalg:
            @staticmethod
            def norm(vector):
                return math.sqrt(sum(x**2 for x in vector))
        return MockLinalg()
    
    @staticmethod
    def argmax(data):
        """Return index of maximum value."""
        if isinstance(data, list) and data:
            return data.index(max(data))
        return 0
    
    @staticmethod
    def argsort(data):
        """Return indices that would sort the array."""
        if isinstance(data, list):
            return sorted(range(len(data)), key=lambda i: data[i])
        return []
    
    @staticmethod
    def clip(data, min_val, max_val):
        """Clip values to range."""
        if isinstance(data, list):
            return [max(min_val, min(max_val, x)) for x in data]
        else:
            return max(min_val, min(max_val, data))
    
    @staticmethod
    def where(condition, x, y):
        """Element-wise selection based on condition."""
        if isinstance(condition, list):
            return [x[i] if condition[i] else y[i] for i in range(len(condition))]
        else:
            return x if condition else y
    
    @staticmethod
    def sin(x):
        """Sine function."""
        if isinstance(x, list):
            return [math.sin(val) for val in x]
        return math.sin(x)
    
    @staticmethod
    def cos(x):
        """Cosine function."""
        if isinstance(x, list):
            return [math.cos(val) for val in x]
        return math.cos(x)
    
    @staticmethod
    def exp(x):
        """Exponential function."""
        if isinstance(x, list):
            return [math.exp(val) for val in x]
        return math.exp(x)
    
    @staticmethod
    def sqrt(x):
        """Square root function."""
        if isinstance(x, list):
            return [math.sqrt(val) for val in x]
        return math.sqrt(x)
    
    pi = math.pi


class MockPSUtil:
    """Mock psutil for system monitoring."""
    
    @staticmethod
    def cpu_count():
        return 8
    
    @staticmethod
    def cpu_percent(interval=None):
        return random.uniform(10, 80)
    
    @staticmethod
    def virtual_memory():
        class MockMemory:
            total = 16 * 1024**3  # 16 GB
            used = int(total * random.uniform(0.3, 0.7))
            percent = (used / total) * 100
        return MockMemory()
    
    @staticmethod
    def disk_io_counters():
        class MockDiskIO:
            read_bytes = random.randint(1000000, 10000000)
            write_bytes = random.randint(1000000, 10000000)
        return MockDiskIO()
    
    @staticmethod
    def net_io_counters():
        class MockNetIO:
            bytes_sent = random.randint(1000000, 10000000)
            bytes_recv = random.randint(1000000, 10000000)
        return MockNetIO()


class MockCryptography:
    """Mock cryptography for basic security testing."""
    
    class hazmat:
        class primitives:
            class hashes:
                class SHA3_512:
                    pass
            
            class serialization:
                class Encoding:
                    PEM = "PEM"
                
                class PrivateFormat:
                    PKCS8 = "PKCS8"
                
                class PublicFormat:
                    SubjectPublicKeyInfo = "SubjectPublicKeyInfo"
                
                class NoEncryption:
                    pass
                
                @staticmethod
                def load_pem_public_key(key_data, backend=None):
                    class MockPublicKey:
                        def encrypt(self, data, padding):
                            return b"encrypted_" + data
                        def verify(self, signature, data, padding, algorithm):
                            return True
                    return MockPublicKey()
                
                @staticmethod
                def load_pem_private_key(key_data, password=None, backend=None):
                    class MockPrivateKey:
                        def decrypt(self, data, padding):
                            if data.startswith(b"encrypted_"):
                                return data[10:]  # Remove "encrypted_" prefix
                            return data
                        
                        def sign(self, data, padding, algorithm):
                            return b"signature_" + data[:10]
                        
                        def public_key(self):
                            class MockPublicKey:
                                def public_bytes(self, encoding, format):
                                    return b"public_key_pem_data"
                            return MockPublicKey()
                        
                        def private_bytes(self, encoding, format, encryption):
                            return b"private_key_pem_data"
                    return MockPrivateKey()
            
            class asymmetric:
                class rsa:
                    @staticmethod
                    def generate_private_key(public_exponent, key_size, backend):
                        class MockPrivateKey:
                            def public_key(self):
                                class MockPublicKey:
                                    def public_bytes(self, encoding, format):
                                        return b"public_key_pem_data"
                                return MockPublicKey()
                            
                            def private_bytes(self, encoding, format, encryption):
                                return b"private_key_pem_data"
                        return MockPrivateKey()
                
                class padding:
                    class OAEP:
                        def __init__(self, mgf, algorithm, label):
                            pass
                    
                    class MGF1:
                        def __init__(self, algorithm):
                            pass
                    
                    class PSS:
                        MAX_LENGTH = "MAX_LENGTH"
                        def __init__(self, mgf, salt_length):
                            pass
            
            class ciphers:
                class Cipher:
                    def __init__(self, algorithm, mode, backend):
                        pass
                
                class algorithms:
                    class AES:
                        def __init__(self, key):
                            pass
                
                class modes:
                    class CBC:
                        def __init__(self, iv):
                            pass
            
        class backends:
            @staticmethod
            def default_backend():
                return None


class MockBCrypt:
    """Mock bcrypt for password hashing."""
    
    @staticmethod
    def hashpw(password, salt):
        return b"hashed_" + password
    
    @staticmethod
    def gensalt(rounds=12):
        return b"salt_12_rounds"
    
    @staticmethod
    def checkpw(password, hashed):
        return hashed == b"hashed_" + password


def install_mocks():
    """Install mock modules to avoid import errors."""
    
    # Mock numpy
    sys.modules['numpy'] = MockNumPy()
    sys.modules['numpy.random'] = MockNumPy.random()
    sys.modules['numpy.linalg'] = MockNumPy.linalg()
    
    # Mock pandas (basic)
    class MockPandas:
        @staticmethod
        def DataFrame(data=None):
            return {"data": data}
    sys.modules['pandas'] = MockPandas()
    
    # Mock scikit-learn (basic)
    class MockSklearn:
        class ensemble:
            class RandomForestRegressor:
                def __init__(self, **kwargs):
                    pass
                def fit(self, X, y):
                    pass
                def predict(self, X):
                    return [0.5] * len(X)
    sys.modules['sklearn'] = MockSklearn()
    sys.modules['sklearn.ensemble'] = MockSklearn.ensemble
    
    # Mock psutil
    sys.modules['psutil'] = MockPSUtil()
    
    # Mock cryptography
    mock_crypto = MockCryptography()
    sys.modules['cryptography'] = mock_crypto
    sys.modules['cryptography.hazmat'] = mock_crypto.hazmat
    sys.modules['cryptography.hazmat.primitives'] = mock_crypto.hazmat.primitives
    sys.modules['cryptography.hazmat.primitives.hashes'] = mock_crypto.hazmat.primitives.hashes
    sys.modules['cryptography.hazmat.primitives.serialization'] = mock_crypto.hazmat.primitives.serialization
    sys.modules['cryptography.hazmat.primitives.asymmetric'] = mock_crypto.hazmat.primitives.asymmetric
    sys.modules['cryptography.hazmat.primitives.asymmetric.rsa'] = mock_crypto.hazmat.primitives.asymmetric.rsa
    sys.modules['cryptography.hazmat.primitives.asymmetric.padding'] = mock_crypto.hazmat.primitives.asymmetric.padding
    sys.modules['cryptography.hazmat.primitives.ciphers'] = mock_crypto.hazmat.primitives.ciphers
    sys.modules['cryptography.hazmat.primitives.kdf'] = type('MockKDF', (), {})()
    sys.modules['cryptography.hazmat.primitives.kdf.pbkdf2'] = type('MockPBKDF2', (), {'PBKDF2HMAC': lambda **kw: None})()
    sys.modules['cryptography.hazmat.backends'] = mock_crypto.hazmat.backends
    
    # Mock bcrypt
    sys.modules['bcrypt'] = MockBCrypt()
    
    # Mock scipy (basic)
    class MockScipy:
        @staticmethod
        def stats():
            return type('MockStats', (), {})()
    sys.modules['scipy'] = MockScipy()
    
    print("✅ Mock dependencies installed successfully")


if __name__ == '__main__':
    install_mocks()
    print("Mock dependencies ready for testing")