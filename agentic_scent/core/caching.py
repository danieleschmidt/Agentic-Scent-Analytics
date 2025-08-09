"""
Multi-level caching system for performance optimization.
"""

import asyncio
import json
import time
import hashlib
from typing import Any, Dict, Optional, Union, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: datetime
    ttl: int  # Time to live in seconds
    hit_count: int = 0
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl <= 0:
            return False  # No expiration
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)
    
    def touch(self):
        """Update access timestamp and hit count."""
        self.hit_count += 1


class MemoryCache:
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            self._stats["total_requests"] += 1
            
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        async with self._lock:
            ttl = ttl or self.default_ttl
            
            # Calculate size (rough estimate)
            try:
                size_bytes = len(json.dumps(value, default=str))
            except:
                size_bytes = 1024  # Default estimate
            
            entry = CacheEntry(
                value=value,
                timestamp=datetime.now(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
            
            self._cache[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["total_requests"]
        hit_rate = (self._stats["hits"] / total) if total > 0 else 0.0
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "current_size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_mb": sum(e.size_bytes for e in self._cache.values()) / 1024 / 1024
        }


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 300):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._redis = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import aioredis
                self._redis = aioredis.from_url(self.redis_url)
                await self._redis.ping()
                logger.info("Connected to Redis cache")
            except ImportError:
                logger.warning("aioredis not installed, using mock Redis")
                self._redis = MockRedis()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using mock")
                self._redis = MockRedis()
        
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        redis = await self._get_redis()
        self._stats["total_requests"] += 1
        
        try:
            data = await redis.get(f"cache:{key}")
            if data is None:
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return json.loads(data)
        
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        redis = await self._get_redis()
        ttl = ttl or self.default_ttl
        
        try:
            data = json.dumps(value, default=str)
            await redis.setex(f"cache:{key}", ttl, data)
            return True
        
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        redis = await self._get_redis()
        
        try:
            result = await redis.delete(f"cache:{key}")
            return result > 0
        
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self):
        """Clear all cache entries."""
        redis = await self._get_redis()
        
        try:
            keys = await redis.keys("cache:*")
            if keys:
                await redis.delete(*keys)
        
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["total_requests"]
        hit_rate = (self._stats["hits"] / total) if total > 0 else 0.0
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "redis_url": self.redis_url
        }


class MockRedis:
    """Mock Redis for testing."""
    
    def __init__(self):
        self._data = {}
        self._expiry = {}
    
    async def ping(self):
        return True
    
    async def get(self, key: str):
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._data[key]
            del self._expiry[key]
            return None
        return self._data.get(key)
    
    async def setex(self, key: str, ttl: int, value: str):
        self._data[key] = value
        self._expiry[key] = time.time() + ttl
    
    async def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                if key in self._expiry:
                    del self._expiry[key]
                count += 1
        return count
    
    async def keys(self, pattern: str):
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self._data.keys() if k.startswith(prefix)]
        return [k for k in self._data.keys() if k == pattern]


class MultiLevelCache:
    """
    Multi-level cache system: Memory -> Redis -> Database
    """
    
    def __init__(self, memory_cache: MemoryCache, redis_cache: RedisCache):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.logger = logging.getLogger("multi_level_cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with multi-level fallback."""
        
        # Level 1: Memory cache
        value = await self.memory_cache.get(key)
        if value is not None:
            self.logger.debug(f"Cache hit (memory): {key}")
            return value
        
        # Level 2: Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            self.logger.debug(f"Cache hit (redis): {key}")
            # Populate memory cache
            await self.memory_cache.set(key, value)
            return value
        
        self.logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both caches."""
        
        # Set in both levels
        memory_result = await self.memory_cache.set(key, value, ttl)
        redis_result = await self.redis_cache.set(key, value, ttl)
        
        return memory_result and redis_result
    
    async def delete(self, key: str) -> bool:
        """Delete from both caches."""
        memory_result = await self.memory_cache.delete(key)
        redis_result = await self.redis_cache.delete(key)
        
        return memory_result or redis_result
    
    async def clear(self):
        """Clear both caches."""
        await self.memory_cache.clear()
        await self.redis_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            "memory_cache": self.memory_cache.get_stats(),
            "redis_cache": self.redis_cache.get_stats()
        }


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    
    # Convert args and kwargs to string
    key_parts = []
    
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use class name and key attributes
            key_parts.append(f"{arg.__class__.__name__}:{getattr(arg, 'id', id(arg))}")
        else:
            key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    
    key_string = ":".join(key_parts)
    
    # Hash long keys
    if len(key_string) > 200:
        return hashlib.md5(key_string.encode()).hexdigest()
    
    return key_string


def cached(cache: Union[MemoryCache, RedisCache, MultiLevelCache], 
          ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache: Cache instance to use
        ttl: Time to live in seconds
        key_func: Custom key generation function
    """
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {key}")
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}: {key}")
            
            return result
        
        return wrapper
    return decorator


async def create_optimized_cache(memory_size: int = 1000, 
                               memory_ttl: int = 300,
                               redis_url: Optional[str] = None,
                               redis_ttl: int = 900) -> MultiLevelCache:
    """Create optimized multi-level cache system."""
    
    # Memory cache
    memory_cache = MemoryCache(max_size=memory_size, default_ttl=memory_ttl)
    
    # Redis cache
    if redis_url:
        redis_cache = RedisCache(redis_url=redis_url, default_ttl=redis_ttl)
    else:
        redis_cache = RedisCache(default_ttl=redis_ttl)  # Will use mock
    
    return MultiLevelCache(memory_cache, redis_cache)