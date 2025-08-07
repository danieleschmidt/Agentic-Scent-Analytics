"""
Multi-level caching system for sentiment analysis
"""

import asyncio
import hashlib
import json
import time
import logging
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.models import SentimentResult

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache implementation"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = CacheStats()
        self._lock = threading.RLock()
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            with self._lock:
                if key in self._cache:
                    entry = self._cache[key]
                    
                    # Check expiration
                    if entry['expires_at'] and time.time() > entry['expires_at']:
                        del self._cache[key]
                        self._stats.evictions += 1
                        self._stats.misses += 1
                        return None
                    
                    # Update access time
                    entry['accessed_at'] = time.time()
                    self._stats.hits += 1
                    return entry['value']
                else:
                    self._stats.misses += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._stats.errors += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        try:
            with self._lock:
                # Evict if at capacity
                if len(self._cache) >= self.max_size and key not in self._cache:
                    self._evict_lru()
                
                expires_at = None
                if ttl or self.default_ttl:
                    expires_at = time.time() + (ttl or self.default_ttl)
                
                self._cache[key] = {
                    'value': value,
                    'expires_at': expires_at,
                    'created_at': time.time(),
                    'accessed_at': time.time()
                }
                
                self._stats.sets += 1
                return True
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self._stats.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    self._stats.deletes += 1
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self._stats.errors += 1
            return False
    
    async def clear(self) -> bool:
        try:
            with self._lock:
                self._cache.clear()
                return True
                
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self._stats.errors += 1
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            with self._lock:
                if key in self._cache:
                    entry = self._cache[key]
                    if entry['expires_at'] and time.time() > entry['expires_at']:
                        del self._cache[key]
                        self._stats.evictions += 1
                        return False
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            self._stats.errors += 1
            return False
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return
        
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]['accessed_at']
        )
        del self._cache[lru_key]
        self._stats.evictions += 1
    
    def get_stats(self) -> CacheStats:
        return self._stats


class RedisCache(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 default_ttl: int = 3600, key_prefix: str = "sentiment:"):
        if not REDIS_AVAILABLE:
            raise ImportError("redis library is required for RedisCache")
        
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._stats = CacheStats()
        
        # Initialize Redis connection
        try:
            self._redis = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self._redis.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            redis_key = self._make_key(key)
            value = self._redis.get(redis_key)
            
            if value is not None:
                self._stats.hits += 1
                return json.loads(value)
            else:
                self._stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.errors += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        try:
            redis_key = self._make_key(key)
            serialized_value = json.dumps(value, default=str)
            
            result = self._redis.setex(
                redis_key,
                ttl or self.default_ttl,
                serialized_value
            )
            
            if result:
                self._stats.sets += 1
                return True
            return False
                
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self._stats.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            redis_key = self._make_key(key)
            result = self._redis.delete(redis_key)
            
            if result > 0:
                self._stats.deletes += 1
                return True
            return False
                
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self._stats.errors += 1
            return False
    
    async def clear(self) -> bool:
        try:
            pattern = f"{self.key_prefix}*"
            keys = self._redis.keys(pattern)
            if keys:
                self._redis.delete(*keys)
            return True
                
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            self._stats.errors += 1
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            redis_key = self._make_key(key)
            return bool(self._redis.exists(redis_key))
                
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            self._stats.errors += 1
            return False
    
    def get_stats(self) -> CacheStats:
        return self._stats


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (Redis) tiers"""
    
    def __init__(self, 
                 l1_max_size: int = 1000,
                 l1_ttl: int = 300,  # 5 minutes
                 l2_redis_url: str = None,
                 l2_ttl: int = 3600,  # 1 hour
                 enable_l2: bool = True):
        
        # L1 Cache (In-Memory)
        self.l1 = InMemoryCache(max_size=l1_max_size, default_ttl=l1_ttl)
        
        # L2 Cache (Redis) - optional
        self.l2 = None
        if enable_l2 and l2_redis_url and REDIS_AVAILABLE:
            try:
                self.l2 = RedisCache(redis_url=l2_redis_url, default_ttl=l2_ttl)
                logger.info("Multi-level cache initialized with Redis L2")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis L2 cache: {e}")
        
        if not self.l2:
            logger.info("Single-level cache initialized (memory only)")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache, checking L1 then L2"""
        # Try L1 first
        value = await self.l1.get(key)
        if value is not None:
            logger.debug(f"Cache L1 hit: {key}")
            return value
        
        # Try L2 if available
        if self.l2:
            value = await self.l2.get(key)
            if value is not None:
                logger.debug(f"Cache L2 hit: {key}")
                # Promote to L1
                await self.l1.set(key, value)
                return value
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache (both L1 and L2 if available)"""
        l1_success = await self.l1.set(key, value, ttl)
        l2_success = True
        
        if self.l2:
            l2_success = await self.l2.set(key, value, ttl)
        
        return l1_success or l2_success
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache (both levels)"""
        l1_success = await self.l1.delete(key)
        l2_success = True
        
        if self.l2:
            l2_success = await self.l2.delete(key)
        
        return l1_success or l2_success
    
    async def clear(self) -> bool:
        """Clear all cache levels"""
        l1_success = await self.l1.clear()
        l2_success = True
        
        if self.l2:
            l2_success = await self.l2.clear()
        
        return l1_success and l2_success
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache level"""
        exists_l1 = await self.l1.exists(key)
        if exists_l1:
            return True
        
        if self.l2:
            return await self.l2.exists(key)
        
        return False
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels"""
        stats = {"l1": self.l1.get_stats()}
        
        if self.l2:
            stats["l2"] = self.l2.get_stats()
        
        return stats


class SentimentResultCache:
    """Specialized cache for sentiment analysis results"""
    
    def __init__(self, cache_backend: MultiLevelCache = None):
        self.cache = cache_backend or MultiLevelCache()
        self.cache_version = "v1"  # For cache invalidation
    
    def _generate_cache_key(self, text: str, config_hash: str = None) -> str:
        """Generate cache key for text and configuration"""
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        if config_hash:
            key = f"{self.cache_version}:result:{text_hash}:{config_hash}"
        else:
            key = f"{self.cache_version}:result:{text_hash}"
        
        return key
    
    def _generate_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Generate hash for configuration to use as cache key component"""
        # Sort config for consistent hashing
        sorted_config = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(sorted_config.encode('utf-8')).hexdigest()[:12]
    
    async def get_result(self, text: str, config: Dict[str, Any] = None) -> Optional[SentimentResult]:
        """Get cached sentiment result"""
        try:
            config_hash = self._generate_config_hash(config) if config else None
            cache_key = self._generate_cache_key(text, config_hash)
            
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for text hash: {cache_key}")
                return SentimentResult(**cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    async def set_result(self, text: str, result: SentimentResult, 
                        config: Dict[str, Any] = None, ttl: int = None) -> bool:
        """Cache sentiment result"""
        try:
            config_hash = self._generate_config_hash(config) if config else None
            cache_key = self._generate_cache_key(text, config_hash)
            
            # Convert result to cacheable format
            cached_data = result.dict()
            
            success = await self.cache.set(cache_key, cached_data, ttl)
            if success:
                logger.debug(f"Cached result for text hash: {cache_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
            return False
    
    async def invalidate_cache(self) -> bool:
        """Invalidate all cached results"""
        return await self.cache.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache.get_stats()
        
        # Calculate combined stats if multi-level
        if len(stats) > 1:
            combined_stats = CacheStats()
            for level_stats in stats.values():
                combined_stats.hits += level_stats.hits
                combined_stats.misses += level_stats.misses
                combined_stats.sets += level_stats.sets
                combined_stats.deletes += level_stats.deletes
                combined_stats.evictions += level_stats.evictions
                combined_stats.errors += level_stats.errors
            
            return {
                "levels": {k: v.to_dict() for k, v in stats.items()},
                "combined": combined_stats.to_dict()
            }
        else:
            return {level: stats_obj.to_dict() for level, stats_obj in stats.items()}


# Global cache instance
_global_cache: Optional[SentimentResultCache] = None


def get_global_cache() -> SentimentResultCache:
    """Get or create global cache instance"""
    global _global_cache
    
    if _global_cache is None:
        # Configure based on environment
        import os
        redis_url = os.getenv("REDIS_URL")
        enable_redis = os.getenv("ENABLE_REDIS_CACHE", "false").lower() == "true"
        
        if redis_url and enable_redis and REDIS_AVAILABLE:
            multi_cache = MultiLevelCache(
                l1_max_size=int(os.getenv("L1_CACHE_SIZE", "1000")),
                l1_ttl=int(os.getenv("L1_CACHE_TTL", "300")),
                l2_redis_url=redis_url,
                l2_ttl=int(os.getenv("L2_CACHE_TTL", "3600")),
                enable_l2=True
            )
        else:
            multi_cache = MultiLevelCache(
                l1_max_size=int(os.getenv("L1_CACHE_SIZE", "1000")),
                l1_ttl=int(os.getenv("L1_CACHE_TTL", "300")),
                enable_l2=False
            )
        
        _global_cache = SentimentResultCache(multi_cache)
        logger.info("Global sentiment result cache initialized")
    
    return _global_cache


def clear_global_cache():
    """Clear global cache instance"""
    global _global_cache
    if _global_cache:
        asyncio.create_task(_global_cache.invalidate_cache())
        _global_cache = None