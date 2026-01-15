"""
Cache Module
Implements caching for API responses and computed data
"""

import time
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from config.settings import DATA_STORAGE

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages in-memory caching of data."""
    
    def __init__(self, max_size: int = None, default_ttl: int = None):
        self.cache = {}
        self.max_size = max_size or DATA_STORAGE['MAX_CACHE_SIZE']
        self.default_ttl = default_ttl or DATA_STORAGE['CACHE_DURATION']
        self.access_times = {}
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate cache key."""
        return f"{namespace}:{key}"
    
    def _is_expired(self, cache_entry: Dict) -> bool:
        """Check if cache entry is expired."""
        expiry = cache_entry.get('expiry')
        if not expiry:
            return False
        return datetime.now() > expiry
    
    def _evict_oldest(self):
        """Evict oldest cache entry when max size is reached."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times, key=self.access_times.get)
        self.delete(oldest_key)
    
    def set(self, namespace: str, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in cache.
        
        Args:
            namespace: Cache namespace (e.g., 'price_data', 'news')
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = default)
        
        Returns:
            True if successful
        """
        cache_key = self._generate_key(namespace, key)
        
        # Check cache size
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[cache_key] = {
            'value': value,
            'expiry': expiry,
            'created': datetime.now()
        }
        
        self.access_times[cache_key] = time.time()
        
        logger.debug(f"Cached {cache_key} with TTL {ttl}s")
        return True
    
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._generate_key(namespace, key)
        
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        
        # Check expiry
        if self._is_expired(entry):
            self.delete(cache_key)
            return None
        
        # Update access time
        self.access_times[cache_key] = time.time()
        
        logger.debug(f"Cache hit: {cache_key}")
        return entry['value']
    
    def delete(self, cache_key: str = None, namespace: str = None, key: str = None) -> bool:
        """
        Delete a cache entry.
        
        Args:
            cache_key: Full cache key OR
            namespace: Cache namespace (with key)
            key: Cache key (with namespace)
        
        Returns:
            True if deleted, False if not found
        """
        if not cache_key:
            cache_key = self._generate_key(namespace, key)
        
        if cache_key in self.cache:
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            logger.debug(f"Deleted cache entry: {cache_key}")
            return True
        
        return False
    
    def clear_namespace(self, namespace: str) -> int:
        """
        Clear all entries in a namespace.
        
        Args:
            namespace: Namespace to clear
        
        Returns:
            Number of entries cleared
        """
        prefix = f"{namespace}:"
        keys_to_delete = [k for k in self.cache.keys() if k.startswith(prefix)]
        
        for key in keys_to_delete:
            self.delete(cache_key=key)
        
        logger.info(f"Cleared {len(keys_to_delete)} entries from namespace {namespace}")
        return len(keys_to_delete)
    
    def clear_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self.cache)
        self.cache.clear()
        self.access_times.clear()
        logger.info(f"Cleared all cache ({count} entries)")
        return count
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'max_size': self.max_size,
            'utilization': round(total_entries / self.max_size * 100, 2) if self.max_size > 0 else 0
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            self.delete(cache_key=key)
        
        logger.info(f"Cleaned up {len(expired_keys)} expired entries")
        return len(expired_keys)


# Global cache instance
_cache_instance = None

def get_cache() -> CacheManager:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance
