import pickle
import hashlib
import os
import time
from datetime import datetime, timedelta
from logger.logger import get_logger
from utils.metrics.cache_metrics import CacheMetrics

logger = get_logger("query_cache")

class QueryCache:
    """Simple file-based cache for query results."""
    
    def __init__(self, cache_dir="cache/queries", ttl_hours=24):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        self.metrics = CacheMetrics()
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Query cache initialized: {cache_dir}, TTL: {ttl_hours}h")

    def _get_cache_key(self, query):
        """Generate cache key from query."""
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def get(self, query):
        """Get cached result if available and not expired."""
        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            self.metrics.log_miss(query)
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if expired
            if datetime.now() - cached_data['timestamp'] > self.ttl:
                os.remove(cache_file)
                logger.info(f"Cache expired and removed: {cache_key}")
                self.metrics.log_miss(query)
                return None
            
            logger.info(f"Cache hit: {cache_key}")
            self.metrics.log_hit(query)
            return cached_data['results']
            
        except Exception as e:
            logger.error(f"Error reading cache {cache_key}: {e}")
            self.metrics.log_miss(query)
            return None
        
    def set(self, query, results):
        """Cache query results."""
        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            cached_data = {
                'query': query,
                'results': results,
                'timestamp': datetime.now()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            logger.info(f"Cached results: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error caching results {cache_key}: {e}")
    
    def get_stats(self):
        """Get cache statistics including metrics."""
        # Count files in cache directory
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_entries = len(cache_files)
        
        # Count valid (non-expired) entries
        valid_entries = 0
        for cache_file in cache_files:
            try:
                filepath = os.path.join(self.cache_dir, cache_file)
                with open(filepath, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if datetime.now() - cached_data['timestamp'] <= self.ttl:
                    valid_entries += 1
            except:
                pass
        
        expired_entries = total_entries - valid_entries
        
        # Get metrics summary
        metrics_summary = self.metrics.get_summary()
        
        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'ttl_hours': self.ttl.total_seconds() / 3600,
            'hit_rate': f"{metrics_summary.get('hit_rate_percent', 0):.1f}%",
            'total_requests': metrics_summary.get('total_requests', 0),
            'cache_hits': metrics_summary.get('hits', 0),
            'cache_misses': metrics_summary.get('misses', 0)
        }