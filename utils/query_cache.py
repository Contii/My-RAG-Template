import pickle
import hashlib
import os
from datetime import datetime, timedelta
from logger.logger import get_logger

logger = get_logger("query_cache")

class QueryCache:
    """Simple file-based cache for query results."""
    
    def __init__(self, cache_dir="cache/queries", ttl_hours=24):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
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
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if expired
            if datetime.now() - cached_data['timestamp'] > self.ttl:
                os.remove(cache_file)
                logger.info(f"Cache expired and removed: {cache_key}")
                return None
            
            logger.info(f"Cache hit: {cache_key}")
            return cached_data['results']
            
        except Exception as e:
            logger.error(f"Error reading cache {cache_key}: {e}")
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