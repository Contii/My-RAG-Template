import time
from retriever.reranking_retriever import RerankingRetriever
from utils.query_cache import QueryCache
from logger.logger import get_logger

logger = get_logger("cache_retriever")

class CacheRetriever(RerankingRetriever):
    """Retriever with caching and performance metrics."""
    
    def __init__(self, data_path="data/documents", embeddings_path="data/embeddings", 
                 model_name="all-MiniLM-L6-v2", reranker_model="cross-encoder/ms-marco-MiniLM-L12-v2", 
                 top_k=3, rerank_top_k=10, cache_ttl_hours=24, enable_cache=True):
        super().__init__(data_path, embeddings_path, model_name, reranker_model, top_k, rerank_top_k)
        
        # Initialize cache
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = QueryCache(ttl_hours=cache_ttl_hours)
        
        # Initialize metrics
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'fastest_query': float('inf'),
            'slowest_query': 0.0
        }
        
        logger.info(f"Cache retriever initialized - Cache: {enable_cache}")
    
    def retrieve(self, query):
        """Cached retrieval with performance metrics."""
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Check cache first
        if self.enable_cache:
            cached_results = self.cache.get(query)
            if cached_results:
                self.metrics['cache_hits'] += 1
                response_time = time.time() - start_time
                self._update_metrics(response_time)
                logger.info(f"Cache hit - Response time: {response_time:.3f}s")
                return cached_results
        
        # Cache miss - perform actual retrieval
        if self.enable_cache:
            self.metrics['cache_misses'] += 1
            logger.info("Cache miss - performing retrieval")
        
        results = super().retrieve(query)
        
        # Cache the results
        if self.enable_cache and results:
            self.cache.set(query, results)
        
        response_time = time.time() - start_time
        self._update_metrics(response_time)
        
        logger.info(f"Retrieval complete - Response time: {response_time:.3f}s")
        return results
    
    def _update_metrics(self, response_time):
        """Update performance metrics."""
        self.metrics['total_response_time'] += response_time
        self.metrics['avg_response_time'] = (
            self.metrics['total_response_time'] / self.metrics['total_queries']
        )
        self.metrics['fastest_query'] = min(self.metrics['fastest_query'], response_time)
        self.metrics['slowest_query'] = max(self.metrics['slowest_query'], response_time)
    
    def get_metrics(self):
        """Get comprehensive performance metrics."""
        cache_hit_rate = 0.0
        if self.metrics['total_queries'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_queries']
        
        return {
            'total_queries': self.metrics['total_queries'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': f"{cache_hit_rate:.1%}",
            'avg_response_time': f"{self.metrics['avg_response_time']:.3f}s",
            'fastest_query': f"{self.metrics['fastest_query']:.3f}s" if self.metrics['fastest_query'] != float('inf') else "N/A",
            'slowest_query': f"{self.metrics['slowest_query']:.3f}s",
            'total_documents': len(self.documents) if self.documents else 0
        }
    
    def print_metrics(self):
        """Print formatted performance metrics."""
        metrics = self.get_metrics()
        print("\n=== RETRIEVAL METRICS ===")
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Cache Hit Rate: {metrics['cache_hit_rate']}")
        print(f"Avg Response Time: {metrics['avg_response_time']}")
        print(f"Fastest Query: {metrics['fastest_query']}")
        print(f"Slowest Query: {metrics['slowest_query']}")
        print(f"Total Documents: {metrics['total_documents']}")
        print("========================\n")
    
    def clear_cache(self):
        """Clear query cache."""
        if hasattr(self, 'cache'):
            # Remove all cache files
            import os
            cache_dir = self.cache.cache_dir
            for filename in os.listdir(cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(cache_dir, filename))
            logger.info("Cache cleared")