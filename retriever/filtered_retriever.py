import os
import time
from datetime import datetime
from retriever.cache_retriever import CacheRetriever
from logger.logger import get_logger

logger = get_logger("filtered_retriever")

class FilteredRetriever(CacheRetriever):
    """Retriever with advanced metadata filtering capabilities."""
    
    def __init__(self, data_path="data/documents", embeddings_path="data/embeddings", 
                 model_name="all-MiniLM-L6-v2", reranker_model="cross-encoder/ms-marco-MiniLM-L12-v2", 
                 top_k=3, rerank_top_k=10, cache_ttl_hours=24, enable_cache=True,
                 min_score_threshold=0.3):
        super().__init__(data_path, embeddings_path, model_name, reranker_model, 
                        top_k, rerank_top_k, cache_ttl_hours, enable_cache)
        
        self.min_score_threshold = min_score_threshold
        logger.info(f"Filtered retriever initialized with min_score_threshold: {min_score_threshold}")
    
    def retrieve(self, query, file_types=None, sources=None, date_from=None, date_to=None, min_score=None):
        """Enhanced retrieve with metadata filtering."""
        logger.info(f"Filtered retrieval - file_types: {file_types}, sources: {sources}")
        
        # Use custom cache key that includes filters
        cache_key = self._get_filtered_cache_key(query, file_types, sources, date_from, date_to, min_score)
        
        # Check cache with filters
        if self.enable_cache:
            cached_results = self._get_filtered_cache(cache_key)
            if cached_results:
                self.metrics['cache_hits'] += 1
                logger.info("Filtered cache hit")
                return cached_results
        
        # Perform filtered retrieval
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        if self.enable_cache:
            self.metrics['cache_misses'] += 1
        
        # Get candidates from parent (includes reranking)
        results = super().retrieve(query)
        
        # Apply metadata filters
        filtered_results = self._apply_metadata_filters(
            results, file_types, sources, date_from, date_to, min_score
        )
        
        # Cache filtered results
        if self.enable_cache and filtered_results:
            self._set_filtered_cache(cache_key, filtered_results)
        
        response_time = time.time() - start_time
        self._update_metrics(response_time)
        
        logger.info(f"Filtered retrieval complete - {len(filtered_results)} results after filtering")
        return filtered_results
    
    def _apply_metadata_filters(self, results, file_types=None, sources=None, 
                               date_from=None, date_to=None, min_score=None):
        """Apply metadata filters to results."""
        if not results:
            return results
        
        filtered_results = []
        min_threshold = min_score or self.min_score_threshold
        
        for result in results:
            try:
                # Extract score from result string
                if "[Score:" in result:
                    score_str = result.split("[Score: ")[1].split("]")[0]
                    score = float(score_str)
                else:
                    score = 1.0  # Default score if not found
                
                # Find corresponding document metadata
                doc_metadata = self._find_document_metadata(result)
                if not doc_metadata:
                    continue
                
                # Apply filters
                if not self._passes_filters(doc_metadata, score, file_types, sources, 
                                           date_from, date_to, min_threshold):
                    continue
                
                filtered_results.append(result)
                
            except Exception as e:
                logger.warning(f"Error filtering result: {e}")
                continue
        
        logger.info(f"Applied filters: {len(results)} → {len(filtered_results)} results")
        return filtered_results
    
    def _find_document_metadata(self, result):
        """Find document metadata for a result."""
        # Extract content from result (remove score prefix)
        content = result
        if "] " in result:
            content = result.split("] ", 1)[1]
        
        # Find matching document
        for doc in self.documents:
            if doc['content'] == content:
                return doc
        return None
    
    def _passes_filters(self, doc_metadata, score, file_types=None, sources=None, 
                       date_from=None, date_to=None, min_threshold=0.3):
        """Check if document passes all filters."""
        
        # Score filter
        if score < min_threshold:
            return False
        
        # File type filter
        if file_types and doc_metadata.get('file_type') not in file_types:
            return False
        
        # Source filter
        if sources and doc_metadata.get('source') not in sources:
            return False
        
        # Date filters (if file has modification date)
        if date_from or date_to:
            file_path = doc_metadata.get('file_path')
            if file_path and os.path.exists(file_path):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if date_from and file_mtime < date_from:
                    return False
                
                if date_to and file_mtime > date_to:
                    return False
        
        return True
    
    def _get_filtered_cache_key(self, query, file_types, sources, date_from, date_to, min_score):
        """Generate cache key including filters."""
        import hashlib
        
        cache_data = {
            'query': query.lower().strip(),
            'file_types': sorted(file_types) if file_types else None,
            'sources': sorted(sources) if sources else None,
            'date_from': date_from.isoformat() if date_from else None,
            'date_to': date_to.isoformat() if date_to else None,
            'min_score': min_score
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_filtered_cache(self, cache_key):
        """Get from cache using custom key."""
        if hasattr(self, 'cache'):
            cache_file = os.path.join(self.cache.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                try:
                    import pickle
                    from datetime import datetime
                    
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Check if expired
                    if datetime.now() - cached_data['timestamp'] > self.cache.ttl:
                        os.remove(cache_file)
                        return None
                    
                    return cached_data['results']
                except:
                    return None
        return None
    
    def _set_filtered_cache(self, cache_key, results):
        """Set cache using custom key."""
        if hasattr(self, 'cache'):
            cache_file = os.path.join(self.cache.cache_dir, f"{cache_key}.pkl")
            try:
                import pickle
                from datetime import datetime
                
                cached_data = {
                    'results': results,
                    'timestamp': datetime.now()
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
            except:
                pass
    
    def get_available_file_types(self):
        """Get list of available file types."""
        return list(set(doc.get('file_type', 'unknown') for doc in self.documents))
    
    def get_available_sources(self):
        """Get list of available sources."""
        return list(set(doc.get('source', 'unknown') for doc in self.documents))
    
    def print_filter_info(self):
        """Print information about available filters."""
        print("\n=== AVAILABLE FILTERS ===")
        print(f"File Types: {self.get_available_file_types()}")
        print(f"Sources: {self.get_available_sources()}")
        print(f"Min Score Threshold: {self.min_score_threshold}")
        print(f"Total Documents: {len(self.documents)}")
        print("========================\n")