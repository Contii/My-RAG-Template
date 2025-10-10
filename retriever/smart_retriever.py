import time
import hashlib
from sentence_transformers import CrossEncoder
from retriever.semantic_retriever import SemanticRetriever
from utils.query_cache import QueryCache
from logger.logger import get_logger

logger = get_logger("smart_retriever")

class SmartRetriever:
    """Retriever with optional reranking, caching, and filtering components."""
    
    def __init__(self, data_path="data/documents", embeddings_path="data/embeddings",
                 model_name="all-MiniLM-L6-v2", top_k=3,
                 # Optional components
                 use_reranking=True, reranker_model="ms-marco-MiniLM-L-12-v2", rerank_top_k=10,
                 use_cache=True, cache_ttl_hours=24,
                 use_filters=True, min_score_threshold=0.3):
        
        # Base semantic retriever (always needed)
        self.base_retriever = SemanticRetriever(data_path, embeddings_path, model_name, top_k)
        self.top_k = top_k
        
        # Optional reranking
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = CrossEncoder(reranker_model)
            self.rerank_top_k = rerank_top_k
            logger.info(f"Reranking enabled: {reranker_model}")
        
        # Optional caching  
        self.use_cache = use_cache
        if use_cache:
            self.cache = QueryCache(ttl_hours=cache_ttl_hours)
            logger.info("Caching enabled")
        
        # Optional filtering
        self.use_filters = use_filters
        self.min_score_threshold = min_score_threshold
        if use_filters:
            logger.info("Filtering enabled")
        
        # Metrics
        self.metrics = {'total_queries': 0, 'cache_hits': 0, 'cache_misses': 0}
        
        logger.info(f"SmartRetriever initialized - Rerank: {use_reranking}, Cache: {use_cache}, Filters: {use_filters}")
    
    def retrieve(self, query, file_types=None, sources=None, min_score=None):
        """Main retrieval method with all optional features."""
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Build cache key including filters
        cache_key = self._build_cache_key(query, file_types, sources, min_score)
        
        # 1. Check cache first (if enabled)
        if self.use_cache and cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                logger.info(f"Cache hit for: {query[:30]}...")
                return cached_result
            else:
                self.metrics['cache_misses'] += 1
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # 2. Get initial candidates from semantic search
        # Get more candidates if we're going to rerank
        semantic_results = self.base_retriever.retrieve(query)
        
        # Convert to candidates format for processing
        candidates = self._parse_semantic_results(semantic_results)
        
        # 3. Apply filters early (if enabled)
        if self.use_filters and (file_types or sources or min_score):
            candidates = self._apply_filters(candidates, file_types, sources, min_score)
            logger.info(f"Applied filters, {len(candidates)} candidates remain")
        
        # 4. Rerank candidates (if enabled)
        if self.use_reranking and len(candidates) > 1:
            candidates = self._rerank_candidates(query, candidates)
            logger.info("Reranking completed")
        
        # 5. Take final top-k and format results
        final_results = self._format_results(candidates[:self.top_k])
        
        # 6. Cache the final result (if enabled)
        if self.use_cache and cache_key:
            self.cache.set(cache_key, final_results)
        
        response_time = time.time() - start_time
        logger.info(f"Retrieval completed in {response_time:.3f}s, returned {len(final_results)} results")
        print(f"Retrieval completed in {response_time:.3f}s, returned {len(final_results)} results")
        
        return final_results
    
    def _parse_semantic_results(self, semantic_results):
        """Convert semantic results to internal candidate format."""
        candidates = []
        for result in semantic_results:
            # Extract score and content from semantic result
            score = self._extract_score(result)
            content = self._extract_content(result)
            
            # Find document metadata
            doc_metadata = self._find_document_metadata(content)
            
            candidates.append({
                'content': content,
                'semantic_score': score,
                'metadata': doc_metadata or {},
                'original_result': result
            })
        
        return candidates
    
    def _apply_filters(self, candidates, file_types, sources, min_score):
        """Apply metadata filters to candidates."""
        filtered = []
        min_threshold = min_score or self.min_score_threshold
        
        for candidate in candidates:
            score = candidate['semantic_score']
            metadata = candidate['metadata']
            
            # Apply filters
            if score < min_threshold:
                continue
            if file_types and metadata.get('file_type') not in file_types:
                continue
            if sources and metadata.get('source') not in sources:
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _rerank_candidates(self, query, candidates):
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return candidates
        
        # Prepare for reranking
        contents = [candidate['content'] for candidate in candidates]
        query_doc_pairs = [(query, content) for content in contents]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # Add rerank scores to candidates
        for candidate, score in zip(candidates, rerank_scores):
            candidate['rerank_score'] = float(score)
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return candidates
    
    def _format_results(self, candidates):
        """Format candidates back to result strings."""
        results = []
        for candidate in candidates:
            # Build result string with scores
            score_parts = []
            
            if 'rerank_score' in candidate:
                score_parts.append(f"Rerank: {candidate['rerank_score']:.3f}")
            
            score_parts.append(f"Semantic: {candidate['semantic_score']:.3f}")
            
            # Add file type if available
            file_type = candidate['metadata'].get('file_type', 'unknown')
            
            score_str = "[" + "][".join(score_parts) + "]"
            result = f"{score_str}[{file_type}] {candidate['content']}"
            results.append(result)
        
        return results
    
    def _build_cache_key(self, query, file_types, sources, min_score):
        """Build cache key including all parameters."""
        if not self.use_cache:
            return None
        
        cache_data = {
            'query': query.lower().strip(),
            'file_types': sorted(file_types) if file_types else None,
            'sources': sorted(sources) if sources else None,
            'min_score': min_score,
            'use_reranking': self.use_reranking,
            'use_filters': self.use_filters
        }
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _extract_score(self, result):
        """Extract score from result string."""
        try:
            if "[Score:" in result:
                return float(result.split("[Score: ")[1].split("]")[0])
        except:
            pass
        return 1.0
    
    def _extract_content(self, result):
        """Extract content from result string."""
        content = result
        # Remove score and file type prefixes
        if "] " in result:
            parts = result.split("] ")
            content = "] ".join(parts[1:]) if len(parts) > 1 else result
        return content
    
    def _find_document_metadata(self, content):
        """Find document metadata for filtering."""
        for doc in self.base_retriever.documents:
            if doc['content'] == content:
                return doc
        return None
    
    # Delegate methods to base retriever
    def get_available_file_types(self):
        return list(set(doc.get('file_type', 'unknown') for doc in self.base_retriever.documents))
    
    def get_available_sources(self):
        return list(set(doc.get('source', 'unknown') for doc in self.base_retriever.documents))
    
    def get_metrics(self):
        """Get performance metrics."""
        cache_hit_rate = 0.0
        if self.metrics['total_queries'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_queries']
        
        return {
            **self.metrics,
            'cache_hit_rate': f"{cache_hit_rate:.1%}",
            'components': {
                'reranking': self.use_reranking,
                'caching': self.use_cache, 
                'filtering': self.use_filters
            }
        }
    
    def print_metrics(self):
        """Print formatted performance metrics."""
        metrics = self.get_metrics()
        print("\n=== SMART RETRIEVER METRICS ===")
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Cache Hit Rate: {metrics['cache_hit_rate']}")
        print(f"Components: {metrics['components']}")
        print("==============================\n")