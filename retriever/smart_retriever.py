import time
import hashlib
from datetime import datetime
from sentence_transformers import CrossEncoder
from retriever.semantic_retriever import SemanticRetriever
from utils.retrieval_metrics import RetrievalMetrics
from utils.query_cache import QueryCache
from logger.logger import get_logger, log_retrieval_metrics

logger = get_logger("smart_retriever")

class SmartRetriever:
    """Retriever with optional reranking, caching, and filtering components."""
    
    def __init__(self, data_path="data/documents", embeddings_path="data/embeddings",
                 model_name="all-MiniLM-L6-v2", top_k=3,
                 # Optional components
                 use_reranking=True, reranker_model="cross-encoder/ms-marco-MiniLM-L12-v2", rerank_top_k=10,
                 use_cache=True, cache_ttl_hours=24,
                 use_filters=True, min_score_threshold=0.3,
                 use_metrics=True):
        
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
        
        # Optional metrics
        self.use_metrics = use_metrics
        if use_metrics:
            self.metrics = RetrievalMetrics()
            logger.info("Metrics tracking enabled")
        else:
            self.metrics = None
        
        logger.info(f"SmartRetriever initialized - Rerank: {use_reranking}, Cache: {use_cache}, Filters: {use_filters}")
    
    def retrieve(self, query, file_types=None, sources=None, min_score=None, date_from=None, date_to=None):
        """Main retrieval method with all optional features including date filtering."""
        start_time = time.time()
        component_times = {}
        cache_hit = None
        
        # Start detailed metrics tracking (if enabled)
        query_data = None
        if self.metrics:
            query_data = self.metrics.start_query(query, "SmartRetriever")
        
        try:
            # Build cache key including ALL filters
            cache_key = self._build_cache_key(query, file_types, sources, min_score, date_from, date_to)
            
            # 1. Check cache first (if enabled)
            if self.use_cache and cache_key:
                cache_start = time.time()
                cached_result = self.cache.get(cache_key)
                cache_time = time.time() - cache_start
                component_times['cache_check'] = cache_time
                
                if self.metrics:
                    self.metrics.log_component_time(query_data, 'cache_check', cache_time)
                
                if cached_result:
                    cache_hit = True
                    total_time = time.time() - start_time
                    
                    # Log to both systems
                    log_retrieval_metrics(logger, total_time, component_times, len(cached_result), cache_hit)
                    
                    if self.metrics:
                        self.metrics.log_cache_hit(query_data, True)
                        self.metrics.log_results(query_data, cached_result)
                        self.metrics.finish_query(query_data)
                    
                    logger.info(f"Cache hit for: {query[:30]}...")
                    return cached_result
                else:
                    cache_hit = False
                    if self.metrics:
                        self.metrics.log_cache_hit(query_data, False)
            
            logger.info(f"Processing query: {query[:50]}...")
            
            # 2. Get initial candidates from semantic search
            semantic_start = time.time()
            semantic_results = self.base_retriever.retrieve(query)
            semantic_time = time.time() - semantic_start
            component_times['semantic_search'] = semantic_time
            
            if self.metrics:
                self.metrics.log_component_time(query_data, 'semantic_search', semantic_time)
            
            # Convert to candidates format for processing
            candidates = self._parse_semantic_results(semantic_results)
            
            # 3. Apply filters early (if enabled)
            if self.use_filters and (file_types or sources or min_score or date_from or date_to):
                filter_start = time.time()
                candidates = self._apply_filters(candidates, file_types, sources, min_score, date_from, date_to)
                filter_time = time.time() - filter_start
                component_times['filtering'] = filter_time
                
                if self.metrics:
                    self.metrics.log_component_time(query_data, 'filtering', filter_time)
                
                logger.info(f"Applied filters, {len(candidates)} candidates remain")
            
            # 4. Rerank candidates (if enabled)
            if self.use_reranking and len(candidates) > 1:
                rerank_start = time.time()
                candidates = self._rerank_candidates(query, candidates)
                rerank_time = time.time() - rerank_start
                component_times['reranking'] = rerank_time
                
                if self.metrics:
                    self.metrics.log_component_time(query_data, 'reranking', rerank_time)
                
                logger.info("Reranking completed")
            
            # 5. Take final top-k and format results
            final_results = self._format_results(candidates[:self.top_k])
            
            # 6. Cache the final result (if enabled)
            if self.use_cache and cache_key:
                cache_save_start = time.time()
                self.cache.set(cache_key, final_results)
                cache_save_time = time.time() - cache_save_start
                component_times['cache_save'] = cache_save_time
                
                if self.metrics:
                    self.metrics.log_component_time(query_data, 'cache_save', cache_save_time)
            
            # Final timing and logging
            total_time = time.time() - start_time
            
            # Log to integrated logger (system + component metrics)
            log_retrieval_metrics(logger, total_time, component_times, len(final_results), cache_hit)
            
            # Log to detailed metrics (if enabled)
            if self.metrics:
                self.metrics.log_results(query_data, final_results)
                self.metrics.finish_query(query_data)
            
            logger.info(f"Retrieval completed in {total_time:.3f}s, returned {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            total_time = time.time() - start_time
            
            # Log error to both systems
            log_retrieval_metrics(logger, total_time, component_times, 0, cache_hit)
            
            if self.metrics and query_data:
                self.metrics.log_error(query_data, e)
                self.metrics.finish_query(query_data)
            
            logger.error(f"Error during retrieval: {e}")
            raise
    
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
    
    def _apply_filters(self, candidates, file_types, sources, min_score, date_from=None, date_to=None):
        """Apply metadata filters to candidates including date filtering."""
        filtered = []
        min_threshold = min_score or self.min_score_threshold
        
        for candidate in candidates:
            score = candidate['semantic_score']
            metadata = candidate['metadata']
            
            # Apply score filter
            if score < min_threshold:
                continue
                
            # Apply file type filter
            if file_types and metadata.get('file_type') not in file_types:
                continue
                
            # Apply source filter
            if sources and metadata.get('source') not in sources:
                continue
            
            # Apply date filters
            if date_from or date_to:
                doc_date = metadata.get('date') or metadata.get('created_at') or metadata.get('modified_at')
                if doc_date:
                    try:
                        from datetime import datetime
                        if isinstance(doc_date, str):
                            # Try to parse date string
                            doc_date = datetime.fromisoformat(doc_date.replace('Z', '+00:00'))
                        
                        if date_from and doc_date < date_from:
                            continue
                        if date_to and doc_date > date_to:
                            continue
                    except (ValueError, TypeError):
                        # If date parsing fails, skip date filtering for this document
                        logger.warning(f"Could not parse date for document: {doc_date}")
            
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
    
    def _build_cache_key(self, query, file_types, sources, min_score, date_from=None, date_to=None):
        """Build cache key including all parameters."""
        if not self.use_cache:
            return None
        
        cache_data = {
            'query': query.lower().strip(),
            'file_types': sorted(file_types) if file_types else None,
            'sources': sorted(sources) if sources else None,
            'min_score': min_score,
            'date_from': date_from.isoformat() if date_from else None,
            'date_to': date_to.isoformat() if date_to else None,
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
    
    # Metrics methods - unified interface
    def print_metrics_dashboard(self):
        """Print metrics dashboard if metrics are enabled."""
        if self.metrics:
            self.metrics.print_dashboard()
        else:
            print("Detailed metrics tracking is disabled")
    
    def get_performance_insights(self):
        """Get performance insights if metrics are enabled."""
        if self.metrics:
            return self.metrics.get_performance_insights()
        return ["Detailed metrics tracking is disabled"]
    
    def save_metrics_report(self, filepath="logs/retrieval_report.txt"):
        """Save detailed metrics report if metrics are enabled."""
        if not self.metrics:
            logger.info("Detailed metrics tracking disabled, no report to save")
            return
        
        try:
            summary = self.metrics.get_session_summary()
            insights = self.get_performance_insights()
            
            with open(filepath, 'w') as f:
                f.write("SMART RETRIEVER METRICS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("SESSION SUMMARY:\n")
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\nPERFORMANCE INSIGHTS:\n")
                for insight in insights:
                    f.write(f"{insight}\n")
                
                f.write(f"\nGenerated: {datetime.now().isoformat()}\n")
            
            logger.info(f"Metrics report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics report: {e}")