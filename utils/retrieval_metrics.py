import time
import json
from datetime import datetime
from collections import defaultdict
from logger.logger import get_logger

logger = get_logger("retrieval_metrics")

class RetrievalMetrics:
    """Comprehensive metrics tracking for retrieval systems."""
    
    def __init__(self, log_file="logs/retrieval_metrics.json"):
        self.log_file = log_file
        self.session_metrics = {
            'session_id': datetime.now().isoformat(),
            'queries': [],
            'summary': defaultdict(float),
            'component_timings': defaultdict(list),
            'error_counts': defaultdict(int)
        }
        logger.info("Retrieval metrics initialized")
    
    def start_query(self, query, retriever_type):
        """Start tracking a new query."""
        query_data = {
            'query': query,
            'retriever_type': retriever_type,
            'timestamp': datetime.now().isoformat(),
            'start_time': time.time(),
            'components': {},
            'results_count': 0,
            'cache_hit': False,
            'error': None
        }
        return query_data
    
    def log_component_time(self, query_data, component, duration):
        """Log timing for specific components."""
        query_data['components'][component] = duration
        self.session_metrics['component_timings'][component].append(duration)
    
    def log_cache_hit(self, query_data, hit=True):
        """Log cache hit/miss."""
        query_data['cache_hit'] = hit
        cache_key = 'cache_hits' if hit else 'cache_misses'
        self.session_metrics['summary'][cache_key] += 1
    
    def log_results(self, query_data, results):
        """Log query results."""
        query_data['results_count'] = len(results)
        query_data['results'] = [self._extract_result_metadata(r) for r in results]
    
    def log_error(self, query_data, error):
        """Log query error."""
        query_data['error'] = str(error)
        self.session_metrics['error_counts'][type(error).__name__] += 1
    
    def finish_query(self, query_data):
        """Finish tracking query and calculate metrics."""
        end_time = time.time()
        total_time = end_time - query_data['start_time']
        
        query_data['total_time'] = total_time
        query_data['end_time'] = end_time
        
        # Update session summary
        self.session_metrics['summary']['total_queries'] += 1
        self.session_metrics['summary']['total_time'] += total_time
        
        # Add to queries list
        self.session_metrics['queries'].append(query_data)
        
        # Save to file
        self._save_metrics()
        
        logger.info(f"Query completed - Time: {total_time:.3f}s, Results: {query_data['results_count']}")
        
        return query_data
    
    def _extract_result_metadata(self, result):
        """Extract metadata from result string."""
        metadata = {'content_preview': result[:100]}
        
        # Extract scores
        if '[Score:' in result:
            try:
                metadata['semantic_score'] = float(result.split('[Score: ')[1].split(']')[0])
            except:
                pass
        
        if '[Rerank:' in result:
            try:
                metadata['rerank_score'] = float(result.split('[Rerank: ')[1].split(']')[0])
            except:
                pass
        
        # Extract file type
        if '[txt]' in result:
            metadata['file_type'] = 'txt'
        elif '[pdf]' in result:
            metadata['file_type'] = 'pdf'
        elif '[json]' in result:
            metadata['file_type'] = 'json'
        elif '[html]' in result:
            metadata['file_type'] = 'html'
        
        return metadata
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        try:
            import os
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            with open(self.log_file, 'w') as f:
                json.dump(self.session_metrics, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def get_session_summary(self):
        """Get current session summary."""
        summary = dict(self.session_metrics['summary'])
        
        # Calculate averages
        if summary.get('total_queries', 0) > 0:
            summary['avg_response_time'] = summary['total_time'] / summary['total_queries']
            summary['avg_results_per_query'] = sum(q['results_count'] for q in self.session_metrics['queries']) / summary['total_queries']
        
        # Calculate cache hit rate
        total_cache_requests = summary.get('cache_hits', 0) + summary.get('cache_misses', 0)
        if total_cache_requests > 0:
            summary['cache_hit_rate'] = summary.get('cache_hits', 0) / total_cache_requests
        
        # Component timing averages
        summary['component_avg_times'] = {}
        for component, times in self.session_metrics['component_timings'].items():
            if times:
                summary['component_avg_times'][component] = sum(times) / len(times)
        
        return summary
    
    def print_dashboard(self):
        """Print metrics dashboard."""
        summary = self.get_session_summary()
        
        print("\n" + "="*50)
        print("           RETRIEVAL METRICS DASHBOARD")
        print("="*50)
        
        # Basic stats
        print(f"📊 Total Queries: {summary.get('total_queries', 0)}")
        print(f"⏱️  Avg Response Time: {summary.get('avg_response_time', 0):.3f}s")
        print(f"📄 Avg Results/Query: {summary.get('avg_results_per_query', 0):.1f}")
        
        # Cache performance
        cache_hit_rate = summary.get('cache_hit_rate', 0)
        print(f"💾 Cache Hit Rate: {cache_hit_rate:.1%}")
        
        # Component timings
        if summary.get('component_avg_times'):
            print(f"\n🔧 Component Average Times:")
            for component, avg_time in summary['component_avg_times'].items():
                print(f"   {component}: {avg_time:.3f}s")
        
        # Error summary
        if self.session_metrics['error_counts']:
            print(f"\n❌ Errors:")
            for error_type, count in self.session_metrics['error_counts'].items():
                print(f"   {error_type}: {count}")
        
        print("="*50 + "\n")
    
    def get_performance_insights(self):
        """Generate performance insights."""
        insights = []
        summary = self.get_session_summary()
        
        # Response time insights
        avg_time = summary.get('avg_response_time', 0)
        if avg_time > 2.0:
            insights.append("⚠️  Average response time is high (>2s). Consider optimizing retrieval or enabling cache.")
        elif avg_time < 0.1:
            insights.append("✅ Excellent response time! System is well optimized.")
        
        # Cache insights
        cache_rate = summary.get('cache_hit_rate', 0)
        if cache_rate > 0.7:
            insights.append("✅ High cache hit rate! Cache is working effectively.")
        elif cache_rate < 0.3 and summary.get('total_queries', 0) > 5:
            insights.append("⚠️  Low cache hit rate. Consider increasing cache TTL or query similarity.")
        
        # Results insights
        avg_results = summary.get('avg_results_per_query', 0)
        if avg_results < 1:
            insights.append("⚠️  Low average results per query. Check document relevance or reduce filters.")
        elif avg_results > 5:
            insights.append("📄 High number of results per query. Consider tighter filtering or lower top_k.")
        
        return insights