import time
import json
import os
import logging
from datetime import datetime
from collections import defaultdict

"""Retrieval-specific metrics: query tracking, cache stats, component timing."""

class RetrievalMetrics:
    """Track and analyze retrieval performance metrics."""
    
    def __init__(self, metrics_file="logs/retrieval_metrics.json"):
        self.logger = logging.getLogger("retrieval_metrics")

        self.metrics_file = metrics_file
        self.query_data = {}
        self.query_counter = 0
        self.session_start = datetime.now()
        
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        self.logger.info("RetrievalMetrics initialized")
    
    def start_query(self, query_text):
        """Start tracking a new query."""
        self.query_counter += 1
        query_id = f"query_{self.query_counter}"
        
        self.query_data[query_id] = {
            'id': query_id,
            'query': query_text,
            'start_time': datetime.now().isoformat(),
            'component_times': {},
            'cache_hit': None,
            'num_results': 0
        }
        
        self.logger.debug(f"Started tracking query {query_id}: {query_text[:50]}...")
        return query_id
    
    def log_component(self, query_id, component_name, duration):
        """Log timing for a specific component."""
        if query_id in self.query_data:
            self.query_data[query_id]['component_times'][component_name] = duration
            self.logger.debug(f"Query {query_id} - {component_name}: {duration:.3f}s")
    
    def log_cache_status(self, query_id, cache_hit):
        """Log whether query hit cache."""
        if query_id in self.query_data:
            self.query_data[query_id]['cache_hit'] = cache_hit
            status = "HIT" if cache_hit else "MISS"
            self.logger.debug(f"Query {query_id} - Cache {status}")
    
    def finish_query(self, query_id, num_results):
        """Finish tracking a query."""
        if query_id in self.query_data:
            query = self.query_data[query_id]
            query['end_time'] = datetime.now().isoformat()
            query['num_results'] = num_results
            
            # Calculate total time
            start = datetime.fromisoformat(query['start_time'])
            end = datetime.fromisoformat(query['end_time'])
            query['total_time'] = (end - start).total_seconds()
            
            self.logger.info(
                f"Query {query_id} completed - "
                f"Time: {query['total_time']:.3f}s, "
                f"Results: {num_results}, "
                f"Cache: {'HIT' if query['cache_hit'] else 'MISS' if query['cache_hit'] is not None else 'N/A'}"
            )
            
            self._save_to_file()
    
    def get_session_summary(self):
        """Get summary statistics for the current session."""
        if not self.query_data:
            self.logger.debug("No queries to summarize")
            return {
                'total_queries': 0,
                'avg_response_time': 0,
                'avg_results_per_query': 0,
                'cache_hit_rate': 'N/A',
                'component_avg_times': {}
            }
        
        queries = list(self.query_data.values())
        total_queries = len(queries)
        
        # Calculate averages
        total_time = sum(q.get('total_time', 0) for q in queries)
        avg_time = total_time / total_queries if total_queries > 0 else 0
        
        total_results = sum(q.get('num_results', 0) for q in queries)
        avg_results = total_results / total_queries if total_queries > 0 else 0
        
        # Cache hit rate
        cache_queries = [q for q in queries if q.get('cache_hit') is not None]
        if cache_queries:
            cache_hits = sum(1 for q in cache_queries if q['cache_hit'])
            cache_rate = (cache_hits / len(cache_queries)) * 100
            cache_rate_str = f"{cache_rate:.1f}%"
        else:
            cache_rate_str = "N/A"
        
        # Component timing averages
        component_times = defaultdict(list)
        for query in queries:
            for component, time in query.get('component_times', {}).items():
                component_times[component].append(time)
        
        component_avg = {
            comp: sum(times) / len(times) 
            for comp, times in component_times.items()
        }
        
        self.logger.debug(f"Session summary: {total_queries} queries, avg time {avg_time:.3f}s")
        
        return {
            'total_queries': total_queries,
            'avg_response_time': avg_time,
            'avg_results_per_query': avg_results,
            'cache_hit_rate': cache_rate_str,
            'component_avg_times': component_avg
        }
    
    def print_dashboard(self):
        """Print a formatted metrics dashboard."""
        summary = self.get_session_summary()
        
        self.logger.info("Displaying metrics dashboard")
        
        print("\n" + "="*50)
        print("           RETRIEVAL METRICS DASHBOARD")
        print("="*50)
        print(f"📊 Total Queries: {summary['total_queries']}")
        print(f"⏱️  Avg Response Time: {summary['avg_response_time']:.3f}s")
        print(f"📄 Avg Results/Query: {summary['avg_results_per_query']:.1f}")
        print(f"💾 Cache Hit Rate: {summary['cache_hit_rate']}")
        
        if summary['component_avg_times']:
            print(f"\n🔧 Component Average Times:")
            for component, avg_time in summary['component_avg_times'].items():
                print(f"   {component}: {avg_time:.3f}s")
        
        print("="*50)
    
    def get_performance_insights(self):
        """Generate performance insights and recommendations."""
        summary = self.get_session_summary()
        insights = []
        
        # Response time insights
        avg_time = summary['avg_response_time']
        if avg_time > 2.0:
            insight = "⚠️  Average response time is high (>2s). Consider optimizing retrieval or enabling cache."
            insights.append(insight)
            self.logger.warning(insight)
        elif avg_time < 0.5:
            insight = "✅ Excellent response time (<0.5s)."
            insights.append(insight)
            self.logger.info(insight)
        
        # Cache insights
        cache_rate_str = summary['cache_hit_rate']
        if cache_rate_str != 'N/A':
            cache_rate = float(cache_rate_str.rstrip('%'))
            if cache_rate > 50:
                insight = f"✅ Cache hit rate is good ({cache_rate_str}). Cache is working effectively."
                insights.append(insight)
                self.logger.info(insight)
            elif cache_rate > 20:
                insight = f"⚠️  Cache hit rate is moderate ({cache_rate_str}). Consider increasing TTL or reviewing query patterns."
                insights.append(insight)
                self.logger.warning(insight)
            else:
                insight = f"❌ Low cache hit rate ({cache_rate_str}). Review caching strategy."
                insights.append(insight)
                self.logger.warning(insight)
        
        # Component insights
        component_times = summary['component_avg_times']
        if component_times:
            slowest = max(component_times.items(), key=lambda x: x[1])
            if slowest[1] > 1.0:
                insight = f"🔧 {slowest[0]} is the slowest component ({slowest[1]:.3f}s avg). Consider optimization."
                insights.append(insight)
                self.logger.warning(insight)
        
        return insights
    
    def _save_to_file(self):
        """Save metrics to JSON file."""
        try:
            data = {
                'session_start': self.session_start.isoformat(),
                'last_update': datetime.now().isoformat(),
                'queries': list(self.query_data.values())
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.debug(f"Metrics saved to {self.metrics_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save retrieval metrics: {e}")
    
    def save_session_report(self, filepath="logs/retrieval_report.txt"):
        """Save a human-readable session report."""
        summary = self.get_session_summary()
        insights = self.get_performance_insights()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("           RETRIEVAL SESSION REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Queries: {summary['total_queries']}\n")
                f.write(f"Avg Response Time: {summary['avg_response_time']:.3f}s\n")
                f.write(f"Avg Results/Query: {summary['avg_results_per_query']:.1f}\n")
                f.write(f"Cache Hit Rate: {summary['cache_hit_rate']}\n\n")
                
                if summary['component_avg_times']:
                    f.write("Component Average Times:\n")
                    for comp, time in summary['component_avg_times'].items():
                        f.write(f"  - {comp}: {time:.3f}s\n")
                    f.write("\n")
                
                if insights:
                    f.write("Performance Insights:\n")
                    for insight in insights:
                        f.write(f"  {insight}\n")
                
                f.write("\n" + "="*60 + "\n")
            
            self.logger.info(f"Session report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session report: {e}")