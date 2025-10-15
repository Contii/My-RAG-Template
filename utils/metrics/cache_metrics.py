import json
import os
import logging
from datetime import datetime
from collections import defaultdict

"""Cache-specific metrics: hit/miss rates, performance tracking."""

class CacheMetrics:
    """Track and analyze cache performance metrics."""
    
    def __init__(self, metrics_file="logs/cache_metrics.json"):
        self.logger = logging.getLogger("cache_metrics")

        self.metrics_file = metrics_file
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.hit_times = []  # Time saved by cache hits
        self.miss_times = []  # Time spent on cache misses
        self.session_start = datetime.now()
        
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        self.logger.info("CacheMetrics initialized")
    
    def log_hit(self, query, time_saved=None):
        """Log a cache hit."""
        self.hits += 1
        self.total_requests += 1
        
        if time_saved is not None:
            self.hit_times.append(time_saved)
        
        self.logger.debug(f"Cache HIT - Query: {query[:50]}...")
        self._save_to_file()
    
    def log_miss(self, query, retrieval_time=None):
        """Log a cache miss."""
        self.misses += 1
        self.total_requests += 1
        
        if retrieval_time is not None:
            self.miss_times.append(retrieval_time)
        
        self.logger.debug(f"Cache MISS - Query: {query[:50]}...")
        self._save_to_file()
    
    def get_hit_rate(self):
        """Calculate cache hit rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    def get_summary(self):
        """Get summary statistics for cache performance."""
        hit_rate = self.get_hit_rate()
        
        summary = {
            'total_requests': self.total_requests,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': hit_rate,
            'session_start': self.session_start.isoformat()
        }
        
        # Add timing stats if available
        if self.hit_times:
            summary['avg_hit_time_saved'] = sum(self.hit_times) / len(self.hit_times)
            summary['total_time_saved'] = sum(self.hit_times)
        
        if self.miss_times:
            summary['avg_miss_time'] = sum(self.miss_times) / len(self.miss_times)
        
        self.logger.debug(f"Cache summary: {hit_rate:.1f}% hit rate ({self.hits}/{self.total_requests})")
        
        return summary
    
    def print_dashboard(self):
        """Print a formatted cache metrics dashboard."""
        summary = self.get_summary()
        
        self.logger.info("Displaying cache metrics dashboard")
        
        print("\n" + "="*50)
        print("           CACHE METRICS DASHBOARD")
        print("="*50)
        print(f"📊 Total Requests: {summary['total_requests']}")
        print(f"✅ Hits: {summary['hits']}")
        print(f"❌ Misses: {summary['misses']}")
        print(f"📈 Hit Rate: {summary['hit_rate_percent']:.1f}%")
        
        if 'total_time_saved' in summary:
            print(f"⚡ Total Time Saved: {summary['total_time_saved']:.2f}s")
        
        if 'avg_miss_time' in summary:
            print(f"⏱️  Avg Miss Time: {summary['avg_miss_time']:.3f}s")
        
        print("="*50)
    
    def get_performance_insights(self):
        """Generate cache performance insights."""
        summary = self.get_summary()
        insights = []
        
        hit_rate = summary['hit_rate_percent']
        
        if hit_rate > 70:
            insight = f"✅ Excellent cache hit rate ({hit_rate:.1f}%). Cache is highly effective."
            insights.append(insight)
            self.logger.info(insight)
        elif hit_rate > 40:
            insight = f"⚠️  Moderate cache hit rate ({hit_rate:.1f}%). Consider increasing TTL or cache size."
            insights.append(insight)
            self.logger.warning(insight)
        elif hit_rate > 0:
            insight = f"❌ Low cache hit rate ({hit_rate:.1f}%). Review caching strategy or query patterns."
            insights.append(insight)
            self.logger.warning(insight)
        else:
            insight = "❌ No cache hits. Cache may not be enabled or queries are too diverse."
            insights.append(insight)
            self.logger.warning(insight)
        
        # Time savings insight
        if 'total_time_saved' in summary and summary['total_time_saved'] > 0:
            saved = summary['total_time_saved']
            insight = f"⚡ Cache saved {saved:.2f}s total ({saved/summary['total_requests']:.3f}s per request avg)."
            insights.append(insight)
            self.logger.info(insight)
        
        return insights
    
    def _save_to_file(self):
        """Save cache metrics to JSON file."""
        try:
            data = self.get_summary()
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.debug(f"Cache metrics saved to {self.metrics_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save cache metrics: {e}")
    
    def reset(self):
        """Reset all cache metrics."""
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.hit_times.clear()
        self.miss_times.clear()
        self.session_start = datetime.now()
        
        self.logger.info("Cache metrics reset")