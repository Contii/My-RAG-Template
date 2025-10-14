import json
import os
from datetime import datetime
from logger.logger import get_logger
from .system_metrics import SystemMetrics
from .retrieval_metrics import RetrievalMetrics
from .cache_metrics import CacheMetrics
from .generator_metrics import GeneratorMetrics

logger = get_logger("unified_metrics")

"""Unified metrics coordinator - consolidates all metric systems."""

class MetricsCollector:
    """Central coordinator for all RAG system metrics."""
    
    def __init__(self):
        self.system_monitor = SystemMetrics()
        self.retrieval_tracker = RetrievalMetrics()
        self.cache_tracker = CacheMetrics()
        self.generator_tracker = GeneratorMetrics()
        self.session_start = datetime.now()
        
        logger.info("MetricsCollector initialized - all metrics systems ready")
    
    def get_unified_dashboard(self):
        """Get consolidated dashboard with all metrics."""
        system_summary = self.system_monitor.get_summary()
        retrieval_summary = self.retrieval_tracker.get_session_summary()
        cache_summary = self.cache_tracker.get_summary()
        generator_summary = self.generator_tracker.get_summary()
        
        return {
            'session_start': self.session_start.isoformat(),
            'timestamp': datetime.now().isoformat(),
            'system': system_summary,
            'retrieval': retrieval_summary,
            'cache': cache_summary,
            'generator': generator_summary
        }
    
    def print_unified_dashboard(self):
        """Print comprehensive dashboard with all metrics."""
        logger.info("Displaying unified metrics dashboard")
        
        print("\n" + "="*60)
        print("              UNIFIED RAG METRICS DASHBOARD")
        print("="*60)
        print(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # System metrics
        system_summary = self.system_monitor.get_summary()
        if system_summary:
            print("\n🖥️  SYSTEM RESOURCES:")
            print(f"   CPU: Avg {system_summary.get('cpu_avg', 0):.1f}%, Max {system_summary.get('cpu_max', 0):.1f}%")
            print(f"   RAM: Avg {system_summary.get('memory_avg_gb', 0):.2f}GB, Max {system_summary.get('memory_max_gb', 0):.2f}GB")
            
            if system_summary.get('gpu_name'):
                print(f"   GPU: {system_summary['gpu_name']}")
                print(f"        Avg {system_summary.get('gpu_memory_avg_percent', 0):.1f}%, Max {system_summary.get('gpu_memory_max_percent', 0):.1f}%")
        
        # Retrieval metrics
        retrieval_summary = self.retrieval_tracker.get_session_summary()
        print(f"\n🔍 RETRIEVAL:")
        print(f"   Total Queries: {retrieval_summary['total_queries']}")
        print(f"   Avg Response Time: {retrieval_summary['avg_response_time']:.3f}s")
        print(f"   Avg Results/Query: {retrieval_summary['avg_results_per_query']:.1f}")
        
        if retrieval_summary['component_avg_times']:
            print(f"   Component Times:")
            for comp, time in retrieval_summary['component_avg_times'].items():
                print(f"      {comp}: {time:.3f}s")
        
        # Cache metrics
        cache_summary = self.cache_tracker.get_summary()
        print(f"\n💾 CACHE:")
        print(f"   Total Requests: {cache_summary['total_requests']}")
        print(f"   Hit Rate: {cache_summary['hit_rate_percent']:.1f}%")
        print(f"   Hits: {cache_summary['hits']}, Misses: {cache_summary['misses']}")
        
        if cache_summary.get('total_time_saved'):
            print(f"   Time Saved: {cache_summary['total_time_saved']:.2f}s")
        
        # Generator metrics
        generator_summary = self.generator_tracker.get_summary()
        print(f"\n🤖 GENERATOR:")
        print(f"   Total Generations: {generator_summary['total_generations']}")
        print(f"   Successful: {generator_summary['successful_generations']}")
        print(f"   Failed: {generator_summary['failed_generations']}")
        
        if generator_summary.get('avg_duration'):
            print(f"   Avg Duration: {generator_summary['avg_duration']:.2f}s")
            print(f"   Avg Output Length: {generator_summary['avg_output_length']:.0f} chars")
            
            if generator_summary.get('avg_tokens_per_second'):
                print(f"   Avg Speed: {generator_summary['avg_tokens_per_second']:.1f} tokens/s")
        
        print("\n" + "="*60)
    
    def get_all_insights(self):
        """Get performance insights from all metrics systems."""
        insights = {
            'retrieval': self.retrieval_tracker.get_performance_insights(),
            'cache': self.cache_tracker.get_performance_insights(),
            'generator': self.generator_tracker.get_performance_insights()
        }
        
        return insights
    
    def print_all_insights(self):
        """Print all performance insights."""
        logger.info("Displaying performance insights")
        
        insights = self.get_all_insights()
        
        print("\n" + "="*60)
        print("            PERFORMANCE INSIGHTS")
        print("="*60)
        
        if insights['retrieval']:
            print("\n🔍 RETRIEVAL:")
            for insight in insights['retrieval']:
                print(f"   {insight}")
        
        if insights['cache']:
            print("\n💾 CACHE:")
            for insight in insights['cache']:
                print(f"   {insight}")
        
        if insights['generator']:
            print("\n🤖 GENERATOR:")
            for insight in insights['generator']:
                print(f"   {insight}")
        
        print("\n" + "="*60)
    
    def save_unified_report(self, filepath="logs/unified_report.json"):
        """Save comprehensive unified report to JSON."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            report = {
                'session_start': self.session_start.isoformat(),
                'report_generated': datetime.now().isoformat(),
                'dashboard': self.get_unified_dashboard(),
                'insights': self.get_all_insights()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Unified report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save unified report: {e}")
    
    def clear_all_metrics(self):
        """Clear all metrics history."""
        self.system_monitor.clear_history()
        self.cache_tracker.reset()
        self.generator_tracker.clear_history()
        # Note: retrieval_metrics doesn't have clear method yet
        
        self.session_start = datetime.now()
        logger.info("All metrics cleared")