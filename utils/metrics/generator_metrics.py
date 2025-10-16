import json
import os
import logging
from datetime import datetime
from collections import defaultdict

"""Generator-specific metrics: LLM performance, token usage, generation timing."""

class GeneratorMetrics:
    """Track and analyze LLM generation performance metrics."""
    
    def __init__(self, metrics_file="logs/generator_metrics.json"):
        self.logger = logging.getLogger("generator_metrics")

        self.metrics_file = metrics_file
        self.generation_data = {}
        self.generation_counter = 0
        self.session_start = datetime.now()
        
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        self.logger.info("GeneratorMetrics initialized")
    
    def start_generation(self, context_length, question_length):
        """Start tracking a new generation."""
        self.generation_counter += 1
        gen_id = f"gen_{self.generation_counter}"
        
        self.generation_data[gen_id] = {
            'id': gen_id,
            'start_time': datetime.now().isoformat(),
            'context_length': context_length,
            'question_length': question_length,
            'input_length': context_length + question_length
        }
        
        self.logger.debug(f"Started tracking generation {gen_id}")
        return gen_id
    
    def finish_generation(self, gen_id, output_length, duration, num_tokens=None):
        """Finish tracking a generation."""
        if gen_id in self.generation_data:
            gen = self.generation_data[gen_id]
            gen['end_time'] = datetime.now().isoformat()
            gen['output_length'] = output_length
            gen['duration'] = duration
            gen['tokens_per_second'] = num_tokens / duration if num_tokens and duration > 0 else None
            gen['num_tokens'] = num_tokens
            
            self.logger.info(
                f"Generation {gen_id} completed - "
                f"Duration: {duration:.2f}s, "
                f"Output: {output_length} chars"
                + (f", Speed: {gen['tokens_per_second']:.1f} tok/s" if gen['tokens_per_second'] else "")
            )
            
            self._save_to_file()
    
    def log_error(self, gen_id, error_message):
        """Log a generation error."""
        if gen_id in self.generation_data:
            self.generation_data[gen_id]['error'] = error_message
            self.generation_data[gen_id]['end_time'] = datetime.now().isoformat()
            
            self.logger.error(f"Generation {gen_id} failed: {error_message}")
            self._save_to_file()
    
    def get_summary(self):
        """Get summary statistics for generation performance."""
        if not self.generation_data:
            self.logger.debug("No generations to summarize")
            return {
                'total_generations': 0,
                'successful_generations': 0,
                'failed_generations': 0,
                'avg_duration': 0,
                'avg_output_length': 0,
                'avg_tokens_per_second': None
            }
        
        generations = list(self.generation_data.values())
        successful = [g for g in generations if 'error' not in g and 'duration' in g]
        failed = [g for g in generations if 'error' in g]
        
        total_count = len(generations)
        success_count = len(successful)
        fail_count = len(failed)
        
        summary = {
            'total_generations': total_count,
            'successful_generations': success_count,
            'failed_generations': fail_count,
            'success_rate': (success_count / total_count * 100) if total_count > 0 else 0
        }
        
        if successful:
            durations = [g['duration'] for g in successful]
            outputs = [g['output_length'] for g in successful]
            
            summary['avg_duration'] = sum(durations) / len(durations)
            summary['min_duration'] = min(durations)
            summary['max_duration'] = max(durations)
            summary['avg_output_length'] = sum(outputs) / len(outputs)
            
            # Token speed if available
            speeds = [g['tokens_per_second'] for g in successful if g.get('tokens_per_second')]
            if speeds:
                summary['avg_tokens_per_second'] = sum(speeds) / len(speeds)
        
        self.logger.debug(f"Generator summary: {success_count}/{total_count} successful")
        
        return summary
    
    def print_dashboard(self):
        """Print a formatted generator metrics dashboard."""
        summary = self.get_summary()
        
        self.logger.info("Displaying generator metrics dashboard")
        
        print("\n" + "="*50)
        print("         GENERATOR METRICS DASHBOARD")
        print("="*50)
        print(f"📊 Total Generations: {summary['total_generations']}")
        print(f"✅ Successful: {summary['successful_generations']}")
        print(f"❌ Failed: {summary['failed_generations']}")
        print(f"📈 Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        if summary.get('avg_duration'):
            print(f"\n⏱️  Avg Duration: {summary['avg_duration']:.2f}s")
            print(f"📏 Avg Output Length: {summary['avg_output_length']:.0f} chars")
            
            if summary.get('avg_tokens_per_second'):
                print(f"⚡ Avg Speed: {summary['avg_tokens_per_second']:.1f} tokens/s")
        
        print("="*50)
    
    def get_performance_insights(self):
        """Generate generator performance insights."""
        summary = self.get_summary()
        insights = []
        
        # Success rate insights
        success_rate = summary.get('success_rate', 0)
        if success_rate == 100:
            insight = "✅ Perfect generation success rate (100%)."
            insights.append(insight)
            self.logger.info(insight)
        elif success_rate >= 90:
            insight = f"✅ High generation success rate ({success_rate:.1f}%)."
            insights.append(insight)
            self.logger.info(insight)
        elif success_rate >= 70:
            insight = f"⚠️  Moderate generation success rate ({success_rate:.1f}%). Check error logs."
            insights.append(insight)
            self.logger.warning(insight)
        else:
            insight = f"❌ Low generation success rate ({success_rate:.1f}%). Review configuration."
            insights.append(insight)
            self.logger.warning(insight)
        
        # Duration insights
        avg_duration = summary.get('avg_duration')
        if avg_duration:
            if avg_duration > 10:
                insight = f"⚠️  Slow generation speed ({avg_duration:.2f}s avg). Consider model optimization."
                insights.append(insight)
                self.logger.warning(insight)
            elif avg_duration < 2:
                insight = f"✅ Fast generation speed ({avg_duration:.2f}s avg)."
                insights.append(insight)
                self.logger.info(insight)
        
        # Token speed insights
        token_speed = summary.get('avg_tokens_per_second')
        if token_speed:
            if token_speed < 10:
                insight = f"⚠️  Low token generation speed ({token_speed:.1f} tok/s). Check GPU utilization."
                insights.append(insight)
                self.logger.warning(insight)
            elif token_speed > 50:
                insight = f"✅ Excellent token generation speed ({token_speed:.1f} tok/s)."
                insights.append(insight)
                self.logger.info(insight)
        
        return insights
    
    def _save_to_file(self):
        """Save generator metrics to JSON file."""
        try:
            data = {
                'session_start': self.session_start.isoformat(),
                'last_update': datetime.now().isoformat(),
                'generations': list(self.generation_data.values())
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.debug(f"Generator metrics saved to {self.metrics_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save generator metrics: {e}")
    
    def clear_history(self):
        """Clear generation history."""
        self.generation_data.clear()
        self.generation_counter = 0
        self.session_start = datetime.now()
        
        self.logger.info("Generator metrics history cleared")