import psutil
import torch
import logging
from datetime import datetime

"""System-level metrics: CPU, RAM, GPU monitoring."""

class SystemMetrics:
    """Capture and track system resource usage."""
    
    def __init__(self):
        self.logger = logging.getLogger("system_metrics")

        self.history = []
        self.logger.info("SystemMetrics initialized")
    
    def capture(self):
        """Capture current system state."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.used / (1024**3)
        memory_percent = memory_info.percent
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_gb': memory_gb,
            'memory_percent': memory_percent,
            'memory_available_gb': memory_info.available / (1024**3)
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_percent = (gpu_memory_allocated / gpu_memory_total) * 100
                
                metrics.update({
                    'gpu_available': True,
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_allocated_gb': gpu_memory_allocated,
                    'gpu_memory_total_gb': gpu_memory_total,
                    'gpu_memory_percent': gpu_percent
                })
            except Exception as e:
                self.logger.warning(f"Error capturing GPU metrics: {e}")
                metrics['gpu_available'] = False
        else:
            metrics['gpu_available'] = False
        
        self.history.append(metrics)
        return metrics
    
    def get_summary(self):
        """Get summary of system metrics."""
        if not self.history:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.history]
        memory_values = [m['memory_gb'] for m in self.history]
        
        summary = {
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'cpu_min': min(cpu_values),
            'memory_avg_gb': sum(memory_values) / len(memory_values),
            'memory_max_gb': max(memory_values),
            'memory_min_gb': min(memory_values),
            'samples_count': len(self.history)
        }
        
        # GPU summary if available
        if self.history[0].get('gpu_available'):
            gpu_values = [m['gpu_memory_percent'] for m in self.history if m.get('gpu_memory_percent')]
            if gpu_values:
                summary.update({
                    'gpu_name': self.history[0]['gpu_name'],
                    'gpu_memory_avg_percent': sum(gpu_values) / len(gpu_values),
                    'gpu_memory_max_percent': max(gpu_values),
                })
        
        return summary
    
    def format_current_state(self):
        """Format current metrics for logging."""
        if not self.history:
            return "No metrics captured"
        
        current = self.history[-1]
        
        msg = (f"CPU: {current['cpu_percent']:.1f}%, "
               f"RAM: {current['memory_gb']:.2f}GB ({current['memory_percent']:.1f}%)")
        
        if current.get('gpu_available'):
            msg += f", GPU: {current['gpu_memory_percent']:.1f}% ({current['gpu_memory_allocated_gb']:.2f}GB)"
        
        return msg
    
    def clear_history(self):
        """Clear metrics history."""
        self.history.clear()
        self.logger.info("System metrics history cleared")