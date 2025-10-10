import logging
import os
import yaml
import psutil
import torch
import time
import json
from datetime import datetime
from collections import defaultdict

def log_model_loading_metrics(logger, duration):
    """Enhanced model loading metrics."""
    system_metrics = _capture_system_metrics()
    
    logger.info(
        f"MODEL_LOADING - Duration: {duration:.2f}s, "
        f"CPU: {system_metrics['cpu']}%, "
        f"RAM: {system_metrics['memory']:.2f}GB, "
        f"GPU: {system_metrics['gpu']}"
    )
    
    # Also save structured data
    _save_structured_metrics("model_loading", {
        'duration': duration,
        'timestamp': datetime.now().isoformat(),
        **system_metrics
    })

def log_generation_metrics(logger, duration):
    """Enhanced text generation metrics."""
    system_metrics = _capture_system_metrics()
    
    logger.info(
        f"TEXT_GENERATION - Duration: {duration:.2f}s, "
        f"CPU: {system_metrics['cpu']}%, "
        f"RAM: {system_metrics['memory']:.2f}GB, "
        f"GPU: {system_metrics['gpu']}"
    )
    
    # Also save structured data
    _save_structured_metrics("text_generation", {
        'duration': duration,
        'timestamp': datetime.now().isoformat(),
        **system_metrics
    })

def log_retrieval_metrics(logger, duration, component_times, results_count, cache_hit=None):
    """NEW: Enhanced retrieval metrics."""
    system_metrics = _capture_system_metrics()
    
    # Detailed component breakdown
    components_str = ", ".join([f"{k}: {v:.3f}s" for k, v in component_times.items()])
    cache_str = f", Cache: {'HIT' if cache_hit else 'MISS'}" if cache_hit is not None else ""
    
    logger.info(
        f"RETRIEVAL - Duration: {duration:.3f}s, "
        f"Results: {results_count}, "
        f"Components: [{components_str}]{cache_str}, "
        f"CPU: {system_metrics['cpu']}%, "
        f"RAM: {system_metrics['memory']:.2f}GB, "
        f"GPU: {system_metrics['gpu']}"
    )
    
    # Save structured data
    _save_structured_metrics("retrieval", {
        'duration': duration,
        'results_count': results_count,
        'cache_hit': cache_hit,
        'component_times': component_times,
        'timestamp': datetime.now().isoformat(),
        **system_metrics
    })

def _capture_system_metrics():
    """Capture current system state."""
    cpu_percent = psutil.cpu_percent()
    memory_gb = psutil.virtual_memory().used / (1024**3)
    
    if torch.cuda.is_available():
        gpu_percent = (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100
        gpu_gb = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_str = f"{gpu_percent:.1f}% ({gpu_gb:.2f}GB)"
    else:
        gpu_percent = 0
        gpu_gb = 0
        gpu_str = "N/A"
    
    return {
        'cpu': cpu_percent,
        'memory': memory_gb,
        'gpu_percent': gpu_percent,
        'gpu_memory': gpu_gb,
        'gpu': gpu_str
    }

def _save_structured_metrics(operation_type, metrics_data):
    """Save structured metrics to JSON for analysis."""
    try:
        metrics_file = "logs/system_metrics.json"
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        # Load existing data
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {'sessions': []}
        
        # Add new metrics
        if not data['sessions'] or data['sessions'][-1].get('closed', True):
            # Start new session
            data['sessions'].append({
                'session_id': datetime.now().isoformat(),
                'operations': [],
                'closed': False
            })
        
        # Add operation to current session
        data['sessions'][-1]['operations'].append({
            'type': operation_type,
            **metrics_data
        })
        
        # Save back
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    except Exception as e:
        logging.getLogger("logger").warning(f"Failed to save structured metrics: {e}")

def close_metrics_session():
    """Mark current metrics session as closed."""
    try:
        metrics_file = "logs/system_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            if data['sessions'] and not data['sessions'][-1].get('closed', True):
                data['sessions'][-1]['closed'] = True
                data['sessions'][-1]['end_time'] = datetime.now().isoformat()
                
                with open(metrics_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
    except Exception as e:
        logging.getLogger("logger").warning(f"Failed to close metrics session: {e}")

def setup_logger(config_path="config/config.yaml"):
    """
    Setup centralized logger for the RAG system with flexible configuration.
    """
    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        log_config = config.get("logging", {})
    except Exception:
        # Fallback to default settings
        log_config = {}
    
    # Get logging settings
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_to_file = log_config.get("log_to_file", True)
    log_file = log_config.get("log_file", "logs/rag.log")
    log_format = log_config.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Create logs directory if needed
    if log_to_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure handlers with UTF-8 encoding
    handlers = []
    if log_to_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Clear any existing handlers and configure logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )

def get_logger(name):
    """
    Get logger instance for specific component.
    """
    return logging.getLogger(name)