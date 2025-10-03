import logging
import os
import yaml
import time
import psutil
from datetime import datetime

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
    
    # Configure handlers
    handlers = []
    if log_to_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
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

def log_performance_metrics(logger, operation, start_time, end_time):
    """
    Log performance metrics for operations.
    """
    duration = end_time - start_time
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    
    logger.info(f"PERFORMANCE [{operation}] - Duration: {duration:.2f}s, CPU: {cpu_percent}%, Memory: {memory_info.percent}%")

def log_system_info(logger):
    """
    Log system information for monitoring.
    """
    cpu_count = psutil.cpu_count()
    memory_total = psutil.virtual_memory().total / (1024**3)  # GB
    disk_usage = psutil.disk_usage('/').percent
    
    logger.info(f"SYSTEM_INFO - CPUs: {cpu_count}, Memory: {memory_total:.2f}GB, Disk: {disk_usage}%")