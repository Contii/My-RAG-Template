import logging
import os
import yaml
import psutil
import torch

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

def log_model_loading_metrics(logger, duration):
    cpu_percent = psutil.cpu_percent()
    memory_gb = psutil.virtual_memory().used / (1024**3)
    
    if torch.cuda.is_available():
        gpu_percent = (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100
        gpu_gb = torch.cuda.memory_allocated(0) / (1024**3)
        logger.info(f"MODEL_LOADING - Duration: {duration:.2f}s, CPU: {cpu_percent}%, RAM: {memory_gb:.2f}GB, GPU: {gpu_percent:.1f}% ({gpu_gb:.2f}GB)")
    else:
        logger.info(f"MODEL_LOADING - Duration: {duration:.2f}s, CPU: {cpu_percent}%, RAM: {memory_gb:.2f}GB, GPU: N/A")

def log_generation_metrics(logger, duration):
    cpu_percent = psutil.cpu_percent()
    memory_gb = psutil.virtual_memory().used / (1024**3)
    
    if torch.cuda.is_available():
        gpu_percent = (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100
        gpu_gb = torch.cuda.memory_allocated(0) / (1024**3)
        logger.info(f"TEXT_GENERATION - Duration: {duration:.2f}s, CPU: {cpu_percent}%, RAM: {memory_gb:.2f}GB, GPU: {gpu_percent:.1f}% ({gpu_gb:.2f}GB)")
    else:
        logger.info(f"TEXT_GENERATION - Duration: {duration:.2f}s, CPU: {cpu_percent}%, RAM: {memory_gb:.2f}GB, GPU: N/A")