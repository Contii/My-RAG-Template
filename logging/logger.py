import logging
import os
from datetime import datetime


def setup_logger(log_file="logs/rag.log", log_level=logging.INFO):
    """
    Setup centralized logger for the RAG system.
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

def get_logger(name):
    """
    Get logger instance for specific component.
    """
    return logging.getLogger(name)