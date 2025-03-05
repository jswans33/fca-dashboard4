"""
Fallback logger module for when loguru is not available.

This module provides a simple logging implementation that can be used
when the loguru package is not installed.
"""
import logging
import sys
import os
from datetime import datetime

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

def get_logger(name):
    """Get a logger with the given name."""
    return logging.getLogger(name)

def configure_logging(level="INFO"):
    """Configure logging with the given level."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.getLogger().setLevel(numeric_level)
    
    # Add a file handler if not already present
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Check if we already have a FileHandler
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
        logging.getLogger().addHandler(file_handler)