import logging
import os
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR

def setup_logging(case_id=None):
    """Setup logging for both Python and OpenFOAM outputs"""
    
    # Create logs directory
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if case_id is not None:
        log_file = log_dir / f"case_{case_id}_{timestamp}.log"
    else:
        log_file = log_dir / f"main_{timestamp}.log"
    
    # Setup Python logger
    logger = logging.getLogger(f"hayabusa_engine_{case_id}" if case_id else "hayabusa_engine")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler (show progress info)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

def get_openfoam_log_path(case_id, operation):
    """Get path for OpenFOAM log files"""
    log_dir = RESULTS_DIR / "logs" / "openfoam"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"case_{case_id}_{operation}_{timestamp}.log"
