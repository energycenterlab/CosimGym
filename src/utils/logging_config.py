"""
logging_config.py

Centralized logging system for managing logs across federated HELICS simulation processes.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
import threading
from typing import Optional





class FederationLogger:
    """
    Centralized logging system for HELICS federation processes.
    
    Creates structured logs with:
    - Separate log files for each process (broker/federate)
    - Timestamped scenario runs
    - Configurable log levels
    - Process-safe logging
    """
    
    def __init__(self, scenario_name: str, log_base_dir: str = "logs"):
        self.scenario_name = scenario_name
        self.log_base_dir = Path(log_base_dir)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.scenario_log_dir = self.log_base_dir / scenario_name / self.run_timestamp
        self.simulation_id = f'{scenario_name}_{self.run_timestamp}'
        self.scenario_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.scenario_log_dir / "brokers").mkdir(exist_ok=True)
        (self.scenario_log_dir / "federates").mkdir(exist_ok=True)
        (self.scenario_log_dir / "manager").mkdir(exist_ok=True)
        
        self._loggers = {}
        self._lock = threading.Lock()
        
    def setup_manager_logger(self, level=logging.INFO) -> logging.Logger:
        """Setup the main federation manager logger."""
        logger_name = "federation_manager"
        log_file = self.scenario_log_dir / "manager" / "federation_manager.log"
        
        return self._create_logger(
            logger_name=logger_name,
            log_file=log_file,
            log_level=level
        )
    
    def get_broker_logger(self, broker_name: str, federation_name: str) -> logging.Logger:
        """Get or create a logger for a broker process."""
        logger_name = f"{broker_name}_{federation_name}"
        log_file = self.scenario_log_dir / "brokers" / f"{logger_name}.log"
        
        return self._create_logger(
            logger_name=logger_name,
            log_file=log_file,
            log_level=logging.DEBUG
        )
    
    def get_federate_logger(self, federate_name: str, federate_type: str) -> logging.Logger:
        """Get or create a logger for a federate process."""
        logger_name = f"federate_{federate_name}_{federate_type}"
        log_file = self.scenario_log_dir / "federates" / f"{logger_name}.log"
        
        return self._create_logger(
            logger_name=logger_name,
            log_file=log_file,
            log_level=logging.DEBUG
        )
    
    def _create_logger(self, logger_name: str, log_file: Path, log_level) -> logging.Logger:
        """Create a configured logger instance, or update its level if it already exists."""
        # Normalize log_level to int: accepts int, plain string ('DEBUG'), or LogLevel enum
        if not isinstance(log_level, int):
            log_level = logging.getLevelName(str(log_level.value if hasattr(log_level, 'value') else log_level))
        with self._lock:
            if logger_name in self._loggers:
                # Update the level in case it was created with a different (e.g. default) level
                cached = self._loggers[logger_name]
                cached.setLevel(log_level)
                for handler in cached.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.setLevel(log_level)
                return cached
            
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
            
            # Avoid duplicate handlers
            if not logger.handlers:
                # File handler
                file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                file_handler.setLevel(log_level)
                
                # Console handler (optional, for debugging)
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.WARNING)
                
                # Formatter
                formatter = logging.Formatter(
                    '%(asctime)s |  %(levelname)8s | PID:%(process)d | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                
                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)
                
                logger.addHandler(file_handler)
                logger.addHandler(console_handler)
            
            self._loggers[logger_name] = logger
            return logger
    
    def create_run_summary(self, start_time: datetime, end_time: datetime, 
                          status: str, error_msg: Optional[str] = None):
        """Create a summary file for this run."""
        summary = {
            "scenario_name": self.scenario_name,
            "run_timestamp": self.run_timestamp,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "status": status,
            "error_message": error_msg,
            "log_directory": str(self.scenario_log_dir)
        }
        
        summary_file = self.scenario_log_dir / "run_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_log_paths(self) -> dict:
        """Get all log file paths for this run."""
        return {
            "scenario_log_dir": str(self.scenario_log_dir),
            "manager_logs": str(self.scenario_log_dir / "manager"),
            "broker_logs": str(self.scenario_log_dir / "brokers"),
            "federate_logs": str(self.scenario_log_dir / "federates")
        }


def setup_process_logger(process_name: str, process_type: str, 
                        log_file_path: str, log_level: str = "DEBUG") -> logging.Logger:
    """
    Setup logger for individual processes (brokers/federates).
    This function is called within each subprocess.
    """
    logger = logging.getLogger(f"{process_type}_{process_name}")
    
    # Convert string level to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map.get(log_level.upper(), logging.DEBUG))
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | PID:%(process)d | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger