"""
Logging configuration for CloudWatch compatibility.
"""
import logging
from pythonjsonlogger import jsonlogger  # type: ignore


def setup_logger(name: str = "chronos_api") -> logging.Logger:
    """Configure JSON structured logging for CloudWatch compatibility."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # JSON formatter with custom fields
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(model_path)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
