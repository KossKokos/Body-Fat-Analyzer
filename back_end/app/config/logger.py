# app/config/logger.py
import logging
import sys
from logging.config import dictConfig
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

import structlog
from structlog.stdlib import BoundLogger


def setup_logging(
    log_level: str = "INFO",
    log_json: bool = False,
    log_file: str | None= None,
    module_filter: Dict[str, str] | None = None
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_json: Whether to output JSON logs
        log_file: Optional file path for file logging
        module_filter: Dict of module: level for specific modules
    """
    # Default module filters
    module_filter = module_filter or {
        "uvicorn": "WARNING",
        "fastapi": "WARNING",
        "tensorflow": "WARNING",
        "keras": "WARNING",
        "urllib3": "WARNING",
        "matplotlib": "WARNING",
    }
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_json else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "json" if log_json else "console",
            "stream": sys.stdout,
        }
    }
    
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json" if log_json else "console",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        }
    
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            },
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=True),
            },
        },
        "handlers": handlers,
        "loggers": {
            "": {  # Root logger
                "handlers": list(handlers.keys()),
                "level": log_level,
                "propagate": True,
            },
            **{module: {"level": level, "propagate": True} 
               for module, level in module_filter.items()}
        },
    }
    
    dictConfig(log_config)


class AppLogger:
    """Application logger wrapper."""
    
    def __init__(self, name: str | None = None):
        self._logger = structlog.get_logger(name or "app")
    
    def debug(self, msg: str, **kwargs) -> None:
        self._logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        self._logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        self._logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        self._logger.error(msg, **kwargs)
    
    def exception(self, msg: str, exc: Exception | None = None , **kwargs) -> None:
        """Log exception with context."""
        if exc:
            kwargs["exc_info"] = exc
        self._logger.error(msg, **kwargs)
    
    def with_context(self, **context) -> BoundLogger:
        """Return logger with additional context."""
        return self._logger.bind(**context)


# Global logger instance
logger = AppLogger()