# app/utils/logger.py
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

from app.config.logger import logger


@contextmanager
def log_execution_time(operation: str, level: str = "info", **context):
    """
    Context manager to log execution time.
    
    Usage:
        with log_execution_time("data_processing", user_id=123):
            # your code
    """
    start_time = time.time()
    log_method = getattr(logger, level)
    log_method(f"Starting: {operation}", **context)
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        log_method(
            f"Completed: {operation}",
            duration_seconds=round(elapsed, 3),
            **context
        )

def log_function_call(log_level: str = "debug"):
    """
    Decorator to log function calls.
    
    Usage:
        @log_function_call("info")
        def process_data(data):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            log_method = getattr(logger, log_level)
            
            # Log call
            log_method(
                f"Calling {func.__name__}",
                module=func.__module__,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            # Execute and time
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                log_method(
                    f"Completed {func.__name__}",
                    duration_seconds=round(elapsed, 3),
                    module=func.__module__
                )
                return result
                
            except Exception as e:
                elapsed = time.time() - start
                logger.exception(
                    f"Failed {func.__name__}",
                    error=str(e),
                    duration_seconds=round(elapsed, 3),
                    module=func.__module__
                )
                raise
        
        return wrapper
    return decorator


class LoggerMixin:
    """Mixin to add logging to classes."""
    
    @property
    def logger(self):
        module = self.__class__.__module__
        name = self.__class__.__name__
        return logger.with_context(module=module, class_name=name)