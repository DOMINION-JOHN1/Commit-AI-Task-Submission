"""
Production-Grade Logging Module

Structured logging with context, performance tracking, and multiple outputs.
Uses structlog for machine-parseable logs in production.
"""

import logging
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional
from contextlib import contextmanager

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

from .config import get_config


class RAGLogger:
    """
    Production-grade logger with structured logging support.
    
    Features:
    - Structured JSON logging for production
    - Human-readable console output for development
    - Automatic context injection (request_id, timestamp)
    - Performance timing decorators
    - Error tracking with stack traces
    """
    
    def __init__(self, name: str = "scientific_rag"):
        self.name = name
        self.config = get_config().logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging handlers and formatters."""
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (structured JSON for production)
        log_path = Path(self.config.log_dir) / self.config.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        
        if self.config.json_format and STRUCTLOG_AVAILABLE:
            # JSON format for machine parsing
            file_format = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            file_format = console_format
        
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def _format_extra(self, extra: Optional[dict] = None) -> str:
        """Format extra context for logging."""
        if not extra:
            return ""
        return " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with optional context."""
        self.logger.info(f"{message}{self._format_extra(kwargs)}")
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with optional context."""
        self.logger.debug(f"{message}{self._format_extra(kwargs)}")
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with optional context."""
        self.logger.warning(f"{message}{self._format_extra(kwargs)}")
    
    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message with optional exception info."""
        self.logger.error(
            f"{message}{self._format_extra(kwargs)}",
            exc_info=exc_info
        )
    
    def critical(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log critical error with stack trace."""
        self.logger.critical(
            f"{message}{self._format_extra(kwargs)}",
            exc_info=exc_info
        )
    
    @contextmanager
    def timed_operation(self, operation_name: str, **context: Any):
        """
        Context manager for timing operations.
        
        Usage:
            with logger.timed_operation("vector_search", query_length=100):
                results = vector_db.search(query)
        """
        start_time = time.perf_counter()
        self.debug(f"Starting {operation_name}", **context)
        
        try:
            yield
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.error(
                f"Failed {operation_name}",
                duration_ms=round(elapsed * 1000, 2),
                error=str(e),
                exc_info=True,
                **context
            )
            raise
        else:
            elapsed = time.perf_counter() - start_time
            self.info(
                f"Completed {operation_name}",
                duration_ms=round(elapsed * 1000, 2),
                **context
            )
    
    def log_function_call(self, func: Callable) -> Callable:
        """
        Decorator to log function entry, exit, and timing.
        
        Usage:
            @logger.log_function_call
            def my_function(arg1, arg2):
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            start_time = time.perf_counter()
            
            self.debug(f"Calling {func_name}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                self.debug(
                    f"Returned from {func_name}",
                    duration_ms=round(elapsed * 1000, 2)
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                self.error(
                    f"Exception in {func_name}",
                    duration_ms=round(elapsed * 1000, 2),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True
                )
                raise
        
        return wrapper


# Module-level logger instance
_logger: Optional[RAGLogger] = None


def get_logger(name: str = "scientific_rag") -> RAGLogger:
    """Get or create the module logger."""
    global _logger
    if _logger is None:
        _logger = RAGLogger(name)
    return _logger


# Convenience functions for direct imports
def log_info(message: str, **kwargs: Any) -> None:
    get_logger().info(message, **kwargs)


def log_debug(message: str, **kwargs: Any) -> None:
    get_logger().debug(message, **kwargs)


def log_warning(message: str, **kwargs: Any) -> None:
    get_logger().warning(message, **kwargs)


def log_error(message: str, exc_info: bool = False, **kwargs: Any) -> None:
    get_logger().error(message, exc_info=exc_info, **kwargs)
