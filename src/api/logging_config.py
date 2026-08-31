"""
Logging Configuration for FastAPI Application

This module provides consolidated logging configuration to prevent duplicate
exception logging and ensure clean log output with proper trace IDs.
"""

import logging
import sys
import uuid
from typing import Optional, Dict, Any
from contextvars import ContextVar
from datetime import datetime
import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import FastAPI
import uvicorn

# Context variable for request ID propagation
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to inject and propagate request IDs (Trace IDs) through the request lifecycle.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and inject request ID.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or handler
        """
        # Generate or get request ID
        request_id = request.headers.get('X-Request-ID')
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Set in context
        token = request_id_var.set(request_id)
        
        # Add to request state for access in handlers
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            # Add request ID to response headers
            response.headers['X-Request-ID'] = request_id
            return response
        finally:
            # Reset context
            request_id_var.reset(token)


class StructuredLogFormatter(logging.Formatter):
    """
    Custom formatter that adds request IDs and structures log entries.
    """
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, 
                 style: str = '%', json_format: bool = False):
        """
        Initialize structured formatter.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
            style: Format style ('%', '{', or '$')
            json_format: Whether to output JSON format
        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.json_format = json_format
        
        # Default format if not provided
        if not fmt:
            self.fmt = '%(asctime)s | %(levelname)-8s | [%(request_id)s] | %(name)-15s | %(message)s'
        else:
            self.fmt = fmt
    
    def format(self, record):
        """
        Format log record with request ID and structured output.
        
        Args:
            record: Log record
            
        Returns:
            str: Formatted log message
        """
        # Inject request ID from context
        request_id = request_id_var.get()
        if request_id:
            record.request_id = request_id
        else:
            record.request_id = 'NO_REQUEST_ID'
        
        # Format as JSON if requested
        if self.json_format:
            return self._format_json(record)
        
        # Use parent formatter
        return super().format(record)
    
    def _format_json(self, record):
        """
        Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            str: JSON-formatted log entry
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'request_id': getattr(record, 'request_id', 'NO_REQUEST_ID'),
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields if available
        if hasattr(record, 'extra'):
            log_entry['extra'] = record.extra
        
        return json.dumps(log_entry)


def configure_logging(
    level: str = 'INFO',
    json_format: bool = False,
    disable_uvicorn_access: bool = True,
    disable_uvicorn_error: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application logging with consolidated exception handling.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to output logs in JSON format
        disable_uvicorn_access: Whether to disable Uvicorn's access logger
        disable_uvicorn_error: Whether to disable Uvicorn's error logger
        log_file: Path to log file (optional)
    """
    # Convert string level to int
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = StructuredLogFormatter(json_format=json_format)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure Uvicorn loggers to prevent duplicates
    _configure_uvicorn_loggers(
        disable_access=disable_uvicorn_access,
        disable_error=disable_uvicorn_error,
        level=level
    )
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level {level.upper()}")
    if json_format:
        logger.info("Log format: JSON")
    if log_file:
        logger.info(f"Log file: {log_file}")


def _configure_uvicorn_loggers(
    disable_access: bool = True,
    disable_error: bool = True,
    level: str = 'INFO'
) -> None:
    """
    Configure Uvicorn loggers to prevent duplicate logging.
    
    Args:
        disable_access: Disable access logger
        disable_error: Disable error logger
        level: Log level
    """
    # Uvicorn's default loggers
    uvicorn_access = logging.getLogger('uvicorn.access')
    uvicorn_error = logging.getLogger('uvicorn.error')
    uvicorn_logger = logging.getLogger('uvicorn')
    
    # Set log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure Uvicorn error logger
    if disable_error:
        # Disable propagation to root to prevent duplicates
        uvicorn_error.propagate = False
        # Set high level to suppress most messages
        uvicorn_error.setLevel(logging.WARNING)
    else:
        uvicorn_error.setLevel(numeric_level)
        uvicorn_error.propagate = True
    
    # Configure Uvicorn access logger
    if disable_access:
        # Completely disable access logs
        uvicorn_access.disabled = True
        uvicorn_access.propagate = False
    else:
        uvicorn_access.setLevel(numeric_level)
        uvicorn_access.propagate = True
    
    # Configure Uvicorn main logger
    uvicorn_logger.setLevel(numeric_level)
    
    # Remove existing handlers from uvicorn loggers to avoid duplicates
    for logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error']:
        logger_obj = logging.getLogger(logger_name)
        for handler in logger_obj.handlers[:]:
            logger_obj.removeHandler(handler)
    
    # Add console handler to uvicorn error logger if not disabled
    if not disable_error:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter('%(levelname)s:    %(message)s')
        handler.setFormatter(formatter)
        uvicorn_error.addHandler(handler)


def get_logger(name: str, request_id: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with optional request ID.
    
    Args:
        name: Logger name
        request_id: Optional request ID to set in context
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if request_id:
        token = request_id_var.set(request_id)
    
    return logging.getLogger(name)


class ExceptionLogger:
    """
    Utility class for logging exceptions with proper context.
    """
    
    @staticmethod
    def log_error(
        logger: logging.Logger,
        error: Exception,
        message: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log an error with structured information.
        
        Args:
            logger: Logger instance
            error: Exception to log
            message: Optional custom message
            **kwargs: Additional context to include
        """
        error_message = message or str(error)
        error_type = error.__class__.__name__
        
        # Build context
        context = {
            'error_type': error_type,
            'error_message': error_message,
            **kwargs
        }
        
        # Get request ID from context
        request_id = request_id_var.get()
        if request_id:
            context['request_id'] = request_id
        
        # Log with exception info
        logger.error(
            f"{error_type}: {error_message}",
            exc_info=error,
            extra={'context': context}
        )
    
    @staticmethod
    def log_warning(
        logger: logging.Logger,
        message: str,
        **kwargs
    ) -> None:
        """
        Log a warning with structured information.
        
        Args:
            logger: Logger instance
            message: Warning message
            **kwargs: Additional context to include
        """
        context = {
            'message': message,
            **kwargs
        }
        
        request_id = request_id_var.get()
        if request_id:
            context['request_id'] = request_id
        
        logger.warning(message, extra={'context': context})
    
    @staticmethod
    def log_info(
        logger: logging.Logger,
        message: str,
        **kwargs
    ) -> None:
        """
        Log an info message with structured information.
        
        Args:
            logger: Logger instance
            message: Info message
            **kwargs: Additional context to include
        """
        context = {
            'message': message,
            **kwargs
        }
        
        request_id = request_id_var.get()
        if request_id:
            context['request_id'] = request_id
        
        logger.info(message, extra={'context': context})


# ===================== FASTAPI APP INTEGRATION =====================

def setup_logging_middleware(app: FastAPI) -> None:
    """
    Setup logging middleware for FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    # Add request ID middleware
    app.add_middleware(RequestIDMiddleware)
    
    # Configure logging for the app
    configure_logging(
        level='INFO',
        json_format=False,
        disable_uvicorn_access=True,
        disable_uvicorn_error=True
    )
    
    logger = get_logger(__name__)
    logger.info("Logging middleware configured")


# ===================== UPDATED EXCEPTION HANDLER =====================

def create_exception_handler(logger: logging.Logger, include_traceback: bool = False):
    """
    Create a custom exception handler that prevents duplicate logging.
    
    Args:
        logger: Logger instance
        include_traceback: Whether to include traceback in response
        
    Returns:
        function: Exception handler function
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import HTTPException
    
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        """
        Custom HTTP exception handler with consolidated logging.
        """
        # Get request ID from state
        request_id = getattr(request.state, 'request_id', 'NO_REQUEST_ID')
        
        # Prepare error details
        error_detail = {
            'status_code': exc.status_code,
            'error': exc.detail,
            'path': request.url.path,
            'method': request.method,
            'request_id': request_id
        }
        
        # Log once with structured format
        # ⭐ This prevents duplicate logging because we've disabled Uvicorn's default
        # error logger and we're controlling the output format
        logger.error(
            f"HTTP {exc.status_code}: {exc.detail}",
            extra={
                'context': error_detail,
                'request_id': request_id
            }
        )
        
        # Return JSON response
        return JSONResponse(
            status_code=exc.status_code,
            content=error_detail
        )
    
    return custom_http_exception_handler


# ===================== UNIT TESTS =====================

import unittest
from unittest.mock import Mock, patch, MagicMock
import io


class TestLoggingConfig(unittest.TestCase):
    """Unit tests for logging configuration."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = logging.getLogger('test_logger')
        self.logger.handlers.clear()
    
    def test_structured_log_formatter(self):
        """Test structured log formatter."""
        formatter = StructuredLogFormatter()
        
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        self.assertIsInstance(formatted, str)
        self.assertIn('Test message', formatted)
    
    def test_json_formatter(self):
        """Test JSON formatter."""
        formatter = StructuredLogFormatter(json_format=True)
        
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        self.assertIsInstance(formatted, str)
        
        # Should be valid JSON
        import json
        parsed = json.loads(formatted)
        self.assertIn('message', parsed)
        self.assertEqual(parsed['message'], 'Test message')
    
    def test_configure_logging(self):
        """Test logging configuration."""
        configure_logging(
            level='DEBUG',
            json_format=False,
            disable_uvicorn_access=True,
            disable_uvicorn_error=True
        )
        
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)
        self.assertGreater(len(root_logger.handlers), 0)
    
    def test_exception_logger(self):
        """Test ExceptionLogger utility."""
        logger = logging.getLogger('test')
        exception_logger = ExceptionLogger()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            with patch.object(logger, 'error') as mock_error:
                exception_logger.log_error(logger, e, "Custom message")
                mock_error.assert_called_once()
    
    def test_request_id_middleware(self):
        """Test RequestID middleware."""
        from starlette.requests import Request
        from starlette.responses import Response
        
        middleware = RequestIDMiddleware(Mock())
        
        async def call_next(request):
            return Response()
        
        request = Mock(spec=Request)
        request.headers = {}
        request.state = Mock()
        
        # Test dispatch
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Should not raise
        loop.run_until_complete(middleware.dispatch(request, call_next))
    
    def test_create_exception_handler(self):
        """Test exception handler creation."""
        logger = logging.getLogger('test')
        handler = create_exception_handler(logger)
        
        self.assertIsNotNone(handler)
        self.assertTrue(callable(handler))
    
    def test_get_logger_with_request_id(self):
        """Test getting logger with request ID."""
        logger = get_logger('test_logger', 'test-id-123')
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test_logger')


class TestUvicornLoggerConfiguration(unittest.TestCase):
    """Test Uvicorn logger configuration."""
    
    def test_disable_uvicorn_loggers(self):
        """Test disabling Uvicorn loggers."""
        configure_logging(
            disable_uvicorn_access=True,
            disable_uvicorn_error=True
        )
        
        # Access logger should be disabled
        access_logger = logging.getLogger('uvicorn.access')
        self.assertTrue(access_logger.disabled)
        
        # Error logger should not propagate
        error_logger = logging.getLogger('uvicorn.error')
        self.assertFalse(error_logger.propagate)
    
    def test_enable_uvicorn_loggers(self):
        """Test enabling Uvicorn loggers."""
        configure_logging(
            disable_uvicorn_access=False,
            disable_uvicorn_error=False
        )
        
        # Access logger should be enabled
        access_logger = logging.getLogger('uvicorn.access')
        self.assertFalse(access_logger.disabled)
        
        # Error logger should propagate
        error_logger = logging.getLogger('uvicorn.error')
        self.assertTrue(error_logger.propagate)


# ===================== INTEGRATION EXAMPLE =====================

def create_app_with_logging():
    """
    Example FastAPI application with consolidated logging.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    from fastapi import FastAPI, HTTPException, Request
    
    app = FastAPI(title="API with Consolidated Logging")
    
    # Setup logging middleware
    setup_logging_middleware(app)
    
    # Get logger instance
    logger = get_logger('app')
    
    # Create exception handler
    exception_handler = create_exception_handler(logger)
    
    # Register exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return await exception_handler(request, exc)
    
    @app.get('/')
    async def root():
        logger.info("Root endpoint accessed")
        return {"message": "Hello World"}
    
    @app.get('/error')
    async def error_endpoint():
        logger.warning("Error endpoint accessed")
        raise HTTPException(status_code=404, detail="Resource not found")
    
    @app.get('/exception')
    async def exception_endpoint():
        try:
            raise ValueError("Something went wrong")
        except Exception as e:
            ExceptionLogger.log_error(logger, e, "Custom error handler")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    return app


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
