# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import logging

from src.core.logging_setup import JSONFormatter, setup_logging


def test_json_formatter_outputs_valid_json():
    """Verify that JSONFormatter formats log records as JSON with standard fields."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=10,
        msg="Test message with placeholder: %s",
        args=("val",),
        exc_info=None,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "timestamp" in data
    assert data["level"] == "INFO"
    assert data["name"] == "test_logger"
    assert data["message"] == "Test message with placeholder: val"


def test_json_formatter_includes_exception():
    """Verify that JSONFormatter includes formatting for exceptions when present."""
    formatter = JSONFormatter()
    try:
        raise ValueError("Boom exception")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test_file.py",
        lineno=20,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "exception" in data
    assert "ValueError: Boom exception" in data["exception"]


def test_setup_logging_development(monkeypatch):
    """Verify setup_logging configures root logger with text format in dev."""
    monkeypatch.setenv("APP_ENVIRONMENT", "development")
    setup_logging(log_level="DEBUG")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) == 1
    # Check that formatter is standard Formatter (text)
    handler = root_logger.handlers[0]
    assert not isinstance(handler.formatter, JSONFormatter)


def test_setup_logging_production(monkeypatch):
    """Verify setup_logging configures root logger with JSONFormatter in production."""
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    setup_logging(log_level="WARNING")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.WARNING
    assert len(root_logger.handlers) == 1
    # Check that formatter is JSONFormatter
    handler = root_logger.handlers[0]
    assert isinstance(handler.formatter, JSONFormatter)
