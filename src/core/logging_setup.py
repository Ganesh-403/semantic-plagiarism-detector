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

"""src/core/logging_setup.py - Structured logging setup with dictConfig supporting text and JSON output."""

import json
import logging
import logging.config
import os
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs in structured JSON format for aggregation systems (ELK/Datadog)."""

    def format(self, record: logging.LogRecord) -> str:
        # Build the structured JSON log entry
        log_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include exception information if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include stack trace info if present
        if record.stack_info:
            log_data["stack_trace"] = self.formatStack(record.stack_info)

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging using dictConfig.

    Outputs plain text format in development environment, and structured JSON format in production.
    """
    is_production = os.getenv("APP_ENVIRONMENT", "production").lower() == "production"
    level = log_level.upper()

    formatters = {
        "text": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "src.core.logging_setup.JSONFormatter",
        },
    }

    formatter_to_use = "json" if is_production else "text"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": formatter_to_use,
                "stream": "ext://sys.stderr",
            }
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)
