import logging
import sys

def setup_logging(log_level: str = "INFO") -> None:
    """Configures the root logger across the application."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        stream=sys.stderr
    )
