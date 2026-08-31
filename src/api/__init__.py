"""
src/api module
--------------
REST API package providing endpoints for external LMS integrations (Canvas, Moodle, etc.).
"""

try:
    from src.api.app import app

    __all__ = ["app"]
except ImportError:
    app = None
    __all__ = []
