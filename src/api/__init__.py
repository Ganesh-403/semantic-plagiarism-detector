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


"""
API module for the Semantic Plagiarism Detector
"""

from .upload import router as upload_router
from .routes import router as documents_router

__all__ = ['upload_router', 'documents_router']