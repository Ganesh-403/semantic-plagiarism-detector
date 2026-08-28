"""src/api/endpoints - FastAPI endpoint packages for modular API routes."""

from src.api.endpoints.ai_watermark import router as ai_watermark_router

__all__ = ["ai_watermark_router"]
