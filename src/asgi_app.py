"""ASGI entry point for the Streamlit dashboard.

This wraps the Streamlit UI script (app/streamlit_app.py) so we can attach
Starlette middleware. It's needed for one reason: adding an
X-Frame-Options: DENY header to every HTTP response, to prevent this app
from being embedded in an <iframe> on another site (clickjacking).

Streamlit's own .streamlit/config.toml has no setting for custom HTTP
response headers, so this ASGI-level middleware is the officially
supported way to add one (see Streamlit's "Advanced server configuration
with st.App" documentation).

Run with:
    streamlit run asgi_app.py
"""

import streamlit as st
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        return response


app = st.App(
    "app/streamlit_app.py",
    middleware=[Middleware(SecurityHeadersMiddleware)],
)
