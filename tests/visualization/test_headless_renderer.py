"""
tests/visualization/test_headless_renderer.py
----------------------------------------------
Unit tests verifying headless renderer configuration for CI pipelines (Issue #504).
"""

import os


def test_matplotlib_headless_backend_env():
    """Verify that MPLBACKEND environment variable is configured to Agg."""
    assert os.environ.get("MPLBACKEND") == "Agg"


def test_matplotlib_get_backend():
    """Verify that Matplotlib active backend is non-interactive Agg backend."""
    try:
        import matplotlib

        backend = matplotlib.get_backend()
        assert backend.lower() == "agg"
    except ImportError:
        pass
