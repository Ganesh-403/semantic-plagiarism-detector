"""
test_embedding_model_low_ram_fallback_issue_4013.py
---------------------------------------------------
Unit tests for Issue #4013: Fallback to lightweight model (all-MiniLM-L6-v2) on low-memory servers (< 1.5GB RAM).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.core.embedding_model import (
    _LOW_MEMORY_RAM_THRESHOLD_BYTES,
    _check_system_memory_for_model_fallback,
)


def test_fallback_when_available_ram_below_1_5gb(caplog):
    """Verify automatic fallback to all-MiniLM-L6-v2 with a warning log when RAM < 1.5GB."""
    low_ram_bytes = 1000 * 1024 * 1024  # 1.0 GB (< 1.5 GB)

    mock_vm = MagicMock()
    mock_vm.available = low_ram_bytes

    with patch("psutil.virtual_memory", return_value=mock_vm):
        with caplog.at_level(logging.WARNING):
            result = _check_system_memory_for_model_fallback("paraphrase-multilingual-MiniLM-L12-v2")

    assert result == "all-MiniLM-L6-v2"
    assert "Low available system RAM detected" in caplog.text
    assert "Falling back from 'paraphrase-multilingual-MiniLM-L12-v2' to lightweight embedding model 'all-MiniLM-L6-v2'" in caplog.text


def test_no_fallback_when_available_ram_above_1_5gb():
    """Verify primary model is retained when available RAM >= 1.5GB."""
    sufficient_ram_bytes = 4000 * 1024 * 1024  # 4.0 GB (>= 1.5 GB)

    mock_vm = MagicMock()
    mock_vm.available = sufficient_ram_bytes

    with patch("psutil.virtual_memory", return_value=mock_vm):
        result = _check_system_memory_for_model_fallback("paraphrase-multilingual-MiniLM-L12-v2")

    assert result == "paraphrase-multilingual-MiniLM-L12-v2"


def test_no_fallback_when_already_using_lightweight_model():
    """Verify that when target model is already all-MiniLM-L6-v2, no fallback log or change occurs."""
    low_ram_bytes = 500 * 1024 * 1024  # 500 MB

    mock_vm = MagicMock()
    mock_vm.available = low_ram_bytes

    with patch("psutil.virtual_memory", return_value=mock_vm):
        result = _check_system_memory_for_model_fallback("all-MiniLM-L6-v2")

    assert result == "all-MiniLM-L6-v2"
