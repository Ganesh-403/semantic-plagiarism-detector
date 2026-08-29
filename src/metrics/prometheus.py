"""
src/metrics/prometheus.py
-------------------------
Prometheus metrics definitions for the Semantic Plagiarism Detector.
"""

from __future__ import annotations

from prometheus_client import Counter

# Define Counter for authentication failures with reason label
spd_auth_failures_total = Counter(
    "spd_auth_failures_total",
    "Total authentication failures",
    ["reason"],
)
