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

"""Adversarial Text Watermark Domain Model.

Defines data classes for statistical z-score watermark tests, green/red list token entropy,
AI-generated text fingerprinting, and audit report summaries.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class WatermarkTokenDistribution:
    """Represents token frequency breakdown across green/red lists."""

    total_tokens_analyzed: int
    green_list_tokens_count: int
    red_list_tokens_count: int
    observed_green_ratio: float  # Range: 0.0 - 1.0
    expected_green_ratio: float  # Default: 0.50 (for gamma = 0.5)
    entropy_score: float


@dataclass
class WatermarkDetectionMatch:
    """Represents a statistical watermark detection result for an analyzed text snippet."""

    detection_id: str
    document_id: str
    document_title: str
    z_score: float  # Standardized statistical test score
    p_value: float  # Probability value for hypothesis testing
    watermark_confidence_percentage: float  # Range: 0.0 - 100.0%
    is_watermark_present: bool
    model_generator_signature: str  # e.g., 'Kirchenbauer-Logits', 'AAR-Entropy', 'None'
    token_distribution: WatermarkTokenDistribution
    analyzed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WatermarkAuditReport:
    """Audit report summary aggregating adversarial watermark detection queries."""

    report_id: str
    total_documents_analyzed: int
    watermarked_documents_count: int
    average_z_score: float
    highest_confidence_score: float
    report_generated_at: datetime = field(default_factory=datetime.utcnow)
    detections: List[WatermarkDetectionMatch] = field(default_factory=list)
