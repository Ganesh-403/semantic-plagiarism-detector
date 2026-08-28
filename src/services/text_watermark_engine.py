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

"""Adversarial Text Watermark Detector Engine.

Implements statistical hypothesis testing (Kirchenbauer et al. z-score), token entropy
green/red list partitioning, and AI text fingerprinting detection.
"""

import math
import uuid
from datetime import datetime

from src.models.text_watermark_model import (
    WatermarkDetectionMatch,
    WatermarkTokenDistribution,
)


class AdversarialWatermarkEngine:
    """Core analytics engine for detecting statistical LLM text watermarks."""

    @staticmethod
    def calculate_z_score(
        green_count: int, total_count: int, gamma: float = 0.5
    ) -> float:
        """Calculates Kirchenbauer z-score test statistic for watermark detection.

        Formula: z = (S_g - gamma * N) / sqrt(gamma * (1 - gamma) * N)
        """
        if total_count <= 0 or gamma <= 0.0 or gamma >= 1.0:
            return 0.0

        expected_green = gamma * total_count
        variance = gamma * (1.0 - gamma) * total_count
        std_dev = math.sqrt(variance)

        if std_dev == 0.0:
            return 0.0

        return round((green_count - expected_green) / std_dev, 4)

    @staticmethod
    def calculate_p_value(z_score: float) -> float:
        """Approximates p-value from z-score using cumulative standard normal distribution."""
        if z_score <= 0:
            return 0.50
        # Standard normal CDF approximation (Erf formula)
        p_val = 0.5 * (1.0 - math.erf(z_score / math.sqrt(2.0)))
        return round(max(p_val, 0.00001), 5)

    @classmethod
    def analyze_text(
        cls,
        document_id: str,
        document_title: str,
        text_content: str,
        gamma: float = 0.5,
        z_threshold: float = 4.0,
    ) -> WatermarkDetectionMatch:
        """Analyzes text snippet for green-list token bias and computes z-score statistics."""
        tokens = [t.strip() for t in text_content.split() if t.strip()]
        total_tokens = len(tokens)

        if total_tokens == 0:
            token_dist = WatermarkTokenDistribution(0, 0, 0, 0.0, gamma, 0.0)
            return WatermarkDetectionMatch(
                detection_id=f"WM-{uuid.uuid4().hex[:8].upper()}",
                document_id=document_id,
                document_title=document_title,
                z_score=0.0,
                p_value=0.50,
                watermark_confidence_percentage=0.0,
                is_watermark_present=False,
                model_generator_signature="None",
                token_distribution=token_dist,
                analyzed_at=datetime.utcnow(),
            )

        # Deterministic pseudo green-list hash partition simulation based on token length/chars
        green_count = sum(1 for t in tokens if (hash(t) % 100) < (gamma * 100))
        red_count = total_tokens - green_count
        observed_green_ratio = round(green_count / total_tokens, 4)

        z_score = cls.calculate_z_score(green_count, total_tokens, gamma)
        p_value = cls.calculate_p_value(z_score)

        is_present = z_score >= z_threshold
        confidence = min(
            round(max(0.0, (z_score / (z_threshold * 1.5)) * 100), 2), 99.9
        )

        signature = (
            "Kirchenbauer-Logits" if is_present else "Human-Written / Unwatermarked"
        )

        # Entropy calculation
        entropy = 0.0
        if green_count > 0:
            p_g = green_count / total_tokens
            entropy -= p_g * math.log2(p_g)
        if red_count > 0:
            p_r = red_count / total_tokens
            entropy -= p_r * math.log2(p_r)

        token_dist = WatermarkTokenDistribution(
            total_tokens_analyzed=total_tokens,
            green_list_tokens_count=green_count,
            red_list_tokens_count=red_count,
            observed_green_ratio=observed_green_ratio,
            expected_green_ratio=gamma,
            entropy_score=round(entropy, 4),
        )

        return WatermarkDetectionMatch(
            detection_id=f"WM-{uuid.uuid4().hex[:8].upper()}",
            document_id=document_id,
            document_title=document_title,
            z_score=z_score,
            p_value=p_value,
            watermark_confidence_percentage=confidence,
            is_watermark_present=is_present,
            model_generator_signature=signature,
            token_distribution=token_dist,
            analyzed_at=datetime.utcnow(),
        )
