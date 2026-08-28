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

"""
src/core/back_translation_simulator.py
--------------------------------------
Cross-Lingual Back-Translation Attack Simulation Engine.

Simulates the semantic drift introduced when attackers use back-translation
(translating text to a pivot language and back) to obfuscate plagiarized content.
This allows the system to generate adversarial examples and train defense classifiers.
"""

import logging
import random
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Simple synonym dictionary to simulate semantic drift during back-translation
# In a real attack, this would be a full MT model (e.g., MarianMT), but for
# simulation and defense training, we use a lightweight synonym replacement.
SYNONYM_MAP = {
    "important": ["crucial", "vital", "significant", "essential"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "show": ["demonstrate", "illustrate", "reveal", "display"],
    "big": ["large", "massive", "substantial", "considerable"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "good": ["excellent", "superb", "outstanding", "favorable"],
    "bad": ["poor", "terrible", "awful", "dreadful"],
    "think": ["believe", "consider", "assume", "reckon"],
    "make": ["create", "produce", "generate", "construct"],
    "help": ["assist", "aid", "support", "facilitate"],
}


def simulate_back_translation(text: str, drift_probability: float = 0.3) -> str:
    """Simulate back-translation by introducing semantic drift via synonym replacement.

    This mimics the effect of translating text to a pivot language (e.g., German)
    and back to English, which often results in synonym substitution and slight
    structural changes while preserving the core meaning.

    Args:
        text: The original text string.
        drift_probability: Probability (0.0 to 1.0) of replacing a known word
                           with a synonym.

    Returns:
        The back-translated (drifted) text string.
    """
    if not text or not isinstance(text, str):
        return ""

    words = text.split()
    drifted_words = []

    for word in words:
        # Strip punctuation for matching, but preserve it for output
        clean_word = re.sub(r"[^\w\s]", "", word).lower()
        punctuation = word[len(clean_word) :] if len(word) > len(clean_word) else ""

        if clean_word in SYNONYM_MAP and random.random() < drift_probability:
            synonym = random.choice(SYNONYM_MAP[clean_word])
            # Preserve original capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
            drifted_words.append(synonym + punctuation)
        else:
            drifted_words.append(word)

    return " ".join(drifted_words)


def generate_adversarial_batch(
    texts: List[str], num_variants: int = 3, drift_probability: float = 0.3
) -> Dict[str, List[str]]:
    """Generate a batch of back-translated adversarial examples for a list of texts.

    Args:
        texts: List of original text strings.
        num_variants: Number of drifted variants to generate per text.
        drift_probability: Probability of synonym replacement.

    Returns:
        Dictionary mapping original text to a list of drifted variants.
    """
    results = {}
    for text in texts:
        variants = []
        for _ in range(num_variants):
            variant = simulate_back_translation(text, drift_probability)
            variants.append(variant)
        results[text] = variants

    logger.info("Generated %d variants for %d texts.", num_variants, len(texts))
    return results
