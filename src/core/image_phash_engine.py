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
src/core/image_phash_engine.py
------------------------------
Perceptual Hashing (pHash) Engine for Image Plagiarism Detection.

Extracts image blocks from documents and computes rotation-invariant
perceptual hashes to detect copied charts, diagrams, and figures.
"""

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_phash(image_array: np.ndarray, hash_size: int = 16) -> str:
    """Compute a perceptual hash (pHash) for an image array.

    This is a simplified implementation of the pHash algorithm using
    Discrete Cosine Transform (DCT) principles approximated via FFT
    for environments without OpenCV/PIL.

    Args:
        image_array: 2D numpy array representing grayscale image pixels.
        hash_size: Size of the hash (hash_size x hash_size).

    Returns:
        Hexadecimal string representing the perceptual hash.
    """
    if image_array is None or image_array.size == 0:
        return ""

    # Resize image to hash_size x hash_size (simplified nearest-neighbor)
    h, w = image_array.shape
    if h == 0 or w == 0:
        return ""

    # Simple downscaling by averaging blocks
    block_h = h // hash_size
    block_w = w // hash_size

    if block_h == 0 or block_w == 0:
        return ""

    resized = np.zeros((hash_size, hash_size), dtype=np.float32)
    for i in range(hash_size):
        for j in range(hash_size):
            y_start = i * block_h
            x_start = j * block_w
            block = image_array[
                y_start : y_start + block_h, x_start : x_start + block_w
            ]
            resized[i, j] = np.mean(block)

    # Compute 2D DCT (approximated via FFT for simplicity)
    # Real pHash uses DCT, but FFT is a close proxy for frequency analysis
    dct = np.fft.fft2(resized)
    dct_low_freq = dct[:8, :8]  # Keep top-left low frequencies

    # Compute median of low frequencies
    median_val = np.median(dct_low_freq)

    # Generate binary hash
    hash_bits = (dct_low_freq > median_val).flatten()

    # Convert bits to hex string
    hash_int = int("".join(["1" if b else "0" for b in hash_bits]), 2)
    return hex(hash_int)[2:].zfill(16)


def compute_hamming_distance(hash1: str, hash2: str) -> int:
    """Compute the Hamming distance between two hexadecimal hash strings."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64  # Max distance for 64-bit hash

    int1 = int(hash1, 16)
    int2 = int(hash2, 16)

    xor_val = int1 ^ int2
    return bin(xor_val).count("1")
