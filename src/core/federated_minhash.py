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
src/core/federated_minhash.py
-----------------------------
Privacy-Preserving Federated Plagiarism Detection via MinHash and LSH.

Generates MinHash signatures and Locality-Sensitive Hashing (LSH) bands
for document chunks. This allows institutions to check student submissions
against a global corpus without sharing raw, FERPA-protected documents.
"""

import hashlib
import logging
import struct
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Large prime for MinHash hash functions (Mersenne prime)
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _hash_shingles(shingles: set[str], num_hashes: int, seed: int = 42) -> np.ndarray:
    """Compute MinHash signature for a set of shingles.

    Uses a family of linear hash functions: h(x) = (a * x + b) % p.
    """
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MERSENNE_PRIME, size=num_hashes, dtype=np.uint64)
    b = rng.randint(0, _MERSENNE_PRIME, size=num_hashes, dtype=np.uint64)

    signature = np.full(num_hashes, _MAX_HASH, dtype=np.uint64)

    for shingle in shingles:
        # Hash the shingle to a 32-bit integer
        h = int(hashlib.sha1(shingle.encode("utf-8")).hexdigest()[:8], 16)  # nosec

        # Apply all hash functions
        hashes = (a * h + b) % _MERSENNE_PRIME

        # Update signature with minimums
        signature = np.minimum(signature, hashes)

    return signature


def generate_minhash_signature(
    text: str, num_hashes: int = 128, k: int = 3
) -> np.ndarray:
    """Generate a MinHash signature for a text document.

    Args:
        text: The input text.
        num_hashes: Number of hash functions (signature length).
        k: Size of character shingles (k-shingles).

    Returns:
        Numpy array of shape (num_hashes,) containing the MinHash signature.
    """
    if not text or len(text) < k:
        return np.full(num_hashes, _MAX_HASH, dtype=np.uint64)

    # Generate k-character shingles
    shingles = {text[i : i + k] for i in range(len(text) - k + 1)}

    return _hash_shingles(shingles, num_hashes)


def generate_lsh_bands(
    signature: np.ndarray, bands: int, rows_per_band: int
) -> list[bytes]:
    """Divide a MinHash signature into LSH bands for candidate generation.

    Args:
        signature: The MinHash signature array.
        bands: Number of bands (b).
        rows_per_band: Number of rows per band (r). Note: b * r <= num_hashes.

    Returns:
        List of byte strings, each representing the hash of a band.
        Two documents are candidates if they share at least one identical band.
    """
    num_hashes = len(signature)
    if bands * rows_per_band > num_hashes:
        raise ValueError(
            f"bands ({bands}) * rows_per_band ({rows_per_band}) cannot exceed signature length ({num_hashes})."
        )

    lsh_bands = []
    for i in range(bands):
        start = i * rows_per_band
        end = start + rows_per_band
        band_slice = signature[start:end]

        # Hash the band slice to create a compact bucket key
        band_hash = hashlib.sha256(band_slice.tobytes()).digest()
        lsh_bands.append(band_hash)

    return lsh_bands


def estimate_jaccard_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate the Jaccard similarity between two MinHash signatures.

    The probability that two signatures agree at a given position is
    exactly equal to the Jaccard similarity of their underlying shingle sets.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must be of equal length.")

    agreements = np.sum(sig_a == sig_b)
    return float(agreements) / len(sig_a)
