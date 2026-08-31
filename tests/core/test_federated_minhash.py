"""
tests/core/test_federated_minhash.py
------------------------------------
Unit tests for the Federated Plagiarism Detection via MinHash engine.
"""

import pytest
import numpy as np
from src.core.federated_minhash import (
    generate_minhash_signature,
    generate_lsh_bands,
    estimate_jaccard_similarity,
)
from src.security.bloom_filter_exchange import package_lsh_bands, verify_lsh_package


class TestMinHashGeneration:
    """Test suite for MinHash signature generation."""

    def test_generate_signature_shape(self):
        """Verify signature has the correct shape."""
        sig = generate_minhash_signature("Hello world this is a test.", num_hashes=64)
        assert sig.shape == (64,)

    def test_identical_texts_identical_signatures(self):
        """Verify identical texts produce identical MinHash signatures."""
        text = "The quick brown fox jumps over the lazy dog."
        sig_a = generate_minhash_signature(text, num_hashes=128)
        sig_b = generate_minhash_signature(text, num_hashes=128)
        assert np.array_equal(sig_a, sig_b)

    def test_different_texts_different_signatures(self):
        """Verify different texts produce different signatures."""
        text_a = "The quick brown fox jumps over the lazy dog."
        text_b = "A completely different sentence about nothing."
        sig_a = generate_minhash_signature(text_a, num_hashes=128)
        sig_b = generate_minhash_signature(text_b, num_hashes=128)
        assert not np.array_equal(sig_a, sig_b)

    def test_estimate_jaccard_similarity_identical(self):
        """Verify Jaccard similarity is 1.0 for identical signatures."""
        text = "Test sentence for Jaccard similarity."
        sig_a = generate_minhash_signature(text)
        sig_b = generate_minhash_signature(text)
        sim = estimate_jaccard_similarity(sig_a, sig_b)
        assert sim == 1.0

    def test_estimate_jaccard_similarity_disjoint(self):
        """Verify Jaccard similarity is low for completely disjoint texts."""
        text_a = "apple banana cherry"
        text_b = "dog cat mouse"
        sig_a = generate_minhash_signature(text_a, k=3)
        sig_b = generate_minhash_signature(text_b, k=3)
        sim = estimate_jaccard_similarity(sig_a, sig_b)
        assert sim < 0.2


class TestLSHBands:
    """Test suite for LSH band generation."""

    def test_generate_lsh_bands_count(self):
        """Verify the correct number of LSH bands are generated."""
        sig = generate_minhash_signature("Test text.", num_hashes=64)
        bands = generate_lsh_bands(sig, bands=8, rows_per_band=8)
        assert len(bands) == 8

    def test_lsh_bands_invalid_params_raises(self):
        """Verify ValueError is raised if bands * rows > num_hashes."""
        sig = generate_minhash_signature("Test text.", num_hashes=64)
        with pytest.raises(ValueError):
            generate_lsh_bands(sig, bands=10, rows_per_band=10)  # 100 > 64


class TestBloomFilterExchange:
    """Test suite for cryptographic packaging of LSH bands."""

    def test_package_and_verify(self):
        """Verify a packaged LSH band can be successfully verified."""
        bands = [b"band1", b"band2"]
        secret = "super_secret_key"

        package = package_lsh_bands("doc1", bands, "inst1", secret)
        assert verify_lsh_package(package, secret) is True

    def test_verify_fails_with_wrong_key(self):
        """Verify verification fails with an incorrect secret key."""
        bands = [b"band1"]
        package = package_lsh_bands("doc1", bands, "inst1", "correct_key")
        assert verify_lsh_package(package, "wrong_key") is False

    def test_verify_fails_on_tampered_payload(self):
        """Verify verification fails if the payload is tampered with."""
        bands = [b"band1"]
        package = package_lsh_bands("doc1", bands, "inst1", "secret")

        # Tamper with the payload
        package["payload"]["document_id"] = "doc2"
        assert verify_lsh_package(package, "secret") is False
