"""
tests/core/test_av_scan.py
--------------------------
Unit tests for Multimedia Audio-Visual Sync and Dubbing Plagiarism Detection.
"""

import pytest
from src.core.av_sync_extractor import (
    extract_pitch_contour,
    compute_background_noise_hash,
    extract_av_features,
)
from src.core.dubbing_fingerprinter import compute_pitch_similarity, analyze_dubbing


class TestAVSyncExtractor:
    def test_extract_pitch_contour(self):
        # Simulate audio bytes with varying amplitudes
        audio = bytes([i % 256 for i in range(2048)])
        contour = extract_pitch_contour(audio)
        assert len(contour) == 2
        assert all(0.0 <= p <= 1.0 for p in contour)

    def test_compute_background_noise_hash(self):
        # Simulate silence (bytes near 128)
        audio = bytes([128, 129, 127, 128] * 100)
        hash_val = compute_background_noise_hash(audio)
        assert len(hash_val) == 16

    def test_compute_background_noise_hash_loud(self):
        # Simulate loud audio (bytes far from 128)
        audio = bytes([0, 255, 0, 255] * 100)
        hash_val = compute_background_noise_hash(audio)
        assert hash_val == ""


class TestDubbingFingerprinter:
    def test_compute_pitch_similarity_identical(self):
        contour = [0.1, 0.2, 0.3]
        sim = compute_pitch_similarity(contour, contour)
        assert sim == 1.0

    def test_analyze_dubbing_detected(self):
        # Same background noise, different pitch (dubbing)
        audio_base = bytes([128, 129, 127] * 1000)
        feat_a = extract_av_features(b"vid", audio_base)

        # Modify pitch contour manually for test
        feat_b = extract_av_features(b"vid", audio_base)
        feat_b.pitch_contour = [p + 0.5 for p in feat_a.pitch_contour]  # Shift pitch

        result = analyze_dubbing(feat_a, feat_b)
        assert result["noise_match"] is True
        assert result["pitch_similarity"] < 0.6
        assert result["is_dubbed"] is True
