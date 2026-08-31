"""
tests/core/test_audio_prosody.py
--------------------------------
Unit tests for Audio Transcript Prosody and Cadence Fingerprinting.
"""

import pytest
from src.core.audio_transcript_analyzer import (
    extract_pause_durations,
    compute_speech_rate_variance,
)
from src.core.cadence_fingerprinter import compute_cadence_fingerprint, analyze_prosody


class TestAudioTranscriptAnalyzer:
    def test_extract_pause_durations(self):
        transcript = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 1.0, "end": 1.5},
        ]
        pauses = extract_pause_durations(transcript)
        assert len(pauses) == 1
        assert pauses[0] == 0.5

    def test_compute_speech_rate_variance(self):
        transcript = [
            {"word": "a", "start": 0.0, "end": 1.0},
            {"word": "b", "start": 1.0, "end": 2.0},
            {"word": "c", "start": 6.0, "end": 7.0},
        ]
        var = compute_speech_rate_variance(transcript)
        assert var >= 0.0


class TestCadenceFingerprinter:
    def test_compute_cadence_fingerprint_tts(self):
        pauses = [0.2, 0.2, 0.2, 0.2]  # Uniform pauses
        cadence = compute_cadence_fingerprint(pauses)
        assert cadence["is_tts"] is True
        assert cadence["pause_variance"] < 0.01

    def test_analyze_prosody_human(self):
        transcript = [
            {"word": "a", "start": 0.0, "end": 0.5},
            {"word": "b", "start": 1.5, "end": 2.0},  # 1.0s pause
            {"word": "c", "start": 2.1, "end": 2.5},  # 0.1s pause
        ]
        result = analyze_prosody(transcript)
        assert result["is_synthetic"] is False
