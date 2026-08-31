"""
src/core/av_sync_extractor.py
-----------------------------
Audio-Visual Sync and Pitch Contour Extractor.

Processes multimedia metadata to extract audio pitch contours, background
noise frequency fingerprints, and AV sync variance metrics. Uses lightweight
byte-level heuristics to simulate audio/video feature extraction without
requiring heavy dependencies like ffmpeg or librosa.
"""

import math
import hashlib
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AVFeatures:
    """Represents extracted audio-visual forensics features."""

    duration_seconds: float
    pitch_contour: List[float] = field(default_factory=list)
    background_noise_hash: str = ""
    av_sync_variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "pitch_contour_length": len(self.pitch_contour),
            "background_noise_hash": self.background_noise_hash,
            "av_sync_variance": self.av_sync_variance,
        }


def extract_pitch_contour(audio_bytes: bytes, window_size: int = 1024) -> List[float]:
    """Extract a simplified pitch contour from raw audio bytes.

    Uses a zero-crossing rate proxy and byte-entropy heuristic to estimate
    pitch frequency over time windows. This is a lightweight proxy for
    actual FFT-based pitch extraction.
    """
    if not audio_bytes or len(audio_bytes) < window_size:
        return []

    contour = []
    for i in range(0, len(audio_bytes) - window_size, window_size):
        window = audio_bytes[i : i + window_size]

        # Zero-crossing rate proxy: count sign changes (assuming 8-bit signed PCM proxy)
        crossings = 0
        for j in range(1, len(window)):
            if (window[j] > 127) != (window[j - 1] > 127):
                crossings += 1

        # Normalize to a pseudo-frequency (higher crossings = higher pitch)
        pseudo_pitch = crossings / window_size
        contour.append(round(pseudo_pitch, 4))

    return contour


def compute_background_noise_hash(audio_bytes: bytes, sample_rate: int = 44100) -> str:
    """Compute a hash of the low-amplitude background noise.

    Isolates bytes that fall within a low-amplitude threshold (near silence)
    and hashes them to create a background noise fingerprint.
    """
    if not audio_bytes:
        return ""

    # Threshold for "silence" (near 128 for unsigned 8-bit, or near 0 for signed)
    noise_bytes = bytearray()
    threshold = 15

    for byte in audio_bytes:
        # Proxy for low amplitude
        if abs(byte - 128) < threshold:
            noise_bytes.append(byte)

    if not noise_bytes:
        return ""

    return hashlib.sha256(bytes(noise_bytes)).hexdigest()[:16]


def compute_av_sync_variance(video_bytes: bytes, audio_bytes: bytes) -> float:
    """Compute the variance between video frame changes and audio energy.

    High variance indicates that the audio track does not naturally align
    with the visual changes, suggesting dubbing or splicing.
    """
    if not video_bytes or not audio_bytes:
        return 0.0

    # Simplified proxy: compare byte entropy chunks
    chunk_size = 4096
    video_entropy = []
    audio_energy = []

    for i in range(0, min(len(video_bytes), len(audio_bytes)), chunk_size):
        v_chunk = video_bytes[i : i + chunk_size]
        a_chunk = audio_bytes[i : i + chunk_size]

        # Video entropy proxy (unique bytes)
        v_ent = len(set(v_chunk)) / 256.0
        video_entropy.append(v_ent)

        # Audio energy proxy (variance from mean)
        a_mean = sum(a_chunk) / len(a_chunk) if a_chunk else 0
        a_en = sum((b - a_mean) ** 2 for b in a_chunk) / len(a_chunk) if a_chunk else 0
        audio_energy.append(a_en / 10000.0)  # Normalize

    if not video_entropy:
        return 0.0

    # Compute variance of the difference between normalized video and audio features
    diffs = [abs(v - min(a, 1.0)) for v, a in zip(video_entropy, audio_energy)]
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)

    return round(min(1.0, variance * 10), 4)


def extract_av_features(
    video_bytes: bytes, audio_bytes: bytes, duration: float = 10.0
) -> AVFeatures:
    """Extract comprehensive AV forensics features."""
    pitch = extract_pitch_contour(audio_bytes)
    noise_hash = compute_background_noise_hash(audio_bytes)
    sync_var = compute_av_sync_variance(video_bytes, audio_bytes)

    return AVFeatures(
        duration_seconds=duration,
        pitch_contour=pitch,
        background_noise_hash=noise_hash,
        av_sync_variance=sync_var,
    )
