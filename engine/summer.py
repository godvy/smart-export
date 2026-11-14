"""
Audio Summing and Headroom Protection Engine.

Provides high-precision 32-bit float audio summing, channel expansion,
and headroom protection (-0.1 dBFS ceiling) for DAW stems.
"""

from __future__ import annotations

import os
from typing import Sequence
import numpy as np
import soundfile as sf


def expand_to_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Expands mono audio arrays into 2-channel stereo seamlessly.
    - 1D array of shape (N,) -> (N, 2)
    - 2D array of shape (N, 1) -> (N, 2)
    - 2D array of shape (N, 2) -> (N, 2) (unchanged)

    Returns a float32 numpy array with shape (N, 2).
    """
    arr = np.asarray(audio, dtype=np.float32)

    if arr.ndim == 1:
        return np.column_stack((arr, arr))
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            return np.repeat(arr, 2, axis=1)
        elif arr.shape[1] == 2:
            return arr
        elif arr.shape[0] == 1 and arr.shape[1] > 2:
            # Row vector mono (1, N) -> (N, 2)
            flat = arr.ravel()
            return np.column_stack((flat, flat))
        elif arr.shape[0] == 2 and arr.shape[1] != 2:
            # Channels-first stereo (2, N) -> (N, 2)
            return arr.T
        elif arr.shape[1] > 2:
            # Truncate multichannel to first 2 channels
            return arr[:, :2]

    raise ValueError(f"Unsupported audio array shape: {arr.shape}")


def pad_and_sum(tracks: Sequence[np.ndarray]) -> np.ndarray:
    """
    Pads unequal-length tracks to maximum track length and sums audio with float32 math.
    Each input track is automatically expanded to stereo before padding and summing.

    Returns a float32 numpy array of shape (max_length, 2).
    """
    if not tracks:
        return np.zeros((0, 2), dtype=np.float32)

    stereo_tracks = [expand_to_stereo(t) for t in tracks]
    max_len = max(t.shape[0] for t in stereo_tracks)

    if max_len == 0:
        return np.zeros((0, 2), dtype=np.float32)

    summed = np.zeros((max_len, 2), dtype=np.float32)
    for track in stereo_tracks:
        track_len = track.shape[0]
        if track_len > 0:
            summed[:track_len, :] += track

    return summed


def apply_headroom_protection(
    audio: np.ndarray, target_peak_dbfs: float = -0.1
) -> np.ndarray:
    """
    Detects peak amplitude exceeding 1.0 (0 dBFS) and scales cleanly to target dBFS.
    Default target is -0.1 dBFS (~0.98855 linear amplitude).
    If peak amplitude is <= 1.0, the signal is preserved without scaling.

    Returns a float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return arr

    peak = float(np.max(np.abs(arr)))
    if peak > 1.0:
        target_peak_linear = 10.0 ** (target_peak_dbfs / 20.0)
        scale_factor = target_peak_linear / peak
        return (arr * scale_factor).astype(np.float32)

    return arr.astype(np.float32, copy=False)


def sum_audio_files(
    file_paths: Sequence[str],
    output_path: str,
    target_peak_dbfs: float = -0.1,
) -> dict:
    """
    Reads multiple audio files, resamples if needed to match the first track's
    sample rate, sums them with float32 precision, applies headroom protection
    if clipping occurs, and writes a 32-bit float stereo WAV file.

    Returns a telemetry dictionary with processing metadata.
    """
    if not file_paths:
        raise ValueError("Cannot sum an empty list of audio files.")

    loaded_tracks: list[np.ndarray] = []
    sample_rates: list[int] = []

    for path in file_paths:
        data, sr = sf.read(path, dtype="float32")
        loaded_tracks.append(data)
        sample_rates.append(sr)

    target_sr = sample_rates[0]
    processed_tracks: list[np.ndarray] = []

    for data, sr in zip(loaded_tracks, sample_rates):
        stereo = expand_to_stereo(data)
        if sr != target_sr:
            import librosa

            # librosa expects shape (channels, samples)
            resampled = librosa.resample(
                stereo.T, orig_sr=sr, target_sr=target_sr
            ).T
            processed_tracks.append(np.asarray(resampled, dtype=np.float32))
        else:
            processed_tracks.append(stereo)

    summed = pad_and_sum(processed_tracks)
    peak_before = float(np.max(np.abs(summed))) if summed.size > 0 else 0.0

    final_audio = apply_headroom_protection(
        summed, target_peak_dbfs=target_peak_dbfs
    )
    peak_after = (
        float(np.max(np.abs(final_audio))) if final_audio.size > 0 else 0.0
    )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sf.write(
        output_path,
        final_audio,
        target_sr,
        subtype="FLOAT",
        format="WAV",
    )

    duration = float(len(final_audio) / target_sr) if target_sr > 0 else 0.0

    return {
        "output_path": str(output_path),
        "num_tracks": len(file_paths),
        "sample_rate": target_sr,
        "duration": duration,
        "peak_amplitude": peak_before,
        "final_peak_amplitude": peak_after,
        "clipped": bool(peak_before > 1.0),
        "scaled": bool(peak_before > 1.0),
        "shape": tuple(final_audio.shape),
    }


class AudioSummer:
    """Audio summing engine class wrapping core summing operations."""

    def __init__(self, target_peak_dbfs: float = -0.1) -> None:
        self.target_peak_dbfs = target_peak_dbfs

    def expand_to_stereo(self, audio: np.ndarray) -> np.ndarray:
        return expand_to_stereo(audio)

    def pad_and_sum(self, tracks: Sequence[np.ndarray]) -> np.ndarray:
        return pad_and_sum(tracks)

    def apply_headroom_protection(self, audio: np.ndarray) -> np.ndarray:
        return apply_headroom_protection(
            audio, target_peak_dbfs=self.target_peak_dbfs
        )

    def sum_files(
        self, file_paths: Sequence[str], output_path: str
    ) -> dict:
        return sum_audio_files(
            file_paths, output_path, target_peak_dbfs=self.target_peak_dbfs
        )
