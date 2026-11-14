"""
Tests for Audio Summer Engine and Headroom Protection (T-101).

Verifies:
- AC-1: Mono expansion to 2-channel stereo.
- AC-2: Zero-padding of unequal-length tracks and float32 summing.
- AC-3: Headroom protection (-0.1 dBFS ceiling when peak > 1.0).
- AC-4: Writing 32-bit float stereo WAV files.
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.summer import (
    expand_to_stereo,
    pad_and_sum,
    apply_headroom_protection,
    sum_audio_files,
    AudioSummer,
)


# ============================================================================
# AC-1: Stereo Expansion Tests
# ============================================================================


def test_expand_to_stereo_1d() -> None:
    """1D mono array (N,) should expand to (N, 2) with duplicated channels."""
    n_samples = 500
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    mono_1d = np.sin(2 * np.pi * 440 * t)

    stereo = expand_to_stereo(mono_1d)

    assert stereo.shape == (n_samples, 2)
    assert stereo.dtype == np.float32
    assert np.array_equal(stereo[:, 0], mono_1d)
    assert np.array_equal(stereo[:, 1], mono_1d)


def test_expand_to_stereo_2d_mono() -> None:
    """2D mono array (N, 1) should expand to (N, 2) with duplicated channels."""
    n_samples = 300
    mono_2d = np.ones((n_samples, 1), dtype=np.float32) * 0.42

    stereo = expand_to_stereo(mono_2d)

    assert stereo.shape == (n_samples, 2)
    assert stereo.dtype == np.float32
    assert np.allclose(stereo[:, 0], 0.42)
    assert np.allclose(stereo[:, 1], 0.42)


def test_expand_to_stereo_pass_through() -> None:
    """Existing stereo array (N, 2) should pass through unchanged."""
    n_samples = 200
    left = np.full((n_samples,), 0.25, dtype=np.float32)
    right = np.full((n_samples,), 0.75, dtype=np.float32)
    stereo_in = np.column_stack((left, right))

    stereo_out = expand_to_stereo(stereo_in)

    assert stereo_out.shape == (n_samples, 2)
    assert stereo_out.dtype == np.float32
    assert np.array_equal(stereo_out[:, 0], left)
    assert np.array_equal(stereo_out[:, 1], right)


def test_expand_to_stereo_empty() -> None:
    """Empty arrays should produce (0, 2) float32 arrays."""
    empty_1d = np.array([], dtype=np.float32)
    stereo_1d = expand_to_stereo(empty_1d)
    assert stereo_1d.shape == (0, 2)
    assert stereo_1d.dtype == np.float32

    empty_2d = np.empty((0, 1), dtype=np.float32)
    stereo_2d = expand_to_stereo(empty_2d)
    assert stereo_2d.shape == (0, 2)
    assert stereo_2d.dtype == np.float32


def test_expand_to_stereo_unsupported_shape() -> None:
    """Arrays with >2 dimensions should raise ValueError."""
    invalid = np.zeros((10, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        expand_to_stereo(invalid)


# ============================================================================
# AC-2: Padding and Float32 Summing Tests
# ============================================================================


def test_pad_and_sum_equal_length() -> None:
    """Tracks of equal length sum elementwise with float32 math."""
    track1 = np.full((100, 2), 0.2, dtype=np.float32)
    track2 = np.full((100, 2), 0.3, dtype=np.float32)

    summed = pad_and_sum([track1, track2])

    assert summed.shape == (100, 2)
    assert summed.dtype == np.float32
    assert np.allclose(summed, 0.5, atol=1e-6)


def test_pad_and_sum_varying_length() -> None:
    """Unequal length tracks must be zero-padded to maximum track length."""
    t1 = np.full((50, 2), 0.2, dtype=np.float32)
    t2 = np.full((100, 2), 0.3, dtype=np.float32)
    t3 = np.full((80, 2), 0.1, dtype=np.float32)

    summed = pad_and_sum([t1, t2, t3])

    assert summed.shape == (100, 2)
    assert summed.dtype == np.float32

    # Region 0:50 has t1 + t2 + t3 = 0.2 + 0.3 + 0.1 = 0.6
    assert np.allclose(summed[0:50, :], 0.6, atol=1e-6)
    # Region 50:80 has t2 + t3 = 0.3 + 0.1 = 0.4
    assert np.allclose(summed[50:80, :], 0.4, atol=1e-6)
    # Region 80:100 has only t2 = 0.3
    assert np.allclose(summed[80:100, :], 0.3, atol=1e-6)


def test_pad_and_sum_mixed_channel_inputs() -> None:
    """Pad and sum seamlessly accepts mixed mono 1D, mono 2D, and stereo."""
    t_mono1d = np.full((40,), 0.1, dtype=np.float32)       # becomes (40, 2)
    t_mono2d = np.full((60, 1), 0.2, dtype=np.float32)    # becomes (60, 2)
    t_stereo = np.full((30, 2), 0.3, dtype=np.float32)    # stays (30, 2)

    summed = pad_and_sum([t_mono1d, t_mono2d, t_stereo])

    assert summed.shape == (60, 2)
    assert summed.dtype == np.float32
    # 0:30 has 0.1 + 0.2 + 0.3 = 0.6
    assert np.allclose(summed[0:30, :], 0.6, atol=1e-6)
    # 30:40 has 0.1 + 0.2 = 0.3
    assert np.allclose(summed[30:40, :], 0.3, atol=1e-6)
    # 40:60 has 0.2
    assert np.allclose(summed[40:60, :], 0.2, atol=1e-6)


def test_pad_and_sum_empty() -> None:
    """Empty track list returns (0, 2) float32 array."""
    summed = pad_and_sum([])
    assert summed.shape == (0, 2)
    assert summed.dtype == np.float32


# ============================================================================
# AC-3: Headroom Protection Tests
# ============================================================================


def test_headroom_protection_no_clipping() -> None:
    """Signal with peak <= 1.0 (0 dBFS) must not be scaled down."""
    audio = np.array([[-0.8, 0.5], [0.3, -0.9]], dtype=np.float32)
    protected = apply_headroom_protection(audio, target_peak_dbfs=-0.1)

    assert protected.shape == audio.shape
    assert protected.dtype == np.float32
    assert np.array_equal(protected, audio)


def test_headroom_protection_scales_cleanly() -> None:
    """
    Signal exceeding 1.0 peak is scaled down so maximum peak is
    10^(-0.1 / 20) ~= 0.98855 with margin of error < 1e-4.
    """
    # Create signal with positive and negative clipping peaks
    audio = np.array([[-2.0, 1.5], [0.8, -3.0], [2.5, 0.0]], dtype=np.float32)
    initial_peak = float(np.max(np.abs(audio)))
    assert initial_peak == 3.0

    target_peak_dbfs = -0.1
    expected_peak = 10.0 ** (target_peak_dbfs / 20.0)  # ~0.98855309

    protected = apply_headroom_protection(audio, target_peak_dbfs=target_peak_dbfs)

    actual_peak = float(np.max(np.abs(protected)))
    assert abs(actual_peak - expected_peak) < 1e-4
    assert actual_peak <= expected_peak + 1e-5
    assert actual_peak < 1.0

    # Ensure linearity: each element is scaled by exactly expected_peak / initial_peak
    scale = expected_peak / initial_peak
    assert np.allclose(protected, audio * scale, atol=1e-5)


def test_headroom_protection_custom_target() -> None:
    """Supports custom target_peak_dbfs ceilings (e.g. -1.0 dBFS)."""
    audio = np.full((100, 2), 2.0, dtype=np.float32)
    target_dbfs = -1.0
    expected_peak = 10.0 ** (target_dbfs / 20.0)  # ~0.89125

    protected = apply_headroom_protection(audio, target_peak_dbfs=target_dbfs)
    actual_peak = float(np.max(np.abs(protected)))

    assert abs(actual_peak - expected_peak) < 1e-4


def test_headroom_protection_empty() -> None:
    """Empty audio returns empty array without error."""
    empty = np.zeros((0, 2), dtype=np.float32)
    protected = apply_headroom_protection(empty)
    assert protected.shape == (0, 2)
    assert protected.dtype == np.float32


# ============================================================================
# AC-4: End-to-End File Summing & 32-bit Float Stereo WAV Output Tests
# ============================================================================


def test_sum_audio_files_end_to_end(tmp_path: pytest.TempPathFactory) -> None:
    """
    Writes multiple track WAV files, sums them with headroom protection,
    and verifies 32-bit float stereo WAV output.
    """
    sr = 44100
    tmp_dir = str(tmp_path)

    # Track 1: Mono 1D, 1.0 second (44100 samples), peak 0.7
    t1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32) * 0.7
    f1 = os.path.join(tmp_dir, "track1.wav")
    sf.write(f1, t1, sr)

    # Track 2: Stereo, 1.5 seconds (66150 samples), peak 0.8
    t2_left = np.cos(2 * np.pi * 220 * np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)).astype(np.float32) * 0.8
    t2_right = np.sin(2 * np.pi * 330 * np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)).astype(np.float32) * 0.8
    t2 = np.column_stack((t2_left, t2_right))
    f2 = os.path.join(tmp_dir, "track2.wav")
    sf.write(f2, t2, sr)

    # Track 3: Mono 2D (shape (N, 1)), 0.5 second (22050 samples), peak 0.6
    t3 = (np.ones((int(sr * 0.5), 1), dtype=np.float32) * 0.6)
    f3 = os.path.join(tmp_dir, "track3.wav")
    sf.write(f3, t3, sr)

    out_file = os.path.join(tmp_dir, "stems", "summed_master.wav")

    # Run summing
    result = sum_audio_files([f1, f2, f3], out_file)

    # Verify result telemetry dictionary
    assert os.path.exists(out_file)
    assert result["num_tracks"] == 3
    assert result["sample_rate"] == sr
    assert result["peak_amplitude"] > 1.0
    assert result["clipped"] is True
    assert result["scaled"] is True

    # Inspect WAV file metadata via soundfile.info
    info = sf.info(out_file)
    assert info.format == "WAV"
    assert info.subtype == "FLOAT"
    assert info.channels == 2
    assert info.samplerate == sr
    assert info.frames == int(sr * 1.5)

    # Read back audio data and verify properties
    data, read_sr = sf.read(out_file, dtype="float32")
    assert read_sr == sr
    assert data.dtype == np.float32
    assert data.shape == (int(sr * 1.5), 2)

    # Verify maximum peak is clamped to -0.1 dBFS (~0.98855) within 1e-4 margin
    expected_peak = 10.0 ** (-0.1 / 20.0)
    actual_peak = float(np.max(np.abs(data)))
    assert abs(actual_peak - expected_peak) < 1e-4
    assert actual_peak <= 1.0


def test_sum_audio_files_empty_raises(tmp_path: pytest.TempPathFactory) -> None:
    """Empty list of file paths raises ValueError."""
    out_file = str(tmp_path / "out.wav")
    with pytest.raises(ValueError, match="empty list"):
        sum_audio_files([], out_file)


def test_sum_audio_files_different_sample_rates(tmp_path: pytest.TempPathFactory) -> None:
    """Verifies that files with differing sample rates are resampled to match the first file."""
    tmp_dir = str(tmp_path)
    # File 1 at 44100 Hz (1 second)
    sr1 = 44100
    t1 = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr1, endpoint=False)) * 0.4).astype(np.float32)
    f1 = os.path.join(tmp_dir, "t44k.wav")
    sf.write(f1, t1, sr1)

    # File 2 at 48000 Hz (1 second)
    sr2 = 48000
    t2 = (np.cos(2 * np.pi * 440 * np.linspace(0, 1, sr2, endpoint=False)) * 0.4).astype(np.float32)
    f2 = os.path.join(tmp_dir, "t48k.wav")
    sf.write(f2, t2, sr2)

    out_file = os.path.join(tmp_dir, "resampled_sum.wav")
    result = sum_audio_files([f1, f2], out_file)

    assert os.path.exists(out_file)
    assert result["sample_rate"] == sr1
    info = sf.info(out_file)
    assert info.samplerate == sr1
    assert info.subtype == "FLOAT"
    assert info.channels == 2


def test_audio_summer_class_interface(tmp_path: pytest.TempPathFactory) -> None:
    """Verifies AudioSummer class interface works seamlessly."""
    summer = AudioSummer(target_peak_dbfs=-0.1)

    # Stereo expansion via class
    mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    stereo = summer.expand_to_stereo(mono)
    assert stereo.shape == (3, 2)

    # Pad and sum via class
    summed = summer.pad_and_sum([mono, stereo])
    assert summed.shape == (3, 2)

    # Headroom protection via class
    clipping = np.full((10, 2), 2.5, dtype=np.float32)
    protected = summer.apply_headroom_protection(clipping)
    peak = float(np.max(np.abs(protected)))
    assert abs(peak - 10.0 ** (-0.1 / 20.0)) < 1e-4

    # File summing via class
    sr = 44100
    tmp_dir = str(tmp_path)
    f1 = os.path.join(tmp_dir, "t1.wav")
    sf.write(f1, mono, sr)
    out_f = os.path.join(tmp_dir, "out_class.wav")

    res = summer.sum_files([f1], out_f)
    assert os.path.exists(out_f)
    assert res["num_tracks"] == 1
