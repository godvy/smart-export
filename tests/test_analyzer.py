"""
Unit and integration tests for Audio Conditioning, DSP Feature Extraction,
and Instrument Classification Engine (T-201).

Acceptance Criteria:
  - AC-1: Trims silence via -40 dB gate and resamples to 16 kHz mono slice (up to 5 seconds).
  - AC-2: Calculates spectral centroid in Hz and maps to 5 frequency bands (Sub-bass, Bass, Low-mid, High-mid, High).
  - AC-3: Extracts 12-bin mean chroma profile capturing musical pitch distribution.
  - AC-4: Classifies instrument with generic label filtering and maps to the 10 defined families.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.analyzer import (
    FREQUENCY_BANDS,
    GENERIC_LABELS,
    PITCH_CLASSES,
    TARGET_FAMILIES,
    TARGET_SAMPLING_RATE,
    TrackAnalysis,
    YamnetClassifier,
    analyze_track,
    analyze_tracks,
    calculate_spectral_centroid,
    classify_by_dsp,
    classify_by_keywords,
    classify_instrument,
    condition_audio,
    extract_chroma,
    filter_generic_labels,
    is_generic_label,
    load_audio,
    map_frequency_band,
    map_label_to_family,
)


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def make_sine_wave(freq: float, duration: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    """Generates a pure sine wave numpy array."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return amplitude * np.sin(2.0 * np.pi * freq * t, dtype=np.float32)


# ============================================================================
# AC-1: Audio Conditioning & Slicing Tests
# ============================================================================

class TestAudioConditioning:
    """Tests evaluating AC-1: silence trimming, mono conversion, resampling, slicing."""

    def test_stereo_to_mono_conversion_2d_leading_channels(self) -> None:
        """Shape (2, N) stereo must be converted to 1D mono of shape (N,)."""
        sr = 16000
        left = make_sine_wave(440, 1.0, sr)
        right = make_sine_wave(880, 1.0, sr)
        stereo = np.stack([left, right], axis=0)  # (2, N)

        mono, out_sr = condition_audio(stereo, sr=sr)
        assert mono.ndim == 1
        assert out_sr == TARGET_SAMPLING_RATE
        assert len(mono) == len(left)
        # Value at index 100 should be approximately (left + right) / 2
        expected_sample = (left[100] + right[100]) / 2.0
        assert math.isclose(mono[100], expected_sample, abs_tol=1e-3)

    def test_stereo_to_mono_conversion_2d_trailing_channels(self) -> None:
        """Shape (N, 2) stereo (soundfile style) must be converted to 1D mono."""
        sr = 16000
        left = make_sine_wave(440, 1.0, sr)
        right = make_sine_wave(880, 1.0, sr)
        stereo = np.stack([left, right], axis=1)  # (N, 2)

        mono, out_sr = condition_audio(stereo, sr=sr)
        assert mono.ndim == 1
        assert out_sr == TARGET_SAMPLING_RATE
        assert len(mono) == len(left)

    def test_silence_trimming_40db_gate(self) -> None:
        """Leading and trailing silence below -40 dB must be stripped."""
        sr = 16000
        signal = make_sine_wave(1000, 1.0, sr, amplitude=0.8)  # ~ -2 dBFS
        pre_silence = np.zeros(int(sr * 0.5), dtype=np.float32)  # 500ms silence
        post_silence = np.zeros(int(sr * 0.5), dtype=np.float32)  # 500ms silence

        padded = np.concatenate([pre_silence, signal, post_silence])
        orig_len = len(padded)

        conditioned, _ = condition_audio(padded, sr=sr, top_db=40.0)
        # Conditioned length should be significantly shorter than original length
        assert len(conditioned) < orig_len
        # Should be approximately 1.0 second (16000 samples +- frame boundary buffer)
        assert abs(len(conditioned) - sr) < 2500

    def test_resampling_from_44100_to_16000(self) -> None:
        """Signals at 44.1 kHz must be resampled cleanly to 16 kHz."""
        orig_sr = 44100
        signal = make_sine_wave(440, 2.0, orig_sr)  # 2 seconds = 88200 samples
        conditioned, out_sr = condition_audio(signal, sr=orig_sr, target_sr=16000)

        assert out_sr == 16000
        # Expected samples: 2.0 * 16000 = 32000 samples (+- boundary filter delay)
        assert abs(len(conditioned) - 32000) < 500

    def test_resampling_from_48000_to_16000(self) -> None:
        """Signals at 48 kHz must be resampled cleanly to 16 kHz."""
        orig_sr = 48000
        signal = make_sine_wave(440, 2.0, orig_sr)
        conditioned, out_sr = condition_audio(signal, sr=orig_sr, target_sr=16000)

        assert out_sr == 16000
        assert abs(len(conditioned) - 32000) < 500

    def test_slice_bounding_to_maximum_5_seconds(self) -> None:
        """Signals longer than 5 seconds must be bounded to at most 5 seconds (80,000 samples)."""
        sr = 16000
        long_signal = make_sine_wave(440, 12.0, sr)  # 12 seconds
        conditioned, out_sr = condition_audio(long_signal, sr=sr, max_duration_sec=5.0)

        assert out_sr == 16000
        assert len(conditioned) == 5 * 16000  # Exactly 80000 samples

    def test_short_audio_retains_native_length_without_unwanted_padding(self) -> None:
        """Signals shorter than 5 seconds must NOT be zero-padded to 5 seconds."""
        sr = 16000
        short_signal = make_sine_wave(440, 1.5, sr)  # 1.5 seconds = 24000 samples
        conditioned, _ = condition_audio(short_signal, sr=sr, max_duration_sec=5.0)

        assert len(conditioned) <= 24500
        assert len(conditioned) < 5 * 16000

    def test_complete_silence_input_does_not_crash(self) -> None:
        """Pure silence array must be handled safely without exceptions."""
        sr = 16000
        silent = np.zeros(16000, dtype=np.float32)
        conditioned, out_sr = condition_audio(silent, sr=sr)

        assert out_sr == 16000
        assert isinstance(conditioned, np.ndarray)
        assert len(conditioned) <= 16000

    def test_empty_input_array(self) -> None:
        """Empty input array must return empty array without exception."""
        conditioned, out_sr = condition_audio(np.array([], dtype=np.float32), sr=16000)
        assert len(conditioned) == 0
        assert out_sr == 16000


# ============================================================================
# AC-2: Spectral Centroid & Frequency Band Mapping Tests
# ============================================================================

class TestSpectralCentroidAndBands:
    """Tests evaluating AC-2: Spectral centroid calculation and 5-band mapping."""

    @pytest.mark.parametrize(
        "freq, expected_band",
        [
            (40.0, "Sub-bass"),      # < 60 Hz
            (100.0, "Bass"),         # 60 - 250 Hz
            (1000.0, "Low-mid"),     # 250 - 2000 Hz
            (3500.0, "High-mid"),    # 2000 - 6000 Hz
            (7000.0, "High"),        # >= 6000 Hz
        ],
    )
    def test_spectral_centroid_calculation_accuracy(self, freq: float, expected_band: str) -> None:
        """Calculates centroid for pure tones and verifies matching frequency band."""
        sr = 16000
        sine = make_sine_wave(freq, 1.0, sr)
        centroid = calculate_spectral_centroid(sine, sr)

        # Centroid should be within 15% of the pure tone frequency
        assert math.isclose(centroid, freq, rel_tol=0.15)
        band = map_frequency_band(centroid)
        assert band == expected_band

    @pytest.mark.parametrize(
        "centroid, expected_band",
        [
            (0.0, "Sub-bass"),
            (30.0, "Sub-bass"),
            (59.99, "Sub-bass"),
            (60.0, "Bass"),
            (150.0, "Bass"),
            (249.99, "Bass"),
            (250.0, "Low-mid"),
            (1000.0, "Low-mid"),
            (1999.99, "Low-mid"),
            (2000.0, "High-mid"),
            (4500.0, "High-mid"),
            (5999.99, "High-mid"),
            (6000.0, "High"),
            (8000.0, "High"),
            (15000.0, "High"),
        ],
    )
    def test_band_mapping_boundary_precision(self, centroid: float, expected_band: str) -> None:
        """Verifies strict boundary definitions for all 5 frequency bands."""
        assert map_frequency_band(centroid) == expected_band

    def test_silent_audio_returns_zero_centroid_and_sub_bass(self) -> None:
        """Silence must return centroid 0.0 and Sub-bass band."""
        silent = np.zeros(8000, dtype=np.float32)
        centroid = calculate_spectral_centroid(silent, sr=16000)
        assert centroid == 0.0
        assert map_frequency_band(centroid) == "Sub-bass"


# ============================================================================
# AC-3: Chroma Profile Extraction Tests
# ============================================================================

class TestChromaExtraction:
    """Tests evaluating AC-3: 12-bin mean chroma profile capturing musical pitch distribution."""

    def test_chroma_dimension_and_range(self) -> None:
        """Chroma vector must contain exactly 12 non-negative floats."""
        sr = 16000
        signal = make_sine_wave(440, 1.0, sr)
        chroma = extract_chroma(signal, sr)

        assert isinstance(chroma, list)
        assert len(chroma) == 12
        for bin_val in chroma:
            assert isinstance(bin_val, float)
            assert bin_val >= 0.0

    def test_chroma_pitch_class_a_440(self) -> None:
        """440 Hz is A4; index 9 (A) in [C, C#, D, D#, E, F, F#, G, G#, A, A#, B] must be peak."""
        sr = 16000
        sine_a = make_sine_wave(440.0, 1.0, sr)
        chroma = extract_chroma(sine_a, sr)

        peak_idx = int(np.argmax(chroma))
        # Pitch class 9 is A
        assert peak_idx == 9
        assert PITCH_CLASSES[peak_idx] == "A"
        assert chroma[9] > 0.5

    def test_chroma_pitch_class_c_261(self) -> None:
        """261.63 Hz is C4; index 0 (C) must be peak."""
        sr = 16000
        sine_c = make_sine_wave(261.63, 1.0, sr)
        chroma = extract_chroma(sine_c, sr)

        peak_idx = int(np.argmax(chroma))
        # Pitch class 0 is C
        assert peak_idx == 0
        assert PITCH_CLASSES[peak_idx] == "C"

    def test_chroma_pitch_class_e_330(self) -> None:
        """329.63 Hz is E4; index 4 (E) must be peak."""
        sr = 16000
        sine_e = make_sine_wave(329.63, 1.0, sr)
        chroma = extract_chroma(sine_e, sr)

        peak_idx = int(np.argmax(chroma))
        # Pitch class 4 is E
        assert peak_idx == 4
        assert PITCH_CLASSES[peak_idx] == "E"

    def test_silent_audio_produces_zero_chroma_vector(self) -> None:
        """Pure silence must safely yield 12 zeros."""
        silent = np.zeros(8000, dtype=np.float32)
        chroma = extract_chroma(silent, sr=16000)

        assert len(chroma) == 12
        assert all(val == 0.0 for val in chroma)


# ============================================================================
# AC-4: Generic Label Filtering & 10 Families Classification Tests
# ============================================================================

class TestGenericLabelFiltering:
    """Tests evaluating AC-4: Generic label filtering and suppression."""

    @pytest.mark.parametrize(
        "label",
        [
            "music",
            "sound effect",
            "sound effects",
            "sfx",
            "instrument",
            "musical instrument",
            "silence",
            "background noise",
            "audio",
            "sound",
            "noise",
            "ambient",
        ],
    )
    def test_identifies_generic_labels(self, label: str) -> None:
        """Generic labels must be identified as generic."""
        assert is_generic_label(label) is True

    @pytest.mark.parametrize(
        "label",
        [
            "Snare drum",
            "Electric guitar",
            "Piano",
            "Violin",
            "Trumpet",
            "Flute",
            "Vocals",
            "Bass guitar",
            "Synthesizer",
        ],
    )
    def test_identifies_non_generic_instruments(self, label: str) -> None:
        """Actual instruments must NOT be marked as generic."""
        assert is_generic_label(label) is False

    def test_filter_generic_labels_from_tuples(self) -> None:
        """Suppresses generic labels from ranked predictions."""
        candidates = [
            ("music", 0.98),
            ("sound effect", 0.91),
            ("Snare drum", 0.85),
            ("audio", 0.72),
            ("Hi-hat", 0.65),
        ]
        filtered = filter_generic_labels(candidates)
        assert len(filtered) == 2
        assert filtered[0] == ("Snare drum", 0.85)
        assert filtered[1] == ("Hi-hat", 0.65)

    def test_filter_generic_labels_all_generic_returns_empty(self) -> None:
        """If all candidates are generic, returns empty list."""
        candidates = [("music", 0.9), ("sound", 0.8), ("instrument", 0.7)]
        filtered = filter_generic_labels(candidates)
        assert len(filtered) == 0


class TestTenFamiliesClassification:
    """Tests evaluating AC-4: Instrument classification and mapping to 10 families."""

    @pytest.mark.parametrize(
        "filename, expected_family",
        [
            # Rhythm / Drums
            ("01_Kick_Hard.wav", "Rhythm/Drums"),
            ("Snare_Top_Dry.wav", "Rhythm/Drums"),
            ("HiHat_Closed_16th.wav", "Rhythm/Drums"),
            ("DrumLoop_120bpm.wav", "Rhythm/Drums"),
            ("Perc_Shaker_Loop.wav", "Rhythm/Drums"),
            ("Toms_Fill.wav", "Rhythm/Drums"),
            ("Crash_Cymbal.wav", "Rhythm/Drums"),

            # Bass
            ("Sub_808_Hit_C1.wav", "Bass"),
            ("Electric_Bass_DI.wav", "Bass"),
            ("SynthBass_Acid_Drop.wav", "Bass"),
            ("Upright_Bass_Jazz.wav", "Bass"),

            # Guitars
            ("Acoustic_Guitar_Strum.wav", "Guitars"),
            ("Electric_Gtr_Lead.wav", "Guitars"),
            ("Gtr_Riff_Distorted.wav", "Guitars"),

            # Keys
            ("Grand_Piano_L.wav", "Keys"),
            ("Rhodes_Mark1_Chords.wav", "Keys"),
            ("B3_Organ_Leslie.wav", "Keys"),

            # Synth
            ("Synth_Lead_Saw.wav", "Synth"),
            ("Pad_Atmosphere_Warm.wav", "Synth"),
            ("Pluck_Arp_128bpm.wav", "Synth"),

            # Vocals
            ("Lead_Vocal_Chorus.wav", "Vocals"),
            ("BGV_Harmonies.wav", "Vocals"),
            ("Vox_Adlib_Dry.wav", "Vocals"),

            # Strings
            ("Violin_Solo_Legato.wav", "Strings"),
            ("Cello_Section_C3.wav", "Strings"),
            ("Strings_Ensemble_Staccato.wav", "Strings"),

            # Brass
            ("Trumpet_Lead_Solo.wav", "Brass"),
            ("Trombone_Sforzando.wav", "Brass"),
            ("Brass_Section_Hit.wav", "Brass"),

            # Woodwinds
            ("Flute_Melody.wav", "Woodwinds"),
            ("Tenor_Sax_Solo.wav", "Woodwinds"),
            ("Clarinet_Arpeggio.wav", "Woodwinds"),

            # Other
            ("FX_Riser_Downlifter.wav", "Other"),
            ("Foley_Footsteps.wav", "Other"),
        ],
    )
    def test_filename_keyword_classification_to_ten_families(
        self, filename: str, expected_family: str
    ) -> None:
        """Every target instrument category correctly resolves to one of the 10 defined families."""
        inst, family, conf = classify_instrument(
            y=np.zeros(100, dtype=np.float32),
            sr=16000,
            filename=filename,
        )
        assert family == expected_family
        assert family in TARGET_FAMILIES
        assert conf > 0.70

    def test_dsp_heuristic_fallback_low_frequencies_to_bass(self) -> None:
        """Ambiguous filename with low-frequency audio resolves to Bass family."""
        sr = 16000
        sub_audio = make_sine_wave(50.0, 1.0, sr)  # Centroid ~50 Hz
        inst, family, conf = classify_instrument(sub_audio, sr=sr, filename="Track_01.wav")

        assert family == "Bass"
        assert family in TARGET_FAMILIES

    def test_dsp_heuristic_fallback_high_frequencies_to_drums(self) -> None:
        """Ambiguous filename with ultra-high frequency audio resolves to Rhythm/Drums (cymbals/air)."""
        sr = 16000
        high_audio = make_sine_wave(7000.0, 1.0, sr)  # Centroid >= 6000 Hz
        inst, family, conf = classify_instrument(high_audio, sr=sr, filename="Stem_Unknown.wav")

        assert family == "Rhythm/Drums"
        assert family in TARGET_FAMILIES

    def test_mock_onnx_model_inference_with_generic_suppression(self) -> None:
        """
        When YAMNet ONNX inference returns scores with generic top labels,
        the classifier must suppress them and select the highest non-generic class.
        """
        class_map = ["music", "sound effect", "Snare drum", "silence", "Electric guitar"]
        mock_session = MagicMock()
        # Mock session input name
        mock_input = MagicMock()
        mock_input.name = "waveform"
        mock_input.shape = [1, 80000]
        mock_session.get_inputs.return_value = [mock_input]

        # Return mock logits: "music" highest, but "Snare drum" highest non-generic
        # Frame scores: shape (1, 5)
        mock_scores = np.array([[0.99, 0.95, 0.88, 0.40, 0.30]], dtype=np.float32)
        mock_session.run.return_value = [mock_scores]

        classifier = YamnetClassifier(session=mock_session, class_map=class_map)
        assert classifier.is_model_available is True

        inst, family, conf = classifier.classify(
            y=np.zeros(80000, dtype=np.float32),
            sr=16000,
            filename="Untitled_Track.wav",
        )

        # "music" and "sound effect" must be filtered out!
        assert inst == "Snare drum"
        assert family == "Rhythm/Drums"
        assert math.isclose(conf, 0.88, abs_tol=1e-3)


# ============================================================================
# End-to-End & Integration Tests: analyze_track
# ============================================================================

class TestAnalyzeTrackIntegration:
    """Integration tests for analyze_track and batch analyze_tracks."""

    def test_analyze_track_from_in_memory_array(self) -> None:
        """Verifies end-to-end TrackAnalysis generation from numpy array."""
        sr = 44100
        # 440 Hz tone (Low-mid band, A pitch class), with "Violin" filename
        audio = make_sine_wave(440.0, 2.0, sr)
        analysis = analyze_track(audio, sr=sr, filename="Lead_Violin.wav")

        assert isinstance(analysis, TrackAnalysis)
        assert analysis.filename == "Lead_Violin.wav"
        assert analysis.family == "Strings"
        assert analysis.frequency_band == "Low-mid"
        assert math.isclose(analysis.spectral_centroid, 440.0, rel_tol=0.15)
        assert len(analysis.chroma) == 12
        assert np.argmax(analysis.chroma) == 9  # A

    def test_analyze_track_dict_and_dataclass_interface(self) -> None:
        """TrackAnalysis must satisfy dataclass attribute access and mapping/dict access."""
        sr = 16000
        audio = make_sine_wave(100.0, 1.0, sr)
        analysis = analyze_track(audio, sr=sr, filename="Bass_DI.wav")

        # Dataclass access
        assert analysis.family == "Bass"
        assert analysis.frequency_band == "Bass"

        # Dict / Mapping access
        assert analysis["family"] == "Bass"
        assert analysis["frequency_band"] == "Bass"
        assert analysis.get("family") == "Bass"
        assert "spectral_centroid" in analysis

        # Serialized dict
        d = analysis.to_dict()
        assert d["family"] == "Bass"
        assert isinstance(d["chroma"], list)
        assert len(d["chroma"]) == 12

    def test_analyze_track_from_wav_file(self) -> None:
        """Writes a physical WAV file and verifies full analysis pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "Kick_808_Beat.wav")
            sr = 44100
            # 60 Hz kick tone with silence padding
            kick = make_sine_wave(60.0, 1.0, sr, amplitude=0.8)
            pad = np.zeros(int(sr * 0.5), dtype=np.float32)
            full_audio = np.concatenate([pad, kick, pad])
            sf.write(file_path, full_audio, sr)

            analysis = analyze_track(file_path)

            assert analysis.file_path == file_path
            assert analysis.filename == "Kick_808_Beat.wav"
            assert analysis.family in ("Rhythm/Drums", "Bass")
            assert analysis.frequency_band in ("Sub-bass", "Bass")
            assert analysis.sample_rate == TARGET_SAMPLING_RATE

    def test_batch_analyze_tracks(self) -> None:
        """Batch analyzes a sequence of tracks."""
        sr = 16000
        t1 = make_sine_wave(100, 1.0, sr)
        t2 = make_sine_wave(1000, 1.0, sr)
        results = analyze_tracks([(t1, sr), (t2, sr)])

        assert len(results) == 2
        assert all(isinstance(r, TrackAnalysis) for r in results)
        assert results[0].frequency_band == "Bass"
        assert results[1].frequency_band == "Low-mid"
