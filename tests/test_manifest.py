"""
Unit tests for Export Manifest Generator (Wave 2 Ticket T-202).

Acceptance Criteria:
  - AC-1: Emits Final_Stems/export_manifest.json with ISO timestamp.
  - AC-2: Records output stem files and mapped contributing source tracks.
  - AC-3: Embeds complete DSP attributes (instrument, family, band, centroid Hz, chroma vector) for each track.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.manifest import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_OUTPUT_DIR,
    GROUPING_MODE_NAMES,
    clean_numpy_value,
    current_iso_timestamp,
    ensure_audio_filename,
    export_manifest,
    frequency_band_from_centroid,
    generate_manifest,
    load_manifest,
    normalize_source_track,
    save_manifest,
    validate_manifest,
)
from engine.analyzer import TrackAnalysis
from engine.grouping import TrackInfo, GroupingMode


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_dsp_tracks() -> list[dict[str, Any]]:
    """Generates realistic mock tracks with full DSP attributes."""
    return [
        {
            "filename": "Kick_In.wav",
            "file_path": "stems_raw/Kick_In.wav",
            "instrument": "Bass drum",
            "family": "Rhythm/Drums",
            "frequency_band": "Bass",
            "spectral_centroid_hz": 92.4,
            "chroma_profile": [0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "filename": "Snare_Top.wav",
            "file_path": "stems_raw/Snare_Top.wav",
            "instrument": "Snare drum",
            "family": "Rhythm/Drums",
            "frequency_band": "Low-mid",
            "spectral_centroid_hz": 420.5,
            "chroma_profile": [0.01, 0.04, 0.08, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "filename": "Bass_DI.wav",
            "file_path": "stems_raw/Bass_DI.wav",
            "instrument": "Electric Bass",
            "family": "Bass",
            "frequency_band": "Bass",
            "spectral_centroid_hz": 115.0,
            "chroma_profile": [0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "filename": "Gtr_Lead.wav",
            "file_path": "stems_raw/Gtr_Lead.wav",
            "instrument": "Electric Guitar",
            "family": "Guitars",
            "frequency_band": "High-mid",
            "spectral_centroid_hz": 2450.0,
            "chroma_profile": [0.1, 0.15, 0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]


@pytest.fixture
def mock_stem_mapping(mock_dsp_tracks) -> dict[str, list[dict[str, Any]]]:
    """Generates a multi-stem mapping dictionary."""
    return {
        "Drums.wav": [mock_dsp_tracks[0], mock_dsp_tracks[1]],
        "Bass.wav": [mock_dsp_tracks[2]],
        "Guitars.wav": [mock_dsp_tracks[3]],
    }


# ============================================================================
# AC-1: Manifest Generation & ISO Timestamp Tests
# ============================================================================

class TestAC1ManifestGenerationAndTimestamp:
    """Verifies AC-1: Emits Final_Stems/export_manifest.json with ISO timestamp."""

    def test_current_iso_timestamp_format(self):
        """Timestamp must be ISO 8601 UTC with 'Z' suffix."""
        ts = current_iso_timestamp()
        assert isinstance(ts, str)
        assert ts.endswith("Z")
        # Validate that datetime can parse it as UTC
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    def test_generate_manifest_top_level_metadata(self, mock_stem_mapping):
        """Manifest contains app, version, timestamp, modes, and track totals."""
        manifest = generate_manifest(mock_stem_mapping, grouping_mode=1)

        assert manifest["app"] == "Smart Export"
        assert manifest["version"] == "1.0.0"
        assert isinstance(manifest["timestamp"], str)
        assert manifest["timestamp"].endswith("Z")

        # Validate timestamp parseability
        parsed_dt = datetime.fromisoformat(manifest["timestamp"].replace("Z", "+00:00"))
        assert parsed_dt.year >= 2026

        assert manifest["grouping_mode"] == 1
        assert manifest["grouping_mode_name"] == "By Instrument Family"
        assert manifest["total_output_stems"] == 3
        assert manifest["total_input_tracks"] == 4

    def test_grouping_mode_names(self, mock_stem_mapping):
        """Tests standard grouping modes and custom name resolution."""
        # Mode 0: Rename only
        m0 = generate_manifest(mock_stem_mapping, grouping_mode=0)
        assert m0["grouping_mode"] == 0
        assert m0["grouping_mode_name"] == "Rename Only"

        # Mode 1: By family
        m1 = generate_manifest(mock_stem_mapping, grouping_mode=1)
        assert m1["grouping_mode"] == 1
        assert m1["grouping_mode_name"] == "By Instrument Family"

        # Mode 2: By frequency
        m2 = generate_manifest(mock_stem_mapping, grouping_mode=2)
        assert m2["grouping_mode"] == 2
        assert m2["grouping_mode_name"] == "By Frequency Band"

        # Custom explicit grouping mode name
        m_custom = generate_manifest(
            mock_stem_mapping,
            grouping_mode=99,
            grouping_mode_name="Experimental Custom Cluster",
        )
        assert m_custom["grouping_mode"] == 99
        assert m_custom["grouping_mode_name"] == "Experimental Custom Cluster"

    def test_save_manifest_to_default_directory(self, tmp_path, monkeypatch, mock_stem_mapping):
        """Saves manifest to default Final_Stems/export_manifest.json."""
        monkeypatch.chdir(tmp_path)
        manifest = generate_manifest(mock_stem_mapping)
        out_path = save_manifest(manifest)

        expected_path = os.path.normpath("Final_Stems/export_manifest.json")
        assert os.path.normpath(out_path) == expected_path
        assert os.path.exists(expected_path)

        # Verify saved content is valid JSON
        with open(expected_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["app"] == "Smart Export"
        assert len(data["stems"]) == 3

    def test_save_manifest_to_custom_directory(self, tmp_path, mock_stem_mapping):
        """Saves manifest to custom output directory and verifies file creation."""
        custom_dir = tmp_path / "custom_exports" / "project_alpha"
        manifest = generate_manifest(mock_stem_mapping, output_dir=str(custom_dir))
        out_path = save_manifest(manifest, output_dir=str(custom_dir))

        assert os.path.exists(out_path)
        loaded = load_manifest(out_path)
        assert loaded["total_output_stems"] == 3
        assert loaded["total_input_tracks"] == 4

    def test_export_manifest_convenience_helper(self, tmp_path, mock_stem_mapping):
        """Tests one-step generation and emission."""
        custom_dir = str(tmp_path / "one_step")
        manifest, path = export_manifest(mock_stem_mapping, output_dir=custom_dir, grouping_mode=1)

        assert os.path.exists(path)
        assert manifest["total_output_stems"] == 3
        assert is_valid(manifest)


# ============================================================================
# AC-2: Stem & Source Track Mapping Tests
# ============================================================================

class TestAC2StemAndTrackMapping:
    """Verifies AC-2: Records output stem files and mapped contributing source tracks."""

    def test_records_output_stem_files(self, mock_stem_mapping):
        """Ensures all output stem files are recorded with accurate track counts."""
        manifest = generate_manifest(mock_stem_mapping)
        stems = manifest["stems"]

        assert len(stems) == 3
        stem_names = [s["stem_file"] for s in stems]
        assert "Drums.wav" in stem_names
        assert "Bass.wav" in stem_names
        assert "Guitars.wav" in stem_names

        drums_stem = next(s for s in stems if s["stem_file"] == "Drums.wav")
        assert drums_stem["track_count"] == 2
        assert len(drums_stem["source_tracks"]) == 2

        bass_stem = next(s for s in stems if s["stem_file"] == "Bass.wav")
        assert bass_stem["track_count"] == 1
        assert len(bass_stem["source_tracks"]) == 1

    def test_stem_filename_extension_normalization(self):
        """Ensures stem files without .wav extension are safely normalized."""
        mapping = {
            "Drums": [{"filename": "Kick.wav", "spectral_centroid_hz": 80.0}],
            "Synth_Pad.wav": [{"filename": "Pad.wav", "spectral_centroid_hz": 1200.0}],
            "Strings.flac": [{"filename": "Vln.flac", "spectral_centroid_hz": 1800.0}],
        }
        manifest = generate_manifest(mapping)
        files = [s["stem_file"] for s in manifest["stems"]]

        assert files[0] == "Drums.wav"
        assert files[1] == "Synth_Pad.wav"
        assert files[2] == "Strings.flac"

    def test_dynamic_subgroup_stems_mapped(self):
        """Tests collision split stems like Drums.wav and Drums_Part2.wav."""
        mapping = {
            "Drums.wav": [
                {"filename": "Kick.wav", "spectral_centroid_hz": 70.0},
                {"filename": "Snare.wav", "spectral_centroid_hz": 350.0},
            ],
            "Drums_Part2.wav": [
                {"filename": "Sub_Kick.wav", "spectral_centroid_hz": 72.0},
            ],
        }
        manifest = generate_manifest(mapping)
        assert manifest["total_output_stems"] == 2
        assert manifest["total_input_tracks"] == 3

        stems = {s["stem_file"]: s for s in manifest["stems"]}
        assert "Drums.wav" in stems
        assert "Drums_Part2.wav" in stems
        assert stems["Drums.wav"]["track_count"] == 2
        assert stems["Drums_Part2.wav"]["track_count"] == 1

    def test_empty_mapping_handled_cleanly(self):
        """Empty mapping yields 0 tracks and 0 stems without errors."""
        manifest = generate_manifest({})
        assert manifest["total_output_stems"] == 0
        assert manifest["total_input_tracks"] == 0
        assert manifest["stems"] == []


# ============================================================================
# AC-3: DSP Attribute Embedding Tests
# ============================================================================

class TestAC3DSPAttributeEmbedding:
    """
    Verifies AC-3: Embeds complete DSP attributes (instrument, family, band,
    centroid Hz, chroma vector) for each track.
    """

    def test_all_dsp_attributes_present_in_each_track(self, mock_stem_mapping):
        """Verifies every source track contains all 5 required DSP attributes + file info."""
        manifest = generate_manifest(mock_stem_mapping)

        required_dsp_keys = {
            "instrument",
            "family",
            "frequency_band",
            "spectral_centroid_hz",
            "chroma_profile",
        }
        required_file_keys = {"filename", "file_path"}

        for stem in manifest["stems"]:
            for track in stem["source_tracks"]:
                # Check all DSP keys present
                for key in required_dsp_keys:
                    assert key in track, f"Missing DSP attribute '{key}' in track {track}"
                # Check file identification keys present
                for key in required_file_keys:
                    assert key in track, f"Missing file key '{key}' in track {track}"

                # Verify types
                assert isinstance(track["instrument"], str)
                assert isinstance(track["family"], str)
                assert isinstance(track["frequency_band"], str)
                assert isinstance(track["spectral_centroid_hz"], (int, float))
                assert isinstance(track["chroma_profile"], list)
                assert len(track["chroma_profile"]) == 12

    def test_spectral_centroid_mapping_and_band_fallback(self):
        """Tests mapping from centroid to standard frequency band when band not provided."""
        track_no_band = {
            "filename": "Sub.wav",
            "file_path": "Sub.wav",
            "spectral_centroid": 50.0,  # Sub-bass
        }
        normalized = normalize_source_track(track_no_band)
        assert normalized["frequency_band"] == "Sub-bass"
        assert normalized["spectral_centroid_hz"] == 50.0

        track_high = {
            "filename": "Cymbal.wav",
            "spectral_centroid_hz": 8000.0,
        }
        normalized_high = normalize_source_track(track_high)
        assert normalized_high["frequency_band"] == "High"
        assert normalized_high["spectral_centroid_hz"] == 8000.0

    def test_chroma_vector_dimensions_and_padding(self):
        """Chroma vector must be exactly 12 bins; pads short vectors or truncates long ones."""
        # Short chroma (3 bins)
        t_short = {
            "filename": "test.wav",
            "chroma": [0.1, 0.2, 0.3],
        }
        n_short = normalize_source_track(t_short)
        assert len(n_short["chroma_profile"]) == 12
        assert n_short["chroma_profile"][:3] == [0.1, 0.2, 0.3]
        assert n_short["chroma_profile"][3:] == [0.0] * 9

        # No chroma provided
        t_none = {"filename": "none.wav"}
        n_none = normalize_source_track(t_none)
        assert len(n_none["chroma_profile"]) == 12
        assert n_none["chroma_profile"] == [0.0] * 12

    def test_aliases_for_dsp_attributes(self):
        """Supports various attribute alias names cleanly."""
        t1 = {
            "name": "Kick.wav",
            "filepath": "path/to/Kick.wav",
            "label": "Bass Drum",
            "instrument_family": "Rhythm/Drums",
            "band": "Bass",
            "centroid": 85.5,
            "chroma_vector": [0.1] * 12,
        }
        norm = normalize_source_track(t1)
        assert norm["filename"] == "Kick.wav"
        assert norm["file_path"] == "path/to/Kick.wav"
        assert norm["instrument"] == "Bass Drum"
        assert norm["family"] == "Rhythm/Drums"
        assert norm["frequency_band"] == "Bass"
        assert norm["spectral_centroid_hz"] == 85.5
        assert norm["chroma_profile"] == [0.1] * 12


# ============================================================================
# Numpy Serialization & Clean Typing Tests
# ============================================================================

class TestNumpySerialization:
    """Verifies that numpy floats, ints, and arrays serialize cleanly to JSON."""

    def test_clean_numpy_value_primitives(self):
        """Converts np.float32, np.float64, np.int64, np.bool_ cleanly."""
        assert clean_numpy_value(np.float32(42.5)) == 42.5
        assert isinstance(clean_numpy_value(np.float32(42.5)), float)

        assert clean_numpy_value(np.int64(100)) == 100
        assert isinstance(clean_numpy_value(np.int64(100)), int)

        assert clean_numpy_value(np.bool_(True)) is True
        assert isinstance(clean_numpy_value(np.bool_(True)), bool)

    def test_clean_numpy_arrays(self):
        """Converts numpy ndarray to standard python list of floats."""
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cleaned = clean_numpy_value(arr)
        assert isinstance(cleaned, list)
        assert all(isinstance(x, float) for x in cleaned)
        assert cleaned == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)

    def test_manifest_with_raw_numpy_types_serializes_without_type_error(self, tmp_path):
        """Passing raw numpy types in stems mapping does not raise TypeError in json.dump."""
        raw_numpy_track = {
            "filename": "Numpy_Track.wav",
            "file_path": "path/Numpy_Track.wav",
            "instrument": "Synthesizer",
            "family": "Synth",
            "frequency_band": "High-mid",
            "spectral_centroid_hz": np.float32(2345.67),
            "chroma_profile": np.ones((12,), dtype=np.float32) * 0.5,
        }
        mapping = {"Synth.wav": [raw_numpy_track]}

        # generate_manifest cleans numpy values
        manifest = generate_manifest(mapping)

        # Must serialize cleanly with json.dumps
        json_str = json.dumps(manifest)
        assert isinstance(json_str, str)

        # Must save cleanly to disk
        save_path = save_manifest(manifest, output_dir=str(tmp_path))
        assert os.path.exists(save_path)

        loaded = load_manifest(save_path)
        assert loaded["stems"][0]["source_tracks"][0]["spectral_centroid_hz"] == pytest.approx(2345.67, rel=1e-3)
        assert len(loaded["stems"][0]["source_tracks"][0]["chroma_profile"]) == 12

    def test_nan_and_inf_handling(self):
        """NaN and Inf values are safely converted to 0.0 to maintain valid JSON."""
        nan_track = {
            "filename": "Nan_Track.wav",
            "spectral_centroid_hz": np.nan,
            "chroma_profile": [np.nan] * 12,
        }
        norm = normalize_source_track(nan_track)
        assert norm["spectral_centroid_hz"] == 0.0
        assert norm["chroma_profile"] == [0.0] * 12


# ============================================================================
# Engine Interoperability Tests
# ============================================================================

class TestEngineInteroperability:
    """Verifies seamless interoperability with TrackAnalysis and TrackInfo."""

    def test_interoperability_with_track_analysis(self, tmp_path):
        """Tests passing TrackAnalysis objects directly into stems_mapping."""
        track_analysis_1 = TrackAnalysis(
            file_path="audio/Kick.wav",
            filename="Kick.wav",
            instrument="Kick drum",
            family="Rhythm/Drums",
            frequency_band="Bass",
            spectral_centroid=88.5,
            chroma=[0.02] * 12,
            confidence=0.95,
        )
        track_analysis_2 = TrackAnalysis(
            file_path="audio/Snare.wav",
            filename="Snare.wav",
            instrument="Snare drum",
            family="Rhythm/Drums",
            frequency_band="Low-mid",
            spectral_centroid=350.0,
            chroma=[0.05] * 12,
            confidence=0.92,
        )

        mapping = {"Drums.wav": [track_analysis_1, track_analysis_2]}
        manifest = generate_manifest(mapping)

        assert manifest["total_output_stems"] == 1
        assert manifest["total_input_tracks"] == 2
        stem = manifest["stems"][0]
        assert stem["stem_file"] == "Drums.wav"

        t1 = stem["source_tracks"][0]
        assert t1["filename"] == "Kick.wav"
        assert t1["instrument"] == "Kick drum"
        assert t1["spectral_centroid_hz"] == 88.5
        assert len(t1["chroma_profile"]) == 12

    def test_interoperability_with_track_info(self):
        """Tests passing TrackInfo objects from engine.grouping."""
        track_info = TrackInfo(
            file_path="audio/Bass.wav",
            filename="Bass.wav",
            instrument="Electric Bass",
            family="Bass",
            frequency_band="Bass",
            spectral_centroid=110.0,
            chroma=np.full(12, 0.1, dtype=np.float32),
        )

        mapping = {"Bass.wav": [track_info]}
        manifest = generate_manifest(mapping)

        assert manifest["total_output_stems"] == 1
        assert manifest["total_input_tracks"] == 1
        t = manifest["stems"][0]["source_tracks"][0]
        assert t["filename"] == "Bass.wav"
        assert t["spectral_centroid_hz"] == 110.0
        assert isinstance(t["chroma_profile"], list)


# ============================================================================
# Schema Validation Helper Tests
# ============================================================================

def is_valid(manifest: dict[str, Any]) -> bool:
    valid, _ = validate_manifest(manifest)
    return valid


class TestSchemaValidation:
    """Verifies schema validation helper detects conforming and malformed manifests."""

    def test_valid_manifest_passes(self, mock_stem_mapping):
        manifest = generate_manifest(mock_stem_mapping)
        valid, errors = validate_manifest(manifest)
        assert valid is True
        assert errors == []

    def test_missing_root_key_fails(self, mock_stem_mapping):
        manifest = generate_manifest(mock_stem_mapping)
        del manifest["stems"]
        valid, errors = validate_manifest(manifest)
        assert valid is False
        assert any("Missing required manifest key: 'stems'" in e for e in errors)

    def test_invalid_timestamp_fails(self, mock_stem_mapping):
        manifest = generate_manifest(mock_stem_mapping)
        manifest["timestamp"] = "not-a-timestamp"
        valid, errors = validate_manifest(manifest)
        assert valid is False
        assert any("Invalid ISO 8601 timestamp" in e for e in errors)

    def test_mismatched_track_count_fails(self, mock_stem_mapping):
        manifest = generate_manifest(mock_stem_mapping)
        manifest["stems"][0]["track_count"] = 999
        valid, errors = validate_manifest(manifest)
        assert valid is False
        assert any("track_count" in e for e in errors)

    def test_invalid_chroma_bins_fails(self, mock_stem_mapping):
        manifest = generate_manifest(mock_stem_mapping)
        manifest["stems"][0]["source_tracks"][0]["chroma_profile"] = [0.1, 0.2]  # Only 2 bins
        valid, errors = validate_manifest(manifest)
        assert valid is False
        assert any("chroma_profile" in e for e in errors)
