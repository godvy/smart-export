"""
Unit and integration tests for Spectral Masking & Stem Grouping Engine (T-102).

Acceptance Criteria:
  - AC-1: Supports Grouping Mode 0 (Rename only), Mode 1 (By Instrument family),
          and Mode 2 (By Frequency band).
  - AC-2: Evaluates 15% centroid collision rule: abs(C_A - C_B) / avg(C_A, C_B) < 0.15.
  - AC-3: Dynamically splits colliding tracks into safe subgroups (e.g. Drums.wav, Drums_Part2.wav).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.grouping import (
    TrackInfo,
    GroupingMode,
    GroupingDict,
    StemGrouper,
    as_track_info,
    calculate_centroid_diff,
    tracks_collide,
    resolve_subgroups,
    group_tracks,
    frequency_band_from_centroid,
    normalize_family,
    normalize_frequency_band,
)


# ============================================================================
# AC-2: Spectral Centroid Math & 15% Collision Rule Tests
# ============================================================================

class TestCentroidCollisionRule:
    """Tests evaluating AC-2: 15% centroid collision rule."""

    def test_centroid_diff_identical_frequencies(self) -> None:
        """Identical frequencies produce 0.0 relative difference."""
        assert calculate_centroid_diff(100.0, 100.0) == 0.0
        assert calculate_centroid_diff(1000.0, 1000.0) == 0.0

    def test_centroid_diff_zero_frequencies(self) -> None:
        """Both frequencies zero produces 0.0 difference."""
        assert calculate_centroid_diff(0.0, 0.0) == 0.0

    def test_centroid_diff_negative_raises_value_error(self) -> None:
        """Negative frequencies must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_centroid_diff(-100.0, 100.0)
        with pytest.raises(ValueError, match="non-negative"):
            calculate_centroid_diff(100.0, -50.0)

    def test_centroid_diff_symmetry(self) -> None:
        """diff(A, B) must equal diff(B, A)."""
        diff1 = calculate_centroid_diff(100.0, 150.0)
        diff2 = calculate_centroid_diff(150.0, 100.0)
        assert math.isclose(diff1, diff2, rel_tol=1e-9)
        # Expected: |100 - 150| / 125 = 50 / 125 = 0.4
        assert math.isclose(diff1, 0.4, rel_tol=1e-9)

    def test_centroid_diff_example_from_contract(self) -> None:
        """
        Verify exact example from specification:
        Kick at 100 Hz and 808 at 105 Hz -> diff = 5 / 102.5 ≈ 0.04878 < 0.15.
        """
        diff = calculate_centroid_diff(100.0, 105.0)
        expected = 5.0 / 102.5
        assert math.isclose(diff, expected, rel_tol=1e-6)
        assert diff < 0.15
        assert tracks_collide(100.0, 105.0) is True

    def test_edge_case_diff_014_collides(self) -> None:
        """
        Edge case: Relative difference ≈ 0.14 (< 0.15 threshold) MUST collide.
        c1 = 1000, c2 ≈ 1150.5376 -> diff = 150.5376 / 1075.2688 ≈ 0.1400.
        """
        c1 = 1000.0
        c2 = 2140.0 / 1.86  # gives exact diff of 0.14
        diff = calculate_centroid_diff(c1, c2)
        assert math.isclose(diff, 0.14, rel_tol=1e-4)
        assert tracks_collide(c1, c2, threshold=0.15) is True

    def test_edge_case_diff_016_does_not_collide(self) -> None:
        """
        Edge case: Relative difference ≈ 0.16 (>= 0.15 threshold) MUST NOT collide.
        c1 = 1000, c2 ≈ 1173.913 -> diff = 173.913 / 1086.9565 ≈ 0.1600.
        """
        c1 = 1000.0
        c2 = 2160.0 / 1.84  # gives exact diff of 0.16
        diff = calculate_centroid_diff(c1, c2)
        assert math.isclose(diff, 0.16, rel_tol=1e-4)
        assert tracks_collide(c1, c2, threshold=0.15) is False

    def test_exact_boundary_threshold(self) -> None:
        """
        Boundary condition: diff == threshold (0.15).
        Condition is diff < threshold, so diff == 0.15 is NOT a collision.
        """
        c1 = 1000.0
        c2 = 2150.0 / 1.85  # gives exact diff of 0.15
        diff = calculate_centroid_diff(c1, c2)
        assert math.isclose(diff, 0.15, rel_tol=1e-6)
        assert tracks_collide(c1, c2, threshold=0.15) is False

    def test_tracks_collide_with_track_objects_and_dicts(self) -> None:
        """tracks_collide should accept TrackInfo objects and dicts."""
        t1 = TrackInfo(filename="kick.wav", spectral_centroid=100.0)
        t2 = TrackInfo(filename="808.wav", spectral_centroid=105.0)
        t3 = {"filename": "snare.wav", "spectral_centroid": 250.0}

        assert tracks_collide(t1, t2) is True
        assert tracks_collide(t1, t3) is False
        assert tracks_collide(t2, t3) is False


# ============================================================================
# AC-3: Dynamic Subgroup Partitioning Tests
# ============================================================================

class TestSubgroupPartitioning:
    """Tests evaluating AC-3: Dynamic splitting of colliding tracks into safe subgroups."""

    def test_no_collision_single_stem(self) -> None:
        """
        Tracks with no pairwise collisions remain consolidated in a single stem
        using base_stem_name.
        """
        tracks = [
            TrackInfo(filename="kick.wav", instrument="Kick", spectral_centroid=100.0),
            TrackInfo(filename="snare.wav", instrument="Snare", spectral_centroid=300.0),
            TrackInfo(filename="hihat.wav", instrument="Hi-hat", spectral_centroid=8000.0),
        ]
        res = resolve_subgroups("Drums.wav", tracks)
        assert len(res) == 1
        assert "Drums.wav" in res
        assert len(res["Drums.wav"]) == 3
        # Primary name without part suffix
        assert list(res.keys()) == ["Drums.wav"]

    def test_two_colliding_tracks_split_into_part1_and_part2(self) -> None:
        """
        Kick (100 Hz) and 808 (105 Hz) collide:
        Kick stays in Drums.wav, 808 is pushed to Drums_Part2.wav.
        """
        kick = TrackInfo(filename="kick.wav", instrument="Kick", spectral_centroid=100.0)
        sub808 = TrackInfo(filename="808.wav", instrument="808", spectral_centroid=105.0)

        res = resolve_subgroups("Drums.wav", [kick, sub808])
        assert len(res) == 2
        assert "Drums.wav" in res
        assert "Drums_Part2.wav" in res
        assert res["Drums.wav"] == [kick]
        assert res["Drums_Part2.wav"] == [sub808]

    def test_three_colliding_tracks_split_into_part1_part2_part3(self) -> None:
        """
        Three tracks that all collide pairwise:
        Track 1 -> base name (Part 1)
        Track 2 -> Part 2
        Track 3 -> Part 3
        """
        t1 = TrackInfo(filename="sub1.wav", spectral_centroid=100.0)
        t2 = TrackInfo(filename="sub2.wav", spectral_centroid=104.0)
        t3 = TrackInfo(filename="sub3.wav", spectral_centroid=108.0)

        res = resolve_subgroups("Bass.wav", [t1, t2, t3])
        assert len(res) == 3
        assert list(res.keys()) == ["Bass.wav", "Bass_Part2.wav", "Bass_Part3.wav"]
        assert res["Bass.wav"] == [t1]
        assert res["Bass_Part2.wav"] == [t2]
        assert res["Bass_Part3.wav"] == [t3]

    def test_mixed_collision_and_consolidation(self) -> None:
        """
        4 tracks:
          Kick (100 Hz), 808 (105 Hz), Snare (300 Hz), Clap (310 Hz).
        - Kick and Snare do NOT collide (100 vs 300) -> safe in Drums.wav.
        - 808 and Clap do NOT collide (105 vs 310) -> safe in Drums_Part2.wav.
        """
        kick = TrackInfo(filename="kick.wav", spectral_centroid=100.0)
        sub808 = TrackInfo(filename="808.wav", spectral_centroid=105.0)
        snare = TrackInfo(filename="snare.wav", spectral_centroid=300.0)
        clap = TrackInfo(filename="clap.wav", spectral_centroid=310.0)

        res = resolve_subgroups("Drums.wav", [kick, sub808, snare, clap])
        assert len(res) == 2
        assert "Drums.wav" in res
        assert "Drums_Part2.wav" in res

        # Check internal collision safety in every subgroup
        for stem_name, track_list in res.items():
            for i in range(len(track_list)):
                for j in range(i + 1, len(track_list)):
                    assert not tracks_collide(track_list[i], track_list[j]), (
                        f"Collision detected inside {stem_name} between "
                        f"{track_list[i].filename} and {track_list[j].filename}"
                    )

        assert res["Drums.wav"] == [kick, snare]
        assert res["Drums_Part2.wav"] == [sub808, clap]

    def test_subgroups_without_file_extension(self) -> None:
        """
        If base_stem_name has no file extension (e.g. 'Drums'),
        subgroups should be 'Drums', 'Drums_Part2', etc.
        """
        kick = TrackInfo(filename="kick.wav", spectral_centroid=100.0)
        sub808 = TrackInfo(filename="808.wav", spectral_centroid=105.0)

        res = resolve_subgroups("Drums", [kick, sub808])
        assert "Drums" in res
        assert "Drums_Part2" in res
        assert list(res.keys()) == ["Drums", "Drums_Part2"]

    def test_empty_and_single_track_subgroups(self) -> None:
        """Empty tracks returns empty dict; single track returns base stem."""
        assert resolve_subgroups("Vocals.wav", []) == {}

        vocal = TrackInfo(filename="lead.wav", spectral_centroid=1200.0)
        res = resolve_subgroups("Vocals.wav", [vocal])
        assert len(res) == 1
        assert res["Vocals.wav"] == [vocal]


# ============================================================================
# AC-1: Grouping Modes Tests (Mode 0, Mode 1, Mode 2)
# ============================================================================

class TestGroupingModes:
    """Tests evaluating AC-1: Grouping Mode 0, Mode 1, and Mode 2."""

    def test_mode_0_rename_only_does_not_sum(self) -> None:
        """
        Mode 0 (Rename Only): Does not sum. Each track forms its own stem
        named after detected instrument or clean name.
        """
        tracks = [
            TrackInfo(filename="01_kick.wav", instrument="Kick drum", spectral_centroid=100.0),
            TrackInfo(filename="02_snare.wav", instrument="Snare drum", spectral_centroid=250.0),
            TrackInfo(filename="03_hihat.wav", instrument="Hi-hat", spectral_centroid=7000.0),
        ]
        res = group_tracks(tracks, mode=GroupingMode.RENAME_ONLY)
        assert len(res) == 3
        # Each track forms its own stem with exactly 1 track
        for stem_name, track_list in res.items():
            assert len(track_list) == 1

        assert "Kick drum" in res
        assert "Snare drum" in res
        assert "Hi-hat" in res

    def test_mode_0_duplicate_instruments_get_part_suffixes(self) -> None:
        """
        In Mode 0, duplicate instrument names get distinct stem names via _Part2.
        """
        tracks = [
            TrackInfo(filename="kick_main.wav", instrument="Kick drum", spectral_centroid=100.0),
            TrackInfo(filename="kick_sub.wav", instrument="Kick drum", spectral_centroid=90.0),
        ]
        res = group_tracks(tracks, mode=0)
        assert len(res) == 2
        assert "Kick drum" in res
        assert "Kick drum_Part2" in res
        assert res["Kick drum"][0].filename == "kick_main.wav"
        assert res["Kick drum_Part2"][0].filename == "kick_sub.wav"

    def test_mode_0_fallback_to_filename_when_no_instrument(self) -> None:
        """Tracks with empty/Unknown instrument fall back to clean filename."""
        tracks = [
            TrackInfo(file_path="audio/ambient_layer.wav", instrument="Unknown"),
            TrackInfo(filename="synth_lead.aif", instrument=""),
        ]
        res = group_tracks(tracks, mode=0)
        assert "ambient_layer" in res
        assert "synth_lead" in res

    def test_mode_1_by_instrument_family(self) -> None:
        """
        Mode 1 (By Instrument Family): Group by family (e.g. Drums, Guitars, Bass).
        Collisions within each family are split into subgroups.
        """
        tracks = [
            # Drums family
            TrackInfo(filename="kick.wav", family="Rhythm/Drums", spectral_centroid=100.0),
            TrackInfo(filename="808.wav", family="Rhythm/Drums", spectral_centroid=105.0),  # collides with kick
            TrackInfo(filename="snare.wav", family="Drums", spectral_centroid=300.0),        # safe with kick
            # Bass family
            TrackInfo(filename="bass_synth.wav", family="Bass", spectral_centroid=150.0),
            # Guitars family
            TrackInfo(filename="guitar_lead.wav", family="Guitars", spectral_centroid=1200.0),
            TrackInfo(filename="guitar_rhythm.wav", family="Guitars", spectral_centroid=1220.0), # collides with lead
        ]

        res = group_tracks(tracks, mode=GroupingMode.BY_FAMILY)

        # Drums should have Drums and Drums_Part2
        assert "Drums" in res
        assert "Drums_Part2" in res
        assert len(res["Drums"]) == 2  # kick and snare
        assert len(res["Drums_Part2"]) == 1  # 808

        # Bass should be a single consolidated stem
        assert "Bass" in res
        assert len(res["Bass"]) == 1

        # Guitars should have Guitars and Guitars_Part2
        assert "Guitars" in res
        assert "Guitars_Part2" in res
        assert len(res["Guitars"]) == 1
        assert len(res["Guitars_Part2"]) == 1

    def test_mode_1_with_extension(self) -> None:
        """Mode 1 with extension='.wav' formats stem names with .wav."""
        tracks = [
            TrackInfo(filename="kick.wav", family="Drums", spectral_centroid=100.0),
            TrackInfo(filename="808.wav", family="Drums", spectral_centroid=105.0),
        ]
        res = group_tracks(tracks, mode=1, extension=".wav")
        assert "Drums.wav" in res
        assert "Drums_Part2.wav" in res

    def test_mode_2_by_frequency_band(self) -> None:
        """
        Mode 2 (By Frequency Band): Group by frequency band (Sub-bass, Bass, Low-mid, High-mid, High).
        Collisions within band are split into subgroups.
        """
        tracks = [
            # Low-mid Band (250 - 2000 Hz)
            TrackInfo(filename="vocal.wav", frequency_band="Low-mid", spectral_centroid=800.0),
            TrackInfo(filename="keys.wav", frequency_band="Low-mid", spectral_centroid=820.0), # collides (diff = 20/810 ≈ 0.024)
            # High Band (>= 6000 Hz)
            TrackInfo(filename="shaker.wav", frequency_band="High", spectral_centroid=7000.0),
            TrackInfo(filename="air.wav", frequency_band="High", spectral_centroid=12000.0),   # safe (diff = 5000/9500 ≈ 0.526)
        ]

        res = group_tracks(tracks, mode=GroupingMode.BY_FREQUENCY)

        # Low-mid should split
        assert "Low-mid Band" in res
        assert "Low-mid Band_Part2" in res
        assert len(res["Low-mid Band"]) == 1
        assert len(res["Low-mid Band_Part2"]) == 1

        # High should consolidate
        assert "High Band" in res
        assert len(res["High Band"]) == 2

    def test_mode_string_and_enum_compatibility(self) -> None:
        """group_tracks supports int, Enum, and descriptive string modes."""
        tracks = [TrackInfo(filename="kick.wav", family="Drums", spectral_centroid=100.0)]

        res_int = group_tracks(tracks, mode=1)
        res_enum = group_tracks(tracks, mode=GroupingMode.BY_FAMILY)
        res_str = group_tracks(tracks, mode="family")

        assert "Drums" in res_int
        assert "Drums" in res_enum
        assert "Drums" in res_str

    def test_invalid_mode_raises_value_error(self) -> None:
        """Invalid mode values raise ValueError."""
        tracks = [TrackInfo(filename="kick.wav", spectral_centroid=100.0)]
        with pytest.raises(ValueError, match="Unsupported grouping mode"):
            group_tracks(tracks, mode=99)
        with pytest.raises(ValueError, match="Unrecognized grouping mode"):
            group_tracks(tracks, mode="invalid_mode")


# ============================================================================
# Data Structure & Helper Tests
# ============================================================================

class TestDataStructuresAndHelpers:
    """Tests for TrackInfo, GroupingDict, and normalization helpers."""

    def test_track_info_post_init_and_access(self) -> None:
        """TrackInfo initializes cleanly, supports dict access and to_dict."""
        chroma = np.ones(12, dtype=np.float32)
        track = TrackInfo(
            file_path="/audio/kick.wav",
            instrument="Kick drum",
            family="Rhythm/Drums",
            spectral_centroid=110.0,
            chroma=chroma,
        )
        assert track.filename == "kick.wav"
        assert track.frequency_band == "Bass"  # Inferred 60-250 Hz
        assert track["instrument"] == "Kick drum"
        assert track.get("family") == "Rhythm/Drums"

        d = track.to_dict()
        assert d["spectral_centroid"] == 110.0
        assert isinstance(d["chroma"], list)
        assert len(d["chroma"]) == 12

    def test_as_track_info_from_dict(self) -> None:
        """as_track_info cleanly parses dict inputs."""
        raw = {
            "file_path": "stems/lead.wav",
            "instrument": "Synth Lead",
            "family": "Synth",
            "spectral_centroid": 1500.0,
        }
        track = as_track_info(raw)
        assert isinstance(track, TrackInfo)
        assert track.filename == "lead.wav"
        assert track.spectral_centroid == 1500.0

    def test_grouping_dict_flexible_key_lookup(self) -> None:
        """GroupingDict allows accessing with or without .wav and Band suffix."""
        gd = GroupingDict()
        gd["Drums"] = ["track1"]
        gd["Low-mid Band"] = ["track2"]

        # Access with .wav
        assert gd["Drums.wav"] == ["track1"]
        assert "Drums.wav" in gd
        assert gd.get("Drums.wav") == ["track1"]

        # Access without Band
        assert gd["Low-mid"] == ["track2"]
        assert "Low-mid" in gd
        assert gd["Low-mid.wav"] == ["track2"]

        # Non-existent key
        with pytest.raises(KeyError):
            _ = gd["NonExistent"]
        assert gd.get("NonExistent", None) is None

    def test_frequency_band_boundaries(self) -> None:
        """Test boundary mappings for frequency_band_from_centroid."""
        assert frequency_band_from_centroid(40.0) == "Sub-bass"
        assert frequency_band_from_centroid(100.0) == "Bass"
        assert frequency_band_from_centroid(1000.0) == "Low-mid"
        assert frequency_band_from_centroid(3500.0) == "High-mid"
        assert frequency_band_from_centroid(10000.0) == "High"

    def test_normalize_family(self) -> None:
        """normalize_family maps known families and sanitizes unusual ones."""
        assert normalize_family("Rhythm/Drums") == "Drums"
        assert normalize_family("drums") == "Drums"
        assert normalize_family("Lead Vocal") == "Vocals"
        assert normalize_family("Piano") == "Keys"
        assert normalize_family("Custom/Instrument") == "Custom_Instrument"
        assert normalize_family(None) == "Other"

    def test_stem_grouper_class_interface(self) -> None:
        """StemGrouper OOP wrapper functions correctly."""
        grouper = StemGrouper(collision_threshold=0.15)
        diff = grouper.calculate_centroid_diff(100.0, 105.0)
        assert diff < 0.15
        assert grouper.tracks_collide(100.0, 105.0) is True

        tracks = [
            TrackInfo(filename="kick.wav", family="Drums", spectral_centroid=100.0),
            TrackInfo(filename="808.wav", family="Drums", spectral_centroid=105.0),
        ]
        res = grouper.group_tracks(tracks, mode=GroupingMode.BY_FAMILY)
        assert "Drums" in res
        assert "Drums_Part2" in res
