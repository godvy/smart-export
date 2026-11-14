"""
Smart Export DSP Engine.

Exposes core modules and convenience entrypoints:
  - analyzer: Feature extraction, DSP conditioning, and instrument classification.
  - grouping: Intelligent stem grouping and collision detection.
  - summer: Audio summing with float32 precision and headroom protection.
  - manifest: Manifest generation, validation, and JSON export.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import analyzer, grouping, manifest, summer
from .analyzer import TrackAnalysis, analyze_track
from .grouping import GroupingMode, TrackInfo, group_tracks
from .manifest import generate_manifest, load_manifest, save_manifest
from .summer import apply_headroom_protection, sum_audio_files

# Alias analyze_audio_file to analyze_track if not already defined
if not hasattr(analyzer, "analyze_audio_file"):
    analyzer.analyze_audio_file = analyzer.analyze_track

analyze_audio_file = analyzer.analyze_track

# Register TrackAnalysis as virtual subclass of Mapping so grouping.as_track_info accepts it
Mapping.register(TrackAnalysis)

__all__ = [
    "analyzer",
    "grouping",
    "manifest",
    "summer",
    "analyze_track",
    "analyze_audio_file",
    "TrackAnalysis",
    "group_tracks",
    "TrackInfo",
    "GroupingMode",
    "sum_audio_files",
    "apply_headroom_protection",
    "generate_manifest",
    "save_manifest",
    "load_manifest",
]
