"""
Export Manifest Generator for Smart Export DAW Stems.

Implements Wave 2 Ticket T-202:
  - AC-1: Emits Final_Stems/export_manifest.json with ISO timestamp.
  - AC-2: Records output stem files and mapped contributing source tracks.
  - AC-3: Embeds complete DSP attributes (instrument, family, band, centroid Hz,
          chroma vector) for each track.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np

# ============================================================================
# Architectural Constants
# ============================================================================

DEFAULT_OUTPUT_DIR = "Final_Stems"
DEFAULT_MANIFEST_FILENAME = "export_manifest.json"
APP_NAME = "Smart Export"
APP_VERSION = "1.0.0"

GROUPING_MODE_NAMES: dict[int, str] = {
    0: "Rename Only",
    1: "By Instrument Family",
    2: "By Frequency Band",
}

# Standard 5 frequency bands
FREQUENCY_BANDS: tuple[str, ...] = (
    "Sub-bass",
    "Bass",
    "Low-mid",
    "High-mid",
    "High",
)


# ============================================================================
# Serialization & Normalization Helpers
# ============================================================================

def current_iso_timestamp() -> str:
    """
    Returns current UTC timestamp formatted as ISO 8601 with trailing 'Z'.
    Example: '2026-11-14T08:30:00Z'.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_numpy_value(obj: Any) -> Any:
    """
    Recursively converts numpy primitives, arrays, and complex containers into
    standard Python types (float, int, bool, list, dict) so that json.dump
    never raises TypeError.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return 0.0 if np.isnan(val) or np.isinf(val) else val

    if isinstance(obj, (np.integer, int)):
        return int(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return [clean_numpy_value(x) for x in obj.tolist()]

    if isinstance(obj, (list, tuple)):
        return [clean_numpy_value(x) for x in obj]

    if isinstance(obj, (Path, os.PathLike)):
        return str(obj)

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return clean_numpy_value(obj.to_dict())

    if isinstance(obj, Mapping):
        return {str(k): clean_numpy_value(v) for k, v in obj.items()}

    if hasattr(obj, "__dict__"):
        return clean_numpy_value(obj.__dict__)

    return str(obj)


def frequency_band_from_centroid(centroid: float) -> str:
    """
    Determines frequency band category from spectral centroid in Hz:
      - Sub-bass: < 60 Hz
      - Bass: 60 - 250 Hz
      - Low-mid: 250 - 2000 Hz
      - High-mid: 2000 - 6000 Hz
      - High: >= 6000 Hz
    """
    if centroid < 60.0:
        return "Sub-bass"
    elif centroid < 250.0:
        return "Bass"
    elif centroid < 2000.0:
        return "Low-mid"
    elif centroid < 6000.0:
        return "High-mid"
    else:
        return "High"


def ensure_audio_filename(name: str, default_ext: str = ".wav") -> str:
    """Ensures an audio filename has an appropriate extension (defaults to .wav)."""
    clean_name = str(name).strip()
    _, ext = os.path.splitext(clean_name)
    if not ext:
        return f"{clean_name}{default_ext}"
    return clean_name


def normalize_source_track(track: Any) -> dict[str, Any]:
    """
    Extracts and standardizes DSP and file attributes from a track dict,
    TrackAnalysis, or TrackInfo object into the canonical manifest format.

    Guarantees the presence of:
      - filename: str
      - file_path: str
      - instrument: str
      - family: str
      - frequency_band: str
      - spectral_centroid_hz: float
      - chroma_profile: list[float]
    """
    # 1. Unpack attributes from mapping or object
    def get_attr(keys: Sequence[str], default: Any = None) -> Any:
        for key in keys:
            if isinstance(track, Mapping) and key in track:
                val = track[key]
                if val is not None:
                    return val
            if hasattr(track, key):
                val = getattr(track, key)
                if val is not None:
                    return val
            if hasattr(track, "__getitem__"):
                try:
                    val = track[key]
                    if val is not None:
                        return val
                except (KeyError, TypeError, IndexError, AttributeError):
                    pass
        return default

    # File path and filename
    file_path = str(get_attr(["file_path", "path", "filepath"], "") or "")
    filename = str(get_attr(["filename", "name"], "") or "")
    if not filename and file_path:
        filename = os.path.basename(file_path)
    if not filename:
        filename = "unknown.wav"

    # Instrument and family classification
    instrument = str(get_attr(["instrument", "instrument_name", "label"], "Unknown") or "Unknown")
    family = str(get_attr(["family", "instrument_family"], "Other") or "Other")

    # Spectral centroid (Hz)
    raw_centroid = get_attr(
        ["spectral_centroid_hz", "spectral_centroid", "centroid", "centroid_hz"],
        0.0,
    )
    try:
        centroid_hz = float(raw_centroid)
        if np.isnan(centroid_hz) or np.isinf(centroid_hz) or centroid_hz < 0.0:
            centroid_hz = 0.0
    except (ValueError, TypeError):
        centroid_hz = 0.0

    # Frequency band
    frequency_band = get_attr(["frequency_band", "band", "freq_band"], "")
    if not frequency_band:
        frequency_band = frequency_band_from_centroid(centroid_hz)
    else:
        frequency_band = str(frequency_band)
        # Normalize if it has " Band" suffix
        if frequency_band.endswith(" Band"):
            frequency_band = frequency_band[:-5].strip()

    # Chroma profile (12 pitch classes)
    raw_chroma = get_attr(["chroma_profile", "chroma", "chroma_vector", "pitch_profile"], None)
    if raw_chroma is None:
        chroma_profile = [0.0] * 12
    else:
        if isinstance(raw_chroma, np.ndarray):
            chroma_list = raw_chroma.tolist()
        elif isinstance(raw_chroma, (list, tuple)):
            chroma_list = list(raw_chroma)
        else:
            chroma_list = [0.0] * 12

        # Convert elements to float
        clean_chroma: list[float] = []
        for c in chroma_list:
            try:
                cv = float(c)
                clean_chroma.append(0.0 if (np.isnan(cv) or np.isinf(cv)) else cv)
            except (ValueError, TypeError):
                clean_chroma.append(0.0)

        # Pad or truncate to 12 bins
        if len(clean_chroma) < 12:
            clean_chroma.extend([0.0] * (12 - len(clean_chroma)))
        elif len(clean_chroma) > 12:
            clean_chroma = clean_chroma[:12]
        chroma_profile = clean_chroma

    return {
        "filename": filename,
        "file_path": file_path,
        "instrument": instrument,
        "family": family,
        "frequency_band": frequency_band,
        "spectral_centroid_hz": float(centroid_hz),
        "chroma_profile": [float(p) for p in chroma_profile],
    }


# ============================================================================
# Manifest Core Interface
# ============================================================================

def generate_manifest(
    stems_mapping: Mapping[str, Sequence[Any]],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    grouping_mode: int = 1,
    *,
    timestamp: str | None = None,
    app: str = APP_NAME,
    version: str = APP_VERSION,
    grouping_mode_name: str | None = None,
) -> dict[str, Any]:
    """
    Generates a structured export manifest dictionary from a mapping of output
    stem names to contributing source tracks.

    Args:
        stems_mapping: Mapping of stem filenames (e.g. 'Drums.wav') to sequences
                       of track dictionaries or TrackAnalysis/TrackInfo objects.
        output_dir: Target output directory for stems (default: 'Final_Stems').
        grouping_mode: Integer mode ID (0: Rename only, 1: By family, 2: By frequency).
        timestamp: Optional explicit ISO 8601 UTC timestamp string. Defaults to current time.
        app: Application name identifier (default: 'Smart Export').
        version: Application version identifier (default: '1.0.0').
        grouping_mode_name: Optional explicit display name for grouping mode.

    Returns:
        A JSON-serializable dictionary adhering strictly to the Smart Export
        manifest specification.
    """
    mode_int = int(grouping_mode)
    resolved_mode_name = (
        grouping_mode_name
        or GROUPING_MODE_NAMES.get(mode_int, f"Mode {mode_int}")
    )
    resolved_timestamp = timestamp or current_iso_timestamp()

    stems_list: list[dict[str, Any]] = []
    total_input_tracks = 0

    for stem_name, track_list in stems_mapping.items():
        stem_file = ensure_audio_filename(stem_name)
        source_tracks = [normalize_source_track(t) for t in (track_list or [])]
        track_count = len(source_tracks)
        total_input_tracks += track_count

        stems_list.append(
            {
                "stem_file": stem_file,
                "track_count": track_count,
                "source_tracks": source_tracks,
            }
        )

    manifest: dict[str, Any] = {
        "app": app,
        "version": version,
        "timestamp": resolved_timestamp,
        "grouping_mode": mode_int,
        "grouping_mode_name": resolved_mode_name,
        "total_input_tracks": total_input_tracks,
        "total_output_stems": len(stems_list),
        "stems": stems_list,
    }

    # Ensure all nested values are standard Python types
    return clean_numpy_value(manifest)


def save_manifest(
    manifest: Mapping[str, Any],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: str = DEFAULT_MANIFEST_FILENAME,
    indent: int = 2,
) -> str:
    """
    Saves the manifest dictionary to disk as a JSON file.

    Args:
        manifest: Dictionary structure created by generate_manifest().
        output_dir: Destination directory path (e.g. 'Final_Stems').
        filename: Destination filename (default: 'export_manifest.json').
        indent: Indentation spaces for formatted JSON (default: 2).

    Returns:
        The normalized path to the emitted manifest file.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, filename)
    else:
        manifest_path = filename

    cleaned_manifest = clean_numpy_value(manifest)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_manifest, f, indent=indent, ensure_ascii=False)

    return os.path.normpath(manifest_path)


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """
    Reads and parses a manifest JSON file from disk.

    Args:
        manifest_path: Path to the JSON manifest file.

    Returns:
        The loaded manifest dictionary.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_manifest(
    stems_mapping: Mapping[str, Sequence[Any]],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    grouping_mode: int = 1,
    filename: str = DEFAULT_MANIFEST_FILENAME,
    **kwargs: Any,
) -> tuple[dict[str, Any], str]:
    """
    Convenience function that generates and saves the manifest in one step.

    Returns:
        Tuple of (manifest_dict, manifest_file_path).
    """
    manifest = generate_manifest(
        stems_mapping=stems_mapping,
        output_dir=output_dir,
        grouping_mode=grouping_mode,
        **kwargs,
    )
    saved_path = save_manifest(manifest, output_dir=output_dir, filename=filename)
    return manifest, saved_path


def validate_manifest(manifest: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """
    Validates that a manifest dictionary satisfies the specification schema.

    Returns:
        (is_valid, list_of_errors)
    """
    errors: list[str] = []

    required_keys = (
        "app",
        "version",
        "timestamp",
        "grouping_mode",
        "grouping_mode_name",
        "total_input_tracks",
        "total_output_stems",
        "stems",
    )
    for k in required_keys:
        if k not in manifest:
            errors.append(f"Missing required manifest key: '{k}'")

    # Validate timestamp
    ts = manifest.get("timestamp")
    if not isinstance(ts, str):
        errors.append("Field 'timestamp' must be a string.")
    else:
        # Check ISO format
        try:
            # Handle Z suffix
            clean_ts = ts.replace("Z", "+00:00")
            datetime.fromisoformat(clean_ts)
        except Exception as e:
            errors.append(f"Invalid ISO 8601 timestamp '{ts}': {e}")

    # Validate stems
    stems = manifest.get("stems")
    if not isinstance(stems, list):
        errors.append("Field 'stems' must be a list.")
    else:
        counted_tracks = 0
        for idx, stem in enumerate(stems):
            if not isinstance(stem, dict):
                errors.append(f"stems[{idx}] is not a dictionary.")
                continue

            if "stem_file" not in stem or not stem["stem_file"]:
                errors.append(f"stems[{idx}] missing 'stem_file'.")

            if "track_count" not in stem or not isinstance(stem["track_count"], int):
                errors.append(f"stems[{idx}] missing integer 'track_count'.")

            source_tracks = stem.get("source_tracks")
            if not isinstance(source_tracks, list):
                errors.append(f"stems[{idx}] 'source_tracks' must be a list.")
                continue

            counted_tracks += len(source_tracks)
            if stem.get("track_count") != len(source_tracks):
                errors.append(
                    f"stems[{idx}] 'track_count' ({stem.get('track_count')}) != len(source_tracks) ({len(source_tracks)})."
                )

            for tidx, track in enumerate(source_tracks):
                t_label = f"stems[{idx}].source_tracks[{tidx}]"
                if not isinstance(track, dict):
                    errors.append(f"{t_label} is not a dictionary.")
                    continue

                for field in (
                    "filename",
                    "file_path",
                    "instrument",
                    "family",
                    "frequency_band",
                    "spectral_centroid_hz",
                    "chroma_profile",
                ):
                    if field not in track:
                        errors.append(f"{t_label} missing required field '{field}'.")

                centroid = track.get("spectral_centroid_hz")
                if not isinstance(centroid, (int, float)) or centroid < 0:
                    errors.append(f"{t_label} 'spectral_centroid_hz' must be a non-negative float.")

                chroma = track.get("chroma_profile")
                if not isinstance(chroma, list) or len(chroma) != 12:
                    errors.append(f"{t_label} 'chroma_profile' must be a 12-element list.")

        if manifest.get("total_output_stems") != len(stems):
            errors.append(
                f"'total_output_stems' ({manifest.get('total_output_stems')}) does not match len(stems) ({len(stems)})."
            )

        if manifest.get("total_input_tracks") != counted_tracks:
            errors.append(
                f"'total_input_tracks' ({manifest.get('total_input_tracks')}) does not match total counted tracks ({counted_tracks})."
            )

    return len(errors) == 0, errors
