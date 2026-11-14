"""
Spectral Masking and Stem Grouping Engine.

Implements the 15% centroid collision rule for detecting spectral masking,
dynamic subgroup splitting (e.g. Drums.wav, Drums_Part2.wav), and multiple
grouping modes:
  - Mode 0: Rename only (no summing, individual stems).
  - Mode 1: By Instrument family (Drums, Guitars, Bass, etc.).
  - Mode 2: By Frequency band (Sub-bass, Bass, Low-mid, High-mid, High).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Sequence, Union

import numpy as np


class GroupingMode(IntEnum):
    """Supported stem grouping modes."""

    RENAME_ONLY = 0
    BY_FAMILY = 1
    BY_FREQUENCY = 2


# Standard 10 instrument families
VALID_FAMILIES = (
    "Drums",
    "Bass",
    "Guitars",
    "Keys",
    "Synth",
    "Vocals",
    "Strings",
    "Brass",
    "Woodwinds",
    "Other",
)

FAMILY_MAP = {
    "rhythm/drums": "Drums",
    "rhythm": "Drums",
    "drums": "Drums",
    "drum": "Drums",
    "percussion": "Drums",
    "bass": "Bass",
    "guitars": "Guitars",
    "guitar": "Guitars",
    "keys": "Keys",
    "keyboard": "Keys",
    "piano": "Keys",
    "synth": "Synth",
    "synths": "Synth",
    "synthesizer": "Synth",
    "vocals": "Vocals",
    "vocal": "Vocals",
    "voice": "Vocals",
    "lead vocal": "Vocals",
    "backing vocal": "Vocals",
    "strings": "Strings",
    "string": "Strings",
    "brass": "Brass",
    "woodwinds": "Woodwinds",
    "woodwind": "Woodwinds",
    "other": "Other",
}

# Standard 5 frequency bands
FREQUENCY_BANDS = (
    "Sub-bass",
    "Bass",
    "Low-mid",
    "High-mid",
    "High",
)


def frequency_band_from_centroid(centroid: float) -> str:
    """
    Maps spectral centroid in Hz to standard 5 frequency bands:
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


def normalize_family(family: str | None) -> str:
    """Normalizes family string to one of the 10 canonical families."""
    if not family:
        return "Other"
    clean = family.strip().lower()
    if clean in FAMILY_MAP:
        return FAMILY_MAP[clean]
    # Check substring matches or fallback to Other
    for key, canonical in FAMILY_MAP.items():
        if key in clean:
            return canonical
    # Clean slashes/illegal filename chars
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", family).strip()
    return sanitized if sanitized else "Other"


def normalize_frequency_band(band: str | None, centroid: float = 0.0) -> str:
    """Normalizes frequency band name (e.g. 'Low-mid' or 'Low-mid Band')."""
    if not band and centroid > 0:
        band = frequency_band_from_centroid(centroid)
    elif not band:
        return "Low-mid Band"

    clean = band.strip()
    # Strip existing 'Band' suffix for comparison
    base = clean[:-5].strip() if clean.lower().endswith("band") else clean

    # Match case-insensitively to standard bands
    for std_band in FREQUENCY_BANDS:
        if base.lower() == std_band.lower():
            return f"{std_band} Band"

    # Fallback
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", clean).strip()
    return sanitized if sanitized.endswith(" Band") else f"{sanitized} Band"


@dataclass
class TrackInfo:
    """
    Represents metadata and DSP attributes for a single audio track.
    """

    file_path: str = ""
    filename: str = ""
    instrument: str = "Unknown"
    family: str = "Other"
    frequency_band: str = ""
    spectral_centroid: float = 0.0
    chroma: list[float] | np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.spectral_centroid < 0:
            raise ValueError(
                f"Spectral centroid must be non-negative: {self.spectral_centroid}"
            )
        if not self.filename and self.file_path:
            self.filename = os.path.basename(self.file_path)
        if not self.frequency_band and self.spectral_centroid > 0:
            self.frequency_band = frequency_band_from_centroid(
                self.spectral_centroid
            )
        elif not self.frequency_band:
            self.frequency_band = "Mid"

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        chroma_val = (
            self.chroma.tolist()
            if isinstance(self.chroma, np.ndarray)
            else self.chroma
        )
        return {
            "file_path": self.file_path,
            "filename": self.filename,
            "instrument": self.instrument,
            "family": self.family,
            "frequency_band": self.frequency_band,
            "spectral_centroid": float(self.spectral_centroid),
            "chroma": chroma_val,
        }


def as_track_info(track: TrackInfo | Mapping[str, Any]) -> TrackInfo:
    """Converts a dict or TrackInfo into a validated TrackInfo instance."""
    if isinstance(track, TrackInfo):
        return track
    if isinstance(track, (dict, Mapping)):
        return TrackInfo(
            file_path=str(track.get("file_path", "")),
            filename=str(track.get("filename", "")),
            instrument=str(track.get("instrument", "Unknown")),
            family=str(track.get("family", "Other")),
            frequency_band=str(track.get("frequency_band", "")),
            spectral_centroid=float(track.get("spectral_centroid", 0.0)),
            chroma=track.get("chroma"),
        )
    raise TypeError(f"Expected TrackInfo or dict, got {type(track)}")


class GroupingDict(dict):
    """
    A dict subclass for grouping results supporting flexible lookup.
    Allows accessing keys with or without '.wav', and band names with or without ' Band'.
    """

    def _normalize_key(self, key: str) -> list[str]:
        candidates = [key]
        if key.endswith(".wav"):
            candidates.append(key[:-4])
        else:
            candidates.append(f"{key}.wav")

        for cand in list(candidates):
            if cand.endswith(" Band"):
                candidates.append(cand[:-5])
                candidates.append(f"{cand[:-5]}.wav")
            elif cand.endswith(" Band.wav"):
                candidates.append(f"{cand[:-9]}.wav")
                candidates.append(cand[:-9])
            else:
                candidates.append(f"{cand} Band")
                candidates.append(f"{cand} Band.wav")

        return candidates

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        for cand in self._normalize_key(key):
            if super().__contains__(cand):
                return super().__getitem__(cand)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            for cand in self._normalize_key(key):
                if super().__contains__(cand):
                    return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def calculate_centroid_diff(c1: float, c2: float) -> float:
    """
    Calculates the relative difference between two spectral centroids:
        diff = |C_A - C_B| / ((C_A + C_B) / 2)

    Raises ValueError if either centroid is negative.
    Returns 0.0 if both centroids are zero.
    """
    if c1 < 0 or c2 < 0:
        raise ValueError(f"Centroids must be non-negative: c1={c1}, c2={c2}")

    avg = (c1 + c2) / 2.0
    if avg == 0.0:
        return 0.0

    return abs(c1 - c2) / avg


def _extract_centroid(val: float | TrackInfo | Mapping[str, Any]) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, TrackInfo):
        return float(val.spectral_centroid)
    if isinstance(val, (dict, Mapping)):
        return float(val.get("spectral_centroid", 0.0))
    raise TypeError(f"Cannot extract spectral centroid from {type(val)}")


def tracks_collide(
    c1: float | TrackInfo | Mapping[str, Any],
    c2: float | TrackInfo | Mapping[str, Any],
    threshold: float = 0.15,
) -> bool:
    """
    Evaluates the 15% centroid collision rule:
        diff = |C_A - C_B| / avg(C_A, C_B) < 0.15

    Returns True if tracks collide (spectral masking), False if safe.
    """
    centroid1 = _extract_centroid(c1)
    centroid2 = _extract_centroid(c2)
    diff = calculate_centroid_diff(centroid1, centroid2)
    if math.isclose(diff, threshold, rel_tol=1e-12, abs_tol=1e-12):
        return False
    return diff < threshold


def _format_part_stem_name(base_stem_name: str, part_num: int) -> str:
    """
    Formats the stem name for a given subgroup part number.
    Part 1 preserves base_stem_name.
    Part 2+ appends '_Part{n}' before the file extension if present.
    """
    if part_num == 1:
        return base_stem_name

    root, ext = os.path.splitext(base_stem_name)
    if root.endswith("_Part1"):
        root = root[:-6]
    return f"{root}_Part{part_num}{ext}"


def resolve_subgroups(
    base_stem_name: str,
    tracks: Sequence[TrackInfo | Mapping[str, Any]],
    threshold: float = 0.15,
) -> GroupingDict:
    """
    Dynamically partitions tracks in a stem group into safe subgroups so that
    no two tracks in any subgroup collide (i.e. pairwise diff >= threshold).

    Preserves base_stem_name for the primary group (e.g. Drums.wav),
    and creates subsequent subgroups with _Part2, _Part3 (e.g. Drums_Part2.wav).

    Returns a GroupingDict mapping stem_name -> list of TrackInfo.
    """
    track_objs = [as_track_info(t) for t in tracks]
    if not track_objs:
        return GroupingDict()

    subgroups: list[list[TrackInfo]] = []

    for track in track_objs:
        placed = False
        for subgroup in subgroups:
            # Check collision against all tracks already in this subgroup
            collides_with_subgroup = any(
                tracks_collide(track, existing, threshold=threshold)
                for existing in subgroup
            )
            if not collides_with_subgroup:
                subgroup.append(track)
                placed = True
                break

        if not placed:
            subgroups.append([track])

    result = GroupingDict()
    for idx, subgroup in enumerate(subgroups, start=1):
        stem_name = _format_part_stem_name(base_stem_name, idx)
        result[stem_name] = subgroup

    return result


def _clean_track_name(track: TrackInfo, extension: str = "") -> str:
    """Generates a clean descriptive name for a single track in Mode 0."""
    name = ""
    if track.instrument and track.instrument.lower() not in ("unknown", "other", ""):
        name = track.instrument
    elif track.filename:
        name = os.path.splitext(os.path.basename(track.filename))[0]
    elif track.file_path:
        name = os.path.splitext(os.path.basename(track.file_path))[0]
    else:
        name = "Track"

    name = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    if extension:
        if not extension.startswith("."):
            extension = f".{extension}"
        if not name.endswith(extension):
            name = f"{name}{extension}"
    return name


def group_tracks(
    tracks: Sequence[TrackInfo | Mapping[str, Any]],
    mode: int | GroupingMode = GroupingMode.BY_FAMILY,
    threshold: float = 0.15,
    extension: str = "",
) -> GroupingDict:
    """
    Groups audio tracks into stems based on the selected mode:
      - Mode 0 (Rename Only): Does not sum. Each track forms its own stem named
        after its detected instrument or clean filename.
      - Mode 1 (By Instrument Family): Groups by family (Drums, Guitars, Bass, etc.),
        partitioning collisions into subgroups (_Part2, etc.).
      - Mode 2 (By Frequency Band): Groups by frequency band (Sub-bass, Bass,
        Low-mid, High-mid, High), partitioning collisions into subgroups.

    Parameters:
      tracks: Sequence of TrackInfo instances or dict mappings.
      mode: 0 (Rename Only), 1 (By Family), or 2 (By Frequency Band).
      threshold: Collision threshold for spectral centroid relative difference (default 0.15).
      extension: Optional file extension to append (e.g. '.wav').

    Returns:
      GroupingDict mapping stem_name -> list of TrackInfo.
    """
    track_objs = [as_track_info(t) for t in tracks]
    if not track_objs:
        return GroupingDict()

    if isinstance(mode, str):
        mode_str = mode.lower()
        if "rename" in mode_str or mode_str == "0":
            mode = GroupingMode.RENAME_ONLY
        elif "family" in mode_str or "instrument" in mode_str or mode_str == "1":
            mode = GroupingMode.BY_FAMILY
        elif "frequency" in mode_str or "band" in mode_str or mode_str == "2":
            mode = GroupingMode.BY_FREQUENCY
        else:
            raise ValueError(f"Unrecognized grouping mode string: {mode}")

    try:
        int_mode = int(mode)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid grouping mode: {mode}")

    if int_mode == GroupingMode.RENAME_ONLY:
        # Mode 0: Rename only. Does not sum. Each track forms its own stem.
        result = GroupingDict()
        name_counts: dict[str, int] = {}

        for track in track_objs:
            base_name = _clean_track_name(track, extension=extension)
            root, ext = os.path.splitext(base_name)

            if base_name not in name_counts:
                name_counts[base_name] = 1
                stem_name = base_name
            else:
                name_counts[base_name] += 1
                count = name_counts[base_name]
                stem_name = f"{root}_Part{count}{ext}"

            result[stem_name] = [track]

        return result

    elif int_mode == GroupingMode.BY_FAMILY:
        # Mode 1: By Instrument family
        family_buckets: dict[str, list[TrackInfo]] = {}
        for track in track_objs:
            fam = normalize_family(track.family)
            if fam not in family_buckets:
                family_buckets[fam] = []
            family_buckets[fam].append(track)

        result = GroupingDict()
        for fam, bucket in family_buckets.items():
            base_stem = f"{fam}{extension}" if extension else fam
            subgroups = resolve_subgroups(base_stem, bucket, threshold=threshold)
            for stem_name, sub_tracks in subgroups.items():
                result[stem_name] = sub_tracks

        return result

    elif int_mode == GroupingMode.BY_FREQUENCY:
        # Mode 2: By Frequency band
        band_buckets: dict[str, list[TrackInfo]] = {}
        for track in track_objs:
            band_name = normalize_frequency_band(
                track.frequency_band, track.spectral_centroid
            )
            if band_name not in band_buckets:
                band_buckets[band_name] = []
            band_buckets[band_name].append(track)

        result = GroupingDict()
        for band_name, bucket in band_buckets.items():
            base_stem = f"{band_name}{extension}" if extension else band_name
            subgroups = resolve_subgroups(base_stem, bucket, threshold=threshold)
            for stem_name, sub_tracks in subgroups.items():
                result[stem_name] = sub_tracks

        return result

    else:
        raise ValueError(
            f"Unsupported grouping mode: {int_mode}. "
            "Supported modes: 0 (Rename Only), 1 (By Family), 2 (By Frequency Band)."
        )


class StemGrouper:
    """Object-oriented interface for the Stem Grouping Engine."""

    def __init__(self, collision_threshold: float = 0.15) -> None:
        self.collision_threshold = collision_threshold

    def calculate_centroid_diff(self, c1: float, c2: float) -> float:
        return calculate_centroid_diff(c1, c2)

    def tracks_collide(
        self,
        c1: float | TrackInfo | Mapping[str, Any],
        c2: float | TrackInfo | Mapping[str, Any],
    ) -> bool:
        return tracks_collide(c1, c2, threshold=self.collision_threshold)

    def resolve_subgroups(
        self,
        base_stem_name: str,
        tracks: Sequence[TrackInfo | Mapping[str, Any]],
    ) -> GroupingDict:
        return resolve_subgroups(
            base_stem_name, tracks, threshold=self.collision_threshold
        )

    def group_tracks(
        self,
        tracks: Sequence[TrackInfo | Mapping[str, Any]],
        mode: int | GroupingMode = GroupingMode.BY_FAMILY,
        extension: str = "",
    ) -> GroupingDict:
        return group_tracks(
            tracks,
            mode=mode,
            threshold=self.collision_threshold,
            extension=extension,
        )
