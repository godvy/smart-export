"""
Audio Conditioning, Feature Extraction, and Instrument Classification Engine.

Implements Wave 2 Ticket T-201:
  - AC-1: Trims silence via -40 dB gate and resamples to 16 kHz mono slice (up to 5 seconds).
  - AC-2: Calculates spectral centroid in Hz and maps to 5 frequency bands (Sub-bass, Bass, Low-mid, High-mid, High).
  - AC-3: Extracts 12-bin mean chroma profile capturing musical pitch distribution.
  - AC-4: Classifies instrument with generic label filtering and maps to the 10 defined families.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

# ============================================================================
# Architectural Constants & Canonical Taxonomies
# ============================================================================

TARGET_SAMPLING_RATE = 16000
MAX_ANALYSIS_DURATION_SEC = 5.0
SILENCE_TOP_DB = 40.0

# 10 Canonical Instrument Families
TARGET_FAMILIES: tuple[str, ...] = (
    "Rhythm/Drums",
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

# 5 Standard Frequency Bands
FREQUENCY_BANDS: tuple[str, ...] = (
    "Sub-bass",
    "Bass",
    "Low-mid",
    "High-mid",
    "High",
)

# Standard 12 Musical Pitch Classes
PITCH_CLASSES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

# Generic / non-informative labels that MUST be suppressed
GENERIC_LABELS: frozenset[str] = frozenset({
    "music",
    "musical instrument",
    "instrument",
    "sound effect",
    "sound effects",
    "sfx",
    "silence",
    "background noise",
    "audio",
    "sound",
    "noise",
    "speech",
    "ambient",
    "ambience",
    "white noise",
    "pink noise",
    "static",
    "unknown",
})

try:
    import soxr  # noqa: F401
    _RES_TYPE = "soxr_hq"
except ImportError:
    _RES_TYPE = "scipy"

# Keyword dictionary for robust offline / heuristic filename classification
# Specific instruments and multi-word phrases take precedence over generic modifiers (like 'lead', 'arp')
KEYWORD_MAPPINGS: list[tuple[re.Pattern[str], str, str, float]] = [
    # Multi-word priority phrases
    (re.compile(r"\b(lead_vox|lead_vocal|main_vox|main_vocal)\b", re.IGNORECASE), "Lead Vocals", "Vocals", 0.98),
    (re.compile(r"\b(lead_gtr|lead_guitar)\b", re.IGNORECASE), "Electric Guitar", "Guitars", 0.98),
    (re.compile(r"\b(synth_lead|leadsynth|lead_synth)\b", re.IGNORECASE), "Synth Lead", "Synth", 0.98),
    (re.compile(r"\b(synth_pad|padsynth|pad_synth)\b", re.IGNORECASE), "Synth Pad", "Synth", 0.98),
    (re.compile(r"\b(synth_pluck|plucksynth|pluck_synth)\b", re.IGNORECASE), "Synth Pluck", "Synth", 0.98),
    (re.compile(r"\b(synth_bass|synthbass|acid_bass)\b", re.IGNORECASE), "Synth Bass", "Bass", 0.98),
    (re.compile(r"\b(sub_bass|subbass|808_bass|bass_808)\b", re.IGNORECASE), "Sub / 808 Bass", "Bass", 0.98),

    # Vocals
    (re.compile(r"\b(bgv|backing_vox|backing_vocal|harmonies|harmony|choir|acapella)\b", re.IGNORECASE), "Backing Vocals", "Vocals", 0.95),
    (re.compile(r"\b(vocal|vocals|vox|verse|chorus|adlib|ad-lib|chant|rapping|singing)\b", re.IGNORECASE), "Vocals", "Vocals", 0.90),

    # Rhythm / Drums
    (re.compile(r"\b(kick|kik|bd|bassdrum)\b", re.IGNORECASE), "Kick drum", "Rhythm/Drums", 0.95),
    (re.compile(r"\b(snare|sd|rim|rimshot)\b", re.IGNORECASE), "Snare drum", "Rhythm/Drums", 0.95),
    (re.compile(r"\b(hihat|hi-hat|hh|hat|hats|cymbal|ride|crash|splash)\b", re.IGNORECASE), "Hi-hat / Cymbal", "Rhythm/Drums", 0.95),
    (re.compile(r"\b(tom|toms|floor_tom|rack_tom)\b", re.IGNORECASE), "Tom-tom", "Rhythm/Drums", 0.95),
    (re.compile(r"\b(drum|drums|drumkit|drum_kit|drumloop|breakbeat)\b", re.IGNORECASE), "Drums", "Rhythm/Drums", 0.92),
    (re.compile(r"\b(perc|percussion|bongo|bongos|conga|congas|tambourine|shaker|clap|claps|snap|cowbell|timpani|cabasa|guiro)\b", re.IGNORECASE), "Percussion", "Rhythm/Drums", 0.92),

    # Bass
    (re.compile(r"\b(808|sub)\b", re.IGNORECASE), "Sub / 808 Bass", "Bass", 0.95),
    (re.compile(r"\b(electric_bass|pbass|jbass|slap_bass|picked_bass)\b", re.IGNORECASE), "Electric Bass", "Bass", 0.95),
    (re.compile(r"\b(upright_bass|contrabass|double_bass|acoustic_bass)\b", re.IGNORECASE), "Upright Bass", "Bass", 0.95),
    (re.compile(r"\b(bass|bassline|bass_di|bass_amp|wobble|reese)\b", re.IGNORECASE), "Bass", "Bass", 0.90),

    # Guitars
    (re.compile(r"\b(acoustic_gtr|acoustic_guitar|agtr|nylon|12string)\b", re.IGNORECASE), "Acoustic Guitar", "Guitars", 0.95),
    (re.compile(r"\b(electric_gtr|electric_guitar|egtr|strat|tele|lespaul|telecaster)\b", re.IGNORECASE), "Electric Guitar", "Guitars", 0.95),
    (re.compile(r"\b(guitar|guitars|gtr|gtrs|strum|riff|banjo|ukulele|mandolin)\b", re.IGNORECASE), "Guitar", "Guitars", 0.90),

    # Keys
    (re.compile(r"\b(grand_piano|upright_piano|steinway|yamaha_piano)\b", re.IGNORECASE), "Acoustic Piano", "Keys", 0.95),
    (re.compile(r"\b(rhodes|wurlitzer|wurli|clav|clavier|electric_piano|ep)\b", re.IGNORECASE), "Electric Piano", "Keys", 0.95),
    (re.compile(r"\b(hammond|b3|pipe_organ|church_organ|organ)\b", re.IGNORECASE), "Organ", "Keys", 0.95),
    (re.compile(r"\b(piano|keyboard|keyboards|keys|harpsichord|celesta)\b", re.IGNORECASE), "Piano / Keys", "Keys", 0.90),

    # Strings
    (re.compile(r"\b(violin|violins|fiddle)\b", re.IGNORECASE), "Violin", "Strings", 0.95),
    (re.compile(r"\b(cello|cellos|violoncello)\b", re.IGNORECASE), "Cello", "Strings", 0.95),
    (re.compile(r"\b(viola|violas)\b", re.IGNORECASE), "Viola", "Strings", 0.95),
    (re.compile(r"\b(strings|string_section|orchestra|orchestral|quartet|pizzicato|harp)\b", re.IGNORECASE), "Strings", "Strings", 0.92),

    # Brass
    (re.compile(r"\b(trumpet|trumpets|cornet)\b", re.IGNORECASE), "Trumpet", "Brass", 0.95),
    (re.compile(r"\b(trombone|trombones)\b", re.IGNORECASE), "Trombone", "Brass", 0.95),
    (re.compile(r"\b(french_horn|horn|horns|tuba|euphonium|sousaphone)\b", re.IGNORECASE), "French Horn / Tuba", "Brass", 0.95),
    (re.compile(r"\b(brass|brass_section|horns_section)\b", re.IGNORECASE), "Brass", "Brass", 0.90),

    # Woodwinds
    (re.compile(r"\b(flute|flutes|piccolo)\b", re.IGNORECASE), "Flute", "Woodwinds", 0.95),
    (re.compile(r"\b(sax|saxophone|alto_sax|tenor_sax|baritone_sax|soprano_sax)\b", re.IGNORECASE), "Saxophone", "Woodwinds", 0.95),
    (re.compile(r"\b(clarinet|clarinets)\b", re.IGNORECASE), "Clarinet", "Woodwinds", 0.95),
    (re.compile(r"\b(oboe|oboes|bassoon|bassoons|recorder|harmonica|accordion|woodwind|woodwinds)\b", re.IGNORECASE), "Woodwinds", "Woodwinds", 0.92),

    # Synth (including lone lead / pad / pluck / arp)
    (re.compile(r"\b(synth|synths|synthesizer|poly|monosynth|supersaw|saw_synth|moog)\b", re.IGNORECASE), "Synthesizer", "Synth", 0.90),
    (re.compile(r"\b(lead)\b", re.IGNORECASE), "Synth Lead", "Synth", 0.85),
    (re.compile(r"\b(pad|ambient_pad|drone)\b", re.IGNORECASE), "Synth Pad", "Synth", 0.85),
    (re.compile(r"\b(pluck|arp|arpeggio|arpeggiator)\b", re.IGNORECASE), "Synth Pluck / Arp", "Synth", 0.85),

    # Other / FX
    (re.compile(r"\b(fx|sfx|riser|downlifter|sweep|impact|foley|texture|atmosphere|transition|drop)\b", re.IGNORECASE), "Sound Effect", "Other", 0.85),
]

# AudioSet / YAMNet label mapping dictionary to (instrument, family)
AUDIOSET_LABEL_MAP: dict[str, tuple[str, str]] = {
    # Rhythm / Drums
    "drum kit": ("Drum kit", "Rhythm/Drums"),
    "snare drum": ("Snare drum", "Rhythm/Drums"),
    "bass drum": ("Bass drum", "Rhythm/Drums"),
    "hi-hat": ("Hi-hat", "Rhythm/Drums"),
    "cymbal": ("Cymbal", "Rhythm/Drums"),
    "crash cymbal": ("Crash cymbal", "Rhythm/Drums"),
    "ride cymbal": ("Ride cymbal", "Rhythm/Drums"),
    "drum": ("Drum", "Rhythm/Drums"),
    "percussion": ("Percussion", "Rhythm/Drums"),
    "timpani": ("Timpani", "Rhythm/Drums"),
    "tambourine": ("Tambourine", "Rhythm/Drums"),
    "bongo": ("Bongo", "Rhythm/Drums"),
    "conga": ("Conga", "Rhythm/Drums"),
    "cowbell": ("Cowbell", "Rhythm/Drums"),
    "clapping": ("Clapping", "Rhythm/Drums"),
    "drum roll": ("Drum roll", "Rhythm/Drums"),
    "steel drum": ("Steel drum", "Rhythm/Drums"),
    "tabla": ("Tabla", "Rhythm/Drums"),

    # Bass
    "bass guitar": ("Bass guitar", "Bass"),
    "electric bass": ("Electric bass", "Bass"),
    "double bass": ("Double bass", "Bass"),
    "acoustic bass guitar": ("Acoustic bass guitar", "Bass"),
    "bass": ("Bass", "Bass"),

    # Guitars
    "acoustic guitar": ("Acoustic guitar", "Guitars"),
    "electric guitar": ("Electric guitar", "Guitars"),
    "guitar": ("Guitar", "Guitars"),
    "plucked string instrument": ("Plucked string instrument", "Guitars"),
    "strum": ("Strum", "Guitars"),
    "banjo": ("Banjo", "Guitars"),
    "ukulele": ("Ukulele", "Guitars"),
    "mandolin": ("Mandolin", "Guitars"),
    "steel guitar": ("Steel guitar", "Guitars"),
    "slide guitar": ("Slide guitar", "Guitars"),

    # Keys
    "piano": ("Piano", "Keys"),
    "electric piano": ("Electric piano", "Keys"),
    "organ": ("Organ", "Keys"),
    "hammond organ": ("Hammond organ", "Keys"),
    "keyboard (musical)": ("Keyboard", "Keys"),
    "harpsichord": ("Harpsichord", "Keys"),
    "clavinet": ("Clavinet", "Keys"),
    "synthesizer piano": ("Electric piano", "Keys"),

    # Synth
    "synthesizer": ("Synthesizer", "Synth"),
    "sampler": ("Sampler", "Synth"),
    "drum machine": ("Drum machine", "Synth"),
    "theremin": ("Theremin", "Synth"),

    # Vocals
    "singing": ("Singing", "Vocals"),
    "vocal": ("Vocal", "Vocals"),
    "female singing": ("Female vocals", "Vocals"),
    "male singing": ("Male vocals", "Vocals"),
    "choir": ("Choir", "Vocals"),
    "yodeling": ("Yodeling", "Vocals"),
    "chant": ("Chant", "Vocals"),
    "acapella": ("Acapella", "Vocals"),
    "rapping": ("Rapping", "Vocals"),
    "humming": ("Humming", "Vocals"),

    # Strings
    "violin": ("Violin", "Strings"),
    "fiddle": ("Fiddle", "Strings"),
    "cello": ("Cello", "Strings"),
    "viola": ("Viola", "Strings"),
    "harp": ("Harp", "Strings"),
    "pizzicato": ("Pizzicato", "Strings"),
    "string instrument": ("String instrument", "Strings"),
    "bowed string instrument": ("Bowed string instrument", "Strings"),
    "orchestra": ("Orchestra", "Strings"),

    # Brass
    "trumpet": ("Trumpet", "Brass"),
    "trombone": ("Trombone", "Brass"),
    "french horn": ("French horn", "Brass"),
    "tuba": ("Tuba", "Brass"),
    "cornet": ("Cornet", "Brass"),
    "brass instrument": ("Brass instrument", "Brass"),

    # Woodwinds
    "flute": ("Flute", "Woodwinds"),
    "saxophone": ("Saxophone", "Woodwinds"),
    "alto saxophone": ("Alto saxophone", "Woodwinds"),
    "tenor saxophone": ("Tenor saxophone", "Woodwinds"),
    "baritone saxophone": ("Baritone saxophone", "Woodwinds"),
    "soprano saxophone": ("Soprano saxophone", "Woodwinds"),
    "clarinet": ("Clarinet", "Woodwinds"),
    "bassoon": ("Bassoon", "Woodwinds"),
    "oboe": ("Oboe", "Woodwinds"),
    "piccolo": ("Piccolo", "Woodwinds"),
    "bagpipes": ("Bagpipes", "Woodwinds"),
    "harmonica": ("Harmonica", "Woodwinds"),
    "accordion": ("Accordion", "Woodwinds"),
    "wind instrument": ("Wind instrument", "Woodwinds"),
}


# ============================================================================
# TrackAnalysis Data Structure
# ============================================================================

@dataclass
class TrackAnalysis:
    """
    Structured outcome of track audio conditioning, DSP feature extraction,
    and instrument classification.
    """

    file_path: str = ""
    filename: str = ""
    instrument: str = "Unknown"
    family: str = "Other"
    frequency_band: str = "Low-mid"
    spectral_centroid: float = 0.0
    chroma: list[float] = field(default_factory=lambda: [0.0] * 12)
    confidence: float = 1.0
    duration: float = 0.0
    sample_rate: int = TARGET_SAMPLING_RATE

    def __post_init__(self) -> None:
        if self.spectral_centroid < 0.0:
            raise ValueError(f"Spectral centroid must be non-negative: {self.spectral_centroid}")
        if not self.filename and self.file_path:
            self.filename = os.path.basename(self.file_path)
        if len(self.chroma) != 12:
            raise ValueError(f"Chroma must contain exactly 12 bins, got {len(self.chroma)}")
        if self.family not in TARGET_FAMILIES:
            # Map known aliases or sanitize
            if self.family in ("Drums", "rhythm", "drums"):
                self.family = "Rhythm/Drums"
            elif self.family in ("Guitars", "guitar", "gtr"):
                self.family = "Guitars"
            elif self.family in ("Keys", "piano", "keyboards"):
                self.family = "Keys"
            elif self.family in ("Synth", "synths"):
                self.family = "Synth"
            elif self.family in ("Vocals", "vocal", "vox"):
                self.family = "Vocals"
            elif self.family in ("Strings", "string"):
                self.family = "Strings"
            elif self.family in ("Brass", "brass"):
                self.family = "Brass"
            elif self.family in ("Woodwinds", "woodwind"):
                self.family = "Woodwinds"
            else:
                self.family = "Other"

    def to_dict(self) -> dict[str, Any]:
        """Serializes analysis attributes to a Python dictionary."""
        return {
            "file_path": self.file_path,
            "filename": self.filename,
            "instrument": self.instrument,
            "family": self.family,
            "frequency_band": self.frequency_band,
            "spectral_centroid": float(self.spectral_centroid),
            "chroma": [float(x) for x in self.chroma],
            "confidence": float(self.confidence),
            "duration": float(self.duration),
            "sample_rate": int(self.sample_rate),
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def to_track_info(self) -> Any:
        """Compatibility helper converting to engine.grouping.TrackInfo if available."""
        try:
            from engine.grouping import TrackInfo
            return TrackInfo(
                file_path=self.file_path,
                filename=self.filename,
                instrument=self.instrument,
                family=self.family,
                frequency_band=self.frequency_band,
                spectral_centroid=self.spectral_centroid,
                chroma=self.chroma,
            )
        except ImportError:
            return self


# ============================================================================
# AC-1: Audio Conditioning & Slicing
# ============================================================================

def condition_audio(
    y: np.ndarray,
    sr: int,
    target_sr: int = TARGET_SAMPLING_RATE,
    max_duration_sec: float = MAX_ANALYSIS_DURATION_SEC,
    top_db: float = SILENCE_TOP_DB,
    slice_mode: str = "first",
) -> tuple[np.ndarray, int]:
    """
    Conditions an audio signal for downstream feature extraction and classification:
      1. Converts multi-channel (stereo) signals to 1D mono.
      2. Trims leading and trailing silence below `top_db` (default 40 dB).
      3. Bounds duration to at most `max_duration_sec` (default 5.0 seconds).
      4. Resamples the signal to `target_sr` (default 16 kHz).

    Parameters:
      y: Audio numpy array of any dimension / channel layout.
      sr: Native sampling rate of `y`.
      target_sr: Target sampling rate (standard 16000 Hz).
      max_duration_sec: Maximum duration limit in seconds (standard 5.0s).
      top_db: Decibels below peak reference to consider silence (standard 40 dB).
      slice_mode: 'first' (first 5 seconds after trim) or 'energy' (highest RMS window).

    Returns:
      A tuple of (conditioned_audio_array, target_sr).
    """
    if y is None or len(y) == 0:
        return np.zeros(0, dtype=np.float32), target_sr

    # Ensure float32 representation
    y_proc = np.asarray(y, dtype=np.float32)

    # 1. Convert multi-channel / stereo audio to 1D mono
    if y_proc.ndim > 1:
        if y_proc.shape[0] in (1, 2) and y_proc.shape[1] > y_proc.shape[0]:
            y_proc = librosa.to_mono(y_proc)
        elif y_proc.shape[1] in (1, 2) and y_proc.shape[0] > y_proc.shape[1]:
            y_proc = librosa.to_mono(y_proc.T)
        else:
            y_proc = np.mean(y_proc, axis=0 if y_proc.shape[0] < y_proc.shape[1] else 1)

    y_proc = np.ascontiguousarray(y_proc, dtype=np.float32)

    # Check for empty or near-absolute silence
    peak = float(np.max(np.abs(y_proc))) if len(y_proc) > 0 else 0.0
    if peak < 1e-8 or len(y_proc) == 0:
        max_samples = int(max_duration_sec * target_sr)
        return np.zeros(min(len(y_proc), max_samples), dtype=np.float32), target_sr

    # 2. Trim silence via -40 dB gate
    try:
        y_trimmed, _ = librosa.effects.trim(y_proc, top_db=top_db)
        if len(y_trimmed) > 0:
            y_proc = y_trimmed
    except Exception:
        # Fallback if trimming fails on edge signals
        pass

    # 3. Limit to max duration at native sample rate before resampling for performance
    native_max_samples = int(max_duration_sec * sr)
    if len(y_proc) > native_max_samples:
        if slice_mode == "energy":
            y_proc = _extract_highest_energy_slice(y_proc, native_max_samples)
        else:
            y_proc = y_proc[:native_max_samples]

    # 4. Resample to target_sr (16 kHz)
    if sr != target_sr:
        try:
            y_resampled = librosa.resample(
                y_proc,
                orig_sr=sr,
                target_sr=target_sr,
                res_type=_RES_TYPE,
            )
        except Exception:
            y_resampled = librosa.resample(
                y_proc,
                orig_sr=sr,
                target_sr=target_sr,
                res_type="scipy",
            )
    else:
        y_resampled = y_proc

    # Ensure strictly bounded to max_duration_sec at target_sr
    final_max_samples = int(max_duration_sec * target_sr)
    if len(y_resampled) > final_max_samples:
        y_resampled = y_resampled[:final_max_samples]

    return np.ascontiguousarray(y_resampled, dtype=np.float32), target_sr


def _extract_highest_energy_slice(y: np.ndarray, window_len: int) -> np.ndarray:
    """Finds and returns the sub-window of length `window_len` with maximal RMS energy."""
    if len(y) <= window_len:
        return y
    step = max(1, window_len // 4)
    best_energy = -1.0
    best_start = 0
    for start in range(0, len(y) - window_len + 1, step):
        segment = y[start : start + window_len]
        energy = float(np.mean(segment * segment))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return y[best_start : best_start + window_len]


def load_audio(
    path: str | Path,
    sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Loads an audio file from disk using soundfile (fast) with librosa fallback.
    Returns (audio_array, native_sampling_rate).
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Audio file not found: {path_str}")

    try:
        data, native_sr = sf.read(path_str, dtype="float32")
        return data, native_sr
    except Exception:
        data, native_sr = librosa.load(path_str, sr=sr, mono=False)
        return data, native_sr


# ============================================================================
# AC-2: Spectral Centroid & Frequency Band Mapping
# ============================================================================

def calculate_spectral_centroid(y: np.ndarray, sr: int) -> float:
    """
    Calculates the mean spectral centroid in Hz across the signal using librosa.
    Returns 0.0 if the signal is silent, empty, or unmeasurable.
    """
    if y is None or len(y) == 0:
        return 0.0

    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        return 0.0

    try:
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_val = float(np.mean(centroids))
        return 0.0 if math.isnan(mean_val) else max(0.0, mean_val)
    except Exception:
        return 0.0


def map_frequency_band(centroid: float) -> str:
    """
    Maps spectral centroid in Hz to standard 5 frequency bands:
      - Sub-bass: < 60 Hz
      - Bass: 60 - 250 Hz
      - Low-mid: 250 - 2,000 Hz
      - High-mid: 2,000 - 6,000 Hz
      - High: >= 6,000 Hz
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


# Semantic alias matching engine.grouping.frequency_band_from_centroid
frequency_band_from_centroid = map_frequency_band


# ============================================================================
# AC-3: Chroma Profile Extraction
# ============================================================================

def extract_chroma(y: np.ndarray, sr: int) -> list[float]:
    """
    Extracts a 12-bin mean chroma profile using `librosa.feature.chroma_stft`.
    Captures musical pitch distribution [C, C#, D, D#, E, F, F#, G, G#, A, A#, B].
    Returns a list of 12 non-negative floats.
    """
    if y is None or len(y) == 0:
        return [0.0] * 12

    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        return [0.0] * 12

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chroma_matrix = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chroma_matrix, axis=1)
        # Normalize and sanitize
        result = [0.0 if math.isnan(x) else max(0.0, float(x)) for x in mean_chroma]
        if len(result) == 12:
            return result
        # Fallback to pad/trim if library returned non-12 bins
        padded = [0.0] * 12
        for idx in range(min(12, len(result))):
            padded[idx] = result[idx]
        return padded
    except Exception:
        return [0.0] * 12


# ============================================================================
# AC-4: Generic Label Filtering & Instrument Classification
# ============================================================================

def is_generic_label(label: str) -> bool:
    """
    Identifies non-informative generic acoustic labels that must be suppressed
    (e.g., 'music', 'sound effect', 'instrument', 'silence', 'audio', 'sound').
    """
    if not label:
        return True
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", label.lower()).strip()
    if cleaned in GENERIC_LABELS:
        return True
    words = cleaned.split()
    if not words:
        return True
    if all(w in GENERIC_LABELS for w in words):
        return True
    return False


def filter_generic_labels(
    labels: Sequence[str | tuple[str, float]],
) -> list[Any]:
    """
    Filters out generic labels from a sequence of labels or (label, score) tuples.
    """
    filtered: list[Any] = []
    for item in labels:
        if isinstance(item, tuple):
            name, score = item
            if not is_generic_label(str(name)):
                filtered.append((str(name), float(score)))
        else:
            if not is_generic_label(str(item)):
                filtered.append(str(item))
    return filtered


def map_label_to_family(label: str) -> tuple[str, str]:
    """
    Maps an arbitrary or AudioSet class label into:
      (canonical_instrument, canonical_10_family)
    """
    if not label:
        return "Unknown", "Other"

    clean = label.strip().lower()

    # Direct match in AudioSet dictionary
    if clean in AUDIOSET_LABEL_MAP:
        return AUDIOSET_LABEL_MAP[clean]

    # Substring / pattern matching
    for pattern, inst, family, _ in KEYWORD_MAPPINGS:
        if pattern.search(clean):
            return inst, family

    # Sanitize and default
    sanitized = re.sub(r'[\\/*?:"<>|]', " ", label).strip().title()
    return (sanitized or "Other"), "Other"


def classify_by_keywords(text: str) -> tuple[str, str, float] | None:
    """
    Performs deterministic keyword extraction on filenames or track titles.
    Suppresses generic names and returns (instrument, family, confidence) or None.
    """
    if not text:
        return None

    # Replace path separators and extensions
    base_name = os.path.splitext(os.path.basename(text))[0]
    normalized = re.sub(r"[_\-.\s\(\)\[\]]+", " ", base_name).strip()

    # If the normalized text is purely generic, reject
    if is_generic_label(normalized):
        return None

    for pattern, instrument, family, confidence in KEYWORD_MAPPINGS:
        if pattern.search(normalized):
            return instrument, family, confidence

    return None


def classify_by_dsp(
    y: np.ndarray,
    sr: int,
    centroid: float | None = None,
) -> tuple[str, str, float]:
    """
    Deterministic DSP heuristic classifier based on spectral centroid,
    spectral flatness, zero crossing rate, and chroma concentration.
    Used when filename is ambiguous and ML weights are absent.
    """
    if y is None or len(y) == 0 or np.max(np.abs(y)) < 1e-8:
        return "Silence", "Other", 0.50

    if centroid is None or centroid <= 0.0:
        centroid = calculate_spectral_centroid(y, sr)

    # 1. Sub-bass & Bass ranges (< 250 Hz)
    if centroid < 60.0:
        return "Sub Bass", "Bass", 0.85
    elif centroid < 250.0:
        return "Bass", "Bass", 0.80

    # 2. Extreme High range (> 6000 Hz) -> Cymbals / Air / Hats
    if centroid >= 6000.0:
        return "Cymbal / Air", "Rhythm/Drums", 0.75

    # 3. Mid frequencies: extract spectral flatness and zero crossing rate
    try:
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    except Exception:
        flatness = 0.0
        zcr = 0.0

    # Percussive / noisy signal with high flatness or high ZCR
    if flatness > 0.15 or zcr > 0.12:
        if centroid > 2500.0:
            return "Snare / Percussion", "Rhythm/Drums", 0.70
        return "Percussion", "Rhythm/Drums", 0.65

    # Tonal content (chroma peak analysis)
    try:
        chroma = extract_chroma(y, sr)
        max_chroma = max(chroma) if chroma else 0.0
    except Exception:
        max_chroma = 0.0

    if centroid < 800.0:
        if max_chroma > 0.6:
            return "Keys / Acoustic", "Keys", 0.65
        return "Low-mid Instrument", "Other", 0.50
    elif centroid < 2500.0:
        if max_chroma > 0.6:
            return "Guitar / Lead", "Guitars", 0.60
        return "Mid Instrument", "Other", 0.50
    else:
        return "Synth / Lead", "Synth", 0.60


class YamnetClassifier:
    """
    YAMNet Onnx Runtime Inference Engine with robust fallback to keyword
    matching and DSP heuristics.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        session: Any = None,
        class_map: Sequence[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.session = session
        self.class_map = list(class_map) if class_map else None
        self._initialized = False

        if session is not None or model_path is not None:
            self._init_session()

    def _init_session(self) -> None:
        if self.session is not None:
            self._initialized = True
            return
        if self.model_path and os.path.exists(str(self.model_path)):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(str(self.model_path))
                self._initialized = True
            except Exception:
                self._initialized = False

    @property
    def is_model_available(self) -> bool:
        return self._initialized and self.session is not None

    def predict(
        self,
        y: np.ndarray,
        sr: int = TARGET_SAMPLING_RATE,
    ) -> list[tuple[str, float]]:
        """
        Runs model inference if ONNX session is active, filtering out generic labels.
        Returns a sorted list of (label, score) tuples.
        """
        if not self.is_model_available or self.session is None:
            return []

        try:
            # Prepare 16kHz float32 input waveform
            waveform = np.asarray(y, dtype=np.float32)
            if waveform.ndim == 1:
                input_tensor = waveform
            else:
                input_tensor = waveform.flatten()

            input_name = self.session.get_inputs()[0].name
            # ONNX models may expect [N] or [1, N]
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 2:
                input_data = np.expand_dims(input_tensor, axis=0)
            else:
                input_data = input_tensor

            outputs = self.session.run(None, {input_name: input_data})
            scores = outputs[0]  # Shape: (frames, classes) or (classes,)
            if scores.ndim > 1:
                mean_scores = np.mean(scores, axis=0)
            else:
                mean_scores = scores

            results: list[tuple[str, float]] = []
            for idx, score in enumerate(mean_scores):
                class_name = self.class_map[idx] if (self.class_map and idx < len(self.class_map)) else f"class_{idx}"
                results.append((class_name, float(score)))

            # Sort descending
            results.sort(key=lambda x: x[1], reverse=True)

            # Suppress generic labels
            filtered = filter_generic_labels(results)
            return filtered
        except Exception:
            return []

    def classify(
        self,
        y: np.ndarray,
        sr: int = TARGET_SAMPLING_RATE,
        filename: str = "",
        centroid: float | None = None,
    ) -> tuple[str, str, float]:
        """
        Hierarchical classification pipeline:
          1. Model inference (if ONNX available), with generic suppression.
          2. Filename keyword analysis (deterministic, generic filtered).
          3. DSP heuristics (centroid, flatness, ZCR, chroma).
        Returns (instrument, family, confidence).
        """
        # 1. Model inference
        if self.is_model_available:
            predictions = self.predict(y, sr)
            if predictions:
                top_label, top_score = predictions[0]
                inst, family = map_label_to_family(top_label)
                if family != "Other" or top_score >= 0.5:
                    return inst, family, top_score

        # 2. Filename keyword analysis
        if filename:
            kw_match = classify_by_keywords(filename)
            if kw_match:
                inst, family, conf = kw_match
                return inst, family, conf

        # 3. DSP heuristics
        return classify_by_dsp(y, sr, centroid)


# Global default classifier instance
_DEFAULT_CLASSIFIER = YamnetClassifier()


def classify_instrument(
    y: np.ndarray,
    sr: int = TARGET_SAMPLING_RATE,
    filename: str = "",
    model_path: str | None = None,
    centroid: float | None = None,
) -> tuple[str, str, float]:
    """
    Classifies an audio signal's instrument and 10-family category.
    Returns (instrument_name, family_name, confidence_score).
    """
    if model_path:
        classifier = YamnetClassifier(model_path=model_path)
    else:
        classifier = _DEFAULT_CLASSIFIER

    return classifier.classify(y, sr=sr, filename=filename, centroid=centroid)


# ============================================================================
# Main Analysis Entrypoints
# ============================================================================

def analyze_track(
    audio: str | Path | np.ndarray,
    sr: int | None = None,
    filename: str | None = None,
    model_path: str | None = None,
    max_duration_sec: float = MAX_ANALYSIS_DURATION_SEC,
    top_db: float = SILENCE_TOP_DB,
) -> TrackAnalysis:
    """
    Performs comprehensive audio conditioning, feature extraction, and
    classification on an audio track.

    Accepts either:
      - A file path (str or Path) to an audio file (.wav, .flac, .mp3, etc.).
      - A numpy audio array with corresponding sample rate `sr`.

    Returns:
      A `TrackAnalysis` instance matching all acceptance criteria.
    """
    file_path = ""
    native_sr = sr or TARGET_SAMPLING_RATE

    if isinstance(audio, (str, Path)):
        file_path = str(audio)
        filename = filename or os.path.basename(file_path)
        y, native_sr = load_audio(file_path, sr=sr)
    elif isinstance(audio, np.ndarray):
        y = audio
        filename = filename or "audio.wav"
    else:
        raise TypeError(f"Expected str, Path, or np.ndarray, got {type(audio)}")

    duration = float(len(y) / native_sr) if native_sr > 0 and len(y) > 0 else 0.0

    # AC-1: Condition audio (trim silence, resample to 16 kHz, slice to 5s)
    y_cond, cond_sr = condition_audio(
        y,
        sr=native_sr,
        target_sr=TARGET_SAMPLING_RATE,
        max_duration_sec=max_duration_sec,
        top_db=top_db,
    )

    # AC-2: Spectral Centroid & Frequency Band Mapping
    centroid = calculate_spectral_centroid(y_cond, cond_sr)
    band = map_frequency_band(centroid)

    # AC-3: 12-bin mean Chroma profile
    chroma = extract_chroma(y_cond, cond_sr)

    # AC-4: Instrument & Family classification with generic suppression
    instrument, family, confidence = classify_instrument(
        y_cond,
        sr=cond_sr,
        filename=filename,
        model_path=model_path,
        centroid=centroid,
    )

    return TrackAnalysis(
        file_path=file_path,
        filename=filename,
        instrument=instrument,
        family=family,
        frequency_band=band,
        spectral_centroid=centroid,
        chroma=chroma,
        confidence=confidence,
        duration=duration,
        sample_rate=cond_sr,
    )


def analyze_tracks(
    tracks: Sequence[str | Path | tuple[np.ndarray, int]],
    model_path: str | None = None,
) -> list[TrackAnalysis]:
    """
    Batch analysis for multiple tracks.
    """
    results: list[TrackAnalysis] = []
    for item in tracks:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], np.ndarray):
            y_arr, sr_val = item
            results.append(analyze_track(y_arr, sr=sr_val, model_path=model_path))
        else:
            results.append(analyze_track(item, model_path=model_path))
    return results
