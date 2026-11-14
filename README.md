# Smart Export — Standalone Cross-DAW Stem Consolidator

**Smart Export** is a production-grade, standalone desktop application built for music producers and mixing engineers. It consolidates 40–80 raw recorded multitrack sessions exported from any DAW (Ableton, FL Studio, Logic, REAPER, Pro Tools) into 5–8 clean, collision-free mix stems with sample-accurate summing and intelligent spectral masking detection.

---

## ⚡ Key Capabilities

1. **Local Audio Conditioning & DSP Extraction (`engine/analyzer.py`)**
   - Automatically gates pre-roll dead air using a `-40 dB` top-gate (`librosa.effects.trim`).
   - Converts audio to a 16 kHz mono slice (up to 5 seconds) for feature extraction.
   - Computes exact **Spectral Centroid** in Hz and categorizes into 5 frequency bands:
     - `< 60 Hz`: Sub-bass
     - `60 – 250 Hz`: Bass
     - `250 – 2,000 Hz`: Low-mid
     - `2,000 – 6,000 Hz`: High-mid
     - `> 6,000 Hz`: High
   - Computes a 12-bin mean **Chroma Profile** (`librosa.feature.chroma_stft`) capturing musical pitch distribution.
   - High-confidence on-device instrument classification (suppressing generic labels) mapped to 10 instrument families:
     `Rhythm/Drums`, `Bass`, `Guitars`, `Keys`, `Synth`, `Vocals`, `Strings`, `Brass`, `Woodwinds`, `Other`.

2. **The 15% Centroid Spectral Masking Engine (`engine/grouping.py`)**
   - Identifies acoustic collision risk between tracks destined for the same stem:
     $$\frac{|\text{Centroid}_A - \text{Centroid}_B|}{\text{Average Centroid}} < 0.15$$
   - Automatically prevents frequency masking by partitioning colliding tracks into clean subgroups (e.g., `Drums.wav` and `Drums_Part2.wav`).
   - Supports 3 grouping workflows:
     - `Mode 1`: Stems by Instrument Family (e.g. Drums, Guitars, Bass)
     - `Mode 2`: Stems by Frequency Band (e.g. Low-mid Band, High Band)
     - `Mode 0`: Rename Only (raw renaming without summing)

3. **Sample-Accurate 32-Bit Float Audio Summer (`engine/summer.py`)**
   - Seamlessly expands mono tracks to dual-channel stereo.
   - Zero-pads tracks of unequal length to the maximum session duration.
   - Performs sample-accurate floating point summing: $\text{Summed}(t) = \sum \text{Track}_i(t)$.
   - **Headroom Protection**: Automatically limits peak amplitudes exceeding 1.0 (0 dBFS) to `-0.1 dBFS` to guarantee zero digital clipping or distortion.
   - Writes consolidated stems to `/Final_Stems/` as 32-bit float stereo WAV files.

4. **Export Manifest (`engine/manifest.py`)**
   - Generates `/Final_Stems/export_manifest.json` recording session timestamps, stem mappings, and complete track DSP attributes (family, band, centroid Hz, and 12-bin chroma).

5. **Desktop Shell & Native Bridge (`app.py` & `ui/`)**
   - Powered by `pywebview`: zero local HTTP web servers, zero open terminal windows, zero curl commands.
   - Sleek dark-mode interface styled with Tailwind CSS, Lucide icons, and semantic tokens from `DESIGN.md`.
   - Native OS folder selector dialog and 1-click reveal output folder.

---

## 📁 Repository Layout

```text
smart-export/
├── app.py                      # Desktop entrypoint; launches pywebview with AppBridge API
├── requirements.txt            # Locked project dependencies
├── MEMORY.md                   # Multi-agent orchestrator ledger & ticket DAG
├── SKILL.md                    # Core architectural invariants
├── DESIGN.md                   # Semantic design system tokens
├── engine/
│   ├── __init__.py             # Unified engine exports
│   ├── analyzer.py             # Feature extraction (Centroid, Chroma, Gate, Families)
│   ├── grouping.py             # 15% Centroid masking & stem subgroup resolution
│   ├── summer.py               # 32-bit float audio summing engine (NumPy / SoundFile)
│   └── manifest.py             # export_manifest.json generator
├── ui/
│   ├── index.html              # Modern dark-mode layout
│   ├── app.js                  # Frontend state, drag-drop, & pywebview bridge calls
│   └── style.css               # Animations, glows, and token-compliant CSS rules
├── tests/
│   ├── test_summing.py         # Audio summing, stereo expansion, & headroom tests
│   ├── test_masking.py         # 15% centroid collision tests & subgroup partitioning
│   ├── test_analyzer.py        # DSP, silence gate, and instrument classification tests
│   ├── test_manifest.py        # Manifest serialization & metadata tests
│   ├── test_bridge.py          # End-to-end AppBridge orchestration tests
│   └── test_ui_tokens.js       # Zero-token deterministic fence and DOM validation suite
└── scripts/
    └── fence.py                # Deterministic scope and design token verification fence
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Ensure Python 3.11+ is installed. Activate your virtual environment and install dependencies:

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launching the Desktop Application
Run the desktop app directly:

```bash
python app.py
```

### 3. Using Smart Export
1. Click **"Browse Folder"** or drag-and-drop a folder of exported DAW multitrack WAV files into the dropzone.
2. Select your desired grouping mode:
   - **By Instrument** (recommended for standard stem mixing)
   - **By Frequency** (ideal for mastering and dynamic range processing)
   - **Rename Only** (cleans track names without summing)
3. Click **"RUN SMART EXPORT"**.
4. Monitor the live progress bar as tracks are analyzed, collisions are resolved, and stems are summed.
5. On completion, click **"Open Output Folder"** to inspect your consolidated stems and `export_manifest.json` in `/Final_Stems/`.

---

## 🧪 Running the Verification Suite

Run all unit and integration tests across the engine:

```bash
pytest tests/
```

Run the deterministic design token fence:

```bash
node tests/test_ui_tokens.js
```
