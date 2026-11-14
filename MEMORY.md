# MEMORY LEDGER: Project Orchestration State

## 1. System & Architecture Overview
- **Project Name:** Smart Export (Standalone Cross-DAW Stem Consolidator)
- **Target Spec:**
  - Standalone desktop audio tool (`pywebview` native bridge, zero local web servers).
  - High-precision 32-bit float audio summing with -0.1 dBFS headroom protection.
  - Spectral collision detection via 15% centroid rule + Librosa DSP + YAMNet instrument classification (10 families).
  - Dark-mode responsive UI (Tailwind CSS, Lucide icons, DESIGN.md tokens).
- **Stack & Environment:**
  - Runtime: Python 3.14 (venv at `.venv`)
  - Audio/DSP: `numpy`, `soundfile`, `scipy`, `librosa`
  - Machine Learning: Google YAMNet (local on-device inference via ONNX Runtime / TF)
  - UI Shell: `pywebview` Desktop GUI + HTML5/CSS/JS frontend
- **Git State:**
  - Base: `main`
  - Active Branch: `main`
  - GitHub Remote: Authenticated as `godvy`. Candidate remote: `godvy/Smart-export-for-DAWs-plugin` or new `godvy/smart-export`.

## 2. Dependency Waves & Ticket DAG
```yaml
waves:
  - wave: 1
    description: "Wave 1: Foundation (Audio Summing & Spectral Masking Engine)"
    tickets:
      - id: "T-101"
        wave: 1
        domain: "backend"
        allowed_files:
          - "engine/summer.py"
          - "tests/test_summing.py"
        acceptance_criteria:
          - "AC-1: Expands mono audio arrays into 2-channel stereo seamlessly."
          - "AC-2: Pads unequal-length tracks to maximum track length and sums audio with float32 math."
          - "AC-3: Detects peak amplitude exceeding 1.0 (0 dBFS) and scales cleanly to -0.1 dBFS."
          - "AC-4: Writes 32-bit float stereo WAV files to output destination."
        eval: ".venv/Scripts/pytest tests/test_summing.py"

      - id: "T-102"
        wave: 1
        domain: "backend"
        allowed_files:
          - "engine/grouping.py"
          - "tests/test_masking.py"
        acceptance_criteria:
          - "AC-1: Supports Grouping Mode 0 (Rename only), Mode 1 (By Instrument family), and Mode 2 (By Frequency band)."
          - "AC-2: Evaluates 15% centroid collision rule: abs(C_A - C_B) / avg(C_A, C_B) < 0.15."
          - "AC-3: Dynamically splits colliding tracks into safe subgroups (e.g. Drums.wav, Drums_Part2.wav)."
        eval: ".venv/Scripts/pytest tests/test_masking.py"

  - wave: 2
    description: "Wave 2: Core DSP & ML (Feature Extraction, YAMNet & Manifest)"
    tickets:
      - id: "T-201"
        wave: 2
        domain: "backend"
        allowed_files:
          - "engine/analyzer.py"
          - "tests/test_analyzer.py"
        acceptance_criteria:
          - "AC-1: Trims silence via -40 dB gate and resamples to 16 kHz mono slice (up to 5 seconds)."
          - "AC-2: Calculates spectral centroid in Hz and maps to 5 frequency bands (Sub-bass, Bass, Low-mid, High-mid, High)."
          - "AC-3: Extracts 12-bin mean chroma profile capturing musical pitch distribution."
          - "AC-4: Classifies instrument with generic label filtering and maps to the 10 defined families."
        eval: ".venv/Scripts/pytest tests/test_analyzer.py"

      - id: "T-202"
        wave: 2
        domain: "backend"
        allowed_files:
          - "engine/manifest.py"
          - "tests/test_manifest.py"
        acceptance_criteria:
          - "AC-1: Emits Final_Stems/export_manifest.json with ISO timestamp."
          - "AC-2: Records output stem files and mapped contributing source tracks."
          - "AC-3: Embeds complete DSP attributes (instrument, family, band, centroid Hz, chroma vector) for each track."
        eval: ".venv/Scripts/pytest tests/test_manifest.py"

  - wave: 3
    description: "Wave 3: UI & Shell Integration (Dark Mode UI & Native Desktop Bridge)"
    tickets:
      - id: "T-301"
        wave: 3
        domain: "frontend"
        allowed_files:
          - "ui/index.html"
          - "ui/app.js"
          - "ui/style.css"
          - "tests/test_ui_tokens.js"
        acceptance_criteria:
          - "AC-1: Modern dark-mode UI with pulsing drag-and-drop folder drop zone."
          - "AC-2: Mode selector pill ([By Instrument] [By Frequency] [Rename Only]) and Auto-rename toggle."
          - "AC-3: Live animated progress bar with track-by-track status and success screen with 1-click reveal button."
          - "AC-4: 100% token compliant with DESIGN.md (zero raw hex or arbitrary px)."
        eval: "node tests/test_ui_tokens.js"

      - id: "T-302"
        wave: 3
        domain: "infra"
        allowed_files:
          - "app.py"
          - "engine/__init__.py"
          - "tests/test_bridge.py"
        acceptance_criteria:
          - "AC-1: Launches pywebview window with AppBridge API exposing folder selection, scan, and export methods."
          - "AC-2: Orchestrates analyzer -> grouping -> summer -> manifest pipeline with live event callbacks."
          - "AC-3: Operates with zero local HTTP servers and zero open terminal windows."
        eval: ".venv/Scripts/pytest tests/test_bridge.py"
```

## 3. Wave Execution Status
- Current Active Wave: All Waves Complete (Production Ready)
- Completed Waves:
  - Wave 1: Foundation (Audio Summing & Spectral Masking Engine) - 46/46 tests passing
    - T-101: done
    - T-102: done
  - Wave 2: Core DSP & Machine Learning - 124/124 tests passing
    - T-201: done
    - T-202: done
  - Wave 3: UI & Shell Integration - 12/12 pytest + 44/44 DOM token tests passing
    - T-301: done
    - T-302: done
- Blocked Tickets: None
- Total Test Suite: 182 pytest unit tests + 44 UI/DOM assertions (100% pass rate)

## 4. Telemetry & Circuit Breaker Log
| Ticket ID | Attempt | Exit Code | Error Type | Failed Assertion | Status |
|-----------|---------|-----------|------------|------------------|--------|
| T-101     | 1       | 0         | None       | None             | done   |
| T-102     | 1       | 0         | None       | None             | done   |
| T-201     | 1       | 0         | None       | None             | done   |
| T-202     | 1       | 0         | None       | None             | done   |
| T-301     | 1       | 0         | None       | None             | done   |
| T-302     | 1       | 0         | None       | None             | done   |
