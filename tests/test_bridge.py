"""
Unit Tests for Smart Export Desktop Bridge and Orchestration (Ticket T-302).

Verifies:
  - AC-1: Launches pywebview window with AppBridge API exposing folder selection,
          scan, and export methods.
  - AC-2: Orchestrates analyzer -> grouping -> summer -> manifest pipeline
          with live event callbacks.
  - AC-3: Operates with zero local HTTP servers and zero open terminal windows.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import webview

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app
import engine
from app import AppBridge, main


# ============================================================================
# Bridge Initialization and Window Attachment Tests
# ============================================================================


def test_bridge_init_and_set_window() -> None:
    """Tests AppBridge initialization with and without window."""
    bridge = AppBridge()
    assert bridge.window is None
    assert bridge.last_output_dir == "Final_Stems"

    mock_win = MagicMock()
    bridge.set_window(mock_win)
    assert bridge.window is mock_win

    # Test initialization with window directly
    bridge2 = AppBridge(mock_win)
    assert bridge2.window is mock_win


def test_select_folder_with_window() -> None:
    """Tests folder selection when window file dialog returns a directory."""
    mock_win = MagicMock()
    mock_win.create_file_dialog.return_value = ("/selected/audio/folder",)
    bridge = AppBridge(mock_win)

    selected = bridge.select_folder()
    assert selected == "/selected/audio/folder"
    mock_win.create_file_dialog.assert_called_once_with(webview.FOLDER_DIALOG)


def test_select_folder_cancelled_or_no_window() -> None:
    """Tests folder selection returns None when cancelled or no window set."""
    bridge = AppBridge()
    assert bridge.select_folder() is None

    mock_win = MagicMock()
    mock_win.create_file_dialog.return_value = None
    bridge.set_window(mock_win)
    assert bridge.select_folder() is None

    mock_win.create_file_dialog.return_value = ()
    assert bridge.select_folder() is None


# ============================================================================
# Live Event / Progress Notification Tests
# ============================================================================


def test_notify_progress_dispatches_to_window() -> None:
    """Verifies notify_progress escapes strings and evaluates window JS."""
    mock_win = MagicMock()
    bridge = AppBridge(mock_win)

    bridge.notify_progress(45.5, "Processing file 1 'Kick' \\stem\\...")
    mock_win.evaluate_js.assert_called_once()
    call_arg = mock_win.evaluate_js.call_args[0][0]

    assert "window.updateProgress(45.5" in call_arg
    assert "Processing file 1 \\'Kick\\' \\\\stem\\\\..." in call_arg


def test_notify_progress_noop_without_window() -> None:
    """Verifies notify_progress does not raise exception when window is None."""
    bridge = AppBridge()
    # Must not raise
    bridge.notify_progress(50.0, "Testing without window")


# ============================================================================
# Empty Folder and Invalid Input Tests
# ============================================================================


def test_start_export_empty_folder(tmp_path: pytest.TempPathFactory) -> None:
    """Empty folder returns structured error dictionary without crashing."""
    empty_dir = str(tmp_path)
    bridge = AppBridge()

    result = bridge.start_export({"input_dir": empty_dir, "output_dir": "out"})
    assert result == {"success": False, "error": "No audio files found"}


def test_start_export_nonexistent_folder() -> None:
    """Non-existent directory returns structured error dictionary."""
    bridge = AppBridge()
    result = bridge.start_export({"input_dir": "non_existent_dir_12345"})
    assert result == {"success": False, "error": "No audio files found"}


def test_start_export_folder_with_non_audio_files(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Folder containing only non-audio files returns structured error."""
    folder = str(tmp_path)
    with open(os.path.join(folder, "notes.txt"), "w") as f:
        f.write("Not an audio file")
    with open(os.path.join(folder, "image.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")

    bridge = AppBridge()
    result = bridge.start_export({"input_dir": folder})
    assert result == {"success": False, "error": "No audio files found"}


# ============================================================================
# Open Output Folder Tests
# ============================================================================


def test_open_output_folder_path_resolution(tmp_path: pytest.TempPathFactory) -> None:
    """Verifies open_output_folder resolves absolute path and launches platform explorer."""
    test_dir = str(tmp_path / "Final_Stems")
    bridge = AppBridge()

    with patch("sys.platform", "win32"), patch("os.startfile") as mock_startfile:
        res = bridge.open_output_folder(test_dir)
        assert res is True
        mock_startfile.assert_called_once_with(os.path.abspath(test_dir))

    with patch("sys.platform", "darwin"), patch("subprocess.run") as mock_run:
        res = bridge.open_output_folder(test_dir)
        assert res is True
        mock_run.assert_called_once_with(["open", os.path.abspath(test_dir)], check=True)

    with patch("sys.platform", "linux"), patch("subprocess.run") as mock_run:
        res = bridge.open_output_folder(test_dir)
        assert res is True
        mock_run.assert_called_once_with(["xdg-open", os.path.abspath(test_dir)], check=True)


def test_open_output_folder_handles_exception() -> None:
    """Verifies open_output_folder returns False on OS exception."""
    bridge = AppBridge()
    with patch("sys.platform", "win32"), patch("os.startfile", side_effect=OSError("Access denied")):
        res = bridge.open_output_folder("some_dir")
        assert res is False


# ============================================================================
# End-to-End Orchestration Pipeline Tests (AC-1, AC-2, AC-3)
# ============================================================================


def test_end_to_end_audio_export(tmp_path: pytest.TempPathFactory) -> None:
    """
    End-to-end integration test:
      1. Generates 3 synthetic WAV files (Kick @ 90 Hz, Snare @ 1200 Hz, Lead @ 2500 Hz).
      2. Executes AppBridge.start_export with grouping_mode=1.
      3. Asserts success == True, output WAV stems exist in temp_out.
      4. Asserts export_manifest.json exists and accurately records tracks & stems.
      5. Asserts live progress callbacks and completion evaluation on mock window.
    """
    temp_in = str(tmp_path / "raw_tracks")
    temp_out = str(tmp_path / "stems_out")
    os.makedirs(temp_in, exist_ok=True)

    sr = 16000
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)

    # Synthetic Audio 1: Kick drum @ 90 Hz
    kick_signal = np.sin(2 * np.pi * 90 * t).astype(np.float32) * 0.8
    kick_path = os.path.join(temp_in, "Kick.wav")
    sf.write(kick_path, kick_signal, sr)

    # Synthetic Audio 2: Snare drum @ 1200 Hz
    snare_signal = np.sin(2 * np.pi * 1200 * t).astype(np.float32) * 0.8
    snare_path = os.path.join(temp_in, "Snare.wav")
    sf.write(snare_path, snare_signal, sr)

    # Synthetic Audio 3: Lead Synth @ 2500 Hz
    lead_signal = np.sin(2 * np.pi * 2500 * t).astype(np.float32) * 0.8
    lead_path = os.path.join(temp_in, "Lead.wav")
    sf.write(lead_path, lead_signal, sr)

    mock_win = MagicMock()
    bridge = AppBridge(mock_win)

    # Trigger export pipeline
    result = bridge.start_export({
        "input_dir": temp_in,
        "output_dir": temp_out,
        "grouping_mode": 1,
    })

    # Assert returned result structure
    assert result["success"] is True
    assert result["output_dir"] == temp_out
    assert "manifest" in result
    manifest = result["manifest"]

    # Assert output stems exist in temp_out
    out_files = os.listdir(temp_out)
    assert any(f.endswith(".wav") for f in out_files)
    manifest_file = os.path.join(temp_out, "export_manifest.json")
    assert os.path.exists(manifest_file)

    # Read and validate manifest on disk
    with open(manifest_file, "r", encoding="utf-8") as f:
        disk_manifest = json.load(f)

    assert disk_manifest["total_input_tracks"] == 3
    assert disk_manifest["total_output_stems"] >= 1
    assert disk_manifest["grouping_mode"] == 1

    # Verify all 3 input tracks are recorded across the stems
    recorded_sources: list[str] = []
    for stem in disk_manifest["stems"]:
        stem_path = os.path.join(temp_out, stem["stem_file"])
        assert os.path.exists(stem_path), f"Output stem file {stem['stem_file']} missing on disk"
        for src in stem["source_tracks"]:
            recorded_sources.append(src["filename"])

    assert "Kick.wav" in recorded_sources
    assert "Snare.wav" in recorded_sources
    assert "Lead.wav" in recorded_sources

    # Verify window evaluate_js was called for progress and completion
    calls = [call[0][0] for call in mock_win.evaluate_js.call_args_list]
    assert any("window.updateProgress(100" in c for c in calls)
    assert any("window.onExportComplete(" in c for c in calls)


# ============================================================================
# Entrypoint and Zero-Server Architectural Invariant Tests (AC-3)
# ============================================================================


def test_main_entrypoint_creates_window_without_http_server() -> None:
    """
    Verifies main() launches pywebview directly with local html path,
    enforcing zero local HTTP server and zero open terminal windows.
    """
    with patch("webview.create_window") as mock_create_window, patch("webview.start") as mock_start:
        mock_win = MagicMock()
        mock_create_window.return_value = mock_win

        main()

        mock_create_window.assert_called_once_with(
            "Smart Export",
            "ui/index.html",
            js_api=ANY,
            width=960,
            height=720,
            resizable=True,
            background_color="#0a0c10",
        )
        # Verify bridge instance was passed as js_api
        bridge_instance = mock_create_window.call_args[1]["js_api"]
        assert isinstance(bridge_instance, AppBridge)
        assert bridge_instance.window is mock_win

        mock_start.assert_called_once_with(debug=False)
