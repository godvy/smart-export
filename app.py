"""
Smart Export Native Desktop Application and PyWebview Bridge.

Implements Wave 3 Ticket T-302:
  - AC-1: Launches pywebview window with AppBridge API exposing folder selection,
          scan, and export methods.
  - AC-2: Orchestrates analyzer -> grouping -> summer -> manifest pipeline
          with live event callbacks.
  - AC-3: Operates with zero local HTTP servers and zero open terminal windows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import webview

import engine


class AppBridge:
    """
    JS-Python Bridge exposing native system and DSP pipeline methods
    to pywebview frontend client.
    """

    def __init__(self, window: Any = None) -> None:
        self._window: Any = window
        if window is not None:
            try:
                setattr(window, "_serializable", False)
            except Exception:
                pass
        self.last_output_dir: str = "Final_Stems"

    def set_window(self, window: Any) -> None:
        """Attaches active pywebview window instance to bridge."""
        self._window = window
        if window is not None:
            try:
                setattr(window, "_serializable", False)
            except Exception:
                pass

    @property
    def window(self) -> Any:
        return self._window

    @window.setter
    def window(self, val: Any) -> None:
        self._window = val
        if val is not None:
            try:
                setattr(val, "_serializable", False)
            except Exception:
                pass

    def select_folder(self) -> str | None:
        """
        Invokes pywebview's native folder dialog to allow user directory selection.
        Returns the selected folder path as a string, or None if cancelled.
        """
        if self._window is not None:
            try:
                res = self._window.create_file_dialog(webview.FOLDER_DIALOG)
                if res:
                    if isinstance(res, (list, tuple)):
                        return str(res[0]) if len(res) > 0 else None
                    return str(res)
            except Exception:
                pass
        return None

    def notify_progress(self, percent: float, message: str) -> None:
        """
        Dispatches progress updates to the frontend pywebview client.
        Escapes special characters to safely evaluate JS string literals.
        """
        if self._window is not None:
            escaped_message = (
                str(message)
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
            try:
                self._window.evaluate_js(
                    f"window.updateProgress({percent}, '{escaped_message}')"
                )
            except Exception:
                pass

    def start_export(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Orchestrates the Smart Export DSP pipeline:
          1. Scans input_dir for valid audio files.
          2. Emits progress and calls engine.analyzer.analyze_audio_file.
          3. Emits progress and groups tracks with engine.grouping.group_tracks.
          4. Emits progress and sums audio tracks with headroom protection.
          5. Generates and writes export_manifest.json.
          6. Emits completion callback to JS and returns structured result.
        """
        cfg = config or {}
        input_dir = str(cfg.get("input_dir") or cfg.get("folder") or "").strip()
        output_dir = str(cfg.get("output_dir") or "Final_Stems").strip()
        grouping_mode = (
            cfg.get("grouping_mode")
            if "grouping_mode" in cfg
            else cfg.get("mode", 1)
        )
        auto_rename = cfg.get("auto_rename", True)  # noqa: F841

        self.last_output_dir = output_dir

        if not input_dir or not os.path.isdir(input_dir):
            return {"success": False, "error": "No audio files found"}

        audio_extensions = {".wav", ".wave", ".flac", ".aif", ".aiff", ".mp3"}
        filenames = sorted([
            f
            for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in audio_extensions
        ])

        if not filenames:
            return {"success": False, "error": "No audio files found"}

        total = len(filenames)
        analyzed_tracks = []

        # Step 1: Feature Extraction & Classification per track
        for i, filename in enumerate(filenames, 1):
            percent = round((i / total) * 60.0, 1)
            self.notify_progress(percent, f"Analyzing {filename} ({i}/{total})...")
            file_path = os.path.join(input_dir, filename)
            track_analysis = engine.analyzer.analyze_audio_file(file_path)
            analyzed_tracks.append(track_analysis)

        # Step 2: Collision detection & stem grouping
        self.notify_progress(70, "Detecting collisions and grouping stems...")
        groups = engine.grouping.group_tracks(
            analyzed_tracks, mode=grouping_mode, extension=".wav"
        )

        # Step 3: Summing audio tracks with headroom protection
        self.notify_progress(
            80, "Summing audio tracks with -0.1 dBFS headroom protection..."
        )
        os.makedirs(output_dir, exist_ok=True)
        for stem_name, tracks in groups.items():
            output_path = os.path.join(output_dir, stem_name)
            track_paths = [
                getattr(t, "file_path", None) or t.get("file_path", "")
                for t in tracks
            ]
            engine.summer.sum_audio_files(track_paths, output_path=output_path)

        # Step 4: Manifest generation and write
        manifest = engine.manifest.generate_manifest(
            groups, output_dir=output_dir, grouping_mode=grouping_mode
        )
        engine.manifest.save_manifest(manifest, output_dir=output_dir)

        # Step 5: Completion notification
        self.notify_progress(100, "Export complete!")

        if self._window is not None:
            try:
                self._window.evaluate_js(
                    f"window.onExportComplete({json.dumps(manifest)})"
                )
            except Exception:
                pass

        return {
            "success": True,
            "manifest": manifest,
            "output_dir": output_dir,
        }

    def open_output_folder(self, output_dir: str = "Final_Stems") -> bool:
        """
        Resolves absolute path and opens the output folder in native file explorer.
        Returns True if process was started, False on failure.
        """
        resolved_dir = output_dir or self.last_output_dir or "Final_Stems"
        target = os.path.abspath(resolved_dir)
        if not os.path.exists(target):
            try:
                os.makedirs(target, exist_ok=True)
            except OSError:
                pass

        try:
            if sys.platform == "win32":
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", target], check=True)
            else:
                subprocess.run(["xdg-open", target], check=True)
            return True
        except Exception:
            return False


def main() -> None:
    """
    Application entrypoint.
    Launches pywebview window bound to local ui/index.html with AppBridge API.
    Zero local HTTP servers and zero open terminal windows.
    """
    bridge = AppBridge()
    window = webview.create_window(
        "Smart Export",
        "ui/index.html",
        js_api=bridge,
        width=960,
        height=720,
        resizable=True,
        background_color="#0a0c10",
    )
    bridge.set_window(window)
    webview.start(debug=False)


if __name__ == "__main__":
    main()
