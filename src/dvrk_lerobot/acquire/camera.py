"""Stereo capture with driver metadata and backward-compatible host timestamps."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dvrk_lerobot.acquire.stereo_log import StereoTimestampLog
from dvrk_lerobot.acquire.v4l2 import V4L2Capture
from dvrk_lerobot.config import CameraConfig
from dvrk_lerobot.dataset.video import VideoError, finish_encoder, start_encoder

# Fast enough for two live 1080p30 streams; quality is set by the per-camera crf.
RAW_PRESET = "veryfast"


class CameraError(RuntimeError):
    """A V4L2 device could not be opened or stopped producing frames."""


@dataclass
class _Episode:
    """Encoder and timestamp file receiving the live stream for one episode."""

    video: Path
    process: subprocess.Popen
    handle: StereoTimestampLog
    count: int = 0


class Camera:
    """One V4L2 camera read continuously on its own background thread.

    Frames are discarded between episodes, which keeps the driver queue fresh,
    and piped to an H.264 encoder while an episode is active.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: V4L2Capture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None
        self._lock = threading.Lock()
        self._episode: _Episode | None = None
        self._ending: _Episode | None = None
        self._latest: Any = None
        self._info: dict[str, Any] = {}

    def open(self) -> dict[str, Any]:
        """Open the device and validate one frame at the requested size."""
        if self._capture is not None:
            return dict(self._info)
        capture = V4L2Capture(self.config)
        try:
            info = capture.open()
            _, timing = capture.read()
        except Exception as exc:
            capture.close()
            raise CameraError(f"{self.config.device}: {exc}") from exc
        self._capture = capture
        self._info = {
            **info,
            "driver_timestamp_clock": timing.driver_clock,
            "driver_timestamp_source": timing.driver_timestamp_source,
            "driver_reports_start_of_exposure": timing.driver_timestamp_source == "start_of_exposure",
        }
        return dict(self._info)

    def start(self, stop: threading.Event) -> None:
        """Start the continuous capture thread."""
        if self._capture is None:
            raise CameraError(f"{self.config.device}: open() must be called first")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = stop
        self._thread = threading.Thread(
            target=self._loop,
            args=(stop,),
            name=f"camera-{self.config.side}",
            daemon=True,
        )
        self._thread.start()

    def begin(self, video: Path, timestamps: Path) -> None:
        """Start encoding the live stream to *video* and capture times to *timestamps*."""
        if self._episode is not None:
            raise CameraError(f"{self.config.device}: an episode is already active")
        process = start_encoder(
            video,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            crf=self.config.crf,
            preset=RAW_PRESET,
        )
        try:
            log = StereoTimestampLog(timestamps)
        except BaseException:
            process.kill()
            process.communicate()
            raise
        episode = _Episode(video, process, log)
        with self._lock:
            self._episode = episode

    def detach(self) -> None:
        """Stop feeding frames to the active episode at once; end() then flushes it."""
        with self._lock:
            if self._episode is not None:
                self._ending, self._episode = self._episode, None

    def end(self) -> int:
        """Detach the episode, flush its encoder, and return its frame count."""
        self.detach()
        episode, self._ending = self._ending, None
        if episode is None:
            return 0
        try:
            episode.handle.close()
        finally:
            # Flush the encoder even if a timestamp-file close fails.
            self._finish_episode_encoder(episode)
        return episode.count

    def _finish_episode_encoder(self, episode: _Episode) -> None:
        try:
            finish_encoder(episode.process, f"encode {episode.video}")
        except (VideoError, OSError) as exc:
            raise CameraError(f"{self.config.device}: {exc}") from exc

    def stop(self) -> None:
        """Ask the capture thread to stop and close the device."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._capture is not None:
            if self._thread is not None and self._thread.is_alive():
                raise CameraError("camera thread did not stop; device left open to avoid a concurrent close")
            self._capture.close()
            self._capture = None

    def check(self) -> None:
        """Raise the capture thread's failure, if any."""
        if self._error:
            raise CameraError(f"{self.config.device}: {self._error}")

    @property
    def frame_count(self) -> int:
        """Frames encoded so far in the active episode."""
        episode = self._episode
        return episode.count if episode is not None else 0

    @property
    def latest(self) -> Any:
        """Most recent BGR frame for preview, or None before the first read."""
        return self._latest

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def _loop(self, stop: threading.Event) -> None:
        capture = self._capture
        assert capture is not None
        try:
            while not stop.is_set():
                frame, timing = capture.read()
                if stop.is_set():
                    break
                height, width = frame.shape[:2]
                if (width, height) != (self.config.width, self.config.height):
                    self._error = f"frame size changed to {width}x{height}"
                    break
                self._latest = frame
                with self._lock:
                    if self._episode is not None:
                        encode_start = time.monotonic_ns()
                        self._episode.process.stdin.write(frame.data)
                        encode_end = time.monotonic_ns()
                        self._episode.handle.write(self._episode.count, timing, encode_start, encode_end)
                        self._episode.count += 1
        except Exception as exc:  # noqa: BLE001 - surface thread failures
            self._error = str(exc)
        finally:
            capture.close()
            if self._capture is capture:
                self._capture = None
