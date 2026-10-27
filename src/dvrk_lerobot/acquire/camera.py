"""V4L2 stereo camera capture with software timestamps."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2

from dvrk_lerobot.config import CameraConfig


class CameraError(RuntimeError):
    """A V4L2 device could not be opened or stopped producing frames."""


class Camera:
    """One V4L2 camera captured on its own background thread."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None
        self._count = 0
        self._info: dict[str, Any] = {}

    def open(self) -> dict[str, Any]:
        """Open the device and validate one frame at the requested size."""
        if self._capture is not None:
            return dict(self._info)
        capture = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"cannot open {self.config.device}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        if not ok or frame is None or frame.ndim != 3:
            capture.release()
            raise CameraError(f"{self.config.device}: cannot read a color frame")
        height, width = frame.shape[:2]
        if (width, height) != (self.config.width, self.config.height):
            capture.release()
            raise CameraError(
                f"{self.config.device}: negotiated {width}x{height}, "
                f"expected {self.config.width}x{self.config.height}"
            )
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(self.config.fps)
        self._capture = capture
        self._info = {
            "device": self.config.device,
            "width": width,
            "height": height,
            "fps": actual_fps,
            "fourcc": _fourcc(capture),
        }
        return dict(self._info)

    def start(self, directory: Path, stop: threading.Event) -> None:
        """Start writing JPEG frames named by capture timestamp."""
        if self._capture is None:
            raise CameraError(f"{self.config.device}: open() must be called first")
        if self._thread is not None and self._thread.is_alive():
            return
        directory.mkdir(parents=True, exist_ok=True)
        self._stop = stop
        self._thread = threading.Thread(
            target=self._loop,
            args=(directory, stop),
            name=f"camera-{self.config.side}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Ask the capture thread to stop and close the device."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.close()

    def close(self) -> None:
        """Release the V4L2 handle without waiting for the thread."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def check(self) -> None:
        """Raise the capture thread's failure, if any."""
        if self._error:
            raise CameraError(f"{self.config.device}: {self._error}")

    @property
    def frame_count(self) -> int:
        return self._count

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def _loop(self, directory: Path, stop: threading.Event) -> None:
        capture = self._capture
        assert capture is not None
        quality = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        try:
            while not stop.is_set():
                ok, frame = capture.read()
                timestamp = time.time_ns()
                if not ok or frame is None:
                    if not stop.is_set():
                        self._error = "frame read failed"
                    break
                height, width = frame.shape[:2]
                if (width, height) != (self.config.width, self.config.height):
                    self._error = f"frame size changed to {width}x{height}"
                    break
                path = directory / f"{timestamp}.jpg"
                while path.exists():
                    timestamp += 1
                    path = directory / f"{timestamp}.jpg"
                if not cv2.imwrite(str(path), frame, quality):
                    self._error = f"cannot write {path}"
                    break
                self._count += 1
        except Exception as exc:  # noqa: BLE001 - surface thread failures
            self._error = str(exc)
        finally:
            capture.release()
            if self._capture is capture:
                self._capture = None


def _fourcc(capture: cv2.VideoCapture) -> str:
    value = int(capture.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
