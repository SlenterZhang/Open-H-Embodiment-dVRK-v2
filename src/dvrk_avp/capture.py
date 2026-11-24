"""Background V4L2 capture that always holds the newest frame of one camera."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class LatestFrameCamera:
    """Reads a camera on its own thread and keeps only the most recent frame.

    Dropping old frames keeps display latency at most one frame period.
    """

    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        self.device = device
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open {device}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._stamp = frame, time.monotonic()

    def latest(self) -> tuple[np.ndarray | None, float]:
        """Return the newest frame and its monotonic arrival time."""
        with self._lock:
            return self._frame, self._stamp

    def close(self) -> None:
        """Stop the reader thread and release the device."""
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
