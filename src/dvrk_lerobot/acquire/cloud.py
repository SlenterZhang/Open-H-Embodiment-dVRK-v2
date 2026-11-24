"""Bounded background writer for synchronized per-frame colored point clouds."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import numpy as np

from dvrk_lerobot.acquire.camera import CameraError
from dvrk_lerobot.dataset.cloud import write_ply


class CloudWriter:
    """Own copied RGB-D arrays until their PLY and timestamp have been written."""

    def __init__(self, root: Path, info: dict, unit_rays: np.ndarray) -> None:
        root.mkdir()
        self.root, self.unit_rays = root, unit_rays
        self.scale = info["depth_scale"]
        self.count = 0
        self._error: Exception | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=30)
        self._done = threading.Event()
        self._metadata = {
            "format": "binary_little_endian_ply", "units": "meters",
            "coordinate_frame": "color camera: x right, y down, z forward",
            "color": "RGB from paired capture before video compression",
            "depth_scale": self.scale, "serial": info["serial"],
            "intrinsics": info["color_intrinsics"],
            "indexing": "frame_000000.ply matches external video frame 0",
            "status": "recording", "frame_count": 0,
        }
        self._save_metadata()
        self._thread = threading.Thread(target=self._loop, name="cloud-writer", daemon=True)
        self._thread.start()

    def submit(self, index: int, timestamp: int, raw: np.ndarray, bgr: np.ndarray) -> None:
        """Enqueue a frame without blocking capture; fail explicitly on backlog."""
        self.check()
        try:
            self._queue.put_nowait((index, timestamp, raw.copy(), bgr.copy()))
        except queue.Full as exc:
            self._error = CameraError("point-cloud writer cannot keep up (30-frame queue full)")
            raise self._error from exc

    def check(self) -> None:
        if self._error is not None:
            raise CameraError(f"point-cloud recording failed: {self._error}") from self._error

    def close(self) -> None:
        """Drain queued frames and propagate any write failure."""
        self._done.set()
        self._thread.join()
        self.check()

    def _save_metadata(self) -> None:
        temporary = self.root / "metadata.json.tmp"
        temporary.write_text(json.dumps(self._metadata, indent=2) + "\n")
        temporary.replace(self.root / "metadata.json")

    def _loop(self) -> None:
        try:
            with (self.root / "timestamps.csv").open("x", buffering=1) as log:
                log.write("frame_index,timestamp_ns,points\n")
                while not self._done.is_set() or not self._queue.empty():
                    try:
                        index, timestamp, raw, bgr = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    valid = raw != 0
                    xyz = self.unit_rays[valid] * (
                        raw[valid].astype(np.float32) * self.scale
                    )[:, None]
                    destination = self.root / f"frame_{index:06d}.ply"
                    temporary = destination.with_suffix(".ply.tmp")
                    write_ply(temporary, xyz, bgr[valid][:, ::-1])
                    temporary.replace(destination)
                    log.write(f"{index},{timestamp},{len(xyz)}\n")
                    self.count += 1
        except Exception as exc:  # noqa: BLE001 - propagate worker failures to recorder
            self._error = exc
        finally:
            self._metadata.update(frame_count=self.count, status="failed" if self._error else "ok")
            if self._error:
                self._metadata["error"] = str(self._error)
            try:
                self._save_metadata()
            except Exception as exc:  # noqa: BLE001 - propagate worker failures to recorder
                self._error = exc
