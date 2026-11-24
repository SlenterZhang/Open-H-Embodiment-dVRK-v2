"""One D435i RGB-D frameset stream, with depth aligned to color."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from dvrk_lerobot.acquire.camera import CameraError
from dvrk_lerobot.acquire.rgbd_episode import Episode
from dvrk_lerobot.config import RealSenseConfig
from dvrk_lerobot.dataset.cloud import rays


class RealSense:
    """Capture paired color/depth frames in one thread and one device pipeline."""

    def __init__(self, config: RealSenseConfig) -> None:
        self.config = config
        self._pipeline: Any = None
        self._align: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._episode: Episode | None = None
        self._ending: Episode | None = None
        self._latest: np.ndarray | None = None
        self._error: str | None = None
        self._info: dict[str, Any] = {}
        self._unit_rays: np.ndarray | None = None

    def open(self) -> dict[str, Any]:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError("pyrealsense2 is required for D435i RGB-D capture") from exc
        pipeline = rs.pipeline()
        settings = rs.config()
        if self.config.serial:
            settings.enable_device(self.config.serial)
        settings.enable_stream(
            rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps
        )
        settings.enable_stream(
            rs.stream.depth, self.config.width, self.config.height, rs.format.z16, self.config.fps
        )
        started = False
        try:
            profile = pipeline.start(settings)
            started = True
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise CameraError("D435i did not provide a complete RGB-D frameset")
            if (
                color_frame.get_width() != self.config.width
                or color_frame.get_height() != self.config.height
                or depth_frame.get_width() != self.config.width
                or depth_frame.get_height() != self.config.height
            ):
                raise CameraError("D435i RGB-D frames do not match the configured dimensions")
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            if not math.isfinite(scale) or scale <= 0:
                raise CameraError(f"invalid D435i depth scale: {scale}")
            serial = profile.get_device().get_info(rs.camera_info.serial_number)
            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            calibration = {
                "width": intr.width, "height": intr.height,
                "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                "model": str(intr.model).split(".")[-1], "coeffs": list(intr.coeffs),
            }
            if self.config.record_cloud:
                self._unit_rays = rays(calibration)
        except Exception as exc:
            if started:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
            raise CameraError(f"cannot start D435i RGB-D stream: {exc}") from exc
        self._pipeline = pipeline
        self._align = align
        self._info = {
            "serial": serial,
            "color_intrinsics": calibration,
            "record_cloud": self.config.record_cloud,
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
            "depth_scale": float(scale),
            "depth_aligned_to_color": True,
            "depth_invalid_value": 0,
            "timestamp_source": "host_receive_time_ns",
        }
        return dict(self._info)

    def start(self, stop: threading.Event) -> None:
        self._stop = stop
        self._thread = threading.Thread(target=self._loop, args=(stop,), name="camera-realsense", daemon=True)
        self._thread.start()

    def begin(self, color: Path, depth_path: Path, timestamps: Path) -> None:
        episode = Episode.create(
            color, depth_path, timestamps, self.config, self._info, self._unit_rays,
        )
        with self._lock:
            self._episode = episode

    def detach(self) -> None:
        with self._lock:
            if self._episode is not None:
                self._ending, self._episode = self._episode, None

    def end(self) -> int:
        self.detach()
        episode, self._ending = self._ending, None
        if episode is None:
            return 0
        return episode.close()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def check(self) -> None:
        with self._lock:
            if self._episode is not None and self._episode.cloud is not None:
                self._episode.cloud.check()
        if self._error:
            raise CameraError(f"D435i: {self._error}")

    @property
    def latest(self) -> np.ndarray | None:
        return self._latest

    @property
    def frame_count(self) -> int:
        return self._episode.count if self._episode is not None else 0

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def _loop(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
                timestamp = time.time_ns()
                aligned = self._align.process(frames)
                color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.ascontiguousarray(np.asanyarray(color_frame.get_data()))
                depth_values = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()))
                if color.shape != (self.config.height, self.config.width, 3):
                    raise CameraError(f"unexpected D435i color shape: {color.shape}")
                if depth_values.shape != (self.config.height, self.config.width) or depth_values.dtype != np.uint16:
                    raise CameraError(f"unexpected D435i depth format: {depth_values.shape}/{depth_values.dtype}")
                self._latest = color.copy()
                with self._lock:
                    if self._episode is not None:
                        self._episode.color_process.stdin.write(color.tobytes())
                        self._episode.depth_process.stdin.write(depth_values.tobytes())
                        self._episode.timestamps.write(timestamp, color_frame, depth_frame)
                        if self._episode.cloud is not None:
                            self._episode.cloud.submit(
                                self._episode.count, timestamp, depth_values, color,
                            )
                        self._episode.count += 1
        except Exception as exc:  # noqa: BLE001 - surface background thread failures
            if not stop.is_set():
                self._error = str(exc)
