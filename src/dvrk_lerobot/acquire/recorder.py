"""ROS 2 recorder for measured_js / measured_cp and two V4L2 cameras."""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

import rclpy

from dvrk_lerobot.acquire.camera import Camera
from dvrk_lerobot.acquire.metadata import build_meta, check_cameras, new_session_dir
from dvrk_lerobot.acquire.node import RecorderNode
from dvrk_lerobot.acquire.session import RawSession
from dvrk_lerobot.config import AppConfig

WARMUP_TIMEOUT_S = 20.0


def run_record(
    config: AppConfig,
    root: Path,
    *,
    task: str,
    operator: str = "",
    notes: str = "",
    duration: float = 0.0,
) -> Path:
    """Record one raw session and return its directory."""
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init()
    node = RecorderNode(config)
    cameras = {side: Camera(config.cameras[side]) for side in config.camera_sides}
    stop = threading.Event()
    session: RawSession | None = None
    status = "ok"
    previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        print("[record] waiting for robot topics")
        _wait_ready(node, WARMUP_TIMEOUT_S)
        print("[record] opening cameras")
        camera_info = {side: camera.open() for side, camera in cameras.items()}
        check_cameras(camera_info)
        session_dir = new_session_dir(root, task)
        session = RawSession.create(
            session_dir, build_meta(config, session_dir, task, operator, notes, camera_info)
        )
        node.session = session
        for side, camera in cameras.items():
            camera.start(session_dir / "frames" / side, stop)
        _capture_loop(node, cameras, stop, duration)
    except KeyboardInterrupt:
        print("\n[record] interrupted")
    except Exception:
        status = "failed"
        raise
    finally:
        node.session = None
        stop.set()
        for camera in cameras.values():
            camera.stop()
        if session is not None:
            session.finalize(
                status=status,
                ended_ns=time.time_ns(),
                frame_counts={side: camera.frame_count for side, camera in cameras.items()},
                camera_info={
                    side: {**camera.info, "frame_count": camera.frame_count}
                    for side, camera in cameras.items()
                },
                joint_names=node.joint_names,
                frame_ids=node.frame_ids,
            )
            print(f"[record] saved {session.root}")
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm)
    if session is None:
        raise RuntimeError("recording did not create a session")
    return session.root


def _capture_loop(
    node: RecorderNode, cameras: dict[str, Camera], stop: threading.Event, duration: float
) -> None:
    started = time.monotonic()
    last_report = started
    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.02)
        for camera in cameras.values():
            camera.check()
        elapsed = time.monotonic() - started
        if duration > 0 and elapsed >= duration:
            break
        now = time.monotonic()
        if now - last_report >= 5.0:
            counts = " ".join(f"{side}={camera.frame_count}" for side, camera in cameras.items())
            print(f"[record] {elapsed:6.1f}s frames {counts}")
            last_report = now


def _wait_ready(node: RecorderNode, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.ready():
            return
    missing = ", ".join(node.missing()) or "unknown topics"
    raise TimeoutError(f"no data within {timeout:.0f}s; missing: {missing}")
