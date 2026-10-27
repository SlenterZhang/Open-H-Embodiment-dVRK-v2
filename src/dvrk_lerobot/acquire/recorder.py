"""Episode-by-episode recorder: press 's' to start an episode, 'e' to end it, 'q' to quit."""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor

from dvrk_lerobot.acquire import preview
from dvrk_lerobot.acquire.camera import Camera, CameraError
from dvrk_lerobot.acquire.keyboard import Keyboard
from dvrk_lerobot.acquire.metadata import build_meta, check_cameras, new_run_dir
from dvrk_lerobot.acquire.node import RecorderNode
from dvrk_lerobot.acquire.session import RawSession, video_files
from dvrk_lerobot.config import AppConfig

WARMUP_TIMEOUT_S = 20.0
KEYS = "keys (preview window or terminal): [s] start episode   [e] end episode   [q] quit"


def run_record(config: AppConfig, root: Path, *, task: str, notes: str = "") -> list[Path]:
    """Record episodes into <root>/<datetime>/episode_N until 'q'; return the saved ones."""
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init()
    node = RecorderNode(config)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, name="ros-spin", daemon=True)
    cameras = {side: Camera(config.cameras[side]) for side in config.camera_sides}
    stop = threading.Event()
    saved: list[Path] = []
    run_dir: Path | None = None
    session: RawSession | None = None
    number, started, status = 0, 0.0, "aborted"
    previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        print("[record] waiting for robot topics")
        _wait_ready(executor, node, WARMUP_TIMEOUT_S)
        # Spin ROS on its own thread so drawing the preview never delays robot callbacks.
        spinner.start()
        print("[record] opening cameras")
        camera_info = {side: camera.open() for side, camera in cameras.items()}
        check_cameras(camera_info)
        for camera in cameras.values():
            camera.start(stop)
        print(f"[record] ready. {KEYS}")
        with Keyboard() as keyboard:
            while rclpy.ok() and not stop.is_set():
                for camera in cameras.values():
                    camera.check()
                left, right = cameras["left"].latest, cameras["right"].latest
                if left is None or right is None:
                    key = preview.poll_key()
                elif session is None:
                    key = preview.show(left, right, f"IDLE  {len(saved)} saved", False)
                else:
                    frames = cameras["left"].frame_count
                    text = f"REC episode_{number}  {time.monotonic() - started:.1f}s  {frames}"
                    key = preview.show(left, right, text, True)
                key = keyboard.read() or key
                if key == "q":
                    break
                if key == "s" and session is None:
                    run_dir = run_dir or new_run_dir(root)
                    number += 1
                    episode_dir = run_dir / f"episode_{number}"
                    meta = build_meta(config, episode_dir, task, notes, camera_info)
                    session = RawSession.create(episode_dir, meta)
                    _begin(session, node, cameras)
                    started = time.monotonic()
                    print(f"[record] episode_{number} started: {episode_dir}")
                elif key == "e" and session is not None:
                    current, session = session, None
                    saved.append(_end(node, cameras, current, "ok"))
                    print(f"[record] episode_{number} saved ({len(saved)} this run)")
                elif key == "s":
                    print("[record] episode already running; press 'e' to end it")
                elif key == "e":
                    print("[record] no episode running; press 's' to start one")
    except KeyboardInterrupt:
        print("\n[record] interrupted")
    except Exception:
        status = "failed"
        raise
    finally:
        node.attach(None)
        if session is not None:
            try:
                _end(node, cameras, session, status)
            finally:
                print(f"[record] unfinished episode saved as '{status}': {session.root}")
        stop.set()
        for camera in cameras.values():
            camera.stop()
        preview.close()
        executor.shutdown()
        if spinner.is_alive():
            spinner.join(timeout=5.0)
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm)
    return saved


def _begin(session: RawSession, node: RecorderNode, cameras: dict[str, Camera]) -> None:
    # Robot logging brackets the video so edge frames always have robot samples.
    node.attach(session)
    for side, camera in cameras.items():
        camera.begin(*video_files(session.root, side))


def _end(node: RecorderNode, cameras: dict[str, Camera], session: RawSession, status: str) -> Path:
    for camera in cameras.values():
        camera.detach()  # stop both eyes together; flushing an encoder takes time
    counts: dict[str, int] = {}
    error: CameraError | None = None
    for side, camera in cameras.items():
        try:
            counts[side] = camera.end()
        except CameraError as exc:
            counts[side], error, status = 0, exc, "failed"
    node.attach(None)
    session.finalize(
        status=status,
        ended_ns=time.time_ns(),
        frame_counts=counts,
        camera_info={
            side: {**cam.info, "frame_count": counts[side]} for side, cam in cameras.items()
        },
        joint_names=node.joint_names,
        frame_ids=node.frame_ids,
    )
    if error is not None:
        raise error
    return session.root


def _wait_ready(executor: SingleThreadedExecutor, node: RecorderNode, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
        if node.ready():
            return
    missing = ", ".join(node.missing()) or "unknown topics"
    raise TimeoutError(f"no data within {timeout:.0f}s; missing: {missing}")
