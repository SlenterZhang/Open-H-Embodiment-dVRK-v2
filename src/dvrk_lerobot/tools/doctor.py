"""Pre-flight checks for cameras, ffmpeg, and dVRK ROS 2 topics."""

from __future__ import annotations

import shutil
import time
from typing import Any

from dvrk_lerobot.acquire.camera import Camera
from dvrk_lerobot.config import AppConfig


def run_doctor(config: AppConfig, topic_timeout: float = 5.0) -> bool:
    """Run all checks and print a PASS/FAIL report."""
    ok = True
    print("== tools ==")
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        print(f"{'PASS' if path else 'FAIL'} {name}: {path or 'not found'}")
        ok = ok and path is not None

    print("\n== cameras ==")
    for side in config.camera_sides:
        ok = _camera_check(side, config.cameras[side]) and ok

    print("\n== ROS 2 topics ==")
    ok = _topic_check(config, topic_timeout) and ok
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def _camera_check(side: str, config) -> bool:
    camera = Camera(config)
    try:
        info = camera.open()
        print(
            f"PASS {side}: {info['width']}x{info['height']} @ {info['fps']:.1f} fps "
            f"fourcc={info['fourcc']!r} ({config.device})"
        )
        return True
    except Exception as exc:  # noqa: BLE001 - report any device failure
        print(f"FAIL {side}: {exc}")
        return False
    finally:
        camera.stop()


def _topic_check(config: AppConfig, timeout: float) -> bool:
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        print(f"FAIL cannot import ROS 2 Python packages: {exc}")
        return False

    rclpy.init()
    node = rclpy.create_node("dvrk_lerobot_doctor")
    counts: dict[str, int] = {}
    stamps: dict[str, int] = {}
    received: dict[str, int] = {}

    def record(topic: str, message: Any) -> None:
        counts[topic] = counts.get(topic, 0) + 1
        received.setdefault(topic, time.time_ns())
        stamps.setdefault(
            topic, int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        )

    topics: list[str] = []
    try:
        for arm in config.arms:
            topics.extend((arm.js_topic, arm.jaw_topic, arm.cp_topic))
            for js_topic in (arm.js_topic, arm.jaw_topic):
                node.create_subscription(
                    JointState,
                    js_topic,
                    lambda message, topic=js_topic: record(topic, message),
                    qos_profile_sensor_data,
                )
            node.create_subscription(
                PoseStamped,
                arm.cp_topic,
                lambda message, topic=arm.cp_topic: record(topic, message),
                qos_profile_sensor_data,
            )
        print(f"listening for {timeout:.1f}s ...")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    ok = True
    for topic in topics:
        count = counts.get(topic, 0)
        frequency = count / timeout if timeout > 0 else 0.0
        state = "PASS" if count else "FAIL"
        print(f"{state} {topic}: {count} messages ({frequency:.1f} Hz)")
        if count and stamps.get(topic, 0) > 0:
            skew_ms = (received[topic] - stamps[topic]) / 1e6
            print(f"       first receive time minus header stamp: {skew_ms:.1f} ms")
        ok = ok and count > 0
    return ok
