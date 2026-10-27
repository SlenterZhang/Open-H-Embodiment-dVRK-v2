"""ROS 2 subscriptions that forward dVRK telemetry into a raw session."""

from __future__ import annotations

import math
import time

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from dvrk_lerobot.acquire.metadata import stamp_ns
from dvrk_lerobot.acquire.session import RawSession
from dvrk_lerobot.config import AppConfig


class RecorderNode(Node):
    """Subscribe to every configured arm and forward samples to a raw session."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__("dvrk_lerobot_recorder")
        self.config = config
        self.session: RawSession | None = None
        self.joint_names: dict[str, list[str]] = {}
        self.frame_ids: dict[str, str] = {}
        self.seen: dict[str, set[str]] = {arm.name: set() for arm in config.arms}
        self._warned: set[tuple[str, str]] = set()
        for arm in config.arms:
            self.create_subscription(
                JointState,
                arm.js_topic,
                lambda message, current=arm: self._on_js(current, message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                PoseStamped,
                arm.cp_topic,
                lambda message, current=arm: self._on_cp(current, message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                JointState,
                arm.jaw_topic,
                lambda message, current=arm: self._on_jaw(current, message),
                qos_profile_sensor_data,
            )

    def ready(self) -> bool:
        return all({"jp", "cp", "jaw"} <= kinds for kinds in self.seen.values())

    def missing(self) -> list[str]:
        """Human-readable list of topics that have not produced data yet."""
        missing: list[str] = []
        for arm in self.config.arms:
            if "jp" not in self.seen[arm.name]:
                missing.append(arm.js_topic)
            if "cp" not in self.seen[arm.name]:
                missing.append(arm.cp_topic)
            if "jaw" not in self.seen[arm.name]:
                missing.append(arm.jaw_topic)
        return missing

    def _on_js(self, arm, message: JointState) -> None:
        received = time.time_ns()
        stamp = stamp_ns(message.header.stamp)
        self._warn_clock(arm.name, "jp", stamp, received)
        if not message.position:
            self._warn_once(arm.name, "jp-empty", f"{arm.js_topic} has no positions")
            return
        positions = list(message.position)
        names = list(message.name)
        if len(names) != len(positions):
            if names:
                self._warn_once(arm.name, "jp-names", "joint names and positions differ in length")
            names = [f"joint_{index}" for index in range(len(positions))]
        self.joint_names.setdefault(arm.name, names)
        self.seen[arm.name].add("jp")
        if self.session is not None:
            self.session.add_robot(arm.name, "jp", received, stamp, positions)

    def _on_jaw(self, arm, message: JointState) -> None:
        received = time.time_ns()
        stamp = stamp_ns(message.header.stamp)
        self._warn_clock(arm.name, "jaw", stamp, received)
        if len(message.position) != 1:
            self._warn_once(arm.name, "jaw-size", f"{arm.jaw_topic} must have one position")
            return
        self.seen[arm.name].add("jaw")
        if self.session is not None:
            self.session.add_robot(arm.name, "jaw", received, stamp, list(message.position))

    def _on_cp(self, arm, message: PoseStamped) -> None:
        received = time.time_ns()
        stamp = stamp_ns(message.header.stamp)
        self._warn_clock(arm.name, "cp", stamp, received)
        if message.header.frame_id:
            self.frame_ids.setdefault(arm.name, message.header.frame_id)
        pose, orientation = message.pose.position, message.pose.orientation
        values = [
            pose.x,
            pose.y,
            pose.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        if not all(math.isfinite(value) for value in values):
            self._warn_once(arm.name, "cp-finite", f"{arm.cp_topic} has non-finite values")
            return
        self.seen[arm.name].add("cp")
        if self.session is not None:
            self.session.add_robot(arm.name, "cp", received, stamp, values)

    def _warn_clock(self, arm: str, kind: str, stamp_ns: int, received_ns: int) -> None:
        if stamp_ns <= 0:
            self._warn_once(arm, f"{kind}-stamp", "header stamp is zero; using receive time")
        elif abs(stamp_ns - received_ns) > 2_000_000_000:
            self._warn_once(
                arm, f"{kind}-clock", "header stamp differs from wall clock by more than 2 s"
            )

    def _warn_once(self, arm: str, key: str, message: str) -> None:
        if (arm, key) not in self._warned:
            self._warned.add((arm, key))
            self.get_logger().warning(f"{arm}: {message}")
