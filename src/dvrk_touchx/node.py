"""ROS 2 node that teleoperates a dVRK PSM with a TouchX through servo_cp."""

from __future__ import annotations

import time

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, Joy

from dvrk_touchx.config import TeleopConfig
from dvrk_touchx.retarget import Pose, Retarget, matrix_to_quat, quat_to_matrix


def pose_from_msg(message: PoseStamped) -> Pose:
    p, q = message.pose.position, message.pose.orientation
    return np.array([p.x, p.y, p.z]), quat_to_matrix([q.x, q.y, q.z, q.w])


class TouchXTeleop(Node):
    """Button 1 held: PSM follows the TouchX. Button 2 held: jaw closes.

    Every TouchX sample (500 Hz from sawSensablePhantom) is retargeted and sent
    on servo_cp right away; dVRK solves the PSM IK. Following stops when the
    goal drifts more than `max_error` from the measured PSM position.
    """

    def __init__(self, config: TeleopConfig) -> None:
        super().__init__("dvrk_touchx_teleop")
        self.config = config
        self.retarget = Retarget(np.array(config.rotation), config.scale)
        self._touch: Pose | None = None
        self._psm: Pose | None = None
        self._psm_time = 0.0
        self._clutch = False
        self._jaw_closed = False
        self._frame_id = ""
        self._servo_cp = self.create_publisher(PoseStamped, f"{config.psm}/servo_cp", 10)
        self._jaw_move = self.create_publisher(JointState, f"{config.psm}/jaw/move_jp", 10)
        self.create_subscription(
            PoseStamped, f"{config.touchx}/measured_cp", self._on_touch, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, f"{config.psm}/measured_cp", self._on_psm, qos_profile_sensor_data
        )
        # Reliable QoS so a button edge is never dropped.
        self.create_subscription(Joy, f"{config.touchx}/button1", self._on_clutch, 10)
        self.create_subscription(Joy, f"{config.touchx}/button2", self._on_jaw, 10)

    def _on_psm(self, message: PoseStamped) -> None:
        self._psm = pose_from_msg(message)
        self._psm_time = time.monotonic()
        self._frame_id = message.header.frame_id

    def _on_touch(self, message: PoseStamped) -> None:
        self._touch = pose_from_msg(message)
        if not self.retarget.engaged:
            return
        if time.monotonic() - self._psm_time > self.config.max_age:
            self._release(f"{self.config.psm}/measured_cp is stale")
            return
        position, rotation = self.retarget.goal(self._touch)
        error = float(np.linalg.norm(position - self._psm[0]))
        if error > self.config.max_error:
            self._release(f"goal is {error * 1000:.1f} mm from the measured PSM position")
            return
        self._send_cp(position, rotation)

    def _on_clutch(self, message: Joy) -> None:
        pressed = bool(message.buttons) and message.buttons[0] == 1
        if pressed == self._clutch:
            return
        self._clutch = pressed
        if pressed:
            self._engage()
        elif self.retarget.engaged:
            self._release("clutch released")

    def _on_jaw(self, message: Joy) -> None:
        closed = bool(message.buttons) and message.buttons[0] == 1
        if closed == self._jaw_closed:
            return
        self._jaw_closed = closed
        if self.retarget.engaged:
            self._send_jaw()

    def _engage(self) -> None:
        if self._touch is None:
            self.get_logger().warning(f"cannot engage: no data on {self.config.touchx}/measured_cp")
            return
        if self._psm is None or time.monotonic() - self._psm_time > self.config.max_age:
            self.get_logger().warning(f"cannot engage: no recent {self.config.psm}/measured_cp")
            return
        self.retarget.engage(self._touch, self._psm)
        self._send_jaw()
        self.get_logger().info("following")

    def _release(self, reason: str) -> None:
        # dVRK holds the last servo_cp goal, so the PSM stops where it is.
        self.retarget.release()
        self.get_logger().info(f"released: {reason}")

    def _send_cp(self, position: np.ndarray, rotation: np.ndarray) -> None:
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        message.pose.position.x, message.pose.position.y, message.pose.position.z = position
        q = matrix_to_quat(rotation)
        o = message.pose.orientation
        o.x, o.y, o.z, o.w = (float(value) for value in q)
        self._servo_cp.publish(message)

    def _send_jaw(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.position = [self.config.jaw_closed if self._jaw_closed else self.config.jaw_open]
        self._jaw_move.publish(message)
