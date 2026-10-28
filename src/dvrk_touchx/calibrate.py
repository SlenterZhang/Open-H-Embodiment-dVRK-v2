"""Fit the TouchX `rotation` from paired TouchX and PSM movements."""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np

from dvrk_touchx.retarget import fit_rotation

MIN_TOUCH = 0.01  # m, smallest stylus movement kept
MIN_PSM = 0.005  # m, smallest PSM movement kept

DIRECTIONS = ("RIGHT", "LEFT", "UP", "DOWN", "TOWARD YOU", "AWAY FROM YOU")

INSTRUCTIONS = """\
Teleop must NOT be running. Directions are as seen from your seat (or on the
endoscope view if you teleoperate from it). For each direction:
  1. Hold the stylus and the PSM still, press Enter.
  2. Move the stylus 3-5 cm that way, then move the PSM tip (hand clutch on the
     arm) the same way. Move in a straight line. Press Enter.
Type s at the first prompt to skip a direction, r to redo it."""


def to_yaml(rotation: np.ndarray) -> str:
    """`rotation:` block for configs/touchx.yaml (full precision, passes validation)."""
    rows = ("  - [" + ", ".join(f"{v:.9f}" for v in row) + "]" for row in rotation)
    return "rotation:\n" + "\n".join(rows)


def mm(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{v * 1000:+6.1f}" for v in vector) + "] mm"


def record_pairs(listener) -> tuple[np.ndarray, np.ndarray]:
    """Guide one move per direction; return N x 3 TouchX and PSM displacements."""
    touch, psm = [], []
    pending = list(DIRECTIONS)
    while pending:
        direction = pending[0]
        if input(f"{direction}: hold still, Enter (s skip): ").strip() == "s":
            pending.pop(0)
            continue
        start = listener.snapshot()
        input(f"  move stylus {direction}, then PSM tip {direction}, Enter: ")
        end = listener.snapshot()
        if start is None or end is None:
            print("  no data from TouchX or PSM yet, try again")
            continue
        dt, dp = end[0] - start[0], end[1] - start[1]
        print(f"  touch {mm(dt)}   psm {mm(dp)}")
        if np.linalg.norm(dt) < MIN_TOUCH or np.linalg.norm(dp) < MIN_PSM:
            print("  too small, try again")
            continue
        if input("  keep? Enter (r redo): ").strip() == "r":
            continue
        touch.append(dt)
        psm.append(dp)
        pending.pop(0)
    return np.array(touch), np.array(psm)


def report(rotation: np.ndarray, touch: np.ndarray, psm: np.ndarray) -> None:
    """Print the angle between each mapped stylus move and its PSM move, then the YAML."""
    for i, (a, b) in enumerate(zip(touch, psm), 1):
        mapped = rotation @ a
        cos = mapped @ b / (np.linalg.norm(mapped) * np.linalg.norm(b))
        print(f"move {i}: {np.degrees(np.arccos(np.clip(cos, -1, 1))):5.1f} deg off")
    print("\nPaste into configs/touchx.yaml:\n" + to_yaml(rotation))


def main(argv: list[str] | None = None) -> int:
    """Record pairs over ROS 2, fit the rotation and print it."""
    parser = argparse.ArgumentParser(prog="dvrk-touchx-calibrate", description=__doc__)
    parser.add_argument("-c", "--config", default="configs/touchx.yaml")
    args = parser.parse_args(argv)

    from dvrk_touchx.config import load_config

    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    class Listener(Node):
        """Latest TouchX and PSM positions."""

        def __init__(self) -> None:
            super().__init__("dvrk_touchx_calibrate")
            self.touch = self.psm = None
            self.create_subscription(
                PoseStamped, f"{config.touchx}/measured_cp",
                lambda m: setattr(self, "touch", m.pose.position), qos_profile_sensor_data,
            )
            self.create_subscription(
                PoseStamped, f"{config.psm}/measured_cp",
                lambda m: setattr(self, "psm", m.pose.position), qos_profile_sensor_data,
            )

        def snapshot(self):
            if self.touch is None or self.psm is None:
                return None
            return tuple(np.array([p.x, p.y, p.z]) for p in (self.touch, self.psm))

    rclpy.init()
    listener = Listener()
    threading.Thread(target=rclpy.spin, args=(listener,), daemon=True).start()
    print(INSTRUCTIONS)
    try:
        touch, psm = record_pairs(listener)
        rotation = fit_rotation(touch, psm) if len(touch) >= 2 else None
    except (KeyboardInterrupt, EOFError):
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        listener.destroy_node()
        rclpy.try_shutdown()
    if rotation is None:
        print("error: need at least 2 pairs", file=sys.stderr)
        return 1
    report(rotation, touch, psm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
