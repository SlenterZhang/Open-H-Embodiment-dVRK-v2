"""Command line entry point for TouchX teleoperation."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Load the configuration and spin the teleoperation node until Ctrl+C."""
    parser = argparse.ArgumentParser(prog="dvrk-touchx", description=__doc__)
    parser.add_argument("-c", "--config", default="configs/touchx.yaml")
    args = parser.parse_args(argv)

    from dvrk_touchx.config import load_config

    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import rclpy

    from dvrk_touchx.node import TouchXTeleop

    rclpy.init()
    node = TouchXTeleop(config)
    node.get_logger().info(
        f"ready: hold {config.touchx}/button1 to follow, button2 to close the jaw"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
