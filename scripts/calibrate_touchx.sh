#!/usr/bin/env bash
# Fit the TouchX `rotation` from paired TouchX and PSM movements (teleop must be off).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${DVRK_SETUP:-$HOME/ros2_ws/install/setup.bash}"
set -u

export ROS_LOG_DIR="${ROS_LOG_DIR:-$ROOT/.ros}"
mkdir -p "$ROS_LOG_DIR"
cd "$ROOT"
exec uv run --no-sync python -m dvrk_touchx.calibrate "$@"
