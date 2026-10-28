#!/usr/bin/env bash
# Teleoperate a PSM with the TouchX (hold the front button to follow, the back button to close the jaw).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${DVRK_SETUP:-$HOME/ros2_ws/install/setup.bash}"
set -u

export ROS_LOG_DIR="${ROS_LOG_DIR:-$ROOT/.ros}"
mkdir -p "$ROS_LOG_DIR"
cd "$ROOT"
exec uv run --no-sync dvrk-touchx "$@"
