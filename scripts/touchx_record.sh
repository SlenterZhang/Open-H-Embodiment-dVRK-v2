#!/usr/bin/env bash
# Teleoperate with the TouchX in the background and record episodes in the foreground
# (s = start, e = end, q = quit). Arguments go to the recorder; TOUCHX_CONFIG picks the teleop config.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${DVRK_SETUP:-$HOME/ros2_ws/install/setup.bash}"
set -u

export ROS_LOG_DIR="${ROS_LOG_DIR:-$ROOT/.ros}"
mkdir -p "$ROS_LOG_DIR"
cd "$ROOT"

TELEOP_LOG="$ROS_LOG_DIR/touchx_teleop.log"
uv run --no-sync dvrk-touchx -c "${TOUCHX_CONFIG:-configs/touchx.yaml}" >"$TELEOP_LOG" 2>&1 &
TELEOP_PID=$!
trap 'kill -INT "$TELEOP_PID" 2>/dev/null || true; wait "$TELEOP_PID" 2>/dev/null || true' EXIT

sleep 2
if ! kill -0 "$TELEOP_PID" 2>/dev/null; then
  echo "error: TouchX teleop exited, see $TELEOP_LOG" >&2
  cat "$TELEOP_LOG" >&2
  exit 1
fi
echo "TouchX teleop running (pid $TELEOP_PID, log: $TELEOP_LOG)"

uv run --no-sync dvrk-lerobot record "$@"
