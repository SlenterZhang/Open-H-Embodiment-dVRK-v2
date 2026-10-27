#!/usr/bin/env bash
# Create a uv environment that can import the system ROS 2 Humble packages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -d .venv ]; then
    if ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg; then
        echo "error: .venv exists without system site packages." >&2
        echo "remove .venv and re-run this script so rclpy can be imported." >&2
        exit 1
    fi
else
    uv venv --system-site-packages --python /usr/bin/python3 .venv
fi

uv sync
echo "environment ready"
echo "next: source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash"
echo "then: uv run --no-sync dvrk-lerobot doctor"
