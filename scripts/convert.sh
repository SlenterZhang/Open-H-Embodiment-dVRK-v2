#!/usr/bin/env bash
# Align raw episodes in time and convert them into a LeRobot v2.1 (default) or
# v3.0 (--format v3.0) dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run --no-sync dvrk-lerobot convert "$@"
