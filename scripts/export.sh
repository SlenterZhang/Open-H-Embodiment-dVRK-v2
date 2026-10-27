#!/usr/bin/env bash
# Export one or more raw sessions into a LeRobot v2.1 dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run --no-sync dvrk-lerobot export "$@"
