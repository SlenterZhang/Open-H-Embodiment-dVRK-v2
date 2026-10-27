"""Session naming, raw metadata, and camera validation helpers."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from dvrk_lerobot.acquire.camera import CameraError
from dvrk_lerobot.config import AppConfig


def new_session_dir(root: Path, task: str) -> Path:
    """Create a unique, task-labelled session directory under *root*."""
    root.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:40] or "episode"
    base = f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}"
    candidate = root / base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base}-{suffix}"
        suffix += 1
    return candidate


def build_meta(
    config: AppConfig,
    session_dir: Path,
    task: str,
    operator: str,
    notes: str,
    camera_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the initial meta.json payload for one recording."""
    return {
        "session_id": session_dir.name,
        "created_ns": time.time_ns(),
        "task": task,
        "operator": operator,
        "notes": notes,
        "fps": config.dataset.fps,
        "clock": "system",
        "arms": [
            {
                "name": arm.name,
                "js_topic": arm.js_topic,
                "cp_topic": arm.cp_topic,
                "jaw_topic": arm.jaw_topic,
            }
            for arm in config.arms
        ],
        "cameras": {
            side: {
                "device": camera.device,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "fourcc": camera.fourcc,
                "jpeg_quality": camera.jpeg_quality,
                "negotiated": camera_info[side],
            }
            for side, camera in config.cameras.items()
        },
    }


def check_cameras(info: dict[str, dict[str, Any]]) -> None:
    """Require both cameras to agree on geometry and rate."""
    left, right = info["left"], info["right"]
    if (left["width"], left["height"]) != (right["width"], right["height"]):
        raise CameraError("left and right cameras negotiated different resolutions")
    if abs(float(left["fps"]) - float(right["fps"])) > 1.0:
        raise CameraError("left and right cameras negotiated different frame rates")


def stamp_ns(stamp: Any) -> int:
    """Convert a ROS 2 builtin_interfaces/Time to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
