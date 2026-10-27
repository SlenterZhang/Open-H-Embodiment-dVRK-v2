"""Session naming, raw metadata, and camera validation helpers."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dvrk_lerobot.acquire.camera import CameraError
from dvrk_lerobot.config import AppConfig


def new_run_dir(root: Path) -> Path:
    """Create the datetime-named directory that holds one run's episode_1..n."""
    run = root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    return run


def build_meta(
    config: AppConfig,
    session_dir: Path,
    task: str,
    notes: str,
    camera_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the initial meta.json payload for one recording."""
    return {
        "session_id": f"{session_dir.parent.name}/{session_dir.name}",
        "created_ns": time.time_ns(),
        "task": task,
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
                "crf": camera.crf,
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
