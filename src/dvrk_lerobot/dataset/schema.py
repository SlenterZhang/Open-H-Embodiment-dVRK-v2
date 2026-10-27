"""LeRobot Dataset v2.1 feature names, paths, and info.json body."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dvrk_lerobot.config import ArmConfig

CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")

DEFAULT_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def state_names(arms: Sequence[ArmConfig], joint_names: Mapping[str, Sequence[str]]) -> list[str]:
    """Build the flat observation vector names: joints, jaw, then pose."""
    names: list[str] = []
    for arm in arms:
        joints = list(joint_names.get(arm.name, []))
        if not joints:
            raise ValueError(f"no joint names recorded for {arm.name}")
        names.extend(f"{arm.name}.jp.{joint}" for joint in joints)
        names.append(f"{arm.name}.jaw")
        names.extend(f"{arm.name}.cp.{part}" for part in POSE_NAMES)
    return names


def build_features(
    state_names_: Sequence[str], videos: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return info.json features, with video features inserted before defaults."""
    names = list(state_names_)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
        "action": {"dtype": "float32", "shape": [len(names)], "names": list(names)},
    }
    for side in ("left", "right"):
        info = videos[side]
        features[f"observation.images.{side}"] = {
            "dtype": "video",
            "shape": [int(info["height"]), int(info["width"]), 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": int(info["height"]),
                "video.width": int(info["width"]),
                "video.codec": str(info["codec"]),
                "video.pix_fmt": str(info["pix_fmt"]),
                "video.is_depth_map": False,
                "video.fps": int(info["fps"]),
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update({key: dict(value) for key, value in DEFAULT_FEATURES.items()})
    return features


def info_dict(
    *,
    fps: int,
    robot_type: str,
    features: Mapping[str, Mapping[str, Any]],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> dict[str, Any]:
    """Build the complete meta/info.json payload."""
    total_videos = total_episodes * 2
    total_chunks = (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE if total_episodes else 0
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "total_videos": int(total_videos),
        "total_chunks": int(total_chunks),
        "chunks_size": CHUNKS_SIZE,
        "fps": int(fps),
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {key: dict(value) for key, value in features.items()},
    }


def data_path(episode_index: int) -> str:
    """Relative parquet path for one episode."""
    return DATA_PATH.format(episode_chunk=episode_index // CHUNKS_SIZE, episode_index=episode_index)


def video_path(episode_index: int, side: str) -> str:
    """Relative video path for one episode and camera side."""
    return VIDEO_PATH.format(
        episode_chunk=episode_index // CHUNKS_SIZE,
        video_key=f"observation.images.{side}",
        episode_index=episode_index,
    )
