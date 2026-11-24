"""Export aligned episodes into the LeRobot Dataset v2.1 layout."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dvrk_lerobot.align.sync import Episode
from dvrk_lerobot.config import AppConfig
from dvrk_lerobot.dataset import metadata as meta_io
from dvrk_lerobot.dataset import schema, stats, video
from dvrk_lerobot.dataset.parquet import write_episode_table


class ExportError(RuntimeError):
    """The episodes cannot be represented as a consistent v2.1 dataset."""


@dataclass(frozen=True)
class ExportResult:
    """Summary returned to the CLI after a successful export."""

    root: Path
    episodes: int
    frames: int


def export_dataset(episodes: Sequence[Episode], config: AppConfig, output: Path) -> ExportResult:
    """Write all episodes under *output*; refuse to reuse an existing path."""
    items = list(episodes)
    if not items:
        raise ExportError("no episodes to convert")
    names = items[0].state_names
    has_depth = items[0].depth is not None
    if has_depth != (config.realsense is not None):
        raise ExportError("RGB-D presence differs from the conversion config")
    for episode in items:
        if episode.state_names != names:
            raise ExportError("episodes have different observation.state names")
        if len(episode.left) != len(episode.state) or len(episode.right) != len(episode.state):
            raise ExportError(f"{episode.session}: frame count does not match state count")
        if (episode.depth is not None) != has_depth:
            raise ExportError("cannot mix RGB-D and stereo-only episodes")
        if has_depth and (
            episode.external is None or len(episode.external) != len(episode.state)
            or episode.depth_shape != items[0].depth_shape
            or episode.depth_scale != items[0].depth_scale
            or episode.depth_intrinsics != items[0].depth_intrinsics
            or episode.depth_shape != (config.realsense.height, config.realsense.width)
        ):
            raise ExportError(f"{episode.session}: incompatible RGB-D shape, scale, intrinsics, or frame count")

    destination = output.expanduser().absolute()
    partial = destination.with_name(destination.name + ".partial")
    if _exists(destination):
        raise ExportError(f"output already exists: {destination}")
    if _exists(partial):
        raise ExportError(f"partial output already exists: {partial}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()

    tasks: dict[str, int] = {}
    episode_rows: list[dict] = []
    episode_stats: list[dict] = []
    video_info: dict | None = None
    total_frames = 0
    try:
        for episode_index, episode in enumerate(items):
            task = episode.task.strip()
            if not task:
                raise ExportError(f"{episode.session}: task is empty")
            task_index = tasks.setdefault(task, len(tasks))
            indices = {"left": episode.left, "right": episode.right}
            if has_depth:
                indices["external"] = episode.external
            current_video_info = {
                side: video.encode(
                    episode.videos[side], selected,
                    partial / schema.video_path(episode_index, side),
                    fps=config.dataset.fps,
                    crf=config.dataset.video_crf,
                    preset=config.dataset.video_preset,
                )
                for side, selected in indices.items()
            }
            _check_video_pair(current_video_info["left"], current_video_info["right"], episode.session)
            if video_info is None:
                video_info = current_video_info
            elif any(not _same_video(video_info[side], current_video_info[side]) for side in indices):
                raise ExportError(f"{episode.session}: video format differs from episode 0")

            write_episode_table(
                partial / schema.data_path(episode_index),
                state=episode.state,
                action=episode.action,
                timestamp=episode.timestamps,
                episode_index=episode_index,
                task_index=task_index,
                index_offset=total_frames,
                depth_source=episode.depth,
                depth_indices=episode.external,
                depth_shape=episode.depth_shape,
            )
            episode_stats.append(_stats(episode, episode_index, task_index, total_frames))
            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task],
                    "length": len(episode.state),
                }
            )
            total_frames += len(episode.state)

        if video_info is None:
            raise ExportError("no videos were encoded")
        features = schema.build_features(
            names, video_info,
            depth_shape=items[0].depth_shape if has_depth else None,
            depth_scale=items[0].depth_scale if has_depth else None,
            depth_intrinsics=items[0].depth_intrinsics if has_depth else None,
        )
        info = schema.info_dict(
            fps=config.dataset.fps,
            robot_type=config.dataset.robot_type,
            features=features,
            total_episodes=len(items),
            total_frames=total_frames,
            total_tasks=len(tasks),
        )
        ordered_tasks = [(index, task) for task, index in tasks.items()]
        meta_io.write_metadata(
            partial,
            info=info,
            tasks=ordered_tasks,
            episodes=episode_rows,
            episode_stats=episode_stats,
        )
    except Exception as exc:
        # Keep the partial tree for inspection; never delete user data.
        raise ExportError(f"convert failed: {exc}; partial output kept at {partial}") from exc
    os.replace(partial, destination)
    return ExportResult(root=destination, episodes=len(items), frames=total_frames)


def _stats(episode: Episode, episode_index: int, task_index: int, index_offset: int) -> dict:
    frames = len(episode.state)
    result = {
        "observation.state": stats.numeric_stats(episode.state),
        "action": stats.numeric_stats(episode.action),
        "observation.images.left": stats.video_stats(episode.videos["left"], episode.left),
        "observation.images.right": stats.video_stats(episode.videos["right"], episode.right),
        "timestamp": stats.numeric_stats(episode.timestamps),
        "frame_index": stats.numeric_stats(np.arange(frames, dtype=np.int64)),
        "episode_index": stats.numeric_stats(np.full(frames, episode_index, dtype=np.int64)),
        "index": stats.numeric_stats(
            np.arange(index_offset, index_offset + frames, dtype=np.int64)
        ),
        "task_index": stats.numeric_stats(np.full(frames, task_index, dtype=np.int64)),
    }
    if episode.depth is not None:
        result["observation.images.external"] = stats.video_stats(episode.videos["external"], episode.external)
        result["observation.depth.external"] = stats.depth_stats(
            episode.depth, episode.external, episode.depth_shape
        )
    return result


def _check_video_pair(left: dict, right: dict, session: Path) -> None:
    for key in ("width", "height", "codec", "pix_fmt", "fps", "channels"):
        if left[key] != right[key]:
            raise ExportError(f"{session}: left/right video differ in {key}")


def _same_video(reference: dict, other: dict) -> bool:
    keys = ("width", "height", "codec", "pix_fmt", "fps", "channels")
    return all(reference[key] == other[key] for key in keys)


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()
