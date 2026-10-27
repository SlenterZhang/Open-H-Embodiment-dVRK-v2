"""Read-only structural checks for an exported LeRobot v2.1 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from dvrk_lerobot.dataset import schema, video


class VerifyError(RuntimeError):
    """The dataset does not satisfy the expected v2.1 structure."""


def verify_dataset(path: str | Path) -> dict[str, Any]:
    """Validate metadata, parquet tables, and video files; return a summary."""
    root = Path(path)
    info = _read_json(root / "meta" / "info.json")
    if info.get("codebase_version") != schema.CODEBASE_VERSION:
        raise VerifyError(f"unexpected codebase_version: {info.get('codebase_version')}")
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    episode_stats = _read_jsonl(root / "meta" / "episodes_stats.jsonl")
    if len(episode_stats) != len(episodes):
        raise VerifyError("episodes.jsonl and episodes_stats.jsonl disagree")

    features = info["features"]
    state_names = features["observation.state"]["names"]
    dimension = int(features["observation.state"]["shape"][0])
    if len(state_names) != dimension:
        raise VerifyError("observation.state names and shape disagree")
    if list(features["action"]["names"]) != list(state_names):
        raise VerifyError("action names differ from observation.state")
    fps = int(info["fps"])
    task_indices = {int(item["task_index"]) for item in tasks}
    total_frames = 0
    offset = 0
    for episode in episodes:
        index = int(episode["episode_index"])
        length = int(episode["length"])
        table = pq.read_table(root / schema.data_path(index))
        _check_table(table, index, length, offset, dimension, fps, task_indices)
        for side in ("left", "right"):
            relative = schema.video_path(index, side)
            video_path = root / relative
            if not video_path.is_file():
                raise VerifyError(f"missing video: {relative}")
            probed = video.probe(video_path)
            feature = features[f"observation.images.{side}"]
            expected = (feature["shape"][1], feature["shape"][0])
            if (probed["width"], probed["height"]) != expected:
                raise VerifyError(f"{relative}: unexpected video dimensions")
        total_frames += length
        offset += length

    if total_frames != int(info["total_frames"]):
        raise VerifyError("info.json total_frames does not match episode lengths")
    if len(episodes) != int(info["total_episodes"]):
        raise VerifyError("info.json total_episodes does not match episodes.jsonl")
    if len(tasks) != int(info["total_tasks"]):
        raise VerifyError("info.json total_tasks does not match tasks.jsonl")
    if int(info["total_videos"]) != len(episodes) * 2:
        raise VerifyError("info.json total_videos is inconsistent")
    return {
        "root": str(root),
        "episodes": len(episodes),
        "frames": total_frames,
        "tasks": len(tasks),
        "fps": fps,
        "state_dimension": dimension,
    }


def _check_table(
    table,
    episode_index: int,
    length: int,
    offset: int,
    dimension: int,
    fps: int,
    task_indices: set[int],
) -> None:
    expected = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    if table.column_names != expected:
        raise VerifyError(f"episode {episode_index}: columns are {table.column_names}")
    if table.num_rows != length:
        raise VerifyError(f"episode {episode_index}: {table.num_rows} rows, expected {length}")
    state_type = table.schema.field("observation.state").type
    if not hasattr(state_type, "list_size") or state_type.list_size != dimension:
        raise VerifyError(f"episode {episode_index}: state is not a fixed-size list")
    if not np.array_equal(table["episode_index"].to_numpy(), np.full(length, episode_index)):
        raise VerifyError(f"episode {episode_index}: episode_index column is wrong")
    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
        raise VerifyError(f"episode {episode_index}: frame_index column is not 0..N-1")
    if not np.array_equal(table["index"].to_numpy(), np.arange(offset, offset + length)):
        raise VerifyError(f"episode {episode_index}: global index is not contiguous")
    timestamps = table["timestamp"].to_numpy()
    if not np.allclose(timestamps, np.arange(length) / fps, atol=1e-4):
        raise VerifyError(f"episode {episode_index}: timestamps are not k/fps")
    if not set(table["task_index"].to_numpy().tolist()) <= task_indices:
        raise VerifyError(f"episode {episode_index}: unknown task_index")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerifyError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VerifyError(f"missing file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
